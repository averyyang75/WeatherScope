"""
LLM Service - Weather interpretation using Ollama (local) or vLLM (cloud).
Converts numerical precipitation predictions into natural language alerts.
"""

import os
import logging
import subprocess
import time
import math
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "vllm"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_API_PREFIX = os.getenv("VLLM_API_PREFIX", "/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
INTERPRET_MAX_TOKENS = int(os.getenv("INTERPRET_MAX_TOKENS", "192"))
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "WeatherScope/1.0").strip()
NOMINATIM_EMAIL = os.getenv("NOMINATIM_EMAIL", "").strip()

# ============================================================================
# LLM Abstraction Layer
# ============================================================================

class LLMService(ABC):
    """Abstract base class for LLM services."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if LLM service is healthy."""
        pass


class OllamaService(LLMService):
    """Ollama implementation - runs locally on Apple Silicon."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            # Ollama API endpoint
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False


class VLLMService(LLMService):
    """vLLM/OpenAI-compatible implementation."""

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_prefix: str = VLLM_API_PREFIX,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        cleaned_prefix = (api_prefix or "/v1").strip()
        if not cleaned_prefix.startswith("/"):
            cleaned_prefix = f"/{cleaned_prefix}"
        self.api_prefix = cleaned_prefix.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

    def _api_url(self, suffix: str) -> str:
        if not suffix.startswith("/"):
            suffix = f"/{suffix}"
        return f"{self.base_url}{self.api_prefix}{suffix}"

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Prefer text completions, fallback to chat-completions for providers
        # that only expose chat endpoint.
        completion_payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        try:
            response = await self.client.post(
                self._api_url("/completions"),
                json=completion_payload,
            )
            response.raise_for_status()
            choices = response.json().get("choices", [])
            if choices:
                text = choices[0].get("text")
                if isinstance(text, str):
                    return text
            raise ValueError("vLLM completion response missing choices[0].text")
        except httpx.HTTPStatusError as e:
            if e.response is None or e.response.status_code not in (404, 405):
                logger.error(f"vLLM generation error: {e}")
                raise
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise

        chat_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        try:
            response = await self.client.post(
                self._api_url("/chat/completions"),
                json=chat_payload,
            )
            response.raise_for_status()
            choices = response.json().get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    return content
            raise ValueError("vLLM chat response missing choices[0].message.content")
        except Exception as e:
            logger.error(f"vLLM chat generation error: {e}")
            raise

    async def health_check(self) -> bool:
        health_candidates = [
            self._api_url("/models"),
            f"{self.base_url}/models",
            f"{self.base_url}/health",
        ]
        try:
            for candidate in health_candidates:
                response = await self.client.get(candidate)
                if response.status_code == 200:
                    return True
                if response.status_code in (401, 403):
                    # Auth-protected but reachable endpoint.
                    return True
            return False
        except Exception:
            return False


class MockLLMService(LLMService):
    """Mock LLM for testing without actual LLM backend."""

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Extract key info from prompt and generate template response
        if "extreme" in prompt.lower() or "severe" in prompt.lower():
            return """WEATHER ALERT: Severe precipitation detected in the specified region.

Based on the high-resolution WeatherScope analysis, significant rainfall is expected.
Residents should monitor local conditions and be prepared for potential flash flooding.

Key observations:
- Heavy precipitation cells identified
- Rapid accumulation possible
- Urban drainage systems may be overwhelmed

Recommended actions:
- Avoid low-lying areas
- Do not drive through flooded roads
- Monitor local emergency broadcasts

This alert was generated by WeatherScope AI Weather System."""
        else:
            return """WEATHER UPDATE: Normal precipitation patterns detected.

The WeatherScope high-resolution analysis shows typical rainfall conditions for the region.
No significant weather hazards are expected at this time.

Current conditions:
- Light to moderate precipitation
- Normal drainage capacity expected
- Standard precautions advised

This update was generated by WeatherScope AI Weather System."""

    async def health_check(self) -> bool:
        return True


def get_llm_service() -> LLMService:
    """Factory function to get the appropriate LLM service."""
    backend = LLM_BACKEND.lower()

    if backend == "ollama":
        logger.info(f"Using Ollama backend at {OLLAMA_BASE_URL}")
        return OllamaService()
    elif backend == "vllm":
        logger.info(f"Using vLLM backend at {VLLM_BASE_URL} (prefix={VLLM_API_PREFIX})")
        return VLLMService()
    else:
        logger.warning(f"Unknown backend '{backend}', using mock service")
        return MockLLMService()


# ============================================================================
# Ollama Management
# ============================================================================

def is_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_ollama() -> bool:
    """Start Ollama server in the background."""
    try:
        logger.info("Starting Ollama server...")
        # Start ollama serve in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        # Wait for it to start
        for _ in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if is_ollama_running():
                logger.info("Ollama server started successfully")
                return True
        logger.error("Ollama server failed to start within timeout")
        return False
    except FileNotFoundError:
        logger.error("Ollama not found. Install from https://ollama.ai")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False


def is_model_available(model: str) -> bool:
    """Check if the specified model is pulled in Ollama."""
    try:
        import urllib.request
        import json
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = data.get("models", [])
            # Check if model name matches (with or without tag)
            model_base = model.split(":")[0]
            for m in models:
                m_name = m.get("name", "")
                if m_name == model or m_name.startswith(f"{model_base}:"):
                    return True
            return False
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return False


def pull_model(model: str) -> bool:
    """Pull the specified model using Ollama CLI."""
    try:
        logger.info(f"Pulling model '{model}'... (this may take a few minutes)")
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for large models
        )
        if result.returncode == 0:
            logger.info(f"Model '{model}' pulled successfully")
            return True
        else:
            logger.error(f"Failed to pull model: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout pulling model '{model}'")
        return False
    except FileNotFoundError:
        logger.error("Ollama CLI not found. Install from https://ollama.ai")
        return False
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        return False


def ensure_ollama_ready(model: str) -> bool:
    """
    Ensure Ollama is running and the model is available.

    Returns True if ready, False otherwise.
    """
    # Step 1: Check if Ollama is running, start if not
    if not is_ollama_running():
        logger.warning("Ollama is not running")
        if not start_ollama():
            return False
    else:
        logger.info("Ollama server is running")

    # Step 2: Check if model is available, pull if not
    if not is_model_available(model):
        logger.warning(f"Model '{model}' not found")
        if not pull_model(model):
            return False
    else:
        logger.info(f"Model '{model}' is available")

    return True


# ============================================================================
# Weather Alert Generation
# ============================================================================

def build_weather_prompt(prediction_data: Dict) -> str:
    """Build a prompt for weather interpretation."""

    severity = prediction_data.get("severity", "unknown")
    max_precip = prediction_data.get("max_precipitation", 0)
    region = prediction_data.get("region", "the specified area")
    forecast_hour = prediction_data.get("forecast_hour", 0)
    affected_pct = prediction_data.get("affected_percentage", 0)

    prompt = f"""You are a professional meteorologist providing weather alerts. Based on the following high-resolution precipitation forecast data, generate a concise weather alert.

FORECAST DATA:
- Region: {region}
- Forecast Time: +{forecast_hour} hours from now
- Maximum Precipitation: {max_precip:.1f} mm/hr
- Severity Level: {severity.upper()}
- Affected Area: {affected_pct:.1f}% of region

SEVERITY DEFINITIONS:
- NONE: < 10 mm/hr (normal conditions)
- MODERATE: 10-25 mm/hr (increased rainfall)
- SEVERE: 25-50 mm/hr (heavy rain, localized flooding possible)
- EXTREME: > 50 mm/hr (flash flood risk, dangerous conditions)

Please provide:
1. A clear alert headline
2. Brief description of expected conditions
3. Specific risks based on severity
4. Recommended actions for residents

Keep the response concise (under 200 words) and actionable."""

    return prompt


def _flatten_2d(values: List[List[float]]) -> List[float]:
    return [float(v) for row in values for v in row]


def _resolve_downscale_payload(payload: Dict) -> Dict:
    """Allow either direct /downscale payload or wrapped pipeline payload."""
    if isinstance(payload.get("downscale"), dict):
        return payload["downscale"]
    return payload

def _resolve_region_from_weather_condition(payload: Dict[str, Any]) -> Optional[str]:
    """Resolve region label from downscale/extraction metadata."""
    if not isinstance(payload, dict):
        return None

    region_info = payload.get("region_info")
    if isinstance(region_info, dict):
        name = region_info.get("name")
        key = region_info.get("key")
        if isinstance(name, str) and name.strip():
            return name.strip()
        if isinstance(key, str) and key.strip():
            return key.strip()

    for field in ("region_name", "region"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _extract_var_stats(downscale_payload: Dict, var_name: str) -> Dict[str, Optional[float]]:
    """
    Extract min/max/mean for a variable from /downscale output.
    Prefer output_* stats, fallback to input_* stats, then compute from predictions.
    """
    stats = downscale_payload.get("stats", {}).get(var_name, {})
    min_v = stats.get("output_min", stats.get("input_min"))
    max_v = stats.get("output_max", stats.get("input_max"))
    mean_v = stats.get("output_mean", stats.get("input_mean"))

    if min_v is None or max_v is None or mean_v is None:
        predictions = downscale_payload.get("predictions", {}).get(var_name)
        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            flat = _flatten_2d(predictions)
            if flat:
                min_v = min_v if min_v is not None else min(flat)
                max_v = max_v if max_v is not None else max(flat)
                mean_v = mean_v if mean_v is not None else (sum(flat) / len(flat))

    return {"min": min_v, "max": max_v, "mean": mean_v}


def _normalize_temperature_c(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    # FourCastNet/downscale temperature is typically Kelvin; convert when likely K.
    return v - 273.15 if v > 200 else v


def _normalize_pressure_hpa(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    # msl is typically Pa in this pipeline.
    return v / 100 if v > 2000 else v


def _classify_severity(
    t2m_max_c: Optional[float],
    t2m_min_c: Optional[float],
    wind_speed: Optional[float],
    msl_hpa: Optional[float],
) -> str:
    score = 0
    if wind_speed is not None:
        if wind_speed >= 15:
            score += 2
        elif wind_speed >= 10:
            score += 1
    if msl_hpa is not None:
        if msl_hpa < 1000:
            score += 2
        elif msl_hpa < 1010:
            score += 1
    if (t2m_max_c is not None and t2m_max_c >= 35) or (t2m_min_c is not None and t2m_min_c <= -5):
        score += 1

    if score >= 4:
        return "extreme"
    if score >= 2:
        return "severe"
    if score == 1:
        return "moderate"
    return "none"


def build_inference_prompt(
    region: str,
    forecast_hour: int,
    downscale_payload: Dict,
    severity: str,
    max_precipitation: Optional[float] = None,
    customer_text: Optional[str] = None,
    activity: Optional[str] = None,
) -> str:
    """Build a prompt directly from inference-service /downscale output."""
    t2m = _extract_var_stats(downscale_payload, "t2m")
    u10 = _extract_var_stats(downscale_payload, "u10")
    v10 = _extract_var_stats(downscale_payload, "v10")
    msl = _extract_var_stats(downscale_payload, "msl")
    tp = _extract_var_stats(downscale_payload, "tp")

    t2m_min_c = _normalize_temperature_c(t2m["min"])
    t2m_max_c = _normalize_temperature_c(t2m["max"])
    t2m_mean_c = _normalize_temperature_c(t2m["mean"])
    msl_mean_hpa = _normalize_pressure_hpa(msl["mean"])
    tp_max = tp["max"]

    wind_speed = None
    if u10["mean"] is not None and v10["mean"] is not None:
        wind_speed = math.sqrt(float(u10["mean"]) ** 2 + float(v10["mean"]) ** 2)

    method = downscale_payload.get("method", "unknown")
    upscale_factor = downscale_payload.get("upscale_factor", "unknown")
    input_shape = downscale_payload.get("input_shape", "unknown")
    output_shape = downscale_payload.get("output_shape", "unknown")

    def fmt(v: Optional[float], ndigits: int = 1) -> str:
        return f"{v:.{ndigits}f}" if v is not None else "N/A"

    available_vars = set()
    if isinstance(downscale_payload.get("stats"), dict):
        available_vars.update(downscale_payload.get("stats", {}).keys())
    if isinstance(downscale_payload.get("predictions"), dict):
        available_vars.update(downscale_payload.get("predictions", {}).keys())
    available_vars_text = ", ".join(sorted(available_vars)) if available_vars else "unknown"

    customer_text_norm = _normalize_optional_text(customer_text)
    activity_norm = _normalize_optional_text(activity)
    customer_context_lines: List[str] = []
    if customer_text_norm:
        customer_context_lines.append(
            f"- Original customer request: {json.dumps(customer_text_norm, ensure_ascii=True)}"
        )
    if activity_norm:
        customer_context_lines.append(f"- Planned activity: {activity_norm}")

    customer_context_block = ""
    if customer_context_lines:
        customer_context_block = "CUSTOMER CONTEXT:\n" + "\n".join(customer_context_lines) + "\n\n"

    activity_guidance = ""
    if customer_context_lines:
        activity_guidance = (
            "Activity impact requirement:\n"
            "- Explain how the forecast affects the user's planned activity.\n"
            "- Include concrete precautions or plan adjustments for that activity.\n\n"
            "- If activity is not explicit, infer it from the original customer request.\n\n"
        )

    return f"""You are a professional meteorologist providing weather advisories.

Use the following WeatherScope downscaling output to write a concise, actionable advisory:

REGION: {region}
FORECAST TIME: +{forecast_hour}h
SEVERITY (derived): {severity.upper()}

DOWNSCALING METADATA:
- Method: {method}
- Upscale factor: {upscale_factor}x
- Input shape: {input_shape}
- Output shape: {output_shape}
- Available variables: {available_vars_text}

WEATHER SIGNALS (from downscaled output):
- 2m Temperature (C): min {fmt(t2m_min_c)}, max {fmt(t2m_max_c)}, mean {fmt(t2m_mean_c)}
- 10m Wind speed (m/s, from u10/v10 mean): {fmt(wind_speed)}
- Mean Sea Level Pressure (hPa): {fmt(msl_mean_hpa)}
- Max precipitation (mm/hr, tp): {fmt(tp_max)}
- Max precipitation (mm/hr, provided): {fmt(max_precipitation)}

{customer_context_block}{activity_guidance}Interpretation guidance:
- Focus on likely weather impacts for the specified region and forecast time.
- Wind > 10 m/s: strong winds
- MSL < 1000 hPa: potential storm/low-pressure system
- Very high/low temperatures: increased weather stress

Provide:
1) Alert headline
2) 2-3 sentence summary for residents
3) Recommended actions

Keep it under 200 words."""


def derive_severity(
    explicit_severity: Optional[str],
    downscale_payload: Optional[Dict] = None,
    max_precipitation: Optional[float] = None,
) -> str:
    """Derive a severity label from explicit input and/or available inference signals."""
    if explicit_severity:
        return explicit_severity.lower()

    if max_precipitation is not None:
        if max_precipitation > 50:
            return "extreme"
        if max_precipitation >= 25:
            return "severe"
        if max_precipitation >= 10:
            return "moderate"
        return "none"

    if isinstance(downscale_payload, dict):
        t2m = _extract_var_stats(downscale_payload, "t2m")
        u10 = _extract_var_stats(downscale_payload, "u10")
        v10 = _extract_var_stats(downscale_payload, "v10")
        msl = _extract_var_stats(downscale_payload, "msl")
        tp = _extract_var_stats(downscale_payload, "tp")

        wind_speed = None
        if u10["mean"] is not None and v10["mean"] is not None:
            wind_speed = math.sqrt(float(u10["mean"]) ** 2 + float(v10["mean"]) ** 2)

        tp_max = tp["max"]
        if tp_max is not None:
            if tp_max > 50:
                return "extreme"
            if tp_max >= 25:
                return "severe"
            if tp_max >= 10:
                return "moderate"

        return _classify_severity(
            _normalize_temperature_c(t2m["max"]),
            _normalize_temperature_c(t2m["min"]),
            wind_speed,
            _normalize_pressure_hpa(msl["mean"]),
        )

    return "unknown"


LOCATION_BOUNDS_FALLBACK = {
    "north carolina": {"north": 36.6, "south": 33.8, "east": -75.5, "west": -84.3},
    "nc": {"north": 36.6, "south": 33.8, "east": -75.5, "west": -84.3},
    "netherlands": {"north": 53.7, "south": 50.7, "east": 7.3, "west": 3.2},
    "nl": {"north": 53.7, "south": 50.7, "east": 7.3, "west": 3.2},
    # Greater Miami (approximate)
    "miami": {"north": 26.10, "south": 25.30, "east": -80.00, "west": -80.55},
    "miami fl": {"north": 26.10, "south": 25.30, "east": -80.00, "west": -80.55},
    "miami florida": {"north": 26.10, "south": 25.30, "east": -80.00, "west": -80.55},
}

LOCATION_BOUNDS_OVERRIDE = {
    # Metropolitan France only (exclude overseas territories).
    "france": {"north": 51.3056, "south": 41.2611, "east": 9.8282, "west": -5.4518},
    "french republic": {"north": 51.3056, "south": 41.2611, "east": 9.8282, "west": -5.4518},
}


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def _normalize_hours(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return int(round(float(value)))

    text = str(value).strip().lower()
    if not text:
        return None

    if text.isdigit():
        return int(text)
    if text.endswith("h") and text[:-1].strip().isdigit():
        return int(text[:-1].strip())

    unit_match = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h)\b", text)
    if unit_match:
        return int(round(float(unit_match.group(1))))
    day_match = re.search(r"(\d+(?:\.\d+)?)\s*(days?|d)\b", text)
    if day_match:
        return int(round(float(day_match.group(1)) * 24))
    week_match = re.search(r"(\d+(?:\.\d+)?)\s*(weeks?|w)\b", text)
    if week_match:
        return int(round(float(week_match.group(1)) * 24 * 7))

    if "day after tomorrow" in text:
        return 48
    if "tomorrow" in text:
        return 24
    if "tonight" in text:
        return 12
    if "this evening" in text or "this afternoon" in text or "this morning" in text:
        return 6
    if "next week" in text:
        return 168
    if "next weekend" in text:
        return 216
    if "this weekend" in text:
        return 72
    if "today" in text:
        return 0
    return None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = raw.find("{", start + 1)
    return None


def _normalize_coordinate(value: Any, axis: str) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None

    if axis == "lat":
        if num < -90 or num > 90:
            return None
    elif axis == "lon":
        if num < -180 or num > 180:
            return None
    return float(round(num, 4))


def _normalize_bounds(north: Any, south: Any, east: Any, west: Any) -> Dict[str, Optional[float]]:
    n = _normalize_coordinate(north, "lat")
    s = _normalize_coordinate(south, "lat")
    e = _normalize_coordinate(east, "lon")
    w = _normalize_coordinate(west, "lon")
    if None in (n, s, e, w):
        return {"north": None, "south": None, "east": None, "west": None}
    if n <= s:
        return {"north": None, "south": None, "east": None, "west": None}
    return {"north": n, "south": s, "east": e, "west": w}


def _location_override_bounds(location: Optional[str]) -> Dict[str, Optional[float]]:
    if not location:
        return {"north": None, "south": None, "east": None, "west": None}
    key = re.sub(r"\s+", " ", location.strip().lower()).strip(" .,:;")
    bounds = LOCATION_BOUNDS_OVERRIDE.get(key)
    if not bounds:
        return {"north": None, "south": None, "east": None, "west": None}
    return {
        "north": float(bounds["north"]),
        "south": float(bounds["south"]),
        "east": float(bounds["east"]),
        "west": float(bounds["west"]),
    }


def _fallback_location(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\b(?:in|at|near|around)\s+([a-z0-9][a-z0-9\s,.-]{1,80})", text, re.IGNORECASE)
    if not match:
        return None
    candidate = match.group(1).strip(" .,:;")
    candidate = re.split(
        r"\b(?:for|to|on|during|while|with|if|when|because|and)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,:;")
    return _normalize_optional_text(candidate)


def _fallback_activity(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(
        r"\b(?:plan(?:ning)?|going|intend|want|need|will)\s+(?:to\s+)?([a-z][a-z\s-]{2,80})",
        text,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(r"\bfor\s+([a-z][a-z\s-]{2,80})", text, re.IGNORECASE)
    if not match:
        return None
    candidate = match.group(1).strip(" .,:;")
    candidate = re.split(
        r"\b(?:in|at|on|during|while|with|if|when|because|and)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,:;")
    return _normalize_optional_text(candidate)


async def _photon_geocode_bounds(location: Optional[str]) -> Dict[str, Optional[float]]:
    if not location:
        return {"north": None, "south": None, "east": None, "west": None}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://photon.komoot.io/api/",
                params={"q": location, "limit": 1},
            )
            resp.raise_for_status()
            payload = resp.json()

        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list) or not features:
            return {"north": None, "south": None, "east": None, "west": None}

        first = features[0] or {}
        bbox = first.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            west, lat1, east, lat2 = bbox
            bounds = _normalize_bounds(lat2, lat1, east, west)
            if bounds["north"] is None:
                bounds = _normalize_bounds(lat1, lat2, east, west)
            if bounds["north"] is not None:
                return bounds

        properties = first.get("properties") if isinstance(first.get("properties"), dict) else {}
        extent = properties.get("extent")
        if isinstance(extent, list) and len(extent) == 4:
            west, lat1, east, lat2 = extent
            bounds = _normalize_bounds(lat2, lat1, east, west)
            if bounds["north"] is None:
                bounds = _normalize_bounds(lat1, lat2, east, west)
            if bounds["north"] is not None:
                return bounds

        geometry = first.get("geometry") if isinstance(first.get("geometry"), dict) else {}
        coords = geometry.get("coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            lon_f = float(coords[0])
            lat_f = float(coords[1])
            delta = 0.35
            return _normalize_bounds(lat_f + delta, lat_f - delta, lon_f + delta, lon_f - delta)

        return {"north": None, "south": None, "east": None, "west": None}
    except Exception as e:
        logger.warning(f"Photon geocode failed for '{location}': {e}")
        return {"north": None, "south": None, "east": None, "west": None}


async def _nominatim_geocode_bounds(location: Optional[str]) -> Dict[str, Optional[float]]:
    if not location:
        return {"north": None, "south": None, "east": None, "west": None}
    try:
        params: Dict[str, Any] = {
            "q": location,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 0,
        }
        if NOMINATIM_EMAIL:
            params["email"] = NOMINATIM_EMAIL

        headers = {"User-Agent": NOMINATIM_USER_AGENT}

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            payload = resp.json()

        if not isinstance(payload, list) or not payload:
            return {"north": None, "south": None, "east": None, "west": None}

        first = payload[0] or {}
        boundingbox = first.get("boundingbox")
        if isinstance(boundingbox, list) and len(boundingbox) == 4:
            south, north, west, east = boundingbox
            return _normalize_bounds(north, south, east, west)

        lat = first.get("lat")
        lon = first.get("lon")
        if lat is None or lon is None:
            return {"north": None, "south": None, "east": None, "west": None}

        lat_f = float(lat)
        lon_f = float(lon)
        delta = 0.35
        return _normalize_bounds(lat_f + delta, lat_f - delta, lon_f + delta, lon_f - delta)
    except Exception as e:
        logger.warning(f"Nominatim geocode failed for '{location}': {e}")
        return {"north": None, "south": None, "east": None, "west": None}


async def _resolve_geocode_bounds(location: Optional[str]) -> Dict[str, Optional[float]]:
    override_bounds = _location_override_bounds(location)
    if override_bounds.get("north") is not None:
        return override_bounds

    nominatim_bounds = await _nominatim_geocode_bounds(location)
    if nominatim_bounds.get("north") is not None:
        return nominatim_bounds

    photon_bounds = await _photon_geocode_bounds(location)
    if photon_bounds.get("north") is not None:
        return photon_bounds

    return {
        "north": float(LOCATION_BOUNDS_FALLBACK["north carolina"]["north"]),
        "south": float(LOCATION_BOUNDS_FALLBACK["north carolina"]["south"]),
        "east": float(LOCATION_BOUNDS_FALLBACK["north carolina"]["east"]),
        "west": float(LOCATION_BOUNDS_FALLBACK["north carolina"]["west"]),
    }


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LLM Weather Interpretation Service",
    description="Converts precipitation predictions to natural language alerts",
    version="1.0.0"
)

# Global LLM service instance
llm_service: Optional[LLMService] = None


@app.on_event("startup")
async def startup():
    global llm_service

    # If using Ollama, ensure it's running and model is available
    if LLM_BACKEND.lower() == "ollama":
        logger.info("Checking Ollama availability...")
        if ensure_ollama_ready(DEFAULT_MODEL):
            logger.info("Ollama is ready")
        else:
            logger.warning("Ollama setup failed - service will run in degraded mode")

    llm_service = get_llm_service()
    logger.info(f"LLM service initialized: {type(llm_service).__name__}")


# ============================================================================
# Request/Response Models
# ============================================================================

class InterpretRequest(BaseModel):
    """Request for weather interpretation."""
    model_config = ConfigDict(populate_by_name=True)

    region: Optional[str] = Field(None, description="Optional region name (derived from weather_condition when omitted)")
    forecast_hour: int = Field(6, description="Forecast hour")
    max_precipitation: Optional[float] = Field(None, description="Max precipitation in mm/hr")
    severity: Optional[str] = Field(None, description="Severity level")
    affected_percentage: float = Field(0, description="Percentage of area affected")
    include_prediction: bool = Field(False, description="Include raw prediction in response")
    prediction_data: Optional[List[List[float]]] = Field(None, description="Optional raw prediction")
    weather_condition: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional weather-condition payload from inference/downscale output",
        validation_alias=AliasChoices("weather_condition", "inference_output"),
    )
    customer_text: Optional[str] = Field(None, description="Original customer request text")
    activity: Optional[str] = Field(None, description="Optional extracted user activity")

class InterpretResponse(BaseModel):
    """Response with natural language interpretation."""
    alert: str
    severity: str
    region: str
    forecast_hour: int
    generated_at: str
    llm_backend: str
    model: str

class GenerateRequest(BaseModel):
    """Generic text generation request."""
    prompt: str = Field(..., description="Prompt for generation")
    max_tokens: int = Field(512, description="Maximum tokens to generate")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    llm_backend: str
    llm_healthy: bool
    model: str
    ollama_running: Optional[bool] = None
    model_available: Optional[bool] = None


class ExtractContextRequest(BaseModel):
    """Request for extracting structured context from user text."""
    text: str = Field(..., min_length=1, max_length=4000, description="Customer text input")


class ExtractContextResponse(BaseModel):
    """Structured extraction response for dashboard automation."""
    time: Optional[int] = Field(None, description="Forecast horizon in hours from now")
    location: Optional[str] = Field(None, description="Location phrase")
    activity: Optional[str] = Field(None, description="Activity phrase")
    north: Optional[float] = Field(None, description="Northern latitude bound")
    south: Optional[float] = Field(None, description="Southern latitude bound")
    east: Optional[float] = Field(None, description="Eastern longitude bound")
    west: Optional[float] = Field(None, description="Western longitude bound")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    llm_healthy = await llm_service.health_check() if llm_service else False

    # Add Ollama-specific status if using Ollama backend
    ollama_running = None
    model_available = None
    if LLM_BACKEND.lower() == "ollama":
        ollama_running = is_ollama_running()
        model_available = is_model_available(DEFAULT_MODEL) if ollama_running else False

    return HealthResponse(
        status="ok" if llm_healthy else "degraded",
        llm_backend=LLM_BACKEND,
        llm_healthy=llm_healthy,
        model=DEFAULT_MODEL,
        ollama_running=ollama_running,
        model_available=model_available
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LLM Weather Interpretation",
        "version": "1.0.0",
        "backend": LLM_BACKEND,
        "endpoints": ["/health", "/interpret", "/extract-context", "/generate", "/config", "/ollama/status", "/ollama/setup"]
    }


@app.post("/interpret", response_model=InterpretResponse)
async def interpret_weather(request: InterpretRequest):
    """
    Convert precipitation prediction to natural language alert.

    This endpoint takes numerical weather data and generates
    a human-readable weather alert using the configured LLM.
    """
    if llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    try:
        region = request.region
        downscale_payload: Optional[Dict[str, Any]] = None
        weather_condition_payload = request.weather_condition

        if weather_condition_payload is not None:
            downscale_payload = _resolve_downscale_payload(weather_condition_payload)
            if not isinstance(downscale_payload, dict):
                raise HTTPException(status_code=400, detail="weather_condition must be a JSON object")

            inferred_region = _resolve_region_from_weather_condition(downscale_payload)
            region = region or inferred_region or "the specified area"

            severity = derive_severity(
                explicit_severity=request.severity,
                downscale_payload=downscale_payload,
                max_precipitation=request.max_precipitation,
            )
            prompt = build_inference_prompt(
                region=region,
                forecast_hour=request.forecast_hour,
                downscale_payload=downscale_payload,
                severity=severity,
                max_precipitation=request.max_precipitation,
                customer_text=request.customer_text,
                activity=request.activity,
            )
        else:
            # Backward-compatible mode for legacy interpret payloads.
            region = region or "the specified area"
            severity = derive_severity(
                explicit_severity=request.severity,
                max_precipitation=request.max_precipitation,
            )
            prompt = build_weather_prompt(
                {
                    "region": region,
                    "forecast_hour": request.forecast_hour,
                    "max_precipitation": request.max_precipitation or 0,
                    "severity": severity,
                    "affected_percentage": request.affected_percentage,
                }
            )

        # Generate interpretation
        alert = await llm_service.generate(prompt, max_tokens=INTERPRET_MAX_TOKENS)

        response_obj = InterpretResponse(
            alert=alert.strip(),
            severity=severity,
            region=region,
            forecast_hour=request.forecast_hour,
            generated_at=datetime.utcnow().isoformat(),
            llm_backend=LLM_BACKEND,
            model=DEFAULT_MODEL
        )
        return response_obj

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interpretation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    Generic text generation endpoint.
    Useful for custom prompts and testing.
    """
    if llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    try:
        response = await llm_service.generate(request.prompt, request.max_tokens)
        return {
            "response": response,
            "backend": LLM_BACKEND,
            "model": DEFAULT_MODEL
        }
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-context", response_model=ExtractContextResponse)
async def extract_context(request: ExtractContextRequest):
    """
    Extract structured fields from customer text for dashboard automation:
    time (hours), location, activity, and optional geographic bounds.
    """
    if llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    customer_text = request.text.strip()
    if not customer_text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    prompt = (
        "Extract structured weather-intent fields from the customer text.\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{"time": number|null, "location": string|null, "activity": string|null, '
        '"north": number|null, "south": number|null, "east": number|null, "west": number|null}\n'
        "Rules:\n"
        "- time: forecast horizon in HOURS from now (integer or number).\n"
        "- location: city/state/country/region phrase if present.\n"
        "- activity: user plan/activity if present.\n"
        "- north/south/east/west: bounding box for location if confidently inferable; otherwise null.\n"
        "- Use null when unknown.\n"
        f"Customer text: {json.dumps(customer_text, ensure_ascii=True)}"
    )

    try:
        raw_response = await llm_service.generate(prompt, max_tokens=256)
        parsed_obj = _extract_first_json_object(raw_response) or {}
        parsed_lower = {str(k).lower(): v for k, v in parsed_obj.items()}

        location = _normalize_optional_text(parsed_lower.get("location")) or _fallback_location(customer_text)
        activity = _normalize_optional_text(parsed_lower.get("activity")) or _fallback_activity(customer_text)
        parsed_time = _normalize_hours(parsed_lower.get("time"))
        time_hours = parsed_time if parsed_time is not None else _normalize_hours(customer_text)

        bounds = _normalize_bounds(
            parsed_lower.get("north"),
            parsed_lower.get("south"),
            parsed_lower.get("east"),
            parsed_lower.get("west"),
        )
        if bounds["north"] is None:
            resolved_bounds = await _resolve_geocode_bounds(location)
            bounds = _normalize_bounds(
                resolved_bounds.get("north"),
                resolved_bounds.get("south"),
                resolved_bounds.get("east"),
                resolved_bounds.get("west"),
            )

        return ExtractContextResponse(
            time=time_hours,
            location=location,
            activity=activity,
            north=bounds["north"],
            south=bounds["south"],
            east=bounds["east"],
            west=bounds["west"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current LLM configuration."""
    return {
        "backend": LLM_BACKEND,
        "model": DEFAULT_MODEL,
        "ollama_url": OLLAMA_BASE_URL if LLM_BACKEND == "ollama" else None,
        "vllm_url": VLLM_BASE_URL if LLM_BACKEND == "vllm" else None,
        "vllm_api_prefix": VLLM_API_PREFIX if LLM_BACKEND == "vllm" else None,
    }


@app.get("/ollama/status")
async def ollama_status():
    """
    Check Ollama server and model status.

    Returns detailed status of Ollama server and whether the model is available.
    """
    if LLM_BACKEND.lower() != "ollama":
        return {"error": "Not using Ollama backend", "backend": LLM_BACKEND}

    server_running = is_ollama_running()
    model_available = is_model_available(DEFAULT_MODEL) if server_running else False

    return {
        "ollama_url": OLLAMA_BASE_URL,
        "server_running": server_running,
        "model": DEFAULT_MODEL,
        "model_available": model_available,
        "ready": server_running and model_available,
    }


@app.post("/ollama/setup")
async def ollama_setup():
    """
    Ensure Ollama is running and model is pulled.

    This endpoint will:
    1. Start Ollama server if not running
    2. Pull the model if not available

    Note: Model pulling can take several minutes for large models.
    """
    if LLM_BACKEND.lower() != "ollama":
        raise HTTPException(status_code=400, detail="Not using Ollama backend")

    steps = []

    # Check/start server
    if not is_ollama_running():
        steps.append({"step": "start_server", "status": "starting"})
        if start_ollama():
            steps[-1]["status"] = "success"
        else:
            steps[-1]["status"] = "failed"
            return {"success": False, "steps": steps, "error": "Failed to start Ollama server"}
    else:
        steps.append({"step": "start_server", "status": "already_running"})

    # Check/pull model
    if not is_model_available(DEFAULT_MODEL):
        steps.append({"step": "pull_model", "model": DEFAULT_MODEL, "status": "pulling"})
        if pull_model(DEFAULT_MODEL):
            steps[-1]["status"] = "success"
        else:
            steps[-1]["status"] = "failed"
            return {"success": False, "steps": steps, "error": f"Failed to pull model {DEFAULT_MODEL}"}
    else:
        steps.append({"step": "pull_model", "model": DEFAULT_MODEL, "status": "already_available"})

    return {
        "success": True,
        "steps": steps,
        "message": f"Ollama is ready with model {DEFAULT_MODEL}"
    }


# ============================================================================
# Run directly for development
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
