"""
WeatherScope Demo Dashboard
Simple FastAPI proxy + static UI for KubeCon demonstrations.
"""

import os
import json
import re
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field


FOURCASTNET_URL = os.getenv("FOURCASTNET_URL", "http://localhost:8003")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")
MAP_URL = os.getenv("MAP_URL", "http://localhost:8004")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8002")
GOOGLE_GEOCODING_API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY", "").strip()
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "WeatherScope/1.0").strip()
NOMINATIM_EMAIL = os.getenv("NOMINATIM_EMAIL", "").strip()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="WeatherScope Dashboard",
    description="Demo dashboard API/proxy for WeatherScope services",
    version="1.0.0",
)


class DownscaleRequest(BaseModel):
    variables: Dict[str, Any] | None = Field(None, description="Regional variable arrays")
    region: str | None = Field(None, description="Optional region key")
    region_name: str | None = Field(None, description="Optional region name")
    region_info: Dict[str, Any] | None = Field(None, description="Optional region metadata")
    upscale_factor: int = Field(4, description="Upscale factor")
    method: str = Field("bilinear", description="Downscaling method")


class InterpretRequest(BaseModel):
    region: str | None = Field(None, description="Optional region key/name")
    forecast_hour: int = Field(6, description="Forecast hour")
    weather_condition: Dict[str, Any] = Field(..., description="Inference /downscale output")
    customer_text: str | None = Field(None, description="Original customer request text")
    activity: str | None = Field(None, description="Optional user activity extracted from customer text")


class AdvisoryContextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000, description="Customer advisory input text")


class AdvisoryContextResponse(BaseModel):
    time: int | None = Field(None, description="Extracted time in hours from now")
    location: str | None = Field(None, description="Extracted location expression")
    activity: str | None = Field(None, description="Extracted activity expression")
    north: float | None = Field(None, description="Extracted northern latitude bound")
    south: float | None = Field(None, description="Extracted southern latitude bound")
    east: float | None = Field(None, description="Extracted eastern longitude bound")
    west: float | None = Field(None, description="Extracted western longitude bound")


class MapRequest(BaseModel):
    original_variables: Dict[str, Any] = Field(..., description="Original extracted variables")
    downscaled_variables: Dict[str, Any] = Field(..., description="Downscaled variable arrays")
    upscale_factor: int = Field(4, description="Upscale factor label")
    region: str | None = Field(None, description="Optional region key")
    region_name: str | None = Field(None, description="Optional region name (legacy)")
    region_info: Dict[str, Any] | None = Field(None, description="Optional region metadata")


class RegionalMapRequest(BaseModel):
    variables: Dict[str, Any] = Field(..., description="Regional extracted variables")
    stats: Dict[str, Any] | None = Field(None, description="Optional extraction stats")
    region: str | None = Field(None, description="Optional region key")
    region_name: str | None = Field(None, description="Optional region name")
    region_info: Dict[str, Any] | None = Field(None, description="Optional region metadata")
    variable: str = Field("t2m", description="Variable to render")
    step: int | None = Field(None, description="Optional forecast step")


class DownscaleMapRequest(BaseModel):
    predictions: Dict[str, Any] = Field(..., description="Downscaled variable arrays")
    stats: Dict[str, Any] | None = Field(None, description="Optional downscale stats")
    region: str | None = Field(None, description="Optional region key")
    region_name: str | None = Field(None, description="Optional region name")
    region_info: Dict[str, Any] | None = Field(None, description="Optional region metadata")
    variable: str = Field("t2m", description="Variable to render")
    step: int | None = Field(None, description="Optional forecast step")
    method: str | None = Field(None, description="Optional downscale method label")
    upscale_factor: int | None = Field(None, description="Optional upscale factor label")


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


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def _normalize_coordinate(value: Any, axis: str) -> float | None:
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
    return round(float(num), 4)


def _normalize_bounds(north: Any, south: Any, east: Any, west: Any) -> Dict[str, float | None]:
    n = _normalize_coordinate(north, "lat")
    s = _normalize_coordinate(south, "lat")
    e = _normalize_coordinate(east, "lon")
    w = _normalize_coordinate(west, "lon")
    if None in (n, s, e, w):
        return {"north": None, "south": None, "east": None, "west": None}
    if n <= s:
        return {"north": None, "south": None, "east": None, "west": None}
    return {"north": n, "south": s, "east": e, "west": w}


def _fallback_bounds_from_location(location: str | None, text: str) -> Dict[str, float | None]:
    candidates = []
    if location:
        candidates.append(location.lower())
    if text:
        candidates.append(text.lower())
    for phrase, bounds in LOCATION_BOUNDS_FALLBACK.items():
        for candidate in candidates:
            if phrase in candidate:
                return {
                    "north": float(bounds["north"]),
                    "south": float(bounds["south"]),
                    "east": float(bounds["east"]),
                    "west": float(bounds["west"]),
                }
    return {"north": None, "south": None, "east": None, "west": None}


async def _google_geocode_bounds(location: str | None) -> Dict[str, float | None]:
    if not location or not GOOGLE_GEOCODING_API_KEY:
        return {"north": None, "south": None, "east": None, "west": None}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": location, "key": GOOGLE_GEOCODING_API_KEY},
            )
            resp.raise_for_status()
            payload = resp.json()

        if payload.get("status") != "OK":
            return {"north": None, "south": None, "east": None, "west": None}

        results = payload.get("results") or []
        if not results:
            return {"north": None, "south": None, "east": None, "west": None}

        geometry = (results[0] or {}).get("geometry") or {}
        box = geometry.get("bounds") or geometry.get("viewport") or {}
        ne = box.get("northeast") or {}
        sw = box.get("southwest") or {}
        return _normalize_bounds(
            ne.get("lat"),
            sw.get("lat"),
            ne.get("lng"),
            sw.get("lng"),
        )
    except Exception:
        return {"north": None, "south": None, "east": None, "west": None}


async def _nominatim_geocode_bounds(location: str | None) -> Dict[str, float | None]:
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
    except Exception:
        return {"north": None, "south": None, "east": None, "west": None}


async def _resolve_geocode_bounds(location: str | None) -> Dict[str, float | None]:
    google_bounds = await _google_geocode_bounds(location)
    if google_bounds.get("north") is not None:
        return google_bounds
    return await _nominatim_geocode_bounds(location)


def _normalize_hours(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return int(round(float(value)))

    text = str(value).strip().lower()
    if not text:
        return None

    if text.endswith("h") and text[:-1].strip().isdigit():
        return int(text[:-1].strip())
    if text.isdigit():
        return int(text)

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

    in_hours_match = re.search(r"\bin\s+(\d+(?:\.\d+)?)\b", text)
    if in_hours_match:
        return int(round(float(in_hours_match.group(1))))

    return None


def _extract_first_json_object(text: str) -> Dict[str, Any] | None:
    if not text:
        return None

    raw = text.strip()
    try:
        direct = json.loads(raw)
        if isinstance(direct, dict):
            return direct
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


def _fallback_extract_context(text: str) -> Dict[str, Any]:
    time_value: int | None = None
    location_value = None
    activity_value = None

    lower_text = text.lower()

    time_match = re.search(
        r"\b(today|tonight|tomorrow|this morning|this afternoon|this evening|this weekend|next week|next weekend)\b",
        lower_text,
    )
    clock_match = re.search(r"\b\d{1,2}(?::\d{2})?\s?(am|pm)\b", lower_text)
    if time_match and clock_match:
        time_value = _normalize_hours(f"{time_match.group(0)} {clock_match.group(0)}")
    elif time_match:
        time_value = _normalize_hours(time_match.group(0))
    elif clock_match:
        time_value = _normalize_hours(clock_match.group(0))

    if time_value is None:
        in_hour_match = re.search(r"\bin\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h)\b", lower_text)
        if in_hour_match:
            time_value = int(round(float(in_hour_match.group(1))))
    if time_value is None:
        in_day_match = re.search(r"\bin\s+(\d+(?:\.\d+)?)\s*(days?|d)\b", lower_text)
        if in_day_match:
            time_value = int(round(float(in_day_match.group(1)) * 24))

    location_match = re.search(r"\b(?:in|at|near|around)\s+([a-z0-9][a-z0-9\s,.-]{1,80})", text, re.IGNORECASE)
    if location_match:
        candidate = location_match.group(1).strip(" .,:;")
        location_value = re.split(
            r"\b(?:for|to|on|during|while|with|if|when|because|and)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,:;")

    activity_match = re.search(
        r"\b(?:plan(?:ning)?|going|intend|want|need|will)\s+(?:to\s+)?([a-z][a-z\s-]{2,80})",
        text,
        re.IGNORECASE,
    )
    if not activity_match:
        activity_match = re.search(r"\bfor\s+([a-z][a-z\s-]{2,80})", text, re.IGNORECASE)
    if activity_match:
        candidate = activity_match.group(1).strip(" .,:;")
        activity_value = re.split(
            r"\b(?:in|at|on|during|while|with|if|when|because|and)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,:;")

    return {
        "time": time_value,
        "location": _normalize_optional_text(location_value),
        "activity": _normalize_optional_text(activity_value),
    }


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="Dashboard UI not found")
    return FileResponse(
        index,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "services": {
            "fourcastnet": FOURCASTNET_URL,
            "inference": INFERENCE_URL,
            "map": MAP_URL,
            "llm": LLM_URL,
        },
    }


@app.get("/api/regions")
async def regions():
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{FOURCASTNET_URL}/regions")
        resp.raise_for_status()
        return resp.json()


@app.post("/api/forecast")
async def forecast(payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{FOURCASTNET_URL}/forecast", json=payload)
        resp.raise_for_status()
        return resp.json()


@app.get("/api/forecast/{job_id}")
async def forecast_status(job_id: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{FOURCASTNET_URL}/forecast/{job_id}")
        resp.raise_for_status()
        return resp.json()


@app.get("/api/forecast/{job_id}/steps")
async def forecast_steps(job_id: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{FOURCASTNET_URL}/forecast/{job_id}/steps")
        resp.raise_for_status()
        return resp.json()


@app.get("/api/global-map/{job_id}")
async def global_map(job_id: str, variable: str = Query("t2m"), step: int = Query(0, ge=-1)):
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.get(
            f"{FOURCASTNET_URL}/forecast/{job_id}/map",
            params={"variable": variable, "step": step},
        )
        resp.raise_for_status()
        return Response(
            content=resp.content,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={job_id}_{variable}_global_step{step}.png"},
        )


@app.get("/api/cache")
async def cache_status():
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{FOURCASTNET_URL}/cache")
        resp.raise_for_status()
        return resp.json()


@app.get("/api/extract/{job_id}")
async def extract(
    job_id: str,
    region: str | None = Query(None),
    step: int = Query(0),
    hour: float | None = Query(None, ge=0),
    north: float | None = Query(None),
    south: float | None = Query(None),
    east: float | None = Query(None),
    west: float | None = Query(None),
    region_name: str | None = Query(None),
):
    endpoint = f"{FOURCASTNET_URL}/forecast/{job_id}/regional"
    params: Dict[str, Any] = {"step": step}
    if hour is not None:
        endpoint = f"{FOURCASTNET_URL}/forecast/{job_id}/regional/hour"
        params = {"hour": hour}
    has_any_bounds = any(v is not None for v in (north, south, east, west))

    if has_any_bounds:
        if None in (north, south, east, west):
            raise HTTPException(status_code=400, detail="north/south/east/west must all be provided together")
        params.update(
            {
                "north": north,
                "south": south,
                "east": east,
                "west": west,
            }
        )
        if region_name:
            params["region_name"] = region_name
    else:
        if not region:
            raise HTTPException(status_code=400, detail="region is required when bounds are not provided")
        params["region"] = region

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(
            endpoint,
            params=params,
        )
        resp.raise_for_status()
        return resp.json()


@app.get("/api/extract-series/{job_id}")
async def extract_series(
    job_id: str,
    region: str | None = Query(None),
    step_stride: int = Query(1, ge=1, le=24),
    north: float | None = Query(None),
    south: float | None = Query(None),
    east: float | None = Query(None),
    west: float | None = Query(None),
    region_name: str | None = Query(None),
):
    params: Dict[str, Any] = {"step_stride": step_stride}
    has_any_bounds = any(v is not None for v in (north, south, east, west))

    if has_any_bounds:
        if None in (north, south, east, west):
            raise HTTPException(status_code=400, detail="north/south/east/west must all be provided together")
        params.update(
            {
                "north": north,
                "south": south,
                "east": east,
                "west": west,
            }
        )
        if region_name:
            params["region_name"] = region_name
    else:
        if not region:
            raise HTTPException(status_code=400, detail="region is required when bounds are not provided")
        params["region"] = region

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(
            f"{FOURCASTNET_URL}/forecast/{job_id}/regional/series",
            params=params,
        )
        resp.raise_for_status()
        return resp.json()


@app.post("/api/downscale")
async def downscale(request: DownscaleRequest):
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{INFERENCE_URL}/downscale",
            json=request.model_dump(),
        )
        resp.raise_for_status()
        return resp.json()


@app.post("/api/interpret")
async def interpret(request: InterpretRequest):
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{LLM_URL}/interpret", json=request.model_dump())
        resp.raise_for_status()
        return resp.json()


@app.post("/api/advisory/extract", response_model=AdvisoryContextResponse)
async def extract_advisory_context(request: AdvisoryContextRequest):
    customer_text = request.text.strip()
    if not customer_text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{LLM_URL}/extract-context",
            json={"text": customer_text},
        )
        if resp.status_code == 404:
            prompt = (
                "Extract structured weather-intent fields from the customer text.\n"
                "Return ONLY valid JSON with exactly these keys:\n"
                '{"time": number|null, "location": string|null, "activity": string|null, '
                '"north": number|null, "south": number|null, "east": number|null, "west": number|null}\n'
                "Rules:\n"
                "- time is hours from now.\n"
                "- Use null when unknown.\n"
                f"Customer text: {json.dumps(customer_text, ensure_ascii=True)}"
            )
            fallback_resp = await client.post(
                f"{LLM_URL}/generate",
                json={"prompt": prompt, "max_tokens": 256},
            )
            fallback_resp.raise_for_status()
            raw = fallback_resp.json().get("response", "")
            llm_payload = {str(k).lower(): v for k, v in (_extract_first_json_object(raw) or {}).items()}
        else:
            resp.raise_for_status()
            llm_payload = resp.json()

    fallback = _fallback_extract_context(customer_text)
    time_hours = _normalize_hours(llm_payload.get("time"))
    if time_hours is None:
        time_hours = fallback["time"]

    location = _normalize_optional_text(llm_payload.get("location")) or fallback["location"]
    activity = _normalize_optional_text(llm_payload.get("activity")) or fallback["activity"]

    bounds = _normalize_bounds(
        llm_payload.get("north"),
        llm_payload.get("south"),
        llm_payload.get("east"),
        llm_payload.get("west"),
    )
    if bounds["north"] is None:
        fallback_bounds = _fallback_bounds_from_location(location, customer_text)
        bounds = _normalize_bounds(
            fallback_bounds.get("north"),
            fallback_bounds.get("south"),
            fallback_bounds.get("east"),
            fallback_bounds.get("west"),
        )
    if bounds["north"] is None:
        google_bounds = await _resolve_geocode_bounds(location)
        bounds = _normalize_bounds(
            google_bounds.get("north"),
            google_bounds.get("south"),
            google_bounds.get("east"),
            google_bounds.get("west"),
        )

    return AdvisoryContextResponse(
        time=time_hours,
        location=location,
        activity=activity,
        north=bounds["north"],
        south=bounds["south"],
        east=bounds["east"],
        west=bounds["west"],
    )


@app.post("/api/map")
async def map_render(request: MapRequest):
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{MAP_URL}/map",
            json=request.model_dump(),
        )
        resp.raise_for_status()
        return Response(
            content=resp.content,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=dashboard_map.png"},
        )


@app.post("/api/map/regional")
async def map_regional_render(request: RegionalMapRequest):
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{MAP_URL}/map/regional",
            json=request.model_dump(),
        )
        resp.raise_for_status()
        return Response(
            content=resp.content,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=dashboard_regional_map.png"},
        )


@app.post("/api/map/downscale")
async def map_downscale_render(request: DownscaleMapRequest):
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{MAP_URL}/map/downscale",
            json=request.model_dump(),
        )
        resp.raise_for_status()
        return Response(
            content=resp.content,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=dashboard_downscale_map.png"},
        )
