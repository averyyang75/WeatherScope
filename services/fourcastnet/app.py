"""
FourCastNet v2-small Inference Service
Runs natively on macOS with MPS acceleration for Apple Silicon
"""

import os
import subprocess
import json
import threading
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field
import urllib.request
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FourCastNet v2-small Service",
    description="Global weather forecasting using ECMWF ai-models with MPS acceleration",
    version="1.0.0"
)

# ============================================================================
# Configuration
# ============================================================================

ASSETS_DIR = Path(os.getenv("ASSETS_DIR", Path.home() / ".cache" / "ai-models"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/fourcastnet"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ASSET_DIRNAME = "fourcastnetv2-small"
REQUIRED_ASSET_FILES = ("weights.tar", "global_means.npy", "global_stds.npy")
MODEL_ASSET_DIR = ASSETS_DIR / MODEL_ASSET_DIRNAME

# WeatherScope service URL (for pipeline integration)
REGIONALCAST_URL = os.getenv("REGIONALCAST_URL", "http://localhost:8000")

# Ollama URL (for LLM analysis)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Predefined regions for extraction
REGIONS = {
    "NC": {"name": "North Carolina", "north": 36.6, "south": 33.8, "east": -75.5, "west": -84.3},
    "NL": {"name": "Netherlands", "north": 53.7, "south": 50.7, "east": 7.3, "west": 3.2},
}

REGION_ALIASES = {
    "nc": "NC",
    "north carolina": "NC",
    "north_carolina": "NC",
    "north-carolina": "NC",
    "nl": "NL",
    "netherlands": "NL",
    "the netherlands": "NL",
    "netherland": "NL",
}

# Coarse fallback boundaries for non-cartopy environments.
REGION_CONTEXT_BOUNDARIES = {
    "NC": [
        (-84.32, 35.00), (-84.18, 35.28), (-84.05, 35.58), (-83.89, 35.97),
        (-83.62, 36.29), (-83.24, 36.50), (-82.62, 36.56), (-81.75, 36.56),
        (-80.84, 36.56), (-79.93, 36.56), (-79.03, 36.56), (-78.14, 36.56),
        (-77.23, 36.56), (-76.35, 36.56), (-75.62, 36.55), (-75.52, 36.24),
        (-75.56, 35.87), (-75.66, 35.49), (-75.87, 35.16), (-76.23, 34.93),
        (-76.76, 34.73), (-77.43, 34.64), (-78.14, 34.67), (-78.84, 34.76),
        (-79.54, 34.86), (-80.23, 34.94), (-80.94, 34.99), (-81.65, 35.01),
        (-82.35, 35.03), (-83.05, 35.03), (-83.73, 35.02), (-84.32, 35.00),
    ],
    "NL": [
        (3.37, 51.37), (3.70, 51.43), (4.06, 51.58), (4.36, 51.78),
        (4.63, 51.98), (4.90, 52.19), (5.21, 52.45), (5.56, 52.75),
        (5.94, 53.07), (6.35, 53.33), (6.82, 53.51), (7.22, 53.43),
        (7.19, 53.11), (6.95, 52.85), (6.69, 52.53), (6.42, 52.19),
        (6.13, 51.88), (5.79, 51.62), (5.43, 51.45), (5.02, 51.35),
        (4.58, 51.30), (4.19, 51.28), (3.82, 51.28), (3.52, 51.32),
        (3.37, 51.37),
    ],
}

PIPELINE_VARIABLES = {
    "t2m": {"shortName": "2t", "description": "2-meter temperature", "unit": "K"},
    "u10": {"shortName": "10u", "description": "10-meter U wind", "unit": "m/s"},
    "v10": {"shortName": "10v", "description": "10-meter V wind", "unit": "m/s"},
    "msl": {"shortName": "msl", "description": "Mean sea level pressure", "unit": "Pa"},
}

GLOBAL_MAP_VARIABLES = {
    "t2m": {"shortName": "2t", "label": "2m Temperature", "unit": "C", "cmap": "RdYlBu_r"},
    "u10": {"shortName": "10u", "label": "10m U Wind", "unit": "m/s", "cmap": "RdBu_r"},
    "v10": {"shortName": "10v", "label": "10m V Wind", "unit": "m/s", "cmap": "RdBu_r"},
    "msl": {"shortName": "msl", "label": "Mean Sea Level Pressure", "unit": "hPa", "cmap": "viridis"},
    "tp": {"shortName": "tp", "label": "Total Precipitation", "unit": "mm", "cmap": "Blues"},
    "wind": {"label": "10m Wind Speed", "unit": "m/s", "cmap": "YlOrRd"},
}

# Track running forecasts
running_forecasts = {}

# ERA5 download cache: tracks downloads by (date, time, input_source)
# Prevents duplicate downloads and allows jobs to wait on in-progress fetches
era5_cache: Dict[str, Dict] = {}
era5_cache_lock = threading.Lock()

# ============================================================================
# Request/Response Models
# ============================================================================

class ForecastRequest(BaseModel):
    """Request for weather forecast."""
    lead_time: int = Field(24, ge=6, le=240, description="Forecast hours (6-240)")
    date: Optional[str] = Field(None, description="Date YYYYMMDD (default: 5 days ago for CDS)")
    time: str = Field("1200", description="Time HHMM (0000 or 1200)")
    input_source: str = Field("cds", description="Data input: 'cds' (default), 'mars', or 'ecmwf-open-data'")

class ForecastStatus(BaseModel):
    """Status of a forecast job."""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[str] = None
    output_file: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    assets_downloaded: bool
    mps_available: bool
    running_jobs: int
    cached_files: int
    active_downloads: int

# ============================================================================
# Helper Functions
# ============================================================================

def check_assets():
    """Check if model assets are downloaded."""
    # Check if all required files exist in at least one model asset directory.
    asset_dirs = [
        ASSETS_DIR / MODEL_ASSET_DIRNAME,
        Path.home() / ".cache" / "ai-models" / MODEL_ASSET_DIRNAME,
    ]
    for asset_dir in asset_dirs:
        if all((asset_dir / filename).exists() for filename in REQUIRED_ASSET_FILES):
            return True
    return False


def resolve_region_key(region: str) -> Optional[str]:
    """Resolve user-supplied region into canonical region key."""
    if not isinstance(region, str) or not region.strip():
        return None
    normalized = region.strip().lower()
    key = REGION_ALIASES.get(normalized)
    if key in REGIONS:
        return key
    if region in REGIONS:
        return region
    return None


def get_asset_status() -> dict:
    """
    Return detailed asset availability for user-facing error messages.
    """
    asset_dirs = [
        ASSETS_DIR / MODEL_ASSET_DIRNAME,
        Path.home() / ".cache" / "ai-models" / MODEL_ASSET_DIRNAME,
    ]

    # Prefer the first directory if none exist yet.
    best_dir = asset_dirs[0]
    best_found = -1
    best_missing = list(REQUIRED_ASSET_FILES)

    for asset_dir in asset_dirs:
        missing = [f for f in REQUIRED_ASSET_FILES if not (asset_dir / f).exists()]
        found_count = len(REQUIRED_ASSET_FILES) - len(missing)
        if found_count > best_found:
            best_found = found_count
            best_dir = asset_dir
            best_missing = missing

    return {
        "ready": len(best_missing) == 0,
        "asset_dir": str(best_dir),
        "missing_files": best_missing,
        "required_files": list(REQUIRED_ASSET_FILES),
    }

def check_mps():
    """Check if MPS (Metal) is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False

def get_default_date():
    """Get date for forecast initialization (5 days ago for CDS ERA5 availability)."""
    # ERA5 data on CDS has ~5 day lag
    five_days_ago = datetime.utcnow() - timedelta(days=5)
    return five_days_ago.strftime("%Y%m%d")


def get_era5_cache_key(date: str, time: str, input_source: str, lead_time: int) -> str:
    """Generate cache key for ERA5 download."""
    return f"{input_source}_{date}_{time}_{lead_time}h"


def get_cached_grib_path(cache_key: str) -> Path:
    """Get the standard path for a cached GRIB file."""
    return OUTPUT_DIR / f"era5_{cache_key}.grib"


def find_existing_era5(date: str, time: str, input_source: str, lead_time: int) -> Optional[Path]:
    """
    Check if an ERA5 GRIB file already exists for these parameters.

    Returns the path if found, None otherwise.
    """
    cache_key = get_era5_cache_key(date, time, input_source, lead_time)
    cached_path = get_cached_grib_path(cache_key)

    if cached_path.exists():
        logger.info(f"Found cached ERA5 file: {cached_path}")
        return cached_path

    return None


def wait_for_era5_download(cache_key: str, timeout: int = 1800) -> Optional[Path]:
    """
    Wait for an in-progress ERA5 download to complete.

    Args:
        cache_key: The ERA5 cache key to wait for
        timeout: Maximum wait time in seconds (default 30 min)

    Returns:
        Path to the downloaded file, or None if failed/timeout
    """
    start_time = time.time()
    poll_interval = 5  # seconds

    while time.time() - start_time < timeout:
        with era5_cache_lock:
            if cache_key in era5_cache:
                status = era5_cache[cache_key]["status"]
                if status == "completed":
                    return Path(era5_cache[cache_key]["output_file"])
                elif status == "failed":
                    return None

        time.sleep(poll_interval)

    logger.warning(f"Timeout waiting for ERA5 download: {cache_key}")
    return None


def register_era5_download(cache_key: str, output_file: Path) -> bool:
    """
    Register a new ERA5 download. Returns False if already in progress.

    Args:
        cache_key: Unique key for this download
        output_file: Where the file will be saved

    Returns:
        True if registered (should proceed with download)
        False if another download is in progress (should wait)
    """
    with era5_cache_lock:
        if cache_key in era5_cache:
            status = era5_cache[cache_key]["status"]
            if status in ("downloading", "completed"):
                return False

        era5_cache[cache_key] = {
            "status": "downloading",
            "output_file": str(output_file),
            "started_at": datetime.utcnow().isoformat(),
        }
        return True


def complete_era5_download(cache_key: str, success: bool, error: Optional[str] = None):
    """Mark an ERA5 download as completed or failed."""
    with era5_cache_lock:
        if cache_key in era5_cache:
            era5_cache[cache_key]["status"] = "completed" if success else "failed"
            era5_cache[cache_key]["completed_at"] = datetime.utcnow().isoformat()
            if error:
                era5_cache[cache_key]["error"] = error


def analyze_with_llm(region_name: str, extraction: dict, downscale_result: dict) -> Optional[dict]:
    """
    Send downscaled multi-variable results to Ollama for natural language analysis.

    Args:
        region_name: Name of the region (e.g., "Austin, TX")
        extraction: FourCastNet extraction results (multi-variable)
        downscale_result: bilinear downscaling results

    Returns:
        LLM analysis dict or None if unavailable
    """
    try:
        # Extract multi-variable stats
        t2m_stats = extraction.get("variables", {}).get("t2m", {})
        u10_stats = extraction.get("variables", {}).get("u10", {})
        v10_stats = extraction.get("variables", {}).get("v10", {})
        msl_stats = extraction.get("variables", {}).get("msl", {})

        # Convert temperature from Kelvin to Celsius
        t2m_min_c = t2m_stats.get("min", 273) - 273.15
        t2m_max_c = t2m_stats.get("max", 273) - 273.15

        # Calculate wind speed from u10, v10
        import math
        u10_mean = u10_stats.get("mean", 0)
        v10_mean = v10_stats.get("mean", 0)
        wind_speed = math.sqrt(u10_mean**2 + v10_mean**2)

        # MSL pressure in hPa
        msl_mean = msl_stats.get("mean", 101325) / 100  # Pa to hPa

        # Get downscaled stats if available
        ds_stats = downscale_result.get("stats", {}) if isinstance(downscale_result, dict) else {}

        prompt = f"""Analyze the following weather forecast data for {region_name} and provide a brief, actionable weather advisory:

**FourCastNet Global Model Output (0.25° resolution):**
- 2m Temperature: {t2m_min_c:.1f}°C to {t2m_max_c:.1f}°C
- 10m Wind Speed: {wind_speed:.1f} m/s
- Mean Sea Level Pressure: {msl_mean:.1f} hPa

**Bilinear Downscaled Analysis (high resolution):**
- Upscale factor: {downscale_result.get("upscale_factor", 4) if isinstance(downscale_result, dict) else 4}x
- Input shape: {extraction.get("shape", "unknown")}
- Output shape: {downscale_result.get("output_shape", "unknown") if isinstance(downscale_result, dict) else "unknown"}

Weather interpretation:
- Temperature < 0°C: Freezing conditions
- Wind > 10 m/s: Strong winds
- MSL < 1000 hPa: Low pressure system (potential storms)
- MSL > 1020 hPa: High pressure (fair weather)

Provide a 2-3 sentence weather advisory for residents, including temperature feel, wind conditions, and any recommended actions."""

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload.encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return {
                "model": OLLAMA_MODEL,
                "advisory": result.get("response", ""),
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "response_tokens": result.get("eval_count", 0),
            }

    except Exception as e:
        logger.warning(f"Ollama analysis unavailable: {e}")
        return {"error": str(e), "service_url": OLLAMA_URL}


def extract_regional_variables(output_file: Path, region: str, step: int = 0) -> dict:
    """
    Extract t2m/u10/v10/msl regional slices from a forecast GRIB file.
    """
    import xarray as xr
    import numpy as np

    region_key = resolve_region_key(region)
    if not region_key:
        raise HTTPException(status_code=400, detail=f"Unknown region. Available: {list(REGIONS.keys())}")

    bounds = REGIONS[region_key]
    west_360 = bounds["west"] + 360 if bounds["west"] < 0 else bounds["west"]
    east_360 = bounds["east"] + 360 if bounds["east"] < 0 else bounds["east"]

    extracted_vars = {}
    var_stats = {}

    for var_name, var_info in PIPELINE_VARIABLES.items():
        try:
            ds = xr.open_dataset(
                output_file,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": var_info["shortName"]}},
            )

            data_var = None
            for dv in ds.data_vars:
                data_var = ds[dv]
                break

            if data_var is None:
                logger.warning(f"Variable {var_name} ({var_info['shortName']}) not found, skipping")
                ds.close()
                continue

            if "step" in data_var.dims and step < len(data_var.step):
                data_var = data_var.isel(step=step)

            regional = data_var.sel(
                latitude=slice(bounds["north"], bounds["south"]),
                longitude=slice(west_360, east_360),
            )

            var_data = regional.values.astype(np.float32)
            extracted_vars[var_name] = var_data.tolist()
            var_stats[var_name] = {
                "min": float(var_data.min()),
                "max": float(var_data.max()),
                "mean": float(var_data.mean()),
                "std": float(var_data.std()),
                "unit": var_info["unit"],
                "description": var_info["description"],
            }

            ds.close()

        except Exception as e:
            logger.warning(f"Failed to extract {var_name}: {e}")
            continue

    if not extracted_vars:
        raise HTTPException(status_code=500, detail="Failed to extract any variables from GRIB file")

    first_var = list(extracted_vars.values())[0]
    shape = [len(first_var), len(first_var[0])] if first_var else [0, 0]

    region_info = {
        "key": region_key,
        "name": bounds["name"],
        "bounds": {
            "north": bounds["north"],
            "south": bounds["south"],
            "east": bounds["east"],
            "west": bounds["west"],
        },
    }

    return {
        "region": region_key,
        "region_name": bounds["name"],
        "region_info": region_info,
        "shape": shape,
        "variables": extracted_vars,
        "stats": var_stats,
    }


def _extract_global_variable(output_file: Path, short_name: str, step: int):
    """Extract global variable grid and coordinates from GRIB."""
    import xarray as xr
    import numpy as np

    ds = xr.open_dataset(
        output_file,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}},
    )
    try:
        if not ds.data_vars:
            raise HTTPException(status_code=500, detail=f"Variable shortName={short_name} not found in GRIB")

        data_var_name = next(iter(ds.data_vars))
        data_var = ds[data_var_name]

        if "step" in data_var.dims:
            step_count = len(data_var.step)
            if step < 0 or step >= step_count:
                raise HTTPException(status_code=400, detail=f"Invalid step={step}. Available range: 0..{step_count - 1}")
            data_var = data_var.isel(step=step)

        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise HTTPException(status_code=500, detail="GRIB missing latitude/longitude coordinates")

        lats = ds["latitude"].values.astype(np.float32)
        lons = ds["longitude"].values.astype(np.float32)
        values = data_var.values.astype(np.float32)
        return values, lats, lons
    finally:
        ds.close()


def _normalize_global_longitude(values, lons):
    """Convert longitudes to [-180, 180) and reorder values accordingly."""
    import numpy as np

    if lons.ndim != 1:
        return values, lons

    normalized_lons = ((lons + 180.0) % 360.0) - 180.0
    order = np.argsort(normalized_lons)
    if values.ndim == 2 and values.shape[1] == len(lons):
        values = values[:, order]
    return values, normalized_lons[order]


def _select_regional_slice(values, lats, lons, bounds):
    """Extract a regional slice from global arrays."""
    import numpy as np

    south = float(bounds["south"])
    north = float(bounds["north"])
    west = float(bounds["west"])
    east = float(bounds["east"])

    lat_mask = (lats >= south) & (lats <= north)
    if west <= east:
        lon_mask = (lons >= west) & (lons <= east)
    else:
        # Dateline crossing support.
        lon_mask = (lons >= west) | (lons <= east)

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No grid cells found in requested bounds: {bounds}",
        )

    sliced = values[np.ix_(lat_idx, lon_idx)]
    return sliced, lats[lat_idx], lons[lon_idx]


def _geometry_to_lines(geometry: Any) -> List[List[List[float]]]:
    if geometry is None:
        return []

    geom_type = getattr(geometry, "geom_type", "")
    lines: List[List[List[float]]] = []
    if geom_type == "Polygon":
        lines.append([[float(x), float(y)] for x, y in geometry.exterior.coords])
    elif geom_type == "MultiPolygon":
        for part in geometry.geoms:
            lines.append([[float(x), float(y)] for x, y in part.exterior.coords])
    elif geom_type == "LineString":
        lines.append([[float(x), float(y)] for x, y in geometry.coords])
    elif geom_type == "MultiLineString":
        for part in geometry.geoms:
            lines.append([[float(x), float(y)] for x, y in part.coords])
    return lines


def _attr_lower(attrs: Dict[str, Any], *keys: str) -> str:
    lower_attrs = {str(k).lower(): v for k, v in attrs.items()}
    for key in keys:
        value = lower_attrs.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return ""


def _boundary_lines_from_natural_earth(region_key: str | None) -> List[List[List[float]]]:
    if not region_key:
        return []

    try:
        import cartopy.io.shapereader as shpreader
    except Exception:
        return []

    try:
        if region_key == "NC":
            for shape_name in ("admin_1_states_provinces", "admin_1_states_provinces_lakes"):
                try:
                    path = shpreader.natural_earth(resolution="10m", category="cultural", name=shape_name)
                    matches: List[List[List[float]]] = []
                    for record in shpreader.Reader(path).records():
                        attrs = record.attributes
                        name = _attr_lower(attrs, "name", "name_en")
                        admin = _attr_lower(attrs, "admin", "adm0_name", "sovereignt", "geonunit")
                        postal = _attr_lower(attrs, "postal", "iso_3166_2", "hasc")
                        is_nc = (name == "north carolina") or postal.replace(".", "").endswith("nc")
                        is_us = ("united states" in admin) or (admin in {"us", "usa"})
                        if is_nc and (is_us or admin == ""):
                            matches.extend(_geometry_to_lines(record.geometry))
                    if matches:
                        return matches
                except Exception:
                    continue

        if region_key == "NL":
            path = shpreader.natural_earth(
                resolution="10m",
                category="cultural",
                name="admin_0_countries",
            )
            for record in shpreader.Reader(path).records():
                attrs = record.attributes
                candidates = {
                    _attr_lower(attrs, "name", "name_en", "admin"),
                    _attr_lower(attrs, "name_long", "formal_en"),
                    _attr_lower(attrs, "sovereignt", "geounit"),
                }
                if any("netherlands" in value for value in candidates if value):
                    return _geometry_to_lines(record.geometry)
    except Exception as e:
        logger.warning(f"Natural Earth boundary load failed for region={region_key}: {e}")

    return []


def _resolve_context_boundary_lines(region_key: str | None) -> List[List[List[float]]]:
    if not region_key:
        return []

    ne_lines = _boundary_lines_from_natural_earth(region_key)
    if ne_lines:
        return ne_lines

    fallback = REGION_CONTEXT_BOUNDARIES.get(region_key, [])
    if fallback:
        return [fallback]
    return []


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    cached_count = len(list(OUTPUT_DIR.glob("era5_*.grib")))
    with era5_cache_lock:
        active_count = len([v for v in era5_cache.values() if v.get("status") == "downloading"])

    return HealthResponse(
        status="ok",
        assets_downloaded=check_assets(),
        mps_available=check_mps(),
        running_jobs=len([j for j in running_forecasts.values() if j["status"] == "running"]),
        cached_files=cached_count,
        active_downloads=active_count
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FourCastNet v2-small",
        "version": "1.0.0",
        "model": "fourcastnetv2-small",
        "resolution": "0.25° (~25km)",
        "endpoints": [
            "/health",
            "/forecast",
            "/forecast/{job_id}",
            "/forecast/{job_id}/map?variable=t2m&step=0",
            "/forecast/{job_id}/pipeline?region={region}",
            "/forecast/{job_id}/regional?region={region}",
            "/regions",
            "/cache",
            "/download-assets"
        ],
        "features": [
            "ERA5 caching - avoids re-downloading same data",
            "Download deduplication - waits for in-progress fetches",
            "MPS acceleration on Apple Silicon"
        ]
    }

@app.post("/download-assets")
async def download_assets(background_tasks: BackgroundTasks):
    """Download model assets (run once)."""
    if check_assets():
        return {"status": "assets already downloaded"}

    def download():
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ai_models",
                    "--assets",
                    str(MODEL_ASSET_DIR),
                    "--no-assets-sub-directory",
                    "--download-assets",
                    "fourcastnetv2-small",
                ],
                check=True,
                capture_output=True
            )
            logger.info("Assets downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download assets: {e}")

    background_tasks.add_task(download)
    return {"status": "download started", "message": "Check /health to verify completion"}

@app.post("/forecast", response_model=ForecastStatus)
async def create_forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Start a new weather forecast.

    Features:
    - Checks for existing ERA5 GRIB files to avoid re-downloading
    - Waits for in-progress downloads instead of starting duplicates
    - Caches completed downloads for reuse

    Returns a job_id to track progress.
    """
    asset_status = get_asset_status()
    if not asset_status["ready"]:
        missing = ", ".join(asset_status["missing_files"])
        raise HTTPException(
            status_code=503,
            detail=(
                "Model assets incomplete. "
                f"Missing files: {missing}. "
                f"Expected under: {asset_status['asset_dir']}. "
                "Run: ai-models --download-assets fourcastnetv2-small "
                "or call POST /download-assets."
            )
        )

    # Generate job ID and resolve date
    job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    date = request.date or get_default_date()

    # Check for cached ERA5 data (always use time=0000)
    cache_key = get_era5_cache_key(date, "0000", request.input_source, request.lead_time)
    cached_file = find_existing_era5(date, "0000", request.input_source, request.lead_time)

    if cached_file:
        # Use existing file - instant completion
        running_forecasts[job_id] = {
            "status": "completed",
            "progress": "Using cached ERA5 data",
            "output_file": str(cached_file),
            "error": None,
            "started_at": datetime.utcnow().isoformat(),
            "cache_hit": True,
        }
        logger.info(f"Job {job_id}: Using cached file {cached_file}")
        return ForecastStatus(
            job_id=job_id,
            status="completed",
            progress="Using cached ERA5 data",
            output_file=str(cached_file)
        )

    # Standard output path (cached naming for reuse)
    output_file = get_cached_grib_path(cache_key)

    # Initialize job status
    running_forecasts[job_id] = {
        "status": "pending",
        "progress": None,
        "output_file": str(output_file),
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "cache_key": cache_key,
    }

    def run_forecast():
        try:
            # Check if another job is already downloading this ERA5 data
            if not register_era5_download(cache_key, output_file):
                # Another download in progress - wait for it
                running_forecasts[job_id]["status"] = "running"
                running_forecasts[job_id]["progress"] = "Waiting for in-progress ERA5 download..."
                logger.info(f"Job {job_id}: Waiting for existing download {cache_key}")

                result_path = wait_for_era5_download(cache_key)
                if result_path and result_path.exists():
                    running_forecasts[job_id]["status"] = "completed"
                    running_forecasts[job_id]["progress"] = "Reused in-progress download"
                    running_forecasts[job_id]["output_file"] = str(result_path)
                    logger.info(f"Job {job_id}: Reused download from {cache_key}")
                else:
                    running_forecasts[job_id]["status"] = "failed"
                    running_forecasts[job_id]["error"] = "Waited download failed or timed out"
                return

            # We own this download - proceed
            running_forecasts[job_id]["status"] = "running"
            running_forecasts[job_id]["progress"] = "Downloading ERA5 data and running forecast..."

            cmd = [
                sys.executable,
                "-m",
                "ai_models",
                "--assets", str(MODEL_ASSET_DIR),
                "--no-assets-sub-directory",
                "--input", request.input_source,
                "--lead-time", str(request.lead_time),
                "--date", date,
                "--time", "0000",
                "--output", "file",
                "--path", str(output_file),
                "fourcastnetv2-small"
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            # Set environment for PyTorch 2.6+ compatibility
            env = os.environ.copy()
            env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env
            )

            if result.returncode == 0:
                running_forecasts[job_id]["status"] = "completed"
                running_forecasts[job_id]["progress"] = "Forecast complete"
                complete_era5_download(cache_key, success=True)
                logger.info(f"Forecast {job_id} completed successfully")
            else:
                running_forecasts[job_id]["status"] = "failed"
                running_forecasts[job_id]["error"] = result.stderr
                complete_era5_download(cache_key, success=False, error=result.stderr)
                logger.error(f"Forecast {job_id} failed: {result.stderr}")

        except Exception as e:
            running_forecasts[job_id]["status"] = "failed"
            running_forecasts[job_id]["error"] = str(e)
            complete_era5_download(cache_key, success=False, error=str(e))
            logger.error(f"Forecast {job_id} error: {e}")

    background_tasks.add_task(run_forecast)

    return ForecastStatus(
        job_id=job_id,
        status="pending",
        progress="Job queued",
        output_file=str(output_file)
    )

@app.get("/forecast/{job_id}", response_model=ForecastStatus)
async def get_forecast_status(job_id: str):
    """Get the status of a forecast job."""
    if job_id not in running_forecasts:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = running_forecasts[job_id]
    return ForecastStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        output_file=job["output_file"],
        error=job["error"]
    )

@app.get("/forecast/{job_id}/precipitation")
async def get_precipitation(job_id: str, step: int = 0):
    """
    Extract precipitation data from a completed forecast.

    Returns precipitation as JSON array for integration with WeatherScope.
    """
    if job_id not in running_forecasts:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = running_forecasts[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

    output_file = Path(job["output_file"])
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    try:
        import xarray as xr

        # Open GRIB file
        ds = xr.open_dataset(output_file, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})

        # Get total precipitation
        if 'tp' in ds:
            precip = ds['tp'].isel(step=step).values
        elif 'unknown' in ds:
            precip = ds['unknown'].isel(step=step).values
        else:
            # List available variables
            raise HTTPException(
                status_code=400,
                detail=f"Precipitation not found. Available: {list(ds.data_vars)}"
            )

        return {
            "job_id": job_id,
            "step": step,
            "shape": list(precip.shape),
            "min": float(precip.min()),
            "max": float(precip.max()),
            "data": precip.tolist()
        }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="cfgrib not installed. Run: pip install cfgrib eccodes"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{job_id}/map")
async def get_global_map(job_id: str, variable: str = "t2m", step: int = 0):
    """
    Generate a weather PNG map from a completed forecast job.

    Query params:
    - variable: t2m, u10, v10, msl, tp, or wind
    - step: Forecast step index (default 0)
    """
    if job_id not in running_forecasts:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = running_forecasts[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

    output_file = Path(job["output_file"])
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    variable_key = variable.strip().lower()
    if variable_key not in GLOBAL_MAP_VARIABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variable '{variable}'. Available: {list(GLOBAL_MAP_VARIABLES.keys())}",
        )

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            cartopy_available = True
        except Exception:
            cartopy_available = False

        if variable_key == "wind":
            u10, lats, lons = _extract_global_variable(output_file, "10u", step)
            v10, lats_v, lons_v = _extract_global_variable(output_file, "10v", step)
            u10, lons = _normalize_global_longitude(u10, lons)
            v10, lons_v = _normalize_global_longitude(v10, lons_v)
            if (
                u10.shape != v10.shape
                or lats.shape != lats_v.shape
                or lons.shape != lons_v.shape
                or not np.allclose(lats, lats_v)
                or not np.allclose(lons, lons_v)
            ):
                raise HTTPException(status_code=500, detail="Wind components have mismatched grid shapes")
            plot_data = np.sqrt(u10 ** 2 + v10 ** 2)
        else:
            short_name = GLOBAL_MAP_VARIABLES[variable_key]["shortName"]
            plot_data, lats, lons = _extract_global_variable(output_file, short_name, step)
            plot_data, lons = _normalize_global_longitude(plot_data, lons)

        if variable_key == "t2m":
            plot_data = plot_data - 273.15
        elif variable_key == "msl":
            plot_data = plot_data / 100.0
        elif variable_key == "tp":
            plot_data = plot_data * 1000.0

        lat_min = float(np.min(lats))
        lat_max = float(np.max(lats))
        lon_min = float(np.min(lons))
        lon_max = float(np.max(lons))
        origin = "upper" if len(lats) > 1 and lats[0] > lats[-1] else "lower"

        meta = GLOBAL_MAP_VARIABLES[variable_key]
        use_cartopy = cartopy_available

        def add_map_features(ax):
            if not use_cartopy:
                return
            try:
                # Global map: keep coastline/country lines thin.
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="-", edgecolor="black", zorder=3)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=3)
            except Exception as e:
                logger.warning(f"Could not add map features: {e}")

        if use_cartopy:
            fig, ax = plt.subplots(figsize=(11, 8), subplot_kw={"projection": ccrs.PlateCarree()})
            lons_grid, lats_grid = np.meshgrid(lons, lats)
            im = ax.pcolormesh(
                lons_grid,
                lats_grid,
                plot_data,
                cmap=meta["cmap"],
                transform=ccrs.PlateCarree(),
                shading="auto",
                zorder=1,
            )
            extent = [lon_min, lon_max, lat_min, lat_max]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            add_map_features(ax)
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, zorder=4)
            gl.top_labels = False
            gl.right_labels = False
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(
                plot_data,
                cmap=meta["cmap"],
                extent=[lon_min, lon_max, lat_min, lat_max],
                origin=origin,
                aspect="auto",
            )
            ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)

        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(
            f"FourCastNet Global Map | {meta['label']} ({meta['unit']}) | "
            f"job={job_id}, step={step}"
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label(f"{meta['label']} ({meta['unit']})")

        stats_text = (
            f"Min: {float(np.nanmin(plot_data)):.2f}  "
            f"Max: {float(np.nanmax(plot_data)):.2f}  "
            f"Mean: {float(np.nanmean(plot_data)):.2f}"
        )
        fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.03, 1, 1])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.getvalue()

        map_filename = f"{job_id}_map_{variable_key}_global_step{step}.png"

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename={map_filename}"
            },
        )

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Map dependencies missing. Install: pip install matplotlib xarray cfgrib eccodes",
        )
    except Exception as e:
        logger.error(f"Map generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{job_id}/pipeline")
async def run_pipeline(job_id: str, region: str, step: int = 0):
    """
    Pipeline endpoint: Extract multi-variable regional data, downscale with bilinear interpolation, and analyze with LLM.

    This combines:
    1. FourCastNet global forecast output (multiple variables)
    2. Bilinear multi-variable downscaling for high-resolution local forecasts
    3. Ollama LLM for natural language weather advisory

    Variables extracted:
    - t2m: 2-meter temperature (K)
    - u10: 10-meter U wind component (m/s)
    - v10: 10-meter V wind component (m/s)
    - msl: Mean sea level pressure (Pa)

    Args:
        job_id: Forecast job ID
        region: Region name/key (NC, NL)
        step: Forecast step index (default: 0)

    Returns:
        Combined FourCastNet + bilinear downscaling + LLM analysis
    """
    if job_id not in running_forecasts:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    region_key = resolve_region_key(region)
    if not region_key:
        raise HTTPException(status_code=400, detail=f"Unknown region. Available: {list(REGIONS.keys())}")

    job = running_forecasts[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

    output_file = Path(job["output_file"])
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    try:
        extraction = extract_regional_variables(output_file, region_key, step)
        bounds = REGIONS[region_key]
        extracted_vars = extraction["variables"]
        var_stats = extraction["stats"]

        extraction_result = {
            "job_id": job_id,
            "region": extraction["region"],
            "region_name": extraction["region_name"],
            "region_info": extraction["region_info"],
            "step": step,
            "variables": var_stats,
            "extracted_count": len(extracted_vars),
            "shape": extraction["shape"],
        }

        # Send to Downscaler for multi-variable enhancement
        downscale_result = None
        try:
            payload = json.dumps({
                "variables": extracted_vars,
                "region": extraction["region"],
                "region_name": extraction["region_name"],
                "region_info": extraction["region_info"],
                "upscale_factor": 4,
                "method": "bilinear"
            })
            req = urllib.request.Request(
                f"{REGIONALCAST_URL}/downscale",
                data=payload.encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                downscale_result = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"Downscaler unavailable: {e}")
            downscale_result = {"error": str(e), "service_url": REGIONALCAST_URL}

        # Send to Ollama for LLM analysis
        llm_result = analyze_with_llm(bounds['name'], extraction_result, downscale_result)

        # Build summary analysis
        t2m_stats = var_stats.get("t2m", {})
        u10_stats = var_stats.get("u10", {})
        v10_stats = var_stats.get("v10", {})
        msl_stats = var_stats.get("msl", {})

        import math
        wind_speed = math.sqrt(u10_stats.get("mean", 0)**2 + v10_stats.get("mean", 0)**2)

        return {
            "pipeline": "fourcastnet-downscale-llm",
            "extraction": extraction_result,
            "downscale": downscale_result,
            "llm_analysis": llm_result,
            "summary": {
                "region": bounds['name'],
                "temperature_c": {
                    "min": round(t2m_stats.get("min", 273) - 273.15, 1),
                    "max": round(t2m_stats.get("max", 273) - 273.15, 1),
                    "mean": round(t2m_stats.get("mean", 273) - 273.15, 1),
                },
                "wind_speed_ms": round(wind_speed, 1),
                "pressure_hpa": round(msl_stats.get("mean", 101325) / 100, 1),
                "downscale_factor": downscale_result.get("upscale_factor", 4) if isinstance(downscale_result, dict) else 4,
                "output_resolution": f"~{0.25/4:.4f}° (~{25/4:.1f}km)" if isinstance(downscale_result, dict) and "error" not in downscale_result else "N/A",
            }
        }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="cfgrib/xarray not installed. Run: pip install xarray cfgrib eccodes"
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{job_id}/regional")
async def extract_region(job_id: str, region: str, step: int = 0):
    """
    Extract raw regional slices for t2m/u10/v10/msl from a completed forecast.
    """
    if job_id not in running_forecasts:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    region_key = resolve_region_key(region)
    if not region_key:
        raise HTTPException(status_code=400, detail=f"Unknown region. Available: {list(REGIONS.keys())}")

    job = running_forecasts[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

    output_file = Path(job["output_file"])
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    try:
        extraction = extract_regional_variables(output_file, region_key, step)
        response_payload = {
            "job_id": job_id,
            "region": extraction["region"],
            "region_name": extraction["region_name"],
            "region_info": extraction["region_info"],
            "step": step,
            "shape": extraction["shape"],
            "variables": extraction["variables"],
            "stats": extraction["stats"],
            "extracted_count": len(extraction["variables"]),
        }
        return response_payload
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="cfgrib/xarray not installed. Run: pip install xarray cfgrib eccodes",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Region extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regions")
async def list_regions():
    """List available regions for pipeline extraction."""
    return REGIONS


@app.get("/cache")
async def get_cache_status():
    """
    View ERA5 cache status.

    Shows cached GRIB files and in-progress downloads.
    """
    cached_files = []
    for f in OUTPUT_DIR.glob("era5_*.grib"):
        cached_files.append({
            "file": f.name,
            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })

    with era5_cache_lock:
        downloads = dict(era5_cache)

    return {
        "cache_dir": str(OUTPUT_DIR),
        "cached_files": sorted(cached_files, key=lambda x: x["modified"], reverse=True),
        "total_cached": len(cached_files),
        "total_size_mb": sum(f["size_mb"] for f in cached_files),
        "active_downloads": {k: v for k, v in downloads.items() if v.get("status") == "downloading"},
        "download_history": downloads,
    }


@app.delete("/cache/{cache_key}")
async def delete_cached_file(cache_key: str):
    """
    Delete a specific cached ERA5 file.

    Args:
        cache_key: The cache key (e.g., 'cds_20260122_1200_24h')
    """
    cached_path = get_cached_grib_path(cache_key)

    if not cached_path.exists():
        raise HTTPException(status_code=404, detail=f"Cache file not found: {cache_key}")

    # Remove from tracking
    with era5_cache_lock:
        if cache_key in era5_cache:
            del era5_cache[cache_key]

    # Delete file
    cached_path.unlink()
    logger.info(f"Deleted cached file: {cached_path}")

    return {"status": "deleted", "cache_key": cache_key}


@app.delete("/cache")
async def clear_cache():
    """
    Clear all cached ERA5 files.

    Warning: This deletes all cached GRIB files. In-progress downloads will fail.
    """
    deleted = []
    for f in OUTPUT_DIR.glob("era5_*.grib"):
        f.unlink()
        deleted.append(f.name)

    # Clear tracking
    with era5_cache_lock:
        era5_cache.clear()

    logger.info(f"Cleared cache: {len(deleted)} files deleted")

    return {
        "status": "cleared",
        "deleted_count": len(deleted),
        "deleted_files": deleted,
    }


# ============================================================================
# Run directly
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
