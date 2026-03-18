"""
Downscaling Inference Service - FastAPI server for weather variable interpolation.
Provides bilinear and bicubic upscaling for multi-variable weather data.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_VARIABLES = ["t2m", "u10", "v10", "msl"]  # 2m temp, 10m wind u/v, sea level pressure
UPSCALE_FACTOR = int(os.getenv("UPSCALE_FACTOR", "4"))  # 0.25° → ~0.0625° (4x)

REGION_BOUNDS = {
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

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Weather Downscaling Service",
    description="Multi-variable weather downscaling using interpolation",
    version="3.0.0"
)


# ============================================================================
# Request/Response Models
# ============================================================================

class DownscaleRequest(BaseModel):
    """Request model for multi-variable downscaling."""
    variables: Dict[str, Any] | None = Field(
        None,
        description="Dict of variable name -> 2D array. E.g., {'t2m': [[...]], 'u10': [[...]]}"
    )
    region: str | None = Field(None, description="Region key, e.g. NC")
    region_name: str | None = Field(None, description="Region display name")
    region_info: Dict[str, Any] | None = Field(None, description="Region metadata")
    upscale_factor: int = Field(4, description="Upscaling factor (1, 2, 4, or 8)")
    method: str = Field("bilinear", description="Downscaling method: 'bilinear' or 'bicubic'")

    @model_validator(mode="before")
    @classmethod
    def support_wrapped_extraction_payload(cls, data: Any):
        if not isinstance(data, dict):
            return data
        if isinstance(data.get("extraction"), dict):
            extraction = data["extraction"]
            merged = dict(data)
            for field in ("variables", "region", "region_name", "region_info"):
                if merged.get(field) is None and extraction.get(field) is not None:
                    merged[field] = extraction.get(field)
            return merged
        return data


class DownscaleResponse(BaseModel):
    """Response model for multi-variable downscaling."""
    predictions: Dict[str, List[List[float]]]
    input_shape: List[int]
    output_shape: List[int]
    upscale_factor: int
    variables: List[str]
    stats: Dict[str, Dict[str, float]]
    method: str
    region: str | None = None
    region_name: str | None = None
    region_info: Dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    supported_methods: List[str]
    supported_variables: List[str]
    default_upscale_factor: int


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        supported_methods=["bilinear", "bicubic"],
        supported_variables=DEFAULT_VARIABLES,
        default_upscale_factor=UPSCALE_FACTOR
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Weather Downscaling Service",
        "version": "3.0.0",
        "endpoints": [
            "/health",
            "/downscale - Multi-variable interpolation (bilinear/bicubic)"
        ],
        "supported_variables": DEFAULT_VARIABLES,
        "supported_methods": ["bilinear", "bicubic"],
        "default_upscale_factor": UPSCALE_FACTOR
    }


def run_downscale(request: DownscaleRequest):
    """Shared downscaling logic for JSON and map endpoints."""
    method = request.method.lower()
    if method not in ("bilinear", "bicubic"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method: {method}. Use 'bilinear' or 'bicubic'"
        )

    if not isinstance(request.variables, dict):
        raise HTTPException(status_code=400, detail="variables must be provided as a variable-name map")

    var_names = list(request.variables.keys())
    if len(var_names) == 0:
        raise HTTPException(status_code=400, detail="No variables provided")

    arrays = []
    stats = {}

    for var_name in var_names:
        arr = np.array(request.variables[var_name], dtype=np.float32)
        if arr.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Variable {var_name} must be 2D array"
            )
        arrays.append(arr)
        stats[var_name] = {
            "input_mean": float(arr.mean()),
            "input_std": float(arr.std()),
            "input_min": float(arr.min()),
            "input_max": float(arr.max())
        }

    input_shape = list(arrays[0].shape)
    upscale = request.upscale_factor
    predictions = {}
    output_arrays = {}

    for i, var_name in enumerate(var_names):
        arr_tensor = torch.from_numpy(arrays[i][np.newaxis, np.newaxis, :, :])
        upscaled = F.interpolate(
            arr_tensor,
            scale_factor=upscale,
            mode=method,
            align_corners=True if method == "bicubic" else None
        )
        var_output = upscaled.numpy()[0, 0]
        output_arrays[var_name] = var_output
        predictions[var_name] = var_output.tolist()
        stats[var_name]["output_min"] = float(var_output.min())
        stats[var_name]["output_max"] = float(var_output.max())
        stats[var_name]["output_mean"] = float(var_output.mean())

    output_shape = [input_shape[0] * upscale, input_shape[1] * upscale]
    return method, var_names, arrays, output_arrays, predictions, stats, input_shape, output_shape

def resolve_region_key(region_name: str | None) -> str | None:
    """Resolve normalized region key from user input."""
    if not region_name:
        return None

    normalized = region_name.strip().lower()
    key = REGION_ALIASES.get(normalized, normalized)
    if key in REGION_BOUNDS:
        return key

    for candidate_key, candidate in REGION_BOUNDS.items():
        if candidate["name"].lower() == normalized:
            return candidate_key
    return None


def normalize_region_context(
    region: str | None,
    region_name: str | None,
    region_info: Dict[str, Any] | None,
) -> tuple[str | None, str | None, Dict[str, Any] | None]:
    """Normalize region metadata into a consistent shape."""
    info = dict(region_info) if isinstance(region_info, dict) else {}

    if isinstance(region, str) and region.strip():
        info.setdefault("key", region.strip())
    if isinstance(region_name, str) and region_name.strip():
        info.setdefault("name", region_name.strip())

    key_guess = resolve_region_key(info.get("key")) or resolve_region_key(info.get("name"))
    if key_guess and key_guess in REGION_BOUNDS:
        bounds = REGION_BOUNDS[key_guess]
        info["key"] = key_guess
        info["name"] = bounds["name"]
        info["bounds"] = {
            "north": bounds["north"],
            "south": bounds["south"],
            "east": bounds["east"],
            "west": bounds["west"],
        }

    resolved_region = info.get("key") if isinstance(info.get("key"), str) else None
    resolved_name = info.get("name") if isinstance(info.get("name"), str) else None
    return resolved_region, resolved_name, (info if info else None)


@app.post("/downscale", response_model=DownscaleResponse)
async def downscale(request: DownscaleRequest):
    """
    Multi-variable weather downscaling using interpolation.

    Input: Dict of variable names to 2D arrays
           E.g., {"t2m": [[...]], "u10": [[...]], "v10": [[...]], "msl": [[...]]}

    Output: Dict of downscaled variables at higher resolution

    Methods:
    - 'bilinear': Bilinear interpolation (default, preserves values)
    - 'bicubic': Bicubic interpolation (smoother)

    Supported variables: t2m (2m temperature), u10 (10m u-wind),
                         v10 (10m v-wind), msl (mean sea level pressure)
    """
    try:
        method, var_names, _, _, predictions, stats, input_shape, output_shape = run_downscale(request)
        region, region_name, region_info = normalize_region_context(
            request.region,
            request.region_name,
            request.region_info,
        )

        response_obj = DownscaleResponse(
            predictions=predictions,
            input_shape=input_shape,
            output_shape=output_shape,
            upscale_factor=request.upscale_factor,
            variables=var_names,
            stats=stats,
            method=method,
            region=region,
            region_name=region_name,
            region_info=region_info,
        )
        return response_obj

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Downscaling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run directly for development
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
