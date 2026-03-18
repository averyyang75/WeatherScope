"""
WeatherScope Demo Dashboard
Simple FastAPI proxy + static UI for KubeCon demonstrations.
"""

import os
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


@app.get("/api/global-map/{job_id}")
async def global_map(job_id: str, variable: str = Query("t2m"), step: int = Query(0, ge=0)):
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
async def extract(job_id: str, region: str = Query(...), step: int = Query(0)):
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(
            f"{FOURCASTNET_URL}/forecast/{job_id}/regional",
            params={"region": region, "step": step},
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
