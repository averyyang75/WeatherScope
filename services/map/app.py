"""
WeatherScope Map Service
Renders comparison maps from extracted and downscaled weather payloads.
"""

import io
import logging
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


app = FastAPI(
    title="WeatherScope Map Service",
    description="Map renderer for map outputs",
    version="1.0.0",
)


class MapRequest(BaseModel):
    original_variables: Dict[str, Any] = Field(
        ...,
        description="Original variables map or full extract payload",
    )
    downscaled_variables: Dict[str, Any] = Field(
        ...,
        description="Downscaled variable map or full downscale payload",
    )
    upscale_factor: int = Field(4, description="Upscale factor label for map title")
    region: str | None = Field(None, description="Optional region key")
    region_name: str | None = Field(None, description="Optional region name")
    region_info: Dict[str, Any] | None = Field(None, description="Optional region metadata")


class RegionalMapRequest(BaseModel):
    variables: Dict[str, Any] | None = Field(
        None,
        description="Regional variable map from /forecast/{job_id}/regional",
    )
    stats: Dict[str, Any] | None = Field(
        None,
        description="Optional stats map from /forecast/{job_id}/regional",
    )
    region: str | None = Field(None, description="Optional region key")
    region_name: str | None = Field(None, description="Optional region name")
    region_info: Dict[str, Any] | None = Field(None, description="Optional region metadata")
    variable: str = Field("t2m", description="Variable to render, default t2m")
    step: int | None = Field(None, ge=-1, description="Optional forecast step label (-1 for latest)")

    @model_validator(mode="before")
    @classmethod
    def support_wrapped_payload(cls, data: Any):
        if not isinstance(data, dict):
            return data
        for wrapper_key in ("extraction", "regional_variables", "regional"):
            wrapped = data.get(wrapper_key)
            if isinstance(wrapped, dict):
                merged = dict(data)
                merged.pop(wrapper_key, None)
                for k, v in wrapped.items():
                    if merged.get(k) is None:
                        merged[k] = v
                return merged
        return data


class DownscaleMapRequest(BaseModel):
    predictions: Dict[str, Any] | None = Field(
        None,
        description="Downscaled variable map from /downscale output",
    )
    stats: Dict[str, Any] | None = Field(
        None,
        description="Optional stats map from /downscale output",
    )
    region: str | None = Field(None, description="Optional region key")
    region_name: str | None = Field(None, description="Optional region name")
    region_info: Dict[str, Any] | None = Field(None, description="Optional region metadata")
    variable: str = Field("t2m", description="Variable to render, default t2m")
    step: int | None = Field(None, ge=-1, description="Optional step label (-1 for latest)")
    method: str | None = Field(None, description="Optional downscale method label")
    upscale_factor: int | None = Field(None, ge=1, description="Optional upscale factor label")

    @model_validator(mode="before")
    @classmethod
    def support_wrapped_payload(cls, data: Any):
        if not isinstance(data, dict):
            return data
        for wrapper_key in ("downscale", "downscaled_variables"):
            wrapped = data.get(wrapper_key)
            if isinstance(wrapped, dict):
                merged = dict(data)
                merged.pop(wrapper_key, None)
                for k, v in wrapped.items():
                    if merged.get(k) is None:
                        merged[k] = v
                return merged
        return data


REGIONAL_STYLE = {
    "t2m": {"label": "2m Temperature", "unit": "C", "cmap": "RdYlBu_r"},
    "u10": {"label": "10m U-Wind", "unit": "m/s", "cmap": "RdBu_r"},
    "v10": {"label": "10m V-Wind", "unit": "m/s", "cmap": "RdBu_r"},
    "msl": {"label": "Mean Sea Level Pressure", "unit": "hPa", "cmap": "viridis"},
    "tp": {"label": "Total Precipitation", "unit": "mm", "cmap": "Blues"},
}


def resolve_region_key(region_name: str | None) -> str | None:
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


def resolve_region_bounds(region_name: str | None) -> Dict[str, float] | None:
    key = resolve_region_key(region_name)
    if not key:
        return None
    if key in REGION_BOUNDS:
        return REGION_BOUNDS[key]
    normalized = region_name.strip().lower()
    for candidate in REGION_BOUNDS.values():
        if candidate["name"].lower() == normalized:
            return candidate
    return None


def normalize_region_context(
    region: str | None,
    region_name: str | None,
    region_info: Dict[str, Any] | None,
) -> tuple[str | None, str | None, Dict[str, Any] | None]:
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


def region_context_from_payload(payload: Any) -> tuple[str | None, str | None, Dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, None, None
    return normalize_region_context(
        region=payload.get("region"),
        region_name=payload.get("region_name"),
        region_info=payload.get("region_info"),
    )


def geometry_to_lines(geometry: Any) -> List[List[List[float]]]:
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


def attr_lower(attrs: Dict[str, Any], *keys: str) -> str:
    lower_attrs = {str(k).lower(): v for k, v in attrs.items()}
    for key in keys:
        value = lower_attrs.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return ""


def boundary_lines_from_natural_earth(region_key: str | None) -> List[List[List[float]]]:
    if not CARTOPY_AVAILABLE or not region_key:
        return []

    try:
        if region_key == "NC":
            for shape_name in ("admin_1_states_provinces", "admin_1_states_provinces_lakes"):
                try:
                    path = shpreader.natural_earth(resolution="10m", category="cultural", name=shape_name)
                    matches: List[List[List[float]]] = []
                    for record in shpreader.Reader(path).records():
                        attrs = record.attributes
                        name = attr_lower(attrs, "name", "name_en")
                        admin = attr_lower(attrs, "admin", "adm0_name", "sovereignt", "geonunit")
                        postal = attr_lower(attrs, "postal", "iso_3166_2", "hasc")
                        is_nc = (name == "north carolina") or postal.replace(".", "").endswith("nc")
                        is_us = ("united states" in admin) or (admin in {"us", "usa"})
                        if is_nc and (is_us or admin == ""):
                            matches.extend(geometry_to_lines(record.geometry))
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
                    attr_lower(attrs, "name", "name_en", "admin"),
                    attr_lower(attrs, "name_long", "formal_en"),
                    attr_lower(attrs, "sovereignt", "geounit"),
                }
                if any("netherlands" in value for value in candidates if value):
                    return geometry_to_lines(record.geometry)
    except Exception as e:
        logger.warning(f"Natural Earth boundary load failed for region={region_key}: {e}")

    return []


def resolve_context_boundary_lines(region_key: str | None) -> List[List[List[float]]]:
    if not region_key:
        return []
    ne_lines = boundary_lines_from_natural_earth(region_key)
    if ne_lines:
        return ne_lines
    fallback = REGION_CONTEXT_BOUNDARIES.get(region_key, [])
    if fallback:
        return [fallback]
    return []


def _render_single_regional_map(
    variables: Dict[str, Any],
    region: str | None,
    region_name: str | None,
    region_info: Dict[str, Any] | None,
    variable: str,
    step: int | None = None,
    title_prefix: str = "Regional",
    subtitle: str | None = None,
) -> Response:
    if not isinstance(variables, dict) or not variables:
        raise HTTPException(status_code=400, detail="variables must be provided as a variable-name map")

    var_key = variable.strip().lower()
    if var_key not in variables:
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not found. Available: {sorted(variables.keys())}",
        )

    resolved_region, resolved_region_name, resolved_region_info = normalize_region_context(
        region, region_name, region_info
    )

    bounds = None
    if isinstance(resolved_region_info, dict) and isinstance(resolved_region_info.get("bounds"), dict):
        b = resolved_region_info["bounds"]
        if {"north", "south", "east", "west"}.issubset(b.keys()):
            bounds = {
                "north": float(b["north"]),
                "south": float(b["south"]),
                "east": float(b["east"]),
                "west": float(b["west"]),
            }
    if bounds is None:
        bounds = resolve_region_bounds(resolved_region_name or resolved_region)
    if bounds is None:
        raise HTTPException(status_code=400, detail="Region bounds are required for regional map rendering")

    data = np.array(variables[var_key], dtype=np.float32)
    if data.ndim != 2:
        raise HTTPException(status_code=400, detail=f"Variable {var_key} must be a 2D array")

    style = REGIONAL_STYLE.get(var_key, {"label": var_key, "unit": "", "cmap": "viridis"})
    if var_key == "t2m":
        data = data - 273.15
    elif var_key == "msl":
        data = data / 100.0
    elif var_key == "tp":
        data = data * 1000.0

    region_key = resolve_region_key(resolved_region or resolved_region_name)
    use_cartopy = CARTOPY_AVAILABLE
    boundary_lines = resolve_context_boundary_lines(region_key) if not use_cartopy else []
    extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]

    def add_map_features(ax):
        if not use_cartopy:
            return
        try:
            states_10m = cfeature.NaturalEarthFeature(
                "cultural", "admin_1_states_provinces_lines", "10m",
                facecolor="none", edgecolor="black", linewidth=0.9
            )
            ax.add_feature(states_10m, zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=1.0, linestyle="-", edgecolor="black", zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black", zorder=3)
        except Exception as e:
            logger.warning(f"Could not add regional map features: {e}")

    def add_context_boundary(ax):
        for line in boundary_lines:
            if len(line) < 2:
                continue
            lons = [point[0] for point in line]
            lats = [point[1] for point in line]
            kwargs = {"zorder": 6}
            if use_cartopy:
                kwargs["transform"] = ccrs.PlateCarree()
            ax.plot(lons, lats, color="white", linewidth=3.0, linestyle="-", **kwargs)
            ax.plot(lons, lats, color="black", linewidth=1.6, linestyle="-", **kwargs)

    if use_cartopy:
        fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": ccrs.PlateCarree()})
        lons = np.linspace(bounds["west"], bounds["east"], data.shape[1])
        lats = np.linspace(bounds["north"], bounds["south"], data.shape[0])
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        im = ax.pcolormesh(
            lon_grid,
            lat_grid,
            data,
            cmap=style["cmap"],
            transform=ccrs.PlateCarree(),
            shading="auto",
            zorder=1,
        )
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        add_map_features(ax)
        add_context_boundary(ax)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, zorder=4)
        gl.top_labels = False
        gl.right_labels = False
    else:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(
            data,
            cmap=style["cmap"],
            extent=extent,
            origin="upper",
            aspect="auto",
        )
        ax.set_xlim(bounds["west"], bounds["east"])
        ax.set_ylim(bounds["south"], bounds["north"])
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
        add_context_boundary(ax)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title_region = resolved_region_name or resolved_region or "Region"
    subtitle_label = f" | {subtitle}" if subtitle else ""
    ax.set_title(
        f"{title_prefix} Map - {style['label']} ({style['unit']}) - {title_region}{subtitle_label}".rstrip()
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(f"{style['label']} ({style['unit']})")

    stats_text = (
        f"Min: {float(np.nanmin(data)):.2f}  "
        f"Max: {float(np.nanmax(data)):.2f}  "
        f"Mean: {float(np.nanmean(data)):.2f}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    png_bytes = buffer.getvalue()

    filename_region = (resolved_region or "region").replace(" ", "_")
    prefix = title_prefix.strip().lower().replace(" ", "_") or "map"
    filename = f"{prefix}_{filename_region}_{var_key}.png"
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "matplotlib_available": MATPLOTLIB_AVAILABLE,
        "cartopy_available": CARTOPY_AVAILABLE,
    }


@app.get("/")
async def root():
    return {
        "service": "WeatherScope Map Service",
        "version": "1.0.0",
        "endpoints": ["/health", "/map", "/map/regional", "/map/downscale"],
    }


@app.post("/map")
async def render_map(request: MapRequest):
    if not MATPLOTLIB_AVAILABLE:
        raise HTTPException(status_code=500, detail="matplotlib not installed")

    try:
        original_payload = request.original_variables
        downscaled_payload = request.downscaled_variables

        original_map = original_payload.get("variables", original_payload)
        downscaled_map = downscaled_payload.get("predictions", downscaled_payload)

        if not isinstance(original_map, dict) or not isinstance(downscaled_map, dict):
            raise HTTPException(
                status_code=400,
                detail="original_variables and downscaled_variables must be variable-name maps",
            )

        var_names = list(original_map.keys())
        if len(var_names) == 0:
            raise HTTPException(status_code=400, detail="No original variables provided")

        if set(var_names) != set(downscaled_map.keys()):
            raise HTTPException(
                status_code=400,
                detail="Variable keys in original_variables and downscaled_variables must match",
            )

        req_region, req_region_name, req_info = normalize_region_context(
            request.region, request.region_name, request.region_info
        )
        ds_region, ds_region_name, ds_info = region_context_from_payload(downscaled_payload)
        orig_region, orig_region_name, orig_info = region_context_from_payload(original_payload)

        region, region_name, region_info = normalize_region_context(
            req_region or ds_region or orig_region,
            req_region_name or ds_region_name or orig_region_name,
            req_info or ds_info or orig_info,
        )

        bounds = None
        if isinstance(region_info, dict) and isinstance(region_info.get("bounds"), dict):
            b = region_info["bounds"]
            if {"north", "south", "east", "west"}.issubset(b.keys()):
                bounds = {
                    "north": float(b["north"]),
                    "south": float(b["south"]),
                    "east": float(b["east"]),
                    "west": float(b["west"]),
                }
        if bounds is None:
            bounds = resolve_region_bounds(region_name or region)

        region_key = resolve_region_key(region or region_name)
        use_geo = isinstance(bounds, dict) and {"north", "south", "east", "west"}.issubset(bounds.keys())
        use_cartopy = use_geo and CARTOPY_AVAILABLE
        boundary_lines = resolve_context_boundary_lines(region_key) if (use_geo and not use_cartopy) else []

        def add_map_features(ax):
            if not use_cartopy:
                return
            try:
                states_10m = cfeature.NaturalEarthFeature(
                    "cultural", "admin_1_states_provinces_lines", "10m",
                    facecolor="none", edgecolor="black", linewidth=1.0
                )
                ax.add_feature(states_10m, zorder=2)
                ax.add_feature(cfeature.BORDERS, linewidth=1.2, linestyle="-", edgecolor="black", zorder=3)
                ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor="black", zorder=3)
            except Exception as e:
                logger.warning(f"Could not add map features: {e}")

        def add_context_boundary(ax):
            for line in boundary_lines:
                if len(line) < 2:
                    continue
                lons = [point[0] for point in line]
                lats = [point[1] for point in line]
                kwargs = {"zorder": 6}
                if use_cartopy:
                    kwargs["transform"] = ccrs.PlateCarree()
                ax.plot(lons, lats, color="white", linewidth=3.0, linestyle="-", **kwargs)
                ax.plot(lons, lats, color="black", linewidth=1.6, linestyle="-", **kwargs)

        subplot_kwargs = {"projection": ccrs.PlateCarree()} if use_cartopy else {}
        fig, axes = plt.subplots(
            len(var_names),
            2,
            figsize=(12, 4 * len(var_names)),
            constrained_layout=True,
            subplot_kw=subplot_kwargs,
        )
        if len(var_names) == 1:
            axes = np.array([axes])

        style = {
            "t2m": {"label": "2m Temperature (K)", "cmap": "RdYlBu_r"},
            "u10": {"label": "10m U-Wind (m/s)", "cmap": "RdBu_r"},
            "v10": {"label": "10m V-Wind (m/s)", "cmap": "RdBu_r"},
            "msl": {"label": "Mean Sea Level Pressure (Pa)", "cmap": "viridis"},
        }

        for i, var_name in enumerate(var_names):
            original = np.array(original_map[var_name], dtype=np.float32)
            downscaled = np.array(downscaled_map[var_name], dtype=np.float32)
            if original.ndim != 2 or downscaled.ndim != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Variable {var_name} must be 2D in both original and downscaled payloads",
                )

            var_style = style.get(var_name, {"label": var_name, "cmap": "viridis"})
            vmin = min(float(original.min()), float(downscaled.min()))
            vmax = max(float(original.max()), float(downscaled.max()))
            extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]] if use_geo else None

            if use_geo and use_cartopy:
                lons_orig = np.linspace(bounds["west"], bounds["east"], original.shape[1])
                lats_orig = np.linspace(bounds["north"], bounds["south"], original.shape[0])
                lon_grid_orig, lat_grid_orig = np.meshgrid(lons_orig, lats_orig)
                im_left = axes[i, 0].pcolormesh(
                    lon_grid_orig,
                    lat_grid_orig,
                    original,
                    cmap=var_style["cmap"],
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    zorder=1,
                )
                axes[i, 0].set_extent(extent, crs=ccrs.PlateCarree())
                add_map_features(axes[i, 0])
            else:
                im_left = axes[i, 0].imshow(
                    original, cmap=var_style["cmap"], vmin=vmin, vmax=vmax, origin="upper", extent=extent
                )

            axes[i, 0].set_title(f"{var_style['label']} - Original")
            axes[i, 0].set_xlabel("Longitude" if use_geo else "Lon Index")
            axes[i, 0].set_ylabel("Latitude" if use_geo else "Lat Index")

            if use_geo and use_cartopy:
                lons_ds = np.linspace(bounds["west"], bounds["east"], downscaled.shape[1])
                lats_ds = np.linspace(bounds["north"], bounds["south"], downscaled.shape[0])
                lon_grid_ds, lat_grid_ds = np.meshgrid(lons_ds, lats_ds)
                im_right = axes[i, 1].pcolormesh(
                    lon_grid_ds,
                    lat_grid_ds,
                    downscaled,
                    cmap=var_style["cmap"],
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    zorder=1,
                )
                axes[i, 1].set_extent(extent, crs=ccrs.PlateCarree())
                add_map_features(axes[i, 1])
            else:
                im_right = axes[i, 1].imshow(
                    downscaled, cmap=var_style["cmap"], vmin=vmin, vmax=vmax, origin="upper", extent=extent
                )

            axes[i, 1].set_title(f"{var_style['label']} - Downscaled")
            axes[i, 1].set_xlabel("Longitude" if use_geo else "Lon Index")
            axes[i, 1].set_ylabel("Latitude" if use_geo else "Lat Index")

            if use_geo and not use_cartopy:
                axes[i, 0].set_xlim(bounds["west"], bounds["east"])
                axes[i, 0].set_ylim(bounds["south"], bounds["north"])
                axes[i, 1].set_xlim(bounds["west"], bounds["east"])
                axes[i, 1].set_ylim(bounds["south"], bounds["north"])
                add_context_boundary(axes[i, 0])
                add_context_boundary(axes[i, 1])

            fig.colorbar(im_right, ax=[axes[i, 0], axes[i, 1]], shrink=0.8)

        title_label = region_name or region
        title_region = f" - {title_label}" if title_label else ""
        fig.suptitle(f"Downscaling Comparison{title_region} (factor={request.upscale_factor}x)", fontsize=14)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        png_bytes = buffer.getvalue()

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=downscale_comparison.png"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare map generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/map/regional")
async def render_regional_map(request: RegionalMapRequest):
    """
    Render a regional weather map from FourCastNet regional extraction payload.
    Designed to consume output from GET /forecast/{job_id}/regional.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise HTTPException(status_code=500, detail="matplotlib not installed")

    try:
        return _render_single_regional_map(
            variables=request.variables or {},
            region=request.region,
            region_name=request.region_name,
            region_info=request.region_info,
            variable=request.variable,
            step=request.step,
            title_prefix="Regional",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Regional map generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/map/downscale")
async def render_downscale_map(request: DownscaleMapRequest):
    """
    Render a regional weather map from /downscale output payload.
    Designed to consume output from POST /downscale.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise HTTPException(status_code=500, detail="matplotlib not installed")

    try:
        return _render_single_regional_map(
            variables=request.predictions or {},
            region=request.region,
            region_name=request.region_name,
            region_info=request.region_info,
            variable=request.variable,
            step=request.step,
            title_prefix="Downscaled",
            subtitle=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Downscale map generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
