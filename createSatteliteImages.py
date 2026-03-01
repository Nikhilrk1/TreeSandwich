#!/usr/bin/env python3


from __future__ import annotations

from io import BytesIO
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from shapely.geometry import box, Point, Polygon

import numpy as np
import pandas as pd
import requests
from PIL import Image

import geopandas as gpd
import pyproj
from shapely.geometry import box, Point
from shapely.ops import linemerge, substring, transform as shp_transform, nearest_points

import matplotlib.pyplot as plt

import ee  # earthengine-api
from deepforest import main as df_main
from PIL import ImageEnhance


SECTION_LEN_M = 1609.344 / 12.0     
HALF_SIDE_M = SECTION_LEN_M / 2.0  

WORK_CRS = "EPSG:3857"  
WGS84 = "EPSG:4326"

import cv2
from skimage import exposure, img_as_ubyte
from PIL import ImageEnhance, ImageFilter
import numpy as np


def match_histogram_to_reference(img_np, ref_img_np):
    """
    Match img_np color histogram to ref_img_np using skimage.exposure.match_histograms.
    Both arrays expected uint8 RGB (H,W,3). Returns uint8.
    """
    try:
        from skimage.exposure import match_histograms
    except Exception:
        raise RuntimeError("Please install scikit-image (pip install scikit-image) for histogram matching.")
    matched = match_histograms(img_np, ref_img_np, channel_axis=-1)
    matched_u8 = np.clip(img_as_ubyte(matched), 0, 255).astype(np.uint8)
    return matched_u8


def simple_dehaze(img_np, omega=0.95, win=15):
    """
    Quick approximate dehaze. Returns uint8 RGB.
    img_np: HxWx3 uint8 in range 0-255
    omega: amount to recover (0-1). Larger => stronger correction.
    win: atmospheric light local window for dark channel
    """

    I = img_np.astype(np.float32) / 255.0
    min_channel = np.min(I, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dark = cv2.erode(min_channel, kernel)

    flat = dark.ravel()
    n = len(flat)
    topk = max(1, n // 1000)
    inds = np.argpartition(flat, -topk)[-topk:]
    atm_vals = I.reshape(-1, 3)[inds]
    A = atm_vals.mean(axis=0)
    # estimate transmission
    t = 1 - omega * dark
    t = np.clip(t, 0.1, 1.0)[:, :, None]
    # recover radiance
    J = (I - A) / t + A
    J = np.clip(J, 0, 1.0)
    return (J * 255.0).astype(np.uint8)


def apply_clahe_rgb(img_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb


def adjust_gamma_np(img_np, gamma=0.9):
    # gamma < 1 brightens; gamma > 1 darkens
    inv = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return table[img_np]


def unsharp_mask(img_np, amount=1.0, radius=1):
    # amount ~ 0.5-1.5; radius ~ 1-2
    pil = Image.fromarray(img_np)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    sharp = Image.blend(pil, blurred, alpha=-amount)  # negative alpha to sharpen
    # safer alternative manual: pil + (pil-blur)*amount
    return np.array(sharp)

def make_image_look_newer(pil_img: Image.Image,
                          gamma: float = 0.9,
                          clahe_clip: float = 2.0,
                          clahe_grid: tuple = (8, 8),
                          dehaze_amount: float = 0.9,
                          saturation_mult: float = 1.15,
                          contrast_mult: float = 1.05,
                          brightness_mult: float = 1.03,
                          sharpen_amount: float = 0.8,
                          reference_img: Image.Image | None = None) -> Image.Image:

    if pil_img is None:
        return pil_img

    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.uint8)


    if reference_img is not None:
        ref = np.array(reference_img.convert("RGB")).astype(np.uint8)
        try:
            arr = match_histogram_to_reference(arr, ref)
        except Exception:

            pass


    if gamma is not None and gamma != 1.0:
        arr = adjust_gamma_np(arr, gamma=gamma)


    if dehaze_amount is not None and (0.0 < dehaze_amount <= 1.0):
        arr = simple_dehaze(arr, omega=dehaze_amount, win=15)


    arr = apply_clahe_rgb(arr, clip_limit=clahe_clip, tile_grid_size=clahe_grid)


    pil = Image.fromarray(arr)
    if saturation_mult is not None and saturation_mult != 1.0:
        pil = ImageEnhance.Color(pil).enhance(float(saturation_mult))
    if contrast_mult is not None and contrast_mult != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(float(contrast_mult))
    if brightness_mult is not None and brightness_mult != 1.0:
        pil = ImageEnhance.Brightness(pil).enhance(float(brightness_mult))


    if sharpen_amount is not None and sharpen_amount > 0:
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=int(150 * sharpen_amount), threshold=3))

    return pil


def split_into_sections(geom, section_len_m: float) -> List[Any]:
    """Return LineString sections of ~section_len_m along the line (geom units)."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)

    parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
    sections = []
    for part in parts:
        if part.is_empty or part.length == 0:
            continue
        d = 0.0
        while d < part.length:
            end = min(d + section_len_m, part.length)
            seg = substring(part, d, end)
            if seg.geom_type == "LineString" and not seg.is_empty and seg.length > 0:
                sections.append(seg)
            d = end
    return sections


def midpoint_along_line(geom) -> Optional[Point]:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "MultiLineString":
        parts = list(geom.geoms)
        parts.sort(key=lambda p: p.length, reverse=True)
        geom = parts[0]
    return geom.interpolate(geom.length / 2.0)


def make_square_bbox(point: Point, half_side_m: float):
    x, y = point.x, point.y
    return box(x - half_side_m, y - half_side_m, x + half_side_m, y + half_side_m)


def plot_linestring(ax, geom, **kwargs):
    """Plot LineString or MultiLineString onto an axes."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        x, y = geom.xy
        ax.plot(x, y, **kwargs)
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            if not part.is_empty:
                x, y = part.xy
                ax.plot(x, y, **kwargs)


def ensure_rgb(img):
    """Ensure a numpy array or PIL Image is uint8 RGB (H,W,3)."""
    try:
        from PIL import Image as _PilImage  # noqa
        if isinstance(img, _PilImage.Image):
            return np.array(img.convert("RGB")).astype(np.uint8)
    except Exception:
        pass

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if arr.dtype != np.uint8:
        if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_deepforest_model(model_name: str | None = None):
    m = df_main.deepforest()
    if model_name:
        try:
            m.load_model(model_name=model_name)
        except TypeError:
            m.load_model(model_name)
    else:
        try:
            m.load_model()
        except TypeError:
            try:
                m.load()
            except Exception:
                pass
    return m


def detect_trees_deepforest(model, image_array) -> pd.DataFrame:
    """Return DataFrame columns ['xmin','ymin','xmax','ymax','score'] in pixel coords."""
    img = ensure_rgb(image_array)
    preds = None
    try:
        preds = model.predict_image(image=img)
    except TypeError:
        preds = model.predict_image(img)

    if preds is None or len(preds) == 0:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score"])

    cols_map = {c.lower(): c for c in preds.columns}
    for key in ["xmin", "ymin", "xmax", "ymax", "score", "confidence"]:
        if key not in preds.columns and key in cols_map:
            preds[key] = preds[cols_map[key]]

    if "score" not in preds.columns:
        if "confidence" in preds.columns:
            preds["score"] = preds["confidence"]
        else:
            preds["score"] = 1.0

    return preds[["xmin", "ymin", "xmax", "ymax", "score"]].reset_index(drop=True)



def pixel_center_to_lonlat(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_shape: Tuple[int, int],
    extent_lonlat: Tuple[float, float, float, float],
) -> Tuple[float, float]:

    height, width = int(img_shape[0]), int(img_shape[1])
    minlon, maxlon, minlat, maxlat = extent_lonlat

    xres = (maxlon - minlon) / float(width)
    yres = (maxlat - minlat) / float(height)

    px = (xmin + xmax) / 2.0
    py = (ymin + ymax) / 2.0

    lon = minlon + px * xres
    lat = maxlat - py * yres  # origin='upper'
    return lon, lat


def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """Return a UTM EPSG string like 'EPSG:32617' based on lon/lat."""
    zone = int((lon + 180.0) // 6.0) + 1
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"

def pixel_bbox_to_lonlat_polygon(xmin, ymin, xmax, ymax, img_shape, extent_lonlat):
    """Return shapely Polygon in lon/lat for pixel bbox coords (origin upper-left)."""
    height, width = int(img_shape[0]), int(img_shape[1])
    minlon, maxlon, minlat, maxlat = extent_lonlat
    xres = (maxlon - minlon) / float(width)
    yres = (maxlat - minlat) / float(height)

    corners_px = [
        (xmin, ymin),  # top-left (px,py)
        (xmax, ymin),  # top-right
        (xmax, ymax),  # bottom-right
        (xmin, ymax),  # bottom-left
    ]
    corners_lonlat = []
    for px, py in corners_px:
        lon = minlon + px * xres
        lat = maxlat - py * yres 
        corners_lonlat.append((lon, lat))
    return Polygon(corners_lonlat)

def closest_tree_to_line_distance(
    preds: pd.DataFrame,
    line_ll,
    extent_lonlat: Tuple[float, float, float, float],
    img_shape: Tuple[int, int],
    utm_epsg: str,
) -> Optional[Dict[str, Any]]:

    if preds is None or preds.empty or line_ll is None or line_ll.is_empty:
        return None

    to_utm = pyproj.Transformer.from_crs(WGS84, utm_epsg, always_xy=True)
    from_utm = pyproj.Transformer.from_crs(utm_epsg, WGS84, always_xy=True)

    line_utm = shp_transform(to_utm.transform, line_ll)

    best = {
        "distance_m": float("inf"),
        "tree_lon": None,
        "tree_lat": None,
        "line_lon": None,
        "line_lat": None,
        "ntrees": int(len(preds)),
    }

    EPS_M = 1e-2  

    h, w = int(img_shape[0]), int(img_shape[1])

    for _, r in preds.iterrows():

        try:
            xmin = float(r.xmin); ymin = float(r.ymin)
            xmax = float(r.xmax); ymax = float(r.ymax)
        except Exception:
            continue
        if not np.isfinite(xmin + ymin + xmax + ymax):
            continue


        try:
            poly_ll = pixel_bbox_to_lonlat_polygon(xmin, ymin, xmax, ymax, img_shape=(h, w), extent_lonlat=extent_lonlat)
        except Exception:
            continue
        if poly_ll.is_empty:
            continue


        try:
            poly_utm = shp_transform(to_utm.transform, poly_ll)
        except Exception:
            continue
        if not poly_utm.is_valid:
            poly_utm = poly_utm.buffer(0)
        if poly_utm.is_empty:
            continue


        try:
            d_m = poly_utm.distance(line_utm)
            if not np.isfinite(d_m):
                continue
        except Exception:
            continue


        if d_m <= EPS_M:

            continue


        on_poly_utm, on_line_utm = nearest_points(poly_utm, line_utm)
        tree_lon, tree_lat = from_utm.transform(on_poly_utm.x, on_poly_utm.y)
        line_lon, line_lat = from_utm.transform(on_line_utm.x, on_line_utm.y)

        if d_m < best["distance_m"]:
            best["distance_m"] = float(d_m)
            best["tree_lon"] = float(tree_lon)
            best["tree_lat"] = float(tree_lat)
            best["line_lon"] = float(line_lon)
            best["line_lat"] = float(line_lat)

    if not np.isfinite(best["distance_m"]):
        return None
    return best


def init_earth_engine():
    """Initialize EE. If it fails, raise with a helpful message."""
    try:
        ee.Authenticate()
        ee.Initialize(project='earth-engine-project-488823')
    except Exception as e:
        raise RuntimeError(


            "Earth Engine is not initialized. Run `earthengine authenticate` once, "
            "then try again. If you're using a service account, initialize EE accordingly."
        ) from e


def download_naip_tile(
    region: ee.Geometry,
    target_year: int,
    size_px: int,
    lookback_years: int = 5,
    timeout_s: int = 90,
) -> Tuple[Optional[Image.Image], Optional[int], int]:

    for y in range(target_year, target_year - lookback_years - 1, -1):
        start = f"{y}-01-01"
        end = f"{y+1}-01-01"
        col = (
            ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(region)
            .filterDate(start, end)
        )
        count = int(col.size().getInfo())
        if count <= 0:
            continue

        img = col.sort("system:time_start", False).mosaic().select(["R", "G", "B"])
        params = {
            "bands": ["R", "G", "B"],
            "min": 0,
            "max": 255,
            "region": region,
            "dimensions": f"{size_px}x{size_px}",
            "format": "png",
        }
        url = img.getThumbURL(params)
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        pil = Image.open(BytesIO(r.content)).convert("RGB")
        return pil, y, count

    return None, None, 0


def download_s2_tile(
    region: ee.Geometry,
    target_year: int,
    size_px: int,
    cloud_pct: float = 20.0,
    timeout_s: int = 90,
) -> Tuple[Optional[Image.Image], Optional[int], int]:
    """
    Sentinel-2 fallback (10m). Not ideal for DeepForest but useful if NAIP isn't available.
    Returns (PIL image or None, year_used or None, image_count_in_year).
    """
    y = target_year
    start = f"{y}-01-01"
    end = f"{y+1}-01-01"
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    count = int(col.size().getInfo())
    if count <= 0:
        return None, None, 0

    img = col.median().select(["B4", "B3", "B2"])  # RGB
    params = {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "region": region,
        "dimensions": f"{size_px}x{size_px}",
        "format": "png",
    }
    url = img.getThumbURL(params)
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    pil = Image.open(BytesIO(r.content)).convert("RGB")
    return pil, y, count


def download_rgb_tile(
    dataset: str,
    region: ee.Geometry,
    target_year: int,
    size_px: int,
    lookback_years: int,
    s2_cloud_pct: float,
) -> Tuple[Optional[Image.Image], Optional[int], int, str]:
    """Unified entrypoint to download a tile. Returns (image, year_used, count, dataset_used)."""
    dataset = dataset.lower().strip()
    if dataset == "naip":
        img, y_used, count = download_naip_tile(region, target_year, size_px, lookback_years=lookback_years)
        if img is not None:
            return img, y_used, count, "NAIP"

        s2_img, s2_y, s2_count = download_s2_tile(region, target_year, size_px, cloud_pct=s2_cloud_pct)
        return s2_img, s2_y, s2_count, "Sentinel-2"
    elif dataset in ("s2", "sentinel2", "sentinel-2"):
        img, y_used, count = download_s2_tile(region, target_year, size_px, cloud_pct=s2_cloud_pct)
        return img, y_used, count, "Sentinel-2"
    else:
        raise ValueError("Unknown dataset. Use 'naip' (default) or 's2'.")



def save_overlay_png(
    pil_img: Image.Image,
    extent_lonlat: Tuple[float, float, float, float],
    line_ll,
    closest: Optional[Dict[str, Any]],
    out_path: Path,
    size_px: int,
    dpi: int,
    title: Optional[str] = None,
):
    """Save overlay showing imagery, power line, and closest tree-line distance."""
    minlon, maxlon, minlat, maxlat = extent_lonlat

    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax.imshow(pil_img, extent=(minlon, maxlon, minlat, maxlat), origin="upper", zorder=0)

    plot_linestring(ax, line_ll, linewidth=1.0, color="yellow", zorder=10)

    if closest is not None:
        p_tree = (closest["tree_lon"], closest["tree_lat"])
        p_line = (closest["line_lon"], closest["line_lat"])
        ax.plot([p_tree[0], p_line[0]], [p_tree[1], p_line[1]], color="cyan", linewidth=1.0, zorder=20)
        ax.scatter([p_tree[0], p_line[0]], [p_tree[1], p_line[1]], s=18, color="cyan", zorder=21)

        dist_m = float(closest["distance_m"])
        mx, my = (p_tree[0] + p_line[0]) / 2.0, (p_tree[1] + p_line[1]) / 2.0
        ax.text(
            mx,
            my,
            f"{dist_m:.1f} m ({dist_m * 3.28084:.1f} ft)",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
            zorder=22,
        )

    if title:
        ax.set_title(title, fontsize=9)

    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Power line GeoJSON (EPSG:4326 lon/lat)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--size", type=int, default=2048, help="Pixel width/height")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--dataset", type=str, default="naip", help="naip (default) or s2")
    p.add_argument("--s2_cloud", type=float, default=20.0, help="Sentinel-2 cloud filter (percent)")
    p.add_argument("--lookback", type=int, default=5, help="NAIP year lookback if target year has no data")
    p.add_argument("--year_now", type=int, default=None, help="Override 'this year' (default: current year)")
    p.add_argument("--year_past", type=int, default=None, help="Override 'one year ago' (default: year_now-1)")
    p.add_argument("--model-name", type=str, default=None, help="DeepForest model name (optional)")
    p.add_argument("--sc-only", action="store_true", help="Rough South Carolina bbox filter (lon/lat)")
    p.add_argument("--state-shp", type=str, default=None, help="Optional state polygon shapefile/geojson filter")
    p.add_argument("--skip-overlays", action="store_true", help="Skip PNG overlays (CSV only)")
    p.add_argument("--max-backlook", type=int, default=20, help="Max years to step back when searching for an earlier 'past' year")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    init_earth_engine()


    global DEEPFOREST_MODEL
    DEEPFOREST_MODEL = load_deepforest_model(model_name=args.model_name)
    print("DeepForest model loaded.")

    year_now = args.year_now or datetime.now().year
    year_past = args.year_past or (year_now - 1)


    gdf = gpd.read_file(Path(args.input))
    if gdf.empty:
        raise SystemExit("Input has no features.")
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    else:
        gdf = gdf.to_crs(WGS84)


    if args.state_shp:
        state = gpd.read_file(args.state_shp).to_crs(WGS84)
        state_poly = state.unary_union
        gdf = gdf[gdf.intersects(state_poly)]
    elif args.sc_only:
        sc_box = box(-83.3532, 32.0335, -78.4019, 35.2154)
        gdf = gdf[gdf.intersects(sc_box)]

    if gdf.empty:
        raise SystemExit("No features after filtering; nothing to do.")


    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    if gdf.empty:
        raise SystemExit("No line geometries present.")


    gdf_m = gdf.to_crs(WORK_CRS)
    t_3857_to_4326 = pyproj.Transformer.from_crs(WORK_CRS, WGS84, always_xy=True)

    results = []
    stop = False
    comparison_count = 0
    for feat_idx, row in gdf_m.iterrows():
        
        geom_m = row.geometry
        sections = split_into_sections(geom_m, SECTION_LEN_M)

        if (stop == True):
            break
        for sec_idx, section_m in enumerate(sections):

            if (stop == True):
                break
            mp_m = midpoint_along_line(section_m)
            if mp_m is None:
                continue

            bbox_m = make_square_bbox(mp_m, HALF_SIDE_M)


            section_ll = shp_transform(t_3857_to_4326.transform, section_m)
            bbox_ll = shp_transform(t_3857_to_4326.transform, bbox_m)


            minlon, minlat, maxlon, maxlat = bbox_ll.bounds
            extent_ll = (minlon, maxlon, minlat, maxlat)
            region = ee.Geometry.Rectangle([minlon, minlat, maxlon, maxlat], geodesic=False)


            center_ll = shp_transform(t_3857_to_4326.transform, mp_m)
            center_lon, center_lat = float(center_ll.x), float(center_ll.y)


            utm_epsg = utm_epsg_from_lonlat(center_lon, center_lat)

            row_out: Dict[str, Any] = {
                "feature_id": int(feat_idx),
                "section_id": int(sec_idx),
                "center_lon": center_lon,
                "center_lat": center_lat,
                "utm_epsg": utm_epsg,
                "target_year_now": year_now,
                "target_year_past": year_past,
            }


            now_pil, now_year_used, now_count, now_ds = download_rgb_tile(
                args.dataset, region, year_now, args.size, args.lookback, args.s2_cloud
            )

            row_out["now_dataset"] = now_ds
            row_out["now_year_used"] = now_year_used
            row_out["now_images_in_year"] = now_count

            closest_now = None
            if now_pil is None:
                row_out["now_distance_m"] = np.nan
                row_out["now_tree_lon"] = np.nan
                row_out["now_tree_lat"] = np.nan
                row_out["now_line_lon"] = np.nan
                row_out["now_line_lat"] = np.nan
                row_out["now_ntrees"] = 0
            else:
                img_arr = ensure_rgb(now_pil)
                preds = detect_trees_deepforest(DEEPFOREST_MODEL, img_arr)
                closest_now = closest_tree_to_line_distance(
                    preds=preds,
                    line_ll=section_ll,
                    extent_lonlat=extent_ll,
                    img_shape=(img_arr.shape[0], img_arr.shape[1]),
                    utm_epsg=utm_epsg,
                )
                if closest_now is None:
                    row_out["now_distance_m"] = np.nan
                    row_out["now_tree_lon"] = np.nan
                    row_out["now_tree_lat"] = np.nan
                    row_out["now_line_lon"] = np.nan
                    row_out["now_line_lat"] = np.nan
                    row_out["now_ntrees"] = 0
                else:
                    row_out["now_distance_m"] = float(closest_now["distance_m"])
                    row_out["now_tree_lon"] = float(closest_now["tree_lon"])
                    row_out["now_tree_lat"] = float(closest_now["tree_lat"])
                    row_out["now_line_lon"] = float(closest_now["line_lon"])
                    row_out["now_line_lat"] = float(closest_now["line_lat"])
                    row_out["now_ntrees"] = int(closest_now["ntrees"])

            if not args.skip_overlays:
                now_year_tag = now_year_used if now_year_used is not None else year_now
                out_png_now = outdir / f"tile_{feat_idx}_{sec_idx}_now_{now_year_tag}.png"
                title_now = f"{now_ds} target={year_now} used={now_year_tag}"
                save_overlay_png(
                    pil_img=now_pil,
                    extent_lonlat=extent_ll,
                    line_ll=section_ll,
                    closest=closest_now,
                    out_path=out_png_now,
                    size_px=args.size,
                    dpi=args.dpi,
                    title=title_now,
                )


            if args.year_past is not None:
                start_past = args.year_past
            else:
                start_past = (now_year_used - 1) if (now_year_used is not None) else (year_now - 1)
            target_year = int(start_past)

            max_backlook = int(args.max_backlook)
            backlook_cnt = 0

            past_pil = None
            past_year_used = None
            past_count = 0
            past_ds = None
            closest_p = None

            while backlook_cnt < max_backlook:
                past_pil, past_year_used, past_count, past_ds = download_rgb_tile(
                    args.dataset, region, target_year, args.size, args.lookback, args.s2_cloud
                )

                past_pil = ImageEnhance.Sharpness(ImageEnhance.Color(ImageEnhance.Contrast(past_pil).enhance(1.15)).enhance(1.25)).enhance(1.2)

                row_out["past_dataset"] = past_ds
                row_out["past_year_used"] = past_year_used
                row_out["past_images_in_year"] = past_count


                if past_year_used is None:
                    backlook_cnt += 1
                    target_year -= 1
                    continue


                if (now_year_used is not None) and (past_year_used == now_year_used):
                    backlook_cnt += 1
                    target_year = past_year_used - 1
                    print(f"Info: past tile year == now ({past_year_used}); trying earlier year {target_year} ...")
                    continue


                break

            if backlook_cnt >= max_backlook:
                print(f"Warning: exhausted backlook for feature={feat_idx} section={sec_idx}; last past_year_used={past_year_used}")


            if past_pil is None:
                row_out["past_distance_m"] = np.nan
                row_out["past_tree_lon"] = np.nan
                row_out["past_tree_lat"] = np.nan
                row_out["past_line_lon"] = np.nan
                row_out["past_line_lat"] = np.nan
                row_out["past_ntrees"] = 0
            else:
                img_arr = ensure_rgb(past_pil)
                preds = detect_trees_deepforest(DEEPFOREST_MODEL, img_arr)
                closest_p = closest_tree_to_line_distance(
                    preds=preds,
                    line_ll=section_ll,
                    extent_lonlat=extent_ll,
                    img_shape=(img_arr.shape[0], img_arr.shape[1]),
                    utm_epsg=utm_epsg,
                )
                if closest_p is None:
                    row_out["past_distance_m"] = np.nan
                    row_out["past_tree_lon"] = np.nan
                    row_out["past_tree_lat"] = np.nan
                    row_out["past_line_lon"] = np.nan
                    row_out["past_line_lat"] = np.nan
                    row_out["past_ntrees"] = 0
                else:
                    row_out["past_distance_m"] = float(closest_p["distance_m"])
                    row_out["past_tree_lon"] = float(closest_p["tree_lon"])
                    row_out["past_tree_lat"] = float(closest_p["tree_lat"])
                    row_out["past_line_lon"] = float(closest_p["line_lon"])
                    row_out["past_line_lat"] = float(closest_p["line_lat"])
                    row_out["past_ntrees"] = int(closest_p["ntrees"])


            if not args.skip_overlays:
                past_year_tag = past_year_used if past_year_used is not None else target_year
                out_png_past = outdir / f"tile_{feat_idx}_{sec_idx}_past_{past_year_tag}.png"
                title_past = f"{past_ds} target={start_past} used={past_year_tag}"
                save_overlay_png(
                    pil_img=past_pil,
                    extent_lonlat=extent_ll,
                    line_ll=section_ll,
                    closest=closest_p,
                    out_path=out_png_past,
                    size_px=args.size,
                    dpi=args.dpi,
                    title=title_past,
                )

     
            d_now = row_out.get("now_distance_m", np.nan)
            d_past = row_out.get("past_distance_m", np.nan)
            if pd.notna(d_now) and pd.notna(d_past):
                row_out["delta_distance_m"] = float(d_past) - float(d_now)
            else:
                row_out["delta_distance_m"] = np.nan

            results.append(row_out)
            print(f"feature={feat_idx} section={sec_idx} done")

            comparison_count += 1
            if comparison_count >= 1500:
                print(f"Reached max comparisons 10). Stopping early.")
                stop = True
                break
                

    df = pd.DataFrame(results)
    out_csv = outdir / "tree_line_distance_compare.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote", out_csv)


if __name__ == "__main__":
    ee.Initialize(project='earth-engine-project-488823')
    ee.Authenticate(force=True)
    main()