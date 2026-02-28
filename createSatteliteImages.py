import argparse
import math
from pathlib import Path
from io import BytesIO

import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import contextily as ctx
import pyproj
import requests
from PIL import Image

MILE_METERS = 1609.344
DEFAULT_CRS = "EPSG:3857"  # WebMercator for tile services

def midpoint_along_line(geom):
    if geom.is_empty:
        return None
    if geom.geom_type == "MultiLineString":
        parts = list(geom.geoms)
        parts.sort(key=lambda p: p.length, reverse=True)
        geom = parts[0]
    return geom.interpolate(geom.length / 2.0)

def make_square_bbox(point, half_side_m):
    x, y = point.x, point.y
    return box(x - half_side_m, y - half_side_m, x + half_side_m, y + half_side_m)

def meters_per_pixel_for_zoom(lat_deg, zoom):
    initial_resolution = 156543.03392804062
    return initial_resolution * math.cos(math.radians(lat_deg)) / (2 ** zoom)

def choose_zoom_for_extent(center_lat_deg, img_px, target_extent_m, zoom_min=1, zoom_max=19):
    desired_mpp = target_extent_m / img_px
    initial = 156543.03392804062 * math.cos(math.radians(center_lat_deg))
    raw = math.log2(initial / desired_mpp)
    return max(zoom_min, min(zoom_max, int(round(raw))))

def fetch_mapbox_static(lon, lat, size_px, zoom, token):
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{size_px}x{size_px}?access_token={token}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def add_contextily_sat(ax, crs, zoom):
    try:
        provider = ctx.providers.Esri.WorldImagery
        ctx.add_basemap(ax, source=provider, crs=crs, zoom=zoom)
    except Exception:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=crs, zoom=zoom)

def plot_and_save(clipped_gdf, bbox, out_path, size_px, dpi, provider, zoom, mapbox_token):
    minx, miny, maxx, maxy = bbox.bounds
    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    # draw the transmission geometry (on top)
    try:
        clipped_gdf.plot(ax=ax, linewidth=2.0, edgecolor="yellow", zorder=10)
    except Exception:
        pass

    if provider.lower() == "mapbox" and mapbox_token:
        # mapbox expects lon,lat center
        transformer = pyproj.Transformer.from_crs(DEFAULT_CRS, "EPSG:4326", always_xy=True)
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        lon, lat = transformer.transform(cx, cy)
        img = fetch_mapbox_static(lon, lat, size_px, zoom or 17, mapbox_token)
        ax.imshow(img, extent=(minx, maxx, miny, maxy), origin="upper", zorder=0)
    else:
        add_contextily_sat(ax, clipped_gdf.crs.to_string(), zoom or 16)

    ax.axis("off")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--size", type=int, default=1024, help="pixel width/height")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--provider", type=str, default="Esri.WorldImagery",
                   help="Esri.WorldImagery (default) or 'mapbox'")
    p.add_argument("--mapbox_token", type=str, default=None)
    p.add_argument("--sc-only", action="store_true")
    p.add_argument("--state-shp", type=str, default=None)
    p.add_argument("--auto-zoom", action="store_true")
    args = p.parse_args()

    infile = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(infile)  # expects EPSG:4326
    if gdf.empty:
        raise SystemExit("Input has no features")

    # optional state filter
    if args.state_shp:
        state = gpd.read_file(args.state_shp).to_crs("EPSG:4326")
        state_poly = state.unary_union
        gdf = gdf[gdf.intersects(state_poly)]
    elif args.sc_only:
        # rough SC bbox lon/lat
        sc_box = box(-83.3532, 32.0335, -78.4019, 35.2154)
        gdf = gdf[gdf.intersects(sc_box)]

    if gdf.empty:
        raise SystemExit("No features after filtering; nothing to do.")

    # keep only lines
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.head(10)
    if gdf.empty:
        raise SystemExit("No line geometries present")

    # project to WebMercator meters for bbox math and plotting with contextily
    gdf_m = gdf.to_crs(DEFAULT_CRS)
    half = MILE_METERS / 2.0

    # transformer for lat/lon when choosing zoom
    transformer_to_lonlat = pyproj.Transformer.from_crs(DEFAULT_CRS, "EPSG:4326", always_xy=True)

    for idx, row in gdf_m.iterrows():
        geom = row.geometry
        mp = midpoint_along_line(geom)
        if mp is None:
            continue
        bbox = make_square_bbox(mp, half)
        clipped = geom.intersection(bbox)
        clipped_gdf = gpd.GeoDataFrame(geometry=[clipped], crs=gdf_m.crs)

        chosen_zoom = None
        if args.auto_zoom:
            cx, cy = mp.x, mp.y
            lon, lat = transformer_to_lonlat.transform(cx, cy)
            chosen_zoom = choose_zoom_for_extent(lat, args.size, MILE_METERS, zoom_min=1, zoom_max=19)

        # filename
        fname = f"tile_{idx}.png"
        out_path = outdir / fname

        plot_and_save(clipped_gdf, bbox, str(out_path),
                      args.size, args.dpi,
                      args.provider, chosen_zoom, args.mapbox_token)
        print("Saved", out_path)

if __name__ == "__main__":
    main()