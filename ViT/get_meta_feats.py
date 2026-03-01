import ee
import datetime
 
ee.Initialize(project='caramel-park-488923-j2')
 
def batch_average_ndvi_for_year(points, year=None, cloud_thresh=50, scale=20, fast_mode=True):
    """
    points: list of (lon, lat) tuples OR list of dicts {'id':..., 'lon':..., 'lat':...}
    year: int (e.g., 2023). If None, defaults to last full calendar year.
    Returns: list of dicts [{'id': id_or_index, 'lon':..., 'lat':..., 'ndvi': value_or_None}, ...]
    Notes:
      - Runs a single server-side aggregation for the mean NDVI image over the specified calendar year, then samples all points.
      - For many thousands of points use Export.table.toDrive or use paging (getInfo on huge FCs can time out).
    """
    # Speed heuristics
    if fast_mode and scale < 20:
        scale = 20
 
    # Normalize points input to features with IDs
    fc_features = []
    for i, p in enumerate(points):
        if isinstance(p, dict):
            lon = p.get('lon')
            lat = p.get('lat')
            pid = p.get('id', i)
        else:
            lon, lat = p
            pid = i
        feat = ee.Feature(ee.Geometry.Point([lon, lat]), {'id': pid, 'lon': lon, 'lat': lat})
        fc_features.append(feat)
 
    fc = ee.FeatureCollection(fc_features)
 
    # Determine year: default to last full calendar year if not provided
    if year is None:
        import datetime as _dt
        today = _dt.date.today()
        year = today.year - 1
 
    # Dates: server-side ee.Date from Python datetime for the full calendar year
    start = ee.Date(__import__('datetime').datetime(year, 1, 1))
    end = start.advance(1, 'year')  # cover the full calendar year
 
    # Light cloud mask using SCL band (keep mostly good land pixels)
    def mask_s2_sr(image):
        scl = image.select('SCL')
        good = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return image.updateMask(good)
 
    # Build collection once (server side), select only needed bands
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR')
        .filterDate(start, end)
        .filterBounds(fc)     # filter by whole feature collection bounds
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
        .map(mask_s2_sr)
        .select(['B8', 'B4'])
    )
 
    # Compute NDVI per image, then mean over the year (server-side)
    ndvi_col = collection.map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
    mean_ndvi_image = ndvi_col.mean().select('NDVI')
 
    # Sample at all points in one call. geometries=True keeps geometry in result.
    sampled = mean_ndvi_image.sampleRegions(
        collection=fc,
        properties=['id', 'lon', 'lat'],
        scale=scale,
        tileScale=4  # increase if memory errors
    )
 
    # Get results (one RPC). For large numbers of points, use Export.table.toDrive instead.
    try:
        sampled_list = sampled.getInfo()['features']
    except Exception as e:
        # If getInfo fails (too large), tell user to export instead
        raise RuntimeError(
            "Sampling failed or result too large for getInfo(). "
            "For many points use Export.table.toDrive. Error: {}".format(e)
        )
 
    # Convert to friendly Python list
    out = []
    for f in sampled_list:
        props = f.get('properties', {})
        pid = props.get('id')
        lon = props.get('lon')
        lat = props.get('lat')
        ndvi = props.get('NDVI')  # could be None if no valid pixels
        out.append({'id': pid, 'lon': lon, 'lat': lat, 'ndvi': ndvi})
 
    return out
 
def batch_average_precip_last_year(points, year=None, cloud_thresh=50, scale=20, fast_mode=True, workers=8):
    """
    points: list of (lon, lat) tuples OR list of dicts {'id':..., 'lon':..., 'lat':...}
    year: int (e.g., 2023). If None, defaults to the last full calendar year (e.g., if today is 2026-02-28, year=2025).
          If year == current year, the end date will be today (year-to-date).
    Returns: list of dicts [{'id': id_or_index, 'lon':..., 'lat':..., 'precip': value_or_None}, ...]
    Notes:
      - Uses NASA POWER API to compute mean daily precipitation for the requested window.
      - Parameters cloud_thresh, scale, fast_mode accepted for compatibility but not used.
      - Robust parsing: ignores missing/sentinel/negative values and returns mean of valid days or None.
    """
    import requests
    import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
 
    _ = cloud_thresh, scale, fast_mode  # compatibility only
 
    # Normalize input shape
    normalized = []
    for i, p in enumerate(points):
        if isinstance(p, dict):
            lon = p.get('lon')
            lat = p.get('lat')
            pid = p.get('id', i)
        else:
            lon, lat = p
            pid = i
        normalized.append({'id': pid, 'lon': lon, 'lat': lat})
 
    # Date window: follow NDVI-style behavior
    today = datetime.date.today()
 
    if year is None:
        # default to last full calendar year
        year = today.year - 1
 
    # Validate and coerce year
    try:
        year = int(year)
    except Exception:
        raise ValueError("year must be an integer or None")
 
    if year < 1900:
        raise ValueError("year must be >= 1900")
    if year > today.year:
        raise ValueError("year cannot be in the future")
 
    # start: Jan 1 of year
    start = datetime.date(year, 1, 1)
    # end: if current year -> today (year-to-date), else Dec 31 of that year
    if year == today.year:
        end = today
    else:
        end = datetime.date(year, 12, 31)
 
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
 
    POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
 
    def _extract_date_dict(candidate):
        """Return candidate if it looks like a date->value dict (keys are YYYYMMDD-like)."""
        if not isinstance(candidate, dict) or not candidate:
            return None
        for k in candidate.keys():
            if isinstance(k, str) and len(k) == 8 and k.isdigit():
                return candidate
        return None
 
    def _fetch_mean_precip(pt):
        session = requests.Session()
        retries = 3
        backoff = 1.0
        for attempt in range(retries):
            try:
                r = session.get(
                    POWER_ENDPOINT,
                    params={
                        "start": start_str,
                        "end": end_str,
                        "latitude": pt['lat'],
                        "longitude": pt['lon'],
                        "parameters": "PRECTOT",
                        "community": "RE",
                        "format": "JSON",
                    },
                    timeout=30,
                )
                r.raise_for_status()
                j = r.json()
 
                # Primary location for PRECTOT
                props = j.get('properties', {}) or {}
                params = props.get('parameter', {}) or {}
 
                values = params.get('PRECTOT')
                # If PRECTOT not present, try to find a parameter that looks like a date->value dict
                if not _extract_date_dict(values):
                    values = None
                    for k, v in params.items():
                        if _extract_date_dict(v):
                            values = v
                            break
 
                if not _extract_date_dict(values):
                    return None
 
                numeric = []
                for val in values.values():
                    if val is None:
                        continue
                    # Some values may be strings like "nan" — try to coerce cleanly
                    try:
                        f = float(val)
                    except Exception:
                        continue
                    # Discard sentinel / invalid negatives
                    # Keep only physically possible precipitation: >= 0 and not absurdly large
                    if f < 0 or f > 1e4:
                        continue
                    numeric.append(f)
 
                if not numeric:
                    return None
 
                # mean mm/day
                return sum(numeric) / len(numeric)
 
            except Exception:
                # retry with backoff
                time.sleep(backoff)
                backoff *= 2
                continue
 
        return None
 
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_mean_precip, pt): pt for pt in normalized}
        for fut in as_completed(futures):
            pt = futures[fut]
            try:
                mean_precip = fut.result()
            except Exception:
                mean_precip = None
            results.append({
                'id': pt['id'],
                'lon': pt['lon'],
                'lat': pt['lat'],
                'precip': mean_precip
            })
 
    return results
 
def batch_average_temp_summer_winter(points, cloud_thresh=50, scale=20, fast_mode=True, year=None):
    """
    points: list of (lon, lat) tuples OR list of dicts {'id':..., 'lon':..., 'lat':...}
    year: int (e.g., 2023). If None, defaults to last full calendar year.
    Returns: list of dicts:
      [{'id': id_or_index, 'lon':..., 'lat':..., 'summer_avg_c': value_or_None, 'winter_avg_c': value_or_None}, ...]
    Notes:
      - Uses NASA POWER daily T2M via the POWER API (no Google).
      - Summer = Jun, Jul, Aug; Winter = Dec, Jan, Feb (within the same calendar year).
      - Parameters cloud_thresh, scale, fast_mode are accepted for compatibility but not used.
    """
    import requests
    import datetime
 
    # Keep same signature compatibility (these params are unused here)
    _ = cloud_thresh, scale, fast_mode
 
    # Normalize points input to list with IDs (same pattern as the precip function)
    normalized = []
    for i, p in enumerate(points):
        if isinstance(p, dict):
            lon = p.get('lon')
            lat = p.get('lat')
            pid = p.get('id', i)
        else:
            lon, lat = p
            pid = i
        normalized.append({'id': pid, 'lon': lon, 'lat': lat})
 
    # Determine year: default to last full calendar year if not provided
    today = datetime.date.today()
    if year is None:
        year = today.year - 1
 
    # Date range: full calendar year
    start = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
 
    POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
 
    results = []
    for pt in normalized:
        try:
            response = requests.get(
                POWER_ENDPOINT,
                params={
                    "start": start_str,
                    "end": end_str,
                    "latitude": pt['lat'],
                    "longitude": pt['lon'],
                    "parameters": "T2M",
                    "community": "RE",
                    "format": "JSON",
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            values = data.get("properties", {}).get("parameter", {}).get("T2M", {})
 
            summer_vals = []
            winter_vals = []
 
            for date_str, temp in values.items():
                if temp is None:
                    continue
                month = int(date_str[4:6])
                if month in (6, 7, 8):
                    summer_vals.append(temp)
                elif month in (12, 1, 2):
                    winter_vals.append(temp)
 
            summer_avg = sum(summer_vals) / len(summer_vals) if summer_vals else None
            winter_avg = sum(winter_vals) / len(winter_vals) if winter_vals else None
 
        except Exception:
            summer_avg = None
            winter_avg = None
 
        results.append({
            'id': pt['id'],
            'lon': pt['lon'],
            'lat': pt['lat'],
            'summer_avg_c': summer_avg,
            'winter_avg_c': winter_avg
        })
 
    return results
 
if __name__ == "__main__":
    pts = [(-90.5, 33.8), (-89.5, 33.8), (-88.5, 33.8)]
    results = batch_average_ndvi_for_year(pts, year=2021,cloud_thresh=50, scale=20, fast_mode=True)
    for r in results:
        print(r)
    results = batch_average_precip_last_year(pts, year=2021,cloud_thresh=50, scale=20, fast_mode=True)
    for r in results:
        print(r)
    results = batch_average_temp_summer_winter(pts, year=2021,cloud_thresh=50, scale=20, fast_mode=True)
    for r in results:
        print(r)