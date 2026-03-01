const API_BASE = window.location.origin;

const yearRange = document.getElementById("yearRange");
const yearLabel = document.getElementById("yearLabel");
const playBtn = document.getElementById("playBtn");
const topList = document.getElementById("topList");
const segmentTitle = document.getElementById("segmentTitle");
const segmentMeta = document.getElementById("segmentMeta");
const chartSvg = document.getElementById("chart");

const map = L.map("map", { zoomControl: true }).setView([33.8361, -81.1637], 8);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

map.createPane("powerlinesPane");
map.getPane("powerlinesPane").style.zIndex = 330;
map.getPane("powerlinesPane").style.pointerEvents = "none";

const heatLayer = L.layerGroup().addTo(map);

let years = [];
let currentYearIdx = 0;
let playing = false;
let playTimer = null;
let segmentLineLayer = null;
let powerlineLayer = null;
let segmentLookup = new Map();
let activeSegment = null;
let currentGeojson = null;
let hasAutoFit = false;
let powerlineRequestId = 0;
let lastPowerlineBboxKey = "";

function hazardRank(level) {
  const key = String(level || "").toLowerCase();
  if (key === "critical") return 4;
  if (key === "high") return 3;
  if (key === "medium") return 2;
  if (key === "low") return 1;
  return 0;
}

function hazardColor(rank) {
  if (rank >= 4) return "#5D0000";
  if (rank >= 3) return "#8A0606";
  if (rank >= 2) return "#BD2A17";
  if (rank >= 1) return "#DD6540";
  return "#E8A58E";
}

function levelClass(level) {
  const r = hazardRank(level);
  if (r >= 3) return "pill-high";
  if (r === 2) return "pill-medium";
  return "pill-low";
}

function fmtInt(value) {
  const n = Number(value);
  return Number.isFinite(n) ? String(Math.round(n)) : "--";
}

function fmtFloat(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "--";
}

async function getJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return await res.json();
}

function setPlaying(next) {
  playing = next;
  playBtn.textContent = playing ? "Pause" : "Play";
  if (!playing && playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    return;
  }
  if (playing && !playTimer) {
    playTimer = setInterval(() => {
      if (!years.length) return;
      currentYearIdx = (currentYearIdx + 1) % years.length;
      yearRange.value = String(currentYearIdx);
      loadYear();
    }, 1300);
  }
}

function updateYearLabel() {
  yearLabel.textContent = String(years[currentYearIdx] || "-");
}

function collectCoordPairs(node, out) {
  if (!Array.isArray(node)) return;
  if (node.length >= 2 && typeof node[0] === "number" && typeof node[1] === "number") {
    out.push([node[0], node[1]]);
    return;
  }
  for (const part of node) {
    collectCoordPairs(part, out);
  }
}

function geometryInfo(geometry) {
  if (!geometry || !geometry.coordinates) return null;
  const coords = [];
  collectCoordPairs(geometry.coordinates, coords);
  if (!coords.length) return null;

  let minLng = Infinity;
  let minLat = Infinity;
  let maxLng = -Infinity;
  let maxLat = -Infinity;
  let sumLng = 0;
  let sumLat = 0;

  for (const [lng, lat] of coords) {
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    sumLng += lng;
    sumLat += lat;
  }

  const centerLng = sumLng / coords.length;
  const centerLat = sumLat / coords.length;
  return {
    center: [centerLat, centerLng],
    bounds: L.latLngBounds(
      [minLat, minLng],
      [maxLat, maxLng]
    ),
  };
}

function copyBounds(bounds) {
  if (!bounds || !bounds.isValid()) return null;
  return L.latLngBounds(bounds.getSouthWest(), bounds.getNorthEast());
}

function buildSegmentLookup(geojson) {
  segmentLookup = new Map();
  for (const feature of geojson.features || []) {
    const props = feature.properties || {};
    const segmentId = props.segment_id;
    if (!segmentId) continue;
    const info = geometryInfo(feature.geometry);
    if (!info) continue;
    segmentLookup.set(segmentId, {
      segmentId,
      props,
      feature,
      center: info.center,
      bounds: info.bounds,
      hazardRank: hazardRank(props.hazard_level),
    });
  }
}

function fitToDataIfNeeded() {
  if (hasAutoFit || !segmentLookup.size) return;
  let bounds = null;
  for (const item of segmentLookup.values()) {
    bounds = bounds ? bounds.extend(item.bounds) : copyBounds(item.bounds);
  }
  if (bounds && bounds.isValid()) {
    map.fitBounds(bounds.pad(0.2));
    hasAutoFit = true;
  }
}

function clearMapLayers() {
  heatLayer.clearLayers();
  if (segmentLineLayer) {
    map.removeLayer(segmentLineLayer);
    segmentLineLayer = null;
  }
}

function mapBoundsToBbox(bounds) {
  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();
  return `${sw.lng.toFixed(4)},${sw.lat.toFixed(4)},${ne.lng.toFixed(4)},${ne.lat.toFixed(4)}`;
}

function renderPowerlineOverlay(geojson) {
  if (powerlineLayer) {
    map.removeLayer(powerlineLayer);
    powerlineLayer = null;
  }
  powerlineLayer = L.geoJSON(geojson, {
    pane: "powerlinesPane",
    interactive: false,
    style: () => ({
      color: "#0B2F78",
      weight: map.getZoom() >= 8 ? 1.6 : 1.1,
      opacity: 0.82,
    }),
  }).addTo(map);
}

function refreshPowerlineOverlay(force = false) {
  const bbox = mapBoundsToBbox(map.getBounds());
  if (!force && bbox === lastPowerlineBboxKey) {
    return;
  }
  lastPowerlineBboxKey = bbox;

  const requestId = ++powerlineRequestId;
  getJson(`/powerlines/layer?bbox=${encodeURIComponent(bbox)}&max_features=18000`)
    .then((geojson) => {
      if (requestId !== powerlineRequestId) {
        return;
      }
      renderPowerlineOverlay(geojson);
    })
    .catch(() => {
      // Keep app responsive even if optional overlay data is missing.
    });
}

function styleForSegment(item, zoom) {
  const rank = item.hazardRank;
  const color = hazardColor(rank);
  const selected = activeSegment === item.segmentId;
  if (zoom >= 13) {
    return {
      color,
      weight: selected ? 7 : 4,
      opacity: selected ? 1 : 0.9,
    };
  }
  const baseRadius = Math.max(3, 11 - (zoom - 10) * 2.2);
  return {
    radius: selected ? baseRadius + 2 : baseRadius,
    color: "#641010",
    weight: selected ? 2.3 : 1.0,
    fillColor: color,
    fillOpacity: selected ? 0.95 : 0.78,
  };
}

function desiredClusterRadiusPx(zoom) {
  if (zoom <= 5) return 92;
  if (zoom <= 7) return 72;
  if (zoom <= 9) return 56;
  if (zoom <= 10) return 44;
  if (zoom <= 11) return 34;
  return 26;
}

function sqDist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function seedCentroids(points, k) {
  if (!points.length || k <= 0) return [];

  const centroids = [];
  const byHazard = [...points].sort((a, b) => b.item.hazardRank - a.item.hazardRank);
  centroids.push({ x: byHazard[0].x, y: byHazard[0].y });

  while (centroids.length < k) {
    let best = null;
    let bestDist = -1;
    for (const p of points) {
      let nearest = Infinity;
      for (const c of centroids) {
        nearest = Math.min(nearest, sqDist(p, c));
      }
      if (nearest > bestDist) {
        bestDist = nearest;
        best = p;
      }
    }
    if (!best) break;
    centroids.push({ x: best.x, y: best.y });
  }
  return centroids;
}

function clusterSegmentsKMeans(zoom) {
  const items = [...segmentLookup.values()];
  if (!items.length) return [];
  if (items.length === 1) {
    return [
      {
        center: items[0].center,
        count: 1,
        maxRank: items[0].hazardRank,
        maxRisk: Number(items[0].props.risk_score || 0),
        segmentIds: [items[0].segmentId],
        bounds: copyBounds(items[0].bounds),
      },
    ];
  }

  const projected = items.map((item) => {
    const p = map.project(L.latLng(item.center[0], item.center[1]), zoom);
    return { item, x: p.x, y: p.y };
  });

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of projected) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  const area = Math.max((maxX - minX) * (maxY - minY), 1);
  const targetRadius = desiredClusterRadiusPx(zoom);
  const targetArea = Math.PI * targetRadius * targetRadius;
  const roughK = Math.round(area / targetArea);
  const k = Math.max(1, Math.min(projected.length, roughK));

  const centroids = seedCentroids(projected, k);
  if (!centroids.length) return [];

  const assignment = new Array(projected.length).fill(0);
  for (let iter = 0; iter < 12; iter += 1) {
    for (let i = 0; i < projected.length; i += 1) {
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let c = 0; c < centroids.length; c += 1) {
        const dist = sqDist(projected[i], centroids[c]);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = c;
        }
      }
      assignment[i] = bestIdx;
    }

    const sums = centroids.map(() => ({ x: 0, y: 0, n: 0 }));
    for (let i = 0; i < projected.length; i += 1) {
      const a = assignment[i];
      sums[a].x += projected[i].x;
      sums[a].y += projected[i].y;
      sums[a].n += 1;
    }
    for (let c = 0; c < centroids.length; c += 1) {
      if (sums[c].n > 0) {
        centroids[c].x = sums[c].x / sums[c].n;
        centroids[c].y = sums[c].y / sums[c].n;
      }
    }
  }

  const grouped = new Map();
  for (let i = 0; i < projected.length; i += 1) {
    const clusterId = assignment[i];
    if (!grouped.has(clusterId)) {
      grouped.set(clusterId, {
        count: 0,
        latSum: 0,
        lngSum: 0,
        maxRank: 0,
        maxRisk: 0,
        segmentIds: [],
        bounds: null,
      });
    }
    const g = grouped.get(clusterId);
    const item = projected[i].item;
    g.count += 1;
    g.latSum += item.center[0];
    g.lngSum += item.center[1];
    g.maxRank = Math.max(g.maxRank, item.hazardRank);
    g.maxRisk = Math.max(g.maxRisk, Number(item.props.risk_score || 0));
    g.segmentIds.push(item.segmentId);
    g.bounds = g.bounds ? g.bounds.extend(item.bounds) : copyBounds(item.bounds);
  }

  return [...grouped.values()].map((g) => ({
    center: [g.latSum / g.count, g.lngSum / g.count],
    count: g.count,
    maxRank: g.maxRank,
    maxRisk: g.maxRisk,
    segmentIds: g.segmentIds,
    bounds: g.bounds,
  }));
}

function attachSegmentEvents(layer, item) {
  layer.on("click", () => selectSegment(item.segmentId, true));
  layer.bindTooltip(
    `${item.segmentId}<br/>Hazard: ${item.props.hazard_level || "unknown"}<br/>Dist: ${fmtFloat(item.props.vegetation_distance_m)} m`,
    { sticky: true }
  );
}

function renderSegmentLines(zoom) {
  segmentLineLayer = L.geoJSON(currentGeojson, {
    style: (feature) => {
      const props = feature?.properties || {};
      const id = props.segment_id;
      const item = id ? segmentLookup.get(id) : null;
      if (!item) {
        return {
          color: "#B43A1B",
          weight: 4,
          opacity: 0.85,
        };
      }
      return styleForSegment(item, zoom);
    },
    onEachFeature: (feature, layer) => {
      const id = feature?.properties?.segment_id;
      if (!id) return;
      const item = segmentLookup.get(id);
      if (!item) return;
      attachSegmentEvents(layer, item);
    },
  }).addTo(map);
}

function renderAggregatedCircles(zoom) {
  const aggregates = clusterSegmentsKMeans(zoom);
  const targetRadius = desiredClusterRadiusPx(zoom);
  for (const agg of aggregates) {
    const base = Math.max(5, targetRadius * 0.22);
    const radius = base + Math.min(targetRadius * 0.58, Math.sqrt(agg.count) * 2.2);
    const selected = activeSegment && agg.segmentIds.includes(activeSegment);
    const marker = L.circleMarker(agg.center, {
      radius: selected ? radius + 2 : radius,
      color: "#5E0909",
      weight: selected ? 2.2 : 1.1,
      fillColor: hazardColor(agg.maxRank),
      fillOpacity: 0.34 + Math.min(0.56, agg.maxRisk / 150),
    });
    marker.bindTooltip(
      `${agg.count} segments<br/>Max hazard: ${agg.maxRank}<br/>Max risk: ${fmtInt(agg.maxRisk)}`,
      { sticky: true }
    );
    marker.on("click", () => {
      if (agg.segmentIds.length === 1) {
        selectSegment(agg.segmentIds[0], true);
        return;
      }
      if (agg.bounds && agg.bounds.isValid()) {
        map.fitBounds(agg.bounds.pad(0.45), { maxZoom: 12 });
        return;
      }
      map.setView(agg.center, Math.min(12, map.getZoom() + 2));
    });
    marker.addTo(heatLayer);
  }
}

function refreshMapVisualization() {
  if (!currentGeojson) return;
  clearMapLayers();
  const zoom = map.getZoom();
  if (zoom >= 13) {
    renderSegmentLines(zoom);
    return;
  }
  renderAggregatedCircles(zoom);
}

function renderLayer(geojson) {
  currentGeojson = geojson;
  buildSegmentLookup(geojson);
  refreshMapVisualization();
  fitToDataIfNeeded();
}

function renderTopList(items) {
  topList.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "risk-item";
    li.textContent = "No segments returned for this year.";
    topList.appendChild(li);
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "risk-item";
    const trendArrow = item.trend_arrow === "up" ? "▲" : item.trend_arrow === "down" ? "▼" : "▶";
    li.innerHTML = `
      <div class="risk-row">
        <span class="risk-id">${item.segment_id}</span>
        <span class="pill ${levelClass(item.hazard_level)}">${item.hazard_level || "unknown"}</span>
      </div>
      <div class="risk-row">
        <span class="mono">Score ${fmtInt(item.risk_score)}</span>
        <span class="mono">${trendArrow} conf ${fmtFloat(item.risk_confidence, 2)}</span>
      </div>
      <div class="risk-row">
        <span class="mono">+3m growth ${fmtFloat(item.forecast_growth_3m_m)} m</span>
      </div>
      <div class="risk-row">
        <span class="mono">+6m growth ${fmtFloat(item.forecast_growth_6m_m)} m</span>
      </div>
    `;
    li.addEventListener("click", () => selectSegment(item.segment_id, true));
    topList.appendChild(li);
  }
}

function buildPath(points, xScale, yScale) {
  return points
    .map((p, idx) => `${idx === 0 ? "M" : "L"} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`)
    .join(" ");
}

function renderChart(historyRows, forecastRows) {
  const w = 860;
  const h = 210;
  const pad = { l: 52, r: 24, t: 18, b: 28 };
  chartSvg.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const hist = historyRows
    .map((r) => ({ date: r.target_date, y: Number(r.growth_amount_m) }))
    .filter((r) => Number.isFinite(r.y));
  const fct = forecastRows
    .map((r) => ({
      date: r.target_date,
      y: Number(r.predicted_growth_amount_m),
      low: Number(r.growth_lower_m),
      high: Number(r.growth_upper_m),
    }))
    .filter((r) => Number.isFinite(r.y));

  const allDates = [...hist.map((d) => d.date), ...fct.map((d) => d.date)];
  const allY = [
    ...hist.map((d) => d.y),
    ...fct.map((d) => d.y),
    ...fct.map((d) => d.low),
    ...fct.map((d) => d.high),
  ].filter((v) => Number.isFinite(v));

  if (!allDates.length || !allY.length) {
    chartSvg.innerHTML = `<text x="18" y="42" fill="#466151">No vegetation growth values for this segment.</text>`;
    return;
  }

  const uniqueDates = [...new Set(allDates)];
  const xIndex = new Map(uniqueDates.map((d, i) => [d, i]));
  const minY = Math.max(0, Math.min(...allY) - 0.2);
  const maxY = Math.max(...allY) + 0.2;
  const xScale = (x) => pad.l + ((w - pad.l - pad.r) * x) / Math.max(1, uniqueDates.length - 1);
  const yScale = (y) =>
    pad.t + (h - pad.t - pad.b) * (1 - (y - minY) / Math.max(0.0001, maxY - minY));

  const histPts = hist.map((d) => ({ x: xIndex.get(d.date), y: d.y }));
  const fctPts = fct.map((d) => ({ x: xIndex.get(d.date), y: d.y }));

  let bandPath = "";
  if (fct.length) {
    const upper = fct.map((d) => ({ x: xIndex.get(d.date), y: d.high }));
    const lower = [...fct].reverse().map((d) => ({ x: xIndex.get(d.date), y: d.low }));
    const poly = [...upper, ...lower];
    bandPath = buildPath(poly, xScale, yScale) + " Z";
  }

  const gridLines = [0, 1, 2, 3, 4]
    .map((g) => {
      const gy = pad.t + ((h - pad.t - pad.b) * g) / 4;
      return `<line x1="${pad.l}" y1="${gy}" x2="${w - pad.r}" y2="${gy}" stroke="rgba(45,71,56,0.2)" stroke-width="1"/>`;
    })
    .join("");

  const xTicks = uniqueDates
    .map((d, i) => {
      if (uniqueDates.length > 8 && i % 2 !== 0) return "";
      return `<text x="${xScale(i)}" y="${h - 8}" text-anchor="middle" fill="#3E5D4D" font-size="10">${d}</text>`;
    })
    .join("");

  const yTicks = [minY, (minY + maxY) / 2, maxY]
    .map((y) => {
      const py = yScale(y);
      return `<text x="${pad.l - 8}" y="${py + 4}" text-anchor="end" fill="#3E5D4D" font-size="10">${y.toFixed(2)}</text>`;
    })
    .join("");

  chartSvg.innerHTML = `
    <rect x="0" y="0" width="${w}" height="${h}" fill="transparent"></rect>
    ${gridLines}
    <line x1="${pad.l}" y1="${h - pad.b}" x2="${w - pad.r}" y2="${h - pad.b}" stroke="#2F5442" />
    <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h - pad.b}" stroke="#2F5442" />
    ${bandPath ? `<path d="${bandPath}" fill="rgba(213, 131, 34, 0.16)" stroke="none"></path>` : ""}
    <path d="${buildPath(histPts, xScale, yScale)}" fill="none" stroke="#156E48" stroke-width="3.2"></path>
    ${
      fctPts.length
        ? `<path d="${buildPath(fctPts, xScale, yScale)}" fill="none" stroke="#BE5B1C" stroke-width="3.2" stroke-dasharray="8 5"></path>`
        : ""
    }
    ${xTicks}
    ${yTicks}
    <text x="${pad.l + 8}" y="${pad.t + 12}" fill="#1F4A36" font-size="11">Vegetation growth amount (m): history (green), forecast (orange)</text>
  `;
}

async function selectSegment(segmentId, zoomToFeature = false) {
  activeSegment = segmentId;
  segmentTitle.textContent = segmentId;
  segmentMeta.textContent = "loading...";

  if (zoomToFeature && segmentLookup.has(segmentId)) {
    const item = segmentLookup.get(segmentId);
    if (item.bounds && item.bounds.isValid()) {
      map.fitBounds(item.bounds.pad(0.45), { maxZoom: 14 });
    } else {
      map.setView(item.center, 13);
    }
  }

  const layerData = await Promise.allSettled([
    getJson(`/segments/${encodeURIComponent(segmentId)}/timeseries`),
    getJson(`/segments/${encodeURIComponent(segmentId)}/forecast`),
  ]);

  const hist = layerData[0].status === "fulfilled" ? layerData[0].value : [];
  const fct = layerData[1].status === "fulfilled" ? layerData[1].value : [];

  if (hist.length) {
    const latest = hist[hist.length - 1];
    segmentMeta.textContent = `hazard ${latest.hazard_level || "unknown"} | dist ${fmtFloat(latest.vegetation_distance_m)} m`;
  } else {
    segmentMeta.textContent = "no history rows";
  }
  renderChart(hist, fct);
  refreshMapVisualization();
}

async function loadYear() {
  if (!years.length) return;
  updateYearLabel();
  const year = years[currentYearIdx];

  const [layerResult, topResult] = await Promise.allSettled([
    getJson(`/map/layer?year=${encodeURIComponent(year)}`),
    getJson(`/segments/top?year=${encodeURIComponent(year)}&n=20`),
  ]);

  if (layerResult.status === "fulfilled") {
    renderLayer(layerResult.value);
  }
  if (topResult.status === "fulfilled") {
    renderTopList(topResult.value);
  }

  if (!activeSegment && topResult.status === "fulfilled" && topResult.value.length) {
    selectSegment(topResult.value[0].segment_id, false);
  }
  refreshPowerlineOverlay(true);
}

async function initialize() {
  try {
    const timeline = await getJson("/timeline/years");
    years = timeline.years || [];
    if (!years.length) {
      yearLabel.textContent = "no data";
      topList.innerHTML = `<li class="risk-item">No years found in dataset.</li>`;
      return;
    }

    const defaultYear = timeline.default || years[years.length - 1];
    currentYearIdx = Math.max(0, years.indexOf(defaultYear));
    yearRange.max = String(Math.max(0, years.length - 1));
    yearRange.value = String(currentYearIdx);
    updateYearLabel();

    yearRange.addEventListener("input", () => {
      currentYearIdx = Number(yearRange.value);
      loadYear();
    });
    playBtn.addEventListener("click", () => setPlaying(!playing));
    map.on("zoomend", () => {
      refreshMapVisualization();
      refreshPowerlineOverlay(false);
    });
    map.on("moveend", () => refreshPowerlineOverlay(false));

    await loadYear();
  } catch (err) {
    topList.innerHTML = `<li class="risk-item">Failed to load API data: ${String(err)}</li>`;
  }
}

initialize();
