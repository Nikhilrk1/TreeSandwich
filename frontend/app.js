const API_BASE = window.location.origin;

const yearRange = document.getElementById("yearRange");
const yearLabel = document.getElementById("yearLabel");
const playBtn = document.getElementById("playBtn");
const topList = document.getElementById("topList");
const segmentTitle = document.getElementById("segmentTitle");
const segmentMeta = document.getElementById("segmentMeta");
const chartSvg = document.getElementById("chart");
const imageModal = document.getElementById("imageModal");
const imageModalClose = document.getElementById("imageModalClose");
const imageModalMeta = document.getElementById("imageModalMeta");
const imageModalImg = document.getElementById("imageModalImg");


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
let activeSegment = null;
let powerlineLayer = null;
let powerlineRequestId = 0;
let lastPowerlineBboxKey = "";
let predictionRequestId = 0;
let lastPredictionKey = "";
let hasAutoFit = false;

let records = [];
let recordsById = new Map();

function distanceColor(distanceM) {
  const d = Number(distanceM);
  if (!Number.isFinite(d)) return "#DDE7F7";
  const farDistanceM = 20.0;
  const t = 1 - Math.min(Math.max(d, 0), farDistanceM) / farDistanceM;

  // Blue (low risk/far) -> Red (high risk/close)
  const start = { r: 40, g: 96, b: 196 };
  const end = { r: 199, g: 31, b: 55 };
  const r = Math.round(start.r + (end.r - start.r) * t);
  const g = Math.round(start.g + (end.g - start.g) * t);
  const b = Math.round(start.b + (end.b - start.b) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function distanceBand(distanceM) {
  const d = Number(distanceM);
  if (!Number.isFinite(d)) return "unknown";
  if (d <= 2) return "critical";
  if (d <= 5) return "high";
  if (d <= 10) return "medium";
  return "low";
}

function tagTextColor(distanceM) {
  const d = Number(distanceM);
  if (!Number.isFinite(d)) return "#2f2f2f";
  return "#F7FBFF";
}

function fmtFloat(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "--";
}

function buildImageCandidates(imagePath) {
  const raw = String(imagePath || "").trim();
  if (!raw) return [];
  if (/^https?:\/\//i.test(raw)) return [raw];

  const normalized = raw.replace(/\\/g, "/").replace(/^\.?\//, "").replace(/^\/+/, "");
  const encodedPath = normalized
    .split("/")
    .filter((part) => part.length > 0)
    .map((part) => encodeURIComponent(part))
    .join("/");

  const candidates = [
    `${API_BASE}/${encodedPath}`,
    `${API_BASE}/files/${encodedPath}`,
    normalized,
  ];

  return [...new Set(candidates.filter((v) => v))];
}

function closeImageModal() {
  imageModal.classList.add("hidden");
  imageModalImg.removeAttribute("src");
}

function openImageModal(segmentId, imagePath) {
  const path = String(imagePath || "").trim();
  imageModal.classList.remove("hidden");
  imageModalImg.style.display = "none";
  imageModalImg.removeAttribute("src");
  imageModalImg.onload = null;
  imageModalImg.onerror = null;

  if (!path) {
    imageModalMeta.textContent = `${segmentId}: no image path available`;
    return;
  }

  const candidates = buildImageCandidates(path);
  if (!candidates.length) {
    imageModalMeta.textContent = `${segmentId}: no usable image URL candidates`;
    return;
  }

  imageModalMeta.textContent = `${segmentId}: loading image...`;
  imageModalImg.alt = `Segment image for ${segmentId}`;

  const tryAt = (idx) => {
    if (idx >= candidates.length) {
      imageModalMeta.textContent = `${segmentId}: image could not be loaded (${path})`;
      imageModalImg.style.display = "none";
      return;
    }

    const url = candidates[idx];
    imageModalImg.onload = () => {
      imageModalImg.style.display = "block";
      imageModalMeta.textContent = `${segmentId}: ${path}`;
    };
    imageModalImg.onerror = () => {
      tryAt(idx + 1);
    };
    imageModalImg.src = url;
  };

  tryAt(0);
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

function mapBoundsToBbox(bounds) {
  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();
  return `${sw.lng.toFixed(5)},${sw.lat.toFixed(5)},${ne.lng.toFixed(5)},${ne.lat.toFixed(5)}`;
}

function copyBounds(bounds) {
  if (!bounds || !bounds.isValid()) return null;
  return L.latLngBounds(bounds.getSouthWest(), bounds.getNorthEast());
}

function buildRecordFromFeature(feature) {
  const props = feature.properties || {};
  const coords = feature.geometry?.coordinates || [0, 0];
  const center = [Number(coords[1]), Number(coords[0])];
  const minLon = Number(props.bbox_min_lon);
  const minLat = Number(props.bbox_min_lat);
  const maxLon = Number(props.bbox_max_lon);
  const maxLat = Number(props.bbox_max_lat);

  let bounds;
  if ([minLon, minLat, maxLon, maxLat].every(Number.isFinite)) {
    bounds = L.latLngBounds([minLat, minLon], [maxLat, maxLon]);
  } else {
    const half = 0.0035;
    bounds = L.latLngBounds([center[0] - half, center[1] - half], [center[0] + half, center[1] + half]);
  }

  return {
    segmentId: String(props.segment_id),
    label: String(props.label || props.segment_id || ""),
    center,
    bounds,
    distanceM: Number(props.distance_m),
    originalDistanceM: Number(props.original_distance_m),
    newDistanceM: Number(props.new_distance_m),
    growthRate2yM: Number(props.growth_rate_2y_m),
    annualGrowthM: Number(props.annual_growth_m_per_year),
    year: Number(props.year),
    imagePath: props.image_path ? String(props.image_path) : "",
  };
}

function updateRecords(geojson) {
  records = [];
  recordsById = new Map();
  for (const feature of geojson.features || []) {
    const r = buildRecordFromFeature(feature);
    records.push(r);
    recordsById.set(r.segmentId, r);
  }
}

function bboxRadiusPx(record, zoom) {
  const sw = record.bounds.getSouthWest();
  const ne = record.bounds.getNorthEast();
  const p1 = map.project(sw, zoom);
  const p2 = map.project(ne, zoom);
  const width = Math.abs(p2.x - p1.x);
  const height = Math.abs(p2.y - p1.y);
  const radius = Math.max(width, height) / 2.0;
  return Math.max(3, radius);
}

function clusterTargetRadiusPx(zoom) {
  if (zoom <= 4) return 96;
  if (zoom <= 6) return 76;
  if (zoom <= 8) return 58;
  if (zoom <= 10) return 44;
  if (zoom <= 12) return 34;
  if (zoom <= 14) return 26;
  return 20;
}

function sqDist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function seedCentroids(points, k) {
  if (!points.length || k <= 0) return [];
  const centroids = [];
  const byRisk = [...points].sort((a, b) => a.record.distanceM - b.record.distanceM);
  centroids.push({ x: byRisk[0].x, y: byRisk[0].y });

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

function clusterRecordsKMeans(zoom) {
  if (!records.length) return [];
  if (records.length === 1) {
    const r = records[0];
    return [
      {
        center: r.center,
        count: 1,
        minDistanceM: r.distanceM,
        ids: [r.segmentId],
        bounds: copyBounds(r.bounds),
        minRadiusPx: bboxRadiusPx(r, zoom),
      },
    ];
  }

  const points = records.map((record) => {
    const p = map.project(L.latLng(record.center[0], record.center[1]), zoom);
    return { record, x: p.x, y: p.y };
  });

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  const area = Math.max((maxX - minX) * (maxY - minY), 1);
  const targetRadius = clusterTargetRadiusPx(zoom);
  const targetArea = Math.PI * targetRadius * targetRadius;
  const roughK = Math.round(area / targetArea);
  const k = Math.max(1, Math.min(points.length, roughK));

  const centroids = seedCentroids(points, k);
  const assignment = new Array(points.length).fill(0);

  for (let iter = 0; iter < 12; iter += 1) {
    for (let i = 0; i < points.length; i += 1) {
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let c = 0; c < centroids.length; c += 1) {
        const dist = sqDist(points[i], centroids[c]);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = c;
        }
      }
      assignment[i] = bestIdx;
    }

    const sums = centroids.map(() => ({ x: 0, y: 0, n: 0 }));
    for (let i = 0; i < points.length; i += 1) {
      const a = assignment[i];
      sums[a].x += points[i].x;
      sums[a].y += points[i].y;
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
  for (let i = 0; i < points.length; i += 1) {
    const clusterId = assignment[i];
    if (!grouped.has(clusterId)) {
      grouped.set(clusterId, {
        count: 0,
        latSum: 0,
        lonSum: 0,
        minDistanceM: Infinity,
        ids: [],
        bounds: null,
        minRadiusPx: 0,
      });
    }
    const g = grouped.get(clusterId);
    const r = points[i].record;
    g.count += 1;
    g.latSum += r.center[0];
    g.lonSum += r.center[1];
    g.minDistanceM = Math.min(g.minDistanceM, r.distanceM);
    g.ids.push(r.segmentId);
    g.bounds = g.bounds ? g.bounds.extend(r.bounds) : copyBounds(r.bounds);
    g.minRadiusPx = Math.max(g.minRadiusPx, bboxRadiusPx(r, zoom));
  }

  return [...grouped.values()].map((g) => ({
    center: [g.latSum / g.count, g.lonSum / g.count],
    count: g.count,
    minDistanceM: g.minDistanceM,
    ids: g.ids,
    bounds: g.bounds,
    minRadiusPx: g.minRadiusPx,
  }));
}

function renderDistanceCircles() {
  heatLayer.clearLayers();
  if (!records.length) return;

  const zoom = map.getZoom();
  const clusters = clusterRecordsKMeans(zoom);
  const targetRadius = clusterTargetRadiusPx(zoom);

  for (const cluster of clusters) {
    const base = Math.max(6, targetRadius * 0.24);
    const overlapRadius = base + Math.min(targetRadius * 0.65, Math.sqrt(cluster.count) * 2.35);
    const radius = Math.max(overlapRadius, cluster.minRadiusPx);
    const selected = activeSegment && cluster.ids.includes(activeSegment);
    const color = distanceColor(cluster.minDistanceM);

    const marker = L.circleMarker(cluster.center, {
      radius: selected ? radius + 2 : radius,
      color: "#1A2233",
      weight: selected ? 2.2 : 1.0,
      fillColor: color,
      fillOpacity: selected ? 0.94 : 0.74,
    });
    marker.bindTooltip(
      `${cluster.count} areas<br/>Closest distance: ${fmtFloat(cluster.minDistanceM)} m`,
      { sticky: true }
    );
    marker.on("click", () => {
      if (cluster.ids.length === 1) {
        selectSegment(cluster.ids[0], true);
        return;
      }
      if (cluster.bounds && cluster.bounds.isValid()) {
        map.fitBounds(cluster.bounds.pad(0.45), { maxZoom: 14 });
        return;
      }
      map.setView(cluster.center, Math.min(14, map.getZoom() + 2));
    });
    marker.addTo(heatLayer);
  }
}

function fitToDataIfNeeded() {
  if (hasAutoFit || !records.length) return;
  let bounds = null;
  for (const r of records) {
    bounds = bounds ? bounds.extend(r.bounds) : copyBounds(r.bounds);
  }
  if (bounds && bounds.isValid()) {
    map.fitBounds(bounds.pad(0.18));
    hasAutoFit = true;
  }
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
  if (!force && bbox === lastPowerlineBboxKey) return;
  lastPowerlineBboxKey = bbox;

  const reqId = ++powerlineRequestId;
  getJson(`/powerlines/layer?bbox=${encodeURIComponent(bbox)}&max_features=18000`)
    .then((geojson) => {
      if (reqId !== powerlineRequestId) return;
      renderPowerlineOverlay(geojson);
    })
    .catch(() => {
      // Optional overlay.
    });
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
    li.dataset.segmentId = item.segment_id;
    const badgeColor = distanceColor(item.distance_m);
    const badgeTextColor = tagTextColor(item.distance_m);

    const topRow = document.createElement("div");
    topRow.className = "risk-row";

    const idSpan = document.createElement("span");
    idSpan.className = "risk-id";
    idSpan.textContent = item.segment_id;

    const pill = document.createElement("span");
    pill.className = "pill";
    pill.style.background = badgeColor;
    pill.style.color = badgeTextColor;
    pill.style.border = "1px solid rgba(18,33,63,0.28)";
    pill.textContent = distanceBand(item.distance_m);

    topRow.appendChild(idSpan);
    topRow.appendChild(pill);

    const secondRow = document.createElement("div");
    secondRow.className = "risk-row";

    const distanceSpan = document.createElement("span");
    distanceSpan.className = "mono";
    distanceSpan.textContent = `Distance ${fmtFloat(item.distance_m)} m`;

    const imageBtn = document.createElement("button");
    imageBtn.className = "point-icon-btn";
    imageBtn.type = "button";
    imageBtn.textContent = "Img";
    imageBtn.title = "View segment image";
    imageBtn.setAttribute("aria-label", `Open image for ${item.segment_id}`);
    imageBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      openImageModal(item.segment_id, item.image_path);
    });

    const actionsWrap = document.createElement("div");
    actionsWrap.className = "risk-row-actions";
    actionsWrap.appendChild(imageBtn);

    secondRow.appendChild(distanceSpan);
    secondRow.appendChild(actionsWrap);

    li.appendChild(topRow);
    li.appendChild(secondRow);
    li.addEventListener("click", () => selectSegment(item.segment_id, true));
    topList.appendChild(li);
  }
}

function buildPath(points, xScale, yScale) {
  return points
    .map((p, idx) => `${idx === 0 ? "M" : "L"} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`)
    .join(" ");
}

function renderDistanceChart(rows) {
  const w = 860;
  const h = 210;
  const pad = { l: 52, r: 24, t: 18, b: 28 };
  chartSvg.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const sorted = [...rows].sort((a, b) => Number(a.year) - Number(b.year));
  const hist = sorted
    .filter((r) => Number(r.year) <= 2026)
    .map((r) => ({ x: Number(r.year), y: Number(r.distance_m) }))
    .filter((r) => Number.isFinite(r.y));
  const proj = sorted
    .filter((r) => Number(r.year) >= 2026)
    .map((r) => ({ x: Number(r.year), y: Number(r.distance_m) }))
    .filter((r) => Number.isFinite(r.y));

  const all = sorted.map((r) => ({ x: Number(r.year), y: Number(r.distance_m) })).filter((r) => Number.isFinite(r.y));
  if (!all.length) {
    chartSvg.innerHTML = `<text x="18" y="42" fill="#466151">No distance values for this segment.</text>`;
    return;
  }

  const minX = Math.min(...all.map((p) => p.x));
  const maxX = Math.max(...all.map((p) => p.x));
  const minY = Math.max(0, Math.min(...all.map((p) => p.y)) - 0.3);
  const maxY = Math.max(...all.map((p) => p.y)) + 0.3;

  const xScale = (x) => pad.l + ((w - pad.l - pad.r) * (x - minX)) / Math.max(1, maxX - minX);
  const yScale = (y) => pad.t + (h - pad.t - pad.b) * (1 - (y - minY) / Math.max(0.0001, maxY - minY));

  const gridLines = [0, 1, 2, 3, 4]
    .map((g) => {
      const gy = pad.t + ((h - pad.t - pad.b) * g) / 4;
      return `<line x1="${pad.l}" y1="${gy}" x2="${w - pad.r}" y2="${gy}" stroke="rgba(45,71,56,0.2)" stroke-width="1"/>`;
    })
    .join("");

  const xTicks = sorted
    .map((r) => Number(r.year))
    .filter((v, i, arr) => arr.indexOf(v) === i)
    .map((year) => `<text x="${xScale(year)}" y="${h - 8}" text-anchor="middle" fill="#3E5D4D" font-size="10">${year}</text>`)
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
    <path d="${buildPath(hist, xScale, yScale)}" fill="none" stroke="#1E7A53" stroke-width="3.2"></path>
    <path d="${buildPath(proj, xScale, yScale)}" fill="none" stroke="#BE5B1C" stroke-width="3.2" stroke-dasharray="8 5"></path>
    ${xTicks}
    ${yTicks}
  `;
}

async function selectSegment(segmentId, zoomToFeature = false) {
  activeSegment = segmentId;
  segmentTitle.textContent = segmentId;
  segmentMeta.textContent = "loading...";

  // Scroll the corresponding sidebar item into view
  const sidebarItem = topList.querySelector(`[data-segment-id="${CSS.escape(segmentId)}"]`);
  if (sidebarItem) {
    sidebarItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  const local = recordsById.get(segmentId);
  if (zoomToFeature && local?.bounds?.isValid()) {
    map.fitBounds(local.bounds.pad(0.45), { maxZoom: 14 });
  }

  const rows = await getJson(`/segments/${encodeURIComponent(segmentId)}/timeseries`);
  if (rows.length) {
    const current = rows.find((r) => Number(r.year) === Number(years[currentYearIdx])) || rows[rows.length - 1];
    segmentMeta.textContent = `distance ${fmtFloat(current.distance_m)} m | annual growth ${fmtFloat(current.annual_growth_m_per_year)} m`;
  } else {
    segmentMeta.textContent = "no rows";
  }
  renderDistanceChart(rows);
  renderDistanceCircles();
}

async function refreshPredictionLayer(force = false) {
  if (!years.length) return;
  const year = years[currentYearIdx];
  const bbox = mapBoundsToBbox(map.getBounds());
  const key = `${year}|${bbox}`;
  if (!force && key === lastPredictionKey) {
    renderDistanceCircles();
    return;
  }
  lastPredictionKey = key;

  const reqId = ++predictionRequestId;
  const geojson = await getJson(
    `/map/layer?year=${encodeURIComponent(year)}&bbox=${encodeURIComponent(bbox)}&max_features=12000`
  );
  if (reqId !== predictionRequestId) return;
  updateRecords(geojson);
  fitToDataIfNeeded();
  renderDistanceCircles();
}

async function loadYear() {
  if (!years.length) return;
  updateYearLabel();
  const year = years[currentYearIdx];

  const [topResult] = await Promise.allSettled([
    getJson(`/segments/top?year=${encodeURIComponent(year)}`),
    refreshPredictionLayer(true),
    Promise.resolve(refreshPowerlineOverlay(true)),
  ]);

  if (topResult.status === "fulfilled") {
    renderTopList(topResult.value);
    if (!activeSegment && topResult.value.length) {
      selectSegment(topResult.value[0].segment_id, false);
    }
  }
}

async function initialize() {
  try {
    imageModalClose.addEventListener("click", closeImageModal);
    imageModal.addEventListener("click", (ev) => {
      if (ev.target === imageModal) closeImageModal();
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && !imageModal.classList.contains("hidden")) closeImageModal();
    });
    const timeline = await getJson("/timeline/years");
    years = timeline.years || [2025, 2026, 2027, 2028, 2029, 2030];

    if (!years.length) {
      yearLabel.textContent = "no data";
      topList.innerHTML = `<li class="risk-item">No years found.</li>`;
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
    map.on("moveend", () => {
      refreshPredictionLayer(false);
      refreshPowerlineOverlay(false);
    });
    map.on("zoomend", () => {
      renderDistanceCircles();
      refreshPowerlineOverlay(false);
    });

    await loadYear();
  } catch (err) {
    topList.innerHTML = `<li class="risk-item">Failed to load API data: ${String(err)}</li>`;
  }
}

initialize();
