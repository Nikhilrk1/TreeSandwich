const API_BASE = window.location.origin;

const monthRange = document.getElementById("monthRange");
const monthLabel = document.getElementById("monthLabel");
const playBtn = document.getElementById("playBtn");
const minRiskInput = document.getElementById("minRisk");
const topList = document.getElementById("topList");
const segmentTitle = document.getElementById("segmentTitle");
const segmentMeta = document.getElementById("segmentMeta");
const chartSvg = document.getElementById("chart");

const map = L.map("map", { zoomControl: true }).setView([33.8361, -81.1637], 8);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

let months = [];
let currentMonthIdx = 0;
let playing = false;
let playTimer = null;
let geoLayer = null;
let featureBySegment = new Map();
let activeSegment = null;

function riskColor(score) {
  if (score >= 80) return "#A72312";
  if (score >= 70) return "#D6451E";
  if (score >= 60) return "#E67B25";
  if (score >= 50) return "#E0B43D";
  if (score >= 40) return "#88A94F";
  return "#3E8B60";
}

function riskPillClass(level) {
  if (level === "high") return "pill-high";
  if (level === "medium") return "pill-medium";
  return "pill-low";
}

function fmtInt(value) {
  const n = Number(value);
  return Number.isFinite(n) ? String(Math.round(n)) : "--";
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
      if (!months.length) return;
      currentMonthIdx = (currentMonthIdx + 1) % months.length;
      monthRange.value = String(currentMonthIdx);
      loadMonth();
    }, 1300);
  }
}

function updateMonthLabel() {
  monthLabel.textContent = months[currentMonthIdx] || "-";
}

function renderLayer(geojson) {
  featureBySegment = new Map();
  if (geoLayer) {
    geoLayer.remove();
  }

  geoLayer = L.geoJSON(geojson, {
    style: (feature) => {
      const score = feature?.properties?.risk_score || 0;
      return {
        color: riskColor(score),
        weight: activeSegment === feature?.properties?.segment_id ? 6 : 4,
        opacity: 0.9,
      };
    },
    onEachFeature: (feature, layer) => {
      const props = feature.properties || {};
      const segmentId = props.segment_id;
      if (!segmentId) return;
      featureBySegment.set(segmentId, { feature, layer });
      layer.on("click", () => selectSegment(segmentId, true));
      layer.bindTooltip(
        `${segmentId}<br/>Risk: ${Math.round(props.risk_score ?? 0)} (${props.risk_level || "n/a"})`,
        { sticky: true }
      );
    },
  }).addTo(map);

  if (geojson.features && geojson.features.length) {
    const bounds = geoLayer.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds.pad(0.18));
  }
}

function renderTopList(items) {
  topList.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "risk-item";
    li.textContent = "No segments returned for this month.";
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
        <span class="pill ${riskPillClass(item.risk_level)}">${item.risk_level || "n/a"}</span>
      </div>
      <div class="risk-row">
        <span class="mono">Score ${Math.round(item.risk_score ?? 0)}</span>
        <span class="mono">${trendArrow} conf ${Number(item.risk_confidence ?? 0).toFixed(2)}</span>
      </div>
      <div class="risk-row">
        <span class="mono">+3m ${fmtInt(item.forecast_risk_3m)}</span>
        <span class="mono">+6m ${fmtInt(item.forecast_risk_6m)}</span>
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
  const pad = { l: 48, r: 24, t: 18, b: 28 };
  chartSvg.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const hist = historyRows
    .map((r) => ({ date: r.target_date, y: Number(r.ndvi_median) }))
    .filter((r) => Number.isFinite(r.y));
  const fct = forecastRows
    .map((r) => ({
      date: r.target_date,
      y: Number(r.ndvi_median),
      low: Number(r.forecast_lower),
      high: Number(r.forecast_upper),
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
    chartSvg.innerHTML = `<text x="18" y="42" fill="#466151">No NDVI values for this segment.</text>`;
    return;
  }

  const uniqueDates = [...new Set(allDates)];
  const xIndex = new Map(uniqueDates.map((d, i) => [d, i]));
  const minY = Math.min(...allY) - 0.04;
  const maxY = Math.max(...allY) + 0.04;
  const xScale = (x) =>
    pad.l + ((w - pad.l - pad.r) * x) / Math.max(1, uniqueDates.length - 1);
  const yScale = (y) =>
    pad.t + (h - pad.t - pad.b) * (1 - (y - minY) / Math.max(0.0001, maxY - minY));

  const histPts = hist.map((d) => ({ x: xIndex.get(d.date), y: d.y }));
  const fctPts = fct.map((d) => ({ x: xIndex.get(d.date), y: d.y }));

  let bandPath = "";
  if (fct.length) {
    const upper = fct.map((d) => ({ x: xIndex.get(d.date), y: d.high }));
    const lower = [...fct]
      .reverse()
      .map((d) => ({ x: xIndex.get(d.date), y: d.low }));
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
    <text x="${pad.l + 8}" y="${pad.t + 12}" fill="#1F4A36" font-size="11">NDVI history (green) and forecast (orange)</text>
  `;
}

async function selectSegment(segmentId, zoomToFeature = false) {
  activeSegment = segmentId;
  segmentTitle.textContent = segmentId;
  segmentMeta.textContent = "loading...";

  if (zoomToFeature && featureBySegment.has(segmentId)) {
    const item = featureBySegment.get(segmentId);
    map.fitBounds(item.layer.getBounds().pad(0.45), { maxZoom: 13 });
  }

  const layerData = await Promise.allSettled([
    getJson(`/segments/${encodeURIComponent(segmentId)}/timeseries`),
    getJson(`/segments/${encodeURIComponent(segmentId)}/forecast`),
  ]);

  const hist = layerData[0].status === "fulfilled" ? layerData[0].value : [];
  const fct = layerData[1].status === "fulfilled" ? layerData[1].value : [];

  if (hist.length) {
    const latest = hist[hist.length - 1];
    segmentMeta.textContent = `score ${Math.round(latest.risk_score ?? 0)} | ${latest.target_date}`;
  } else {
    segmentMeta.textContent = "no history rows";
  }
  renderChart(hist, fct);
  if (geoLayer) geoLayer.setStyle(geoLayer.options.style);
}

async function loadMonth() {
  if (!months.length) return;
  updateMonthLabel();
  const month = months[currentMonthIdx];
  const minRisk = Number(minRiskInput.value || 0);

  const [layerResult, topResult] = await Promise.allSettled([
    getJson(`/map/layer?date=${encodeURIComponent(month)}&min_risk=${minRisk}`),
    getJson(`/segments/top?date=${encodeURIComponent(month)}&n=20`),
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
}

async function initialize() {
  try {
    const timeline = await getJson("/timeline/months");
    months = timeline.months || [];
    if (!months.length) {
      monthLabel.textContent = "no data";
      topList.innerHTML = `<li class="risk-item">No months found in dataset.</li>`;
      return;
    }

    const defaultMonth = timeline.default || months[months.length - 1];
    currentMonthIdx = Math.max(0, months.indexOf(defaultMonth));
    monthRange.max = String(Math.max(0, months.length - 1));
    monthRange.value = String(currentMonthIdx);
    updateMonthLabel();

    monthRange.addEventListener("input", () => {
      currentMonthIdx = Number(monthRange.value);
      loadMonth();
    });
    minRiskInput.addEventListener("change", loadMonth);
    playBtn.addEventListener("click", () => setPlaying(!playing));

    await loadMonth();
  } catch (err) {
    topList.innerHTML = `<li class="risk-item">Failed to load API data: ${String(err)}</li>`;
  }
}

initialize();
