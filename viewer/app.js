import { parseGlb } from "./glb.js";
import { ViewerRenderer } from "./renderer.js";

const $ = (selector) => document.querySelector(selector);
const canvas = $("#canvas");
const dropZone = $("#drop-zone");
const loading = $("#loading");
const toast = $("#toast");
const appState = {
  model: null,
  session: null,
  results: [],
  activeResult: null,
  replay: null,
  frame: 0,
  playing: false,
  playbackStart: 0,
  playbackStartTick: 0,
  placingFlash: false,
};
let toastTimer = null;

let renderer;
try {
  renderer = new ViewerRenderer(canvas, ({ eye, fps }) => {
    $("#camera-status").textContent = `Camera ${eye.map((value) => Math.round(value)).join(", ")}`;
    $("#fps-status").textContent = `${fps} FPS`;
  });
} catch (error) {
  showError(error);
  throw error;
}

const chooseDemo = () => $("#demo-input").click();
const chooseSession = () => $("#session-input").click();
const chooseGlb = () => $("#file-input").click();
$("#open-demo-button").addEventListener("click", chooseDemo);
$("#empty-open-button").addEventListener("click", chooseDemo);
$("#open-session-button").addEventListener("click", chooseSession);
$("#open-button").addEventListener("click", chooseGlb);
bindFileInput("#demo-input", loadDemoFile);
bindFileInput("#session-input", loadSessionFile);
bindFileInput("#file-input", loadGlbFile);

$("#save-session-button").addEventListener("click", () => {
  const result = appState.activeResult ? `?result=${encodeURIComponent(appState.activeResult.id)}` : "";
  download(`/api/session/export.cs2session${result}`);
});
$("#export-button").addEventListener("click", () => {
  if (!appState.activeResult) return;
  const params = new URLSearchParams();
  if (appState.activeResult.analysisType === "vision") {
    params.set("frame", String(appState.frame));
    params.set("mode", $("#replay-mode").value);
  }
  download(`/api/results/${appState.activeResult.id}/export.glb?${params}`);
});

$("#frame-button").addEventListener("click", () => renderer.frameSelection());
$("#show-all-button").addEventListener("click", () => {
  if (!appState.model) return;
  appState.model.drawables.forEach((item) => { item.visible = true; });
  renderObjectList();
  renderer.requestRender();
});
bindToggle("#wireframe-button", (value) => renderer.setWireframe(value));
bindToggle("#grid-button", (value) => renderer.setGrid(value), true);
bindToggle("#axes-button", (value) => renderer.setAxes(value), true);
$("#shading-select").addEventListener("change", (event) => renderer.setShading(event.target.value));
$("#projection-button").addEventListener("click", () => {
  const button = $("#projection-button");
  const orthographic = button.textContent === "Perspective";
  button.textContent = orthographic ? "Orthographic" : "Perspective";
  button.classList.toggle("active", orthographic);
  renderer.setProjection(orthographic ? "orthographic" : "perspective");
});

for (const button of [$("#vision-tab"), $("#flash-tab")]) {
  button.addEventListener("click", () => setAnalysisType(button.dataset.analysis));
}
$("#round-select").addEventListener("change", configureRound);
$("#time-mode").addEventListener("change", configureTimeMode);
bindTimeControl("#start-time", "#start-slider", true);
bindTimeControl("#end-time", "#end-slider", false);
$("#flash-source").addEventListener("change", configureFlashSource);
$("#flash-event-select").addEventListener("change", previewSelectedFlash);
for (const selector of ["#flash-x", "#flash-y", "#flash-z"]) {
  $(selector).addEventListener("input", previewManualFlash);
}
$("#place-flash-button").addEventListener("click", () => setPlacementMode(!appState.placingFlash));
$("#analyze-button").addEventListener("click", analyzeCurrentSelection);

$("#frame-slider").addEventListener("input", (event) => {
  stopPlayback();
  appState.frame = Number(event.target.value);
  applyReplayFrame();
});
$("#replay-mode").addEventListener("change", applyReplayFrame);
$("#play-button").addEventListener("click", () => appState.playing ? stopPlayback() : startPlayback());

canvas.addEventListener("click", async (event) => {
  if (!appState.placingFlash) return;
  try {
    const ray = renderer.getPickRay(event.clientX, event.clientY);
    const response = await apiJson("/api/pick", { method: "POST", body: JSON.stringify(ray) });
    if (!response.position) throw new Error("The placement ray did not hit the map.");
    setManualPosition(response.position);
    setPlacementMode(false);
  } catch (error) {
    showError(error);
  }
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && appState.placingFlash) { setPlacementMode(false); return; }
  if (event.target.matches("input, select, button")) return;
  const key = event.key.toLowerCase();
  if (key === "o" && event.ctrlKey) { event.preventDefault(); chooseGlb(); }
  if (key === "f") { event.preventDefault(); renderer.frameSelection(); }
  if (key === "w" && event.shiftKey) { event.preventDefault(); clickToggle("#wireframe-button"); }
  if (key === "g") clickToggle("#grid-button");
  if (key === "x") clickToggle("#axes-button");
  if (key === "5") $("#projection-button").click();
  if (key === " ") { event.preventDefault(); $("#play-button").click(); }
});

for (const eventName of ["dragenter", "dragover"]) {
  window.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  window.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
}
window.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const extension = file.name.toLowerCase().split(".").pop();
  if (extension === "dem") loadDemoFile(file);
  else if (extension === "cs2session") loadSessionFile(file);
  else if (extension === "glb") loadGlbFile(file);
  else showError(new Error("Drop a .dem, .cs2session, or .glb file."));
});

function bindFileInput(selector, callback) {
  const input = $(selector);
  input.addEventListener("change", () => {
    if (input.files[0]) callback(input.files[0]);
    input.value = "";
  });
}

async function loadDemoFile(file) {
  if (!file.name.toLowerCase().endsWith(".dem")) return showError(new Error("Choose a .dem file."));
  await uploadAndLoad(file, "/api/demo/load", "Loading demo");
}

async function loadSessionFile(file) {
  if (!file.name.toLowerCase().endsWith(".cs2session")) return showError(new Error("Choose a .cs2session file."));
  await uploadAndLoad(file, "/api/session/load", "Opening session");
}

async function uploadAndLoad(file, endpoint, title) {
  stopPlayback();
  setLoading(true, title, `${file.name} · ${formatBytes(file.size)}`, 0);
  try {
    const job = await apiJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    await waitForJob(job.id);
    await refreshApplicationState(true);
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

async function refreshApplicationState(loadGeometry = false) {
  const state = await apiJson("/api/state");
  appState.session = state.session;
  appState.results = state.results || [];
  if (!appState.session) return;
  installSessionControls();
  if (loadGeometry) await loadUrl("/api/geometry.glb", appState.session.mapName);
  const savedResult = appState.results.at(-1);
  if (savedResult) await loadResult(savedResult.id);
}

function installSessionControls() {
  const session = appState.session;
  $("#session-empty").hidden = true;
  $("#session-controls").hidden = false;
  $("#analysis-section").hidden = false;
  $("#session-map").textContent = session.mapName;
  $("#session-source").textContent = session.sourceName;
  $("#model-name").textContent = `${session.sourceName} · ${session.mapName}`;
  $("#save-session-button").disabled = false;
  const roundSelect = $("#round-select");
  roundSelect.replaceChildren(...session.rounds.map((round) => option(round.number, `Round ${round.number}`)));
  configureRound();
  populateFlashes();
}

function configureRound() {
  if (!appState.session) return;
  const round = selectedRound();
  const playerSelect = $("#player-select");
  playerSelect.replaceChildren();
  const groups = groupBy(round.players, (player) => player.team || player.side || "Unknown team");
  for (const [team, players] of groups) {
    const group = document.createElement("optgroup");
    group.label = displayTeam(team);
    for (const player of players) group.append(option(player.id, player.name));
    playerSelect.append(group);
  }
  const maximum = Math.max(0, round.durationSeconds);
  for (const selector of ["#start-slider", "#end-slider", "#start-time", "#end-time"]) $(selector).max = maximum.toFixed(2);
  $("#start-time").value = "0";
  $("#start-slider").value = "0";
  const defaultEnd = Math.min(10, maximum);
  $("#end-time").value = defaultEnd.toFixed(1);
  $("#end-slider").value = defaultEnd.toFixed(1);
}

function populateFlashes() {
  const select = $("#flash-event-select");
  select.replaceChildren();
  const flashes = appState.session.flashes || [];
  const groups = groupBy(flashes, (flash) => flash.thrower_team || flash.thrower_side || "Unknown team");
  for (const [team, values] of groups) {
    const group = document.createElement("optgroup");
    group.label = displayTeam(team);
    for (const flash of values) {
      const round = appState.session.rounds.find((item) => item.number === flash.round_number);
      const seconds = round ? (flash.tick - round.freezeEndTick) / appState.session.tickRate : 0;
      group.append(option(flash.index, `R${flash.round_number ?? "?"} · ${formatTime(seconds)} · ${flash.thrower || "Unknown"}`));
    }
    select.append(group);
  }
  if (!flashes.length) select.append(option("", "No flash detonations found"));
  previewSelectedFlash();
}

function setAnalysisType(type) {
  $("#vision-tab").classList.toggle("active", type === "vision");
  $("#flash-tab").classList.toggle("active", type === "flash");
  $("#vision-controls").hidden = type !== "vision";
  $("#flash-controls").hidden = type !== "flash";
  if (type === "flash") configureFlashSource();
  else { setPlacementMode(false); renderer.setMarker(null); }
}

function configureTimeMode() {
  const instant = $("#time-mode").value === "instant";
  $("#end-time-field").hidden = instant;
  $("#end-slider").hidden = instant;
}

function bindTimeControl(numberSelector, sliderSelector, isStart) {
  const number = $(numberSelector);
  const slider = $(sliderSelector);
  const sync = (source, target) => {
    target.value = source.value;
    let start = Number($("#start-time").value);
    let end = Number($("#end-time").value);
    if (isStart && start > end) {
      $("#end-time").value = String(start);
      $("#end-slider").value = String(start);
    } else if (!isStart && end < start) {
      $("#start-time").value = String(end);
      $("#start-slider").value = String(end);
    }
  };
  number.addEventListener("input", () => sync(number, slider));
  slider.addEventListener("input", () => sync(slider, number));
}

function configureFlashSource() {
  const manual = $("#flash-source").value === "manual";
  $("#flash-event-field").hidden = manual;
  $("#manual-flash-controls").hidden = !manual;
  if (manual) previewManualFlash(); else { setPlacementMode(false); previewSelectedFlash(); }
}

function previewSelectedFlash() {
  if (!appState.session || $("#flash-source").value !== "event") return;
  const flash = appState.session.flashes.find((item) => String(item.index) === $("#flash-event-select").value);
  renderer.setMarker(flash ? positionArray(flash.position) : null);
}

function previewManualFlash() {
  if ($("#flash-source").value !== "manual") return;
  const position = manualPosition(false);
  renderer.setMarker(position);
}

function setManualPosition(position) {
  ["#flash-x", "#flash-y", "#flash-z"].forEach((selector, index) => { $(selector).value = position[index].toFixed(2); });
  renderer.setMarker(position);
}

function manualPosition(required = true) {
  const selectors = ["#flash-x", "#flash-y", "#flash-z"];
  const values = selectors.map((selector) => Number($(selector).value));
  const valid = values.every((value, index) => Number.isFinite(value) && $(selectors[index]).value !== "");
  if (!valid && required) throw new Error("Enter an X, Y, and Z position or place the flash on the map.");
  return valid ? values : null;
}

function setPlacementMode(enabled) {
  appState.placingFlash = enabled;
  canvas.classList.toggle("placing", enabled);
  $("#place-flash-button").classList.toggle("active", enabled);
  $("#place-flash-button").textContent = enabled ? "Cancel placement" : "Place on map";
  $("#placement-help").hidden = !enabled;
}

async function analyzeCurrentSelection() {
  if (!appState.session) return;
  const isVision = $("#vision-tab").classList.contains("active");
  try {
    let endpoint;
    let request;
    if (isVision) {
      endpoint = "/api/analyze/vision";
      request = {
        playerId: $("#player-select").value,
        roundNumber: Number($("#round-select").value),
        timeMode: $("#time-mode").value,
        startSeconds: Number($("#start-time").value),
        endSeconds: Number($("#end-time").value),
        tickStep: Number($("#tick-step").value),
        fov: Number($("#vision-fov").value),
        maxDistance: Number($("#vision-distance").value),
        samplesPerTriangle: Number($("#vision-samples").value),
      };
    } else {
      endpoint = "/api/analyze/flash";
      const manual = $("#flash-source").value === "manual";
      request = {
        roundNumber: Number($("#round-select").value),
        maxDistance: Number($("#flash-distance").value),
        falloffPower: Number($("#flash-falloff").value),
        samplesPerTriangle: Number($("#flash-samples").value),
        ...(manual ? { position: manualPosition() } : { eventIndex: Number($("#flash-event-select").value) }),
      };
    }
    setLoading(true, isVision ? "Analyzing player vision" : "Analyzing flash coverage", "Preparing rays", 0);
    const job = await apiJson(endpoint, { method: "POST", body: JSON.stringify(request) });
    const completed = await waitForJob(job.id);
    await refreshApplicationState(false);
    await loadResult(completed.resultId);
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

async function waitForJob(jobId) {
  while (true) {
    const job = await apiJson(`/api/jobs/${jobId}`);
    setLoading(true, job.kind === "load-demo" ? "Loading demo" : "Working", job.message, job.progress);
    if (job.status === "complete") return job;
    if (job.status === "error") throw new Error(job.error || "The job failed.");
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

async function loadResult(resultId) {
  const metadata = appState.results.find((item) => item.id === resultId);
  if (!metadata) throw new Error("The completed result was not found.");
  const response = await fetch(`/api/results/${resultId}/data.bin`);
  if (!response.ok) throw new Error("Could not load the analysis result.");
  const buffer = await response.arrayBuffer();
  appState.activeResult = metadata;
  $("#export-button").disabled = false;
  $("#result-section").hidden = false;
  renderResultDetails(metadata);
  if (metadata.analysisType === "vision") installVisionReplay(buffer, metadata);
  else installFlashResult(buffer, metadata);
}

function installVisionReplay(buffer, metadata) {
  const header = parseHeader(buffer, "CSV1");
  const ticksOffset = 24;
  const ticksBytes = header.frameCount * 4;
  const masksBytes = header.frameCount * header.width;
  if (buffer.byteLength !== 24 + ticksBytes + masksBytes * 2) throw new Error("Vision result is truncated.");
  appState.replay = {
    type: "vision", faceCount: header.faceCount, frameCount: header.frameCount, width: header.width,
    ticks: new Uint32Array(buffer, ticksOffset, header.frameCount),
    instant: new Uint8Array(buffer, ticksOffset + ticksBytes, masksBytes),
    cumulative: new Uint8Array(buffer, ticksOffset + ticksBytes + masksBytes, masksBytes),
    metadata,
  };
  renderer.setMarker(null);
  appState.frame = Math.max(0, header.frameCount - 1);
  $("#frame-slider").max = String(Math.max(0, header.frameCount - 1));
  $("#frame-slider").value = String(appState.frame);
  $("#timeline").hidden = header.frameCount <= 1;
  $("#replay-mode").value = metadata.timeMode === "instant" ? "instant" : "cumulative";
  applyReplayFrame();
}

function installFlashResult(buffer, metadata) {
  const header = parseHeader(buffer, "CSF1");
  if (buffer.byteLength !== 24 + header.faceCount) throw new Error("Flash result is truncated.");
  stopPlayback();
  $("#timeline").hidden = true;
  appState.replay = { type: "flash", faceCount: header.faceCount, values: new Uint8Array(buffer, 24, header.faceCount) };
  renderer.setFaceValues(appState.replay.values, "flash");
  renderer.setMarker(positionArray(metadata.flash.position));
  $("#status-summary").textContent = `${formatNumber(header.faceCount)} faces · ${metadata.backend} · ${formatNumber(metadata.testedRays)} rays`;
}

function parseHeader(buffer, expectedMagic) {
  if (buffer.byteLength < 24) throw new Error("Analysis result is truncated.");
  const bytes = new Uint8Array(buffer, 0, 4);
  const magic = String.fromCharCode(...bytes);
  const view = new DataView(buffer);
  if (magic !== expectedMagic || view.getUint32(4, true) !== 1) throw new Error("Unsupported analysis result format.");
  return {
    faceCount: view.getUint32(8, true), frameCount: view.getUint32(12, true), width: view.getUint32(16, true),
  };
}

function applyReplayFrame() {
  const replay = appState.replay;
  if (!replay || replay.type !== "vision" || !replay.frameCount) return;
  appState.frame = Math.max(0, Math.min(appState.frame, replay.frameCount - 1));
  const mode = $("#replay-mode").value;
  const source = replay[mode];
  const offset = appState.frame * replay.width;
  const values = new Uint8Array(replay.faceCount);
  for (let face = 0; face < replay.faceCount; face++) {
    values[face] = ((source[offset + (face >> 3)] >> (face & 7)) & 1) ? 255 : 0;
  }
  renderer.setFaceValues(values, "vision");
  $("#frame-slider").value = String(appState.frame);
  const round = appState.session.rounds.find((item) => item.number === replay.metadata.roundNumber);
  const seconds = round ? (replay.ticks[appState.frame] - round.freezeEndTick) / appState.session.tickRate : 0;
  $("#replay-time").textContent = formatTime(seconds);
  $("#status-summary").textContent = `Frame ${appState.frame + 1}/${replay.frameCount} · tick ${replay.ticks[appState.frame]} · ${replay.metadata.backend}`;
}

function startPlayback() {
  const replay = appState.replay;
  if (!replay || replay.type !== "vision" || replay.frameCount <= 1) return;
  if (appState.frame >= replay.frameCount - 1) appState.frame = 0;
  appState.playing = true;
  appState.playbackStart = performance.now();
  appState.playbackStartTick = replay.ticks[appState.frame];
  $("#play-button").textContent = "Pause";
  requestAnimationFrame(playbackFrame);
}

function playbackFrame(time) {
  if (!appState.playing) return;
  const replay = appState.replay;
  const speed = Number($("#playback-speed").value);
  const targetTick = appState.playbackStartTick + (time - appState.playbackStart) / 1000 * replay.metadata.tickRate * speed;
  while (appState.frame + 1 < replay.frameCount && replay.ticks[appState.frame + 1] <= targetTick) appState.frame++;
  if (appState.frame !== Number($("#frame-slider").value)) applyReplayFrame();
  if (appState.frame >= replay.frameCount - 1) stopPlayback();
  else requestAnimationFrame(playbackFrame);
}

function stopPlayback() {
  appState.playing = false;
  $("#play-button").textContent = "Play";
}

function renderResultDetails(result) {
  const details = $("#result-details");
  const rows = result.analysisType === "vision"
    ? [["Type", result.timeMode === "instant" ? "Vision instant" : "Vision replay"], ["Player", result.playerName], ["Frames", formatNumber(result.frameCount)], ["Backend", result.backend], ["Rays", formatNumber(result.testedRays)]]
    : [["Type", "Flash coverage"], ["Source", result.source], ["Thrower", result.flash.thrower || "Manual"], ["Backend", result.backend], ["Rays", formatNumber(result.testedRays)]];
  details.replaceChildren(...rows.map(([key, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt"); term.textContent = key;
    const definition = document.createElement("dd"); definition.textContent = value;
    row.append(term, definition);
    return row;
  }));
}

async function loadGlbFile(file) {
  if (!file.name.toLowerCase().endsWith(".glb")) return showError(new Error("Choose a .glb file."));
  setLoading(true, file.name, formatBytes(file.size));
  try {
    await installModel(await file.arrayBuffer(), file.name);
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

async function loadUrl(url, name) {
  setLoading(true, name, "Reading geometry");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${name} (${response.status}).`);
  await installModel(await response.arrayBuffer(), name);
}

async function installModel(buffer, name) {
  $("#loading-detail").textContent = "Parsing and uploading geometry";
  await nextPaint();
  const parsed = parseGlb(buffer, name);
  renderer.setModel(parsed);
  appState.model = parsed;
  $("#model-name").textContent = appState.session ? `${appState.session.sourceName} · ${appState.session.mapName}` : name;
  $("#status-summary").textContent = `${formatNumber(parsed.triangleCount)} triangles · ${formatNumber(parsed.vertexCount)} vertices · ${formatBytes(parsed.sourceBytes)}`;
  dropZone.hidden = true;
  renderObjectList();
  canvas.focus();
}

function renderObjectList() {
  const list = $("#object-list");
  if (!appState.model) { list.className = "object-list empty"; list.textContent = "Load a scene to inspect its objects."; return; }
  list.className = "object-list";
  list.replaceChildren();
  for (const item of appState.model.drawables) {
    const row = document.createElement("div");
    row.className = `object-row${renderer.selectedId === item.id ? " selected" : ""}`;
    row.title = "Select; double-click to frame";
    const visibility = document.createElement("button");
    visibility.className = `visibility${item.visible ? "" : " off"}`;
    visibility.textContent = item.visible ? "◉" : "○";
    visibility.addEventListener("click", (event) => {
      event.stopPropagation(); item.visible = !item.visible; renderObjectList(); renderer.requestRender();
    });
    const name = document.createElement("span"); name.className = "name"; name.textContent = item.name;
    const count = document.createElement("span"); count.className = "count"; count.textContent = compactNumber(item.triangleCount);
    row.append(visibility, name, count);
    row.addEventListener("click", () => { renderer.setSelected(item.id); renderObjectList(); });
    row.addEventListener("dblclick", () => { renderer.setSelected(item.id); renderer.frameSelection(); });
    list.append(row);
  }
}

function selectedRound() {
  return appState.session.rounds.find((round) => round.number === Number($("#round-select").value)) || appState.session.rounds[0];
}

function positionArray(position) {
  if (Array.isArray(position)) return position.map(Number);
  return [Number(position.x), Number(position.y), Number(position.z)];
}

function groupBy(values, key) {
  const groups = new Map();
  for (const value of values) {
    const groupKey = key(value);
    if (!groups.has(groupKey)) groups.set(groupKey, []);
    groups.get(groupKey).push(value);
  }
  return groups;
}

function displayTeam(value) {
  return value === "ct" ? "Counter-Terrorists" : value === "t" ? "Terrorists" : value;
}

function option(value, label) {
  const element = document.createElement("option");
  element.value = String(value);
  element.textContent = label;
  return element;
}

function bindToggle(selector, callback, initial = false) {
  const button = $(selector);
  button.dataset.enabled = String(initial);
  button.addEventListener("click", () => {
    const enabled = button.dataset.enabled !== "true";
    button.dataset.enabled = String(enabled);
    button.classList.toggle("active", enabled);
    button.setAttribute("aria-pressed", String(enabled));
    callback(enabled);
  });
}

function clickToggle(selector) { $(selector).click(); }

async function apiJson(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (typeof options.body === "string" && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(url, { ...options, headers });
  let value;
  try { value = await response.json(); } catch { value = {}; }
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status}).`);
  return value;
}

function setLoading(visible, title = "Working…", detail = "", progress = null) {
  loading.hidden = !visible;
  $("#loading-title").textContent = title;
  $("#loading-detail").textContent = detail;
  const bar = $("#job-progress");
  bar.hidden = progress === null;
  if (progress !== null) bar.value = progress;
}

function showError(error) {
  console.error(error);
  toast.textContent = error instanceof Error ? error.message : String(error);
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 8000);
}

function download(url) {
  const link = document.createElement("a");
  link.href = url;
  link.click();
}

const formatNumber = (value) => new Intl.NumberFormat().format(value);
const compactNumber = (value) => new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
const formatTime = (seconds) => {
  const sign = seconds < 0 ? "−" : "";
  const absolute = Math.abs(seconds);
  return `${sign}${Math.floor(absolute / 60)}:${(absolute % 60).toFixed(1).padStart(4, "0")}`;
};
const formatBytes = (bytes) => {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
};
const nextPaint = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

configureTimeMode();
fetch("/viewer-config.json")
  .then((response) => response.json())
  .then(async (config) => {
    if (config.startupJobId) {
      setLoading(true, "Loading input", "Starting", 0);
      await waitForJob(config.startupJobId);
      await refreshApplicationState(true);
      setLoading(false);
    } else if (config.modelUrl) {
      await loadUrl(config.modelUrl, config.modelName || "model.glb");
      setLoading(false);
    } else {
      await refreshApplicationState(false);
    }
  })
  .catch((error) => { setLoading(false); showError(error); });
