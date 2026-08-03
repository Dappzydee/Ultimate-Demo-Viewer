import { parseGlb } from "./glb.js";
import { ViewerRenderer } from "./renderer.js";

const $ = (selector) => document.querySelector(selector);
const canvas = $("#canvas");
const viewport = $("#viewport");
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
  analysisType: "vision",
  selectedFlashIndex: null,
  selectedLineupIndex: null,
  manualFlashPosition: null,
  flashCamera: false,
  lineupCamera: false,
  resultHistory: null,
  flashMover: {
    enabled: false,
    moving: false,
    finalizing: false,
    keys: new Set(),
    fast: false,
    previewInFlight: false,
    pendingPosition: null,
    debounceTimer: null,
    animationFrame: null,
    lastFrameTime: 0,
    epoch: 0,
    revision: 0,
    lastAppliedRevision: 0,
    previewVisible: false,
  },
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
  download("/api/session/export.cs2session");
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

const analysisTypes = ["vision", "flash", "lineup"];
for (const button of analysisTypes.map((name) => $(`#${name}-tab`))) {
  button.addEventListener("click", () => setAnalysisType(button.dataset.analysis));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const direction = ["ArrowDown", "ArrowRight"].includes(event.key) ? 1 : -1;
    const current = analysisTypes.indexOf(button.dataset.analysis);
    const nextType = analysisTypes[(current + direction + analysisTypes.length) % analysisTypes.length];
    setAnalysisType(nextType);
    $(`#${nextType}-tab`).focus();
  });
}
$("#round-select").addEventListener("change", configureRound);
$("#time-mode").addEventListener("change", configureTimeMode);
$("#player-marker-toggle").addEventListener("change", () => {
  if (appState.replay?.type === "vision") applyReplayFrame();
  else renderer.setPlayerMarker(null);
});
bindTimeControl("#start-time", "#start-slider", true);
bindTimeControl("#end-time", "#end-slider", false);
$("#flash-source").addEventListener("change", () => {
  setMoveFlashMode(false);
  setFlashCameraMode(false);
  configureFlashSource();
});
$("#focus-flash-toggle").addEventListener("change", () => {
  if ($("#focus-flash-toggle").checked && appState.analysisType === "flash" && !appState.flashCamera) focusSelectedFlash();
});
$("#flash-camera-button").addEventListener("click", () => setFlashCameraMode(!appState.flashCamera));
$("#placed-flash-origin").addEventListener("change", (event) => copyDemoFlashPosition(Number(event.target.value)));
$("#copy-flash-position-button").addEventListener("click", () => {
  copyDemoFlashPosition(Number($("#placed-flash-origin").value));
});
$("#move-flash-button").addEventListener("click", () => setMoveFlashMode(!appState.flashMover.enabled));
$("#analyze-vision-button").addEventListener("click", () => analyzeCurrentSelection("vision"));
$("#analyze-flash-button").addEventListener("click", () => analyzeCurrentSelection("flash"));
for (const selector of ["#lineup-round-filter", "#lineup-grenade-filter", "#lineup-player-filter", "#lineup-reference-filter"]) {
  $(selector).addEventListener("change", populateLineups);
}
$("#focus-lineup-toggle").addEventListener("change", () => {
  if ($("#focus-lineup-toggle").checked) renderer.frameLineup();
});
$("#frame-lineup-button").addEventListener("click", () => renderer.frameLineup());
$("#lineup-camera-button").addEventListener("click", () => setLineupCameraMode(!appState.lineupCamera));
$("#copy-lineup-commands-button").addEventListener("click", copySelectedLineupCommands);
$("#export-lineups-json-button").addEventListener("click", () => exportFilteredLineups("json"));
$("#export-lineups-cfg-button").addEventListener("click", () => exportFilteredLineups("cfg"));

$("#frame-slider").addEventListener("input", (event) => {
  stopPlayback();
  appState.frame = Number(event.target.value);
  applyReplayFrame();
});
$("#replay-mode").addEventListener("change", applyReplayFrame);
$("#play-button").addEventListener("click", () => appState.playing ? stopPlayback() : startPlayback());

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && appState.lineupCamera) { setLineupCameraMode(false); return; }
  if (event.key === "Escape" && appState.flashMover.enabled) { setMoveFlashMode(false); return; }
  if (event.key === "Escape" && appState.flashCamera) { setFlashCameraMode(false); return; }
  const key = event.key.toLowerCase();
  if (appState.flashMover.enabled && key === "shift") {
    event.preventDefault();
    appState.flashMover.fast = true;
    return;
  }
  if (appState.flashMover.enabled && ["w", "a", "s", "d", "q", "e"].includes(key)) {
    event.preventDefault();
    if (!appState.flashMover.finalizing) beginFlashMovement(key, event.shiftKey);
    return;
  }
  if (event.target.matches("input, select, button")) return;
  if (key === "o" && event.ctrlKey) { event.preventDefault(); chooseGlb(); }
  if (key === "f") { event.preventDefault(); renderer.frameSelection(); }
  if (key === "w" && event.shiftKey) { event.preventDefault(); clickToggle("#wireframe-button"); }
  if (key === "g") clickToggle("#grid-button");
  if (key === "x") clickToggle("#axes-button");
  if (key === "5") $("#projection-button").click();
  if (key === " ") { event.preventDefault(); $("#play-button").click(); }
});

window.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  if (appState.flashMover.enabled && key === "shift") {
    appState.flashMover.fast = false;
    return;
  }
  if (!appState.flashMover.enabled || !["w", "a", "s", "d", "q", "e"].includes(key)) return;
  event.preventDefault();
  endFlashMovementKey(key);
});
window.addEventListener("blur", () => stopFlashMovement());

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
  setMoveFlashMode(false);
  setFlashCameraMode(false);
  setLineupCameraMode(false);
  clearActiveResult();
  appState.manualFlashPosition = null;
  appState.selectedLineupIndex = null;
  setLoading(true, title, `${file.name} · ${formatBytes(file.size)}`, 0);
  try {
    const job = await apiJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    await waitForJob(job.id);
    await refreshApplicationState(true, true);
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

async function refreshApplicationState(loadGeometry = false, loadLatestResult = false) {
  const hadSession = Boolean(appState.session);
  const state = await apiJson("/api/state");
  appState.session = state.session;
  appState.results = state.results || [];
  appState.resultHistory = state.resultHistory || null;
  if (!appState.session) return;
  if (loadGeometry || !hadSession) installSessionControls();
  if (loadGeometry) await loadUrl("/api/geometry.glb", appState.session.mapName);
  renderHistory();
  const savedResult = loadLatestResult
    ? [...appState.results].reverse().find((result) => !result.discarded)
    : null;
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
  configureLineupFilters();
  populateLineups();
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
  const list = $("#flash-event-list");
  list.replaceChildren();
  const flashes = appState.session.flashes || [];
  if (!flashes.some((flash) => flash.index === appState.selectedFlashIndex)) {
    appState.selectedFlashIndex = flashes[0]?.index ?? null;
  }
  const groups = groupBy(flashes, (flash) => flash.thrower_team || flash.thrower_side || "Unknown team");
  for (const [team, values] of groups) {
    const group = document.createElement("div");
    group.className = "flash-team";
    const heading = document.createElement("div");
    heading.className = "flash-team-heading";
    heading.textContent = displayTeam(team);
    group.append(heading);
    for (const flash of values) {
      const round = appState.session.rounds.find((item) => item.number === flash.round_number);
      const seconds = round ? (flash.tick - round.freezeEndTick) / appState.session.tickRate : 0;
      const button = document.createElement("button");
      const selected = flash.index === appState.selectedFlashIndex;
      button.className = `flash-event${selected ? " selected" : ""}`;
      button.type = "button";
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", String(selected));
      button.dataset.flashIndex = String(flash.index);

      const player = document.createElement("span");
      player.className = "flash-event-player";
      player.textContent = flash.thrower || "Unknown player";
      const time = document.createElement("span");
      time.className = "flash-event-time";
      time.textContent = `R${flash.round_number ?? "?"} · ${formatTime(seconds)}`;
      const position = document.createElement("span");
      position.className = "flash-event-position";
      position.textContent = positionArray(flash.position).map((value) => Math.round(value)).join(", ");
      button.append(player, time, position);
      button.addEventListener("click", () => selectFlash(flash.index));
      group.append(button);
    }
    list.append(group);
  }
  if (!flashes.length) {
    const empty = document.createElement("div");
    empty.className = "flash-event-empty";
    empty.textContent = "No flash detonations were found in this demo.";
    list.append(empty);
  }
  populatePlacedFlashOrigins(flashes);
  previewSelectedFlash();
}

function populatePlacedFlashOrigins(flashes) {
  const select = $("#placed-flash-origin");
  select.replaceChildren();
  const groups = groupBy(flashes, (flash) => flash.thrower_team || flash.thrower_side || "Unknown team");
  for (const [team, values] of groups) {
    const group = document.createElement("optgroup");
    group.label = displayTeam(team);
    for (const flash of values) {
      const round = appState.session.rounds.find((item) => item.number === flash.round_number);
      const seconds = round ? (flash.tick - round.freezeEndTick) / appState.session.tickRate : 0;
      group.append(option(flash.index, `R${flash.round_number ?? "?"} · ${formatTime(seconds)} · ${flash.thrower || "Unknown player"}`));
    }
    select.append(group);
  }
  if (!flashes.length) {
    const empty = option("", "No recorded flash pops");
    empty.disabled = true;
    select.append(empty);
  } else {
    select.value = String(appState.selectedFlashIndex ?? flashes[0].index);
  }
  select.disabled = !flashes.length;
  $("#copy-flash-position-button").disabled = !flashes.length;
}

function copyDemoFlashPosition(index) {
  const flash = appState.session?.flashes.find((item) => item.index === index);
  if (!flash) return;
  stopFlashMovement();
  const mover = appState.flashMover;
  mover.revision++;
  mover.lastAppliedRevision = mover.revision;
  mover.pendingPosition = null;
  clearTimeout(mover.debounceTimer);
  mover.debounceTimer = null;
  if (mover.previewVisible) restoreActiveVisualization();
  mover.previewVisible = false;
  setManualPosition(positionArray(flash.position));
  renderer.focusPoint(appState.manualFlashPosition);
  setMoveFlashStatus(`Copied ${flash.thrower || "demo flash"} at R${flash.round_number ?? "?"}. Move it or click Analyze flash.`);
  canvas.focus();
}

function selectFlash(index) {
  appState.selectedFlashIndex = index;
  for (const button of document.querySelectorAll(".flash-event")) {
    const selected = Number(button.dataset.flashIndex) === index;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  }
  previewSelectedFlash();
  if (appState.flashCamera) setFlashCameraMode(true);
  else if ($("#focus-flash-toggle").checked) focusSelectedFlash();
}

function configureLineupFilters() {
  const lineups = appState.session?.lineups || [];
  const replace = (selector, values, label) => {
    const select = $(selector);
    const previous = select.value || "all";
    select.replaceChildren(option("all", label), ...values.map(([value, text]) => option(value, text)));
    select.value = [...select.options].some((item) => item.value === previous) ? previous : "all";
  };
  const rounds = [...new Set(lineups.map((lineup) => lineup.round).filter((value) => value !== null))]
    .sort((left, right) => left - right).map((value) => [value, `Round ${value}`]);
  const grenades = [...new Set(lineups.map((lineup) => lineup.grenade_type))]
    .sort().map((value) => [value, displayGrenade(value)]);
  const players = [...new Map(lineups.map((lineup) => [lineup.thrower_steamid, lineup.thrower || lineup.thrower_steamid])).entries()]
    .sort((left, right) => left[1].localeCompare(right[1]));
  replace("#lineup-round-filter", rounds, "All rounds");
  replace("#lineup-grenade-filter", grenades, "All grenades");
  replace("#lineup-player-filter", players, "All players");
}

function filteredLineups() {
  const lineups = appState.session?.lineups || [];
  const round = $("#lineup-round-filter").value;
  const grenade = $("#lineup-grenade-filter").value;
  const player = $("#lineup-player-filter").value;
  const reference = $("#lineup-reference-filter").value;
  return lineups.map((lineup, index) => ({ lineup, index })).filter(({ lineup }) => (
    (round === "all" || lineup.round === Number(round))
    && (grenade === "all" || lineup.grenade_type === grenade)
    && (player === "all" || lineup.thrower_steamid === player)
    && (reference === "all" || (reference === "fixed") === Boolean(lineup.has_fixed_reference))
  ));
}

function populateLineups() {
  const list = $("#lineup-list");
  const notice = $("#lineup-availability");
  list.replaceChildren();
  const available = appState.session?.lineupsAvailable !== false;
  notice.hidden = available;
  notice.textContent = available ? "" : "This older saved session has no lineup data. Reopen its original demo to detect lineups.";
  const values = available ? filteredLineups() : [];
  $("#export-lineups-json-button").disabled = !values.length;
  $("#export-lineups-cfg-button").disabled = !values.length;
  if (!values.some((item) => item.index === appState.selectedLineupIndex)) {
    appState.selectedLineupIndex = values[0]?.index ?? null;
  }
  const groups = groupBy(values, (item) => item.lineup.round ?? "?");
  for (const [round, groupValues] of groups) {
    const group = document.createElement("div");
    group.className = "lineup-group";
    const heading = document.createElement("div");
    heading.className = "lineup-group-heading";
    heading.textContent = `Round ${round}`;
    group.append(heading);
    for (const { lineup, index } of groupValues) {
      const button = document.createElement("button");
      const selected = index === appState.selectedLineupIndex;
      button.type = "button";
      button.className = `lineup-row${selected ? " selected" : ""}`;
      button.dataset.lineupIndex = String(index);
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", String(selected));
      const title = document.createElement("span");
      title.className = "lineup-row-title";
      title.textContent = `${displayGrenade(lineup.grenade_type)} · ${lineup.thrower || "Unknown"}`;
      const time = document.createElement("span");
      time.className = "lineup-row-time";
      const roundInfo = appState.session.rounds.find((item) => item.number === lineup.round);
      const seconds = roundInfo ? (lineup.T_release - roundInfo.freezeEndTick) / appState.session.tickRate : 0;
      time.textContent = formatTime(seconds);
      const meta = document.createElement("span");
      meta.className = "lineup-row-meta";
      meta.textContent = `${lineup.throw_type.label} · ${lineup.has_fixed_reference ? "fixed reference" : "in motion"}`;
      button.append(title, time, meta);
      button.addEventListener("click", () => selectLineup(index));
      group.append(button);
    }
    list.append(group);
  }
  if (!values.length) {
    const empty = document.createElement("div");
    empty.className = "flash-event-empty";
    empty.textContent = available ? "No grenade lineups match these filters." : "No lineup data in this session.";
    list.append(empty);
    $("#lineup-detail").hidden = true;
    if (appState.analysisType === "lineup") {
      setLineupCameraMode(false);
      renderer.setLineupVisualization(null);
    }
    return;
  }
  selectLineup(
    appState.selectedLineupIndex,
    appState.analysisType === "lineup" && $("#focus-lineup-toggle").checked,
  );
}

function selectedLineup() {
  return appState.selectedLineupIndex === null ? null : appState.session?.lineups?.[appState.selectedLineupIndex] || null;
}

function selectLineup(index, shouldFrame = $("#focus-lineup-toggle").checked) {
  const lineup = appState.session?.lineups?.[index];
  if (!lineup) return;
  appState.selectedLineupIndex = index;
  for (const button of document.querySelectorAll(".lineup-row")) {
    const selected = Number(button.dataset.lineupIndex) === index;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  }
  renderLineupDetails(lineup);
  if (appState.analysisType !== "lineup") return;
  showLineupVisualization(lineup);
  if (appState.lineupCamera) setLineupCameraMode(true);
  else if (shouldFrame) renderer.frameLineup();
}

function renderLineupDetails(lineup) {
  $("#lineup-detail").hidden = false;
  $("#lineup-detail-title").textContent = `${displayGrenade(lineup.grenade_type)} · ${lineup.thrower || "Unknown player"}`;
  $("#lineup-instruction").textContent = lineup.movement_instruction;
  $("#lineup-setpos").textContent = lineup.setpos_command;
  $("#lineup-setang").textContent = lineup.setang_command;
  const speed = Math.hypot(Number(lineup.release_velocity.X), Number(lineup.release_velocity.Y));
  const rows = [
    ["Throw", lineup.throw_type.label],
    ["Release", `Tick ${lineup.T_release} · ${speed.toFixed(1)} u/s`],
    ["Reference", lineup.has_fixed_reference ? `Fixed · tick ${lineup.reference_tick}` : "In-motion fallback"],
    ["Notes", (lineup.notes || []).join("; ") || "No warnings"],
  ];
  $("#lineup-details").replaceChildren(...rows.map(([key, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt"); term.textContent = key;
    const definition = document.createElement("dd"); definition.textContent = value;
    row.append(term, definition);
    return row;
  }));
}

function lineupPosePosition(pose) {
  return [Number(pose.X), Number(pose.Y), Number(pose.Z)];
}

function lineupViewPose(lineup) {
  const pose = lineup.reference_point || lineup.release;
  const duck = Math.max(0, Math.min(1, Number(pose.duck_amount || 0)));
  const position = lineupPosePosition(pose);
  position[2] += 64 + (46 - 64) * duck;
  return { position, yaw: Number(pose.yaw), pitch: Number(pose.pitch) };
}

function showLineupVisualization(lineup) {
  const view = lineupViewPose(lineup);
  renderer.setLineupVisualization({
    reference: lineup.reference_point ? lineupPosePosition(lineup.reference_point) : null,
    release: lineupPosePosition(lineup.release),
    path: (lineup.movement_path || []).map(lineupPosePosition),
    aim: { origin: view.position, yaw: view.yaw, pitch: view.pitch },
  });
}

function setLineupCameraMode(enabled) {
  const lineup = selectedLineup();
  if (enabled && (!lineup || appState.analysisType !== "lineup")) return;
  appState.lineupCamera = Boolean(enabled);
  if (enabled && $("#projection-button").textContent !== "Perspective") $("#projection-button").click();
  renderer.setLineupCamera(enabled ? lineupViewPose(lineup) : null);
  viewport.classList.toggle("lineup-camera-active", appState.lineupCamera);
  const button = $("#lineup-camera-button");
  button.classList.toggle("active", appState.lineupCamera);
  button.setAttribute("aria-pressed", String(appState.lineupCamera));
  button.textContent = appState.lineupCamera ? "Exit aim view" : "View aim";
  $("#projection-button").disabled = appState.lineupCamera || appState.flashCamera;
  if (appState.lineupCamera) canvas.focus();
}

async function copySelectedLineupCommands() {
  const lineup = selectedLineup();
  if (!lineup) return;
  const text = `${lineup.setpos_command}; ${lineup.setang_command}`;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  const button = $("#copy-lineup-commands-button");
  button.textContent = "Copied";
  setTimeout(() => { button.textContent = "Copy commands"; }, 1200);
}

function exportFilteredLineups(format) {
  const values = filteredLineups().map((item) => item.lineup);
  if (!values.length) return;
  const source = appState.session?.sourceName?.replace(/\.[^.]+$/, "") || "demo";
  if (format === "json") {
    downloadBlob(`${source}-lineups.json`, JSON.stringify(values, null, 2), "application/json");
    return;
  }
  const blocks = values.map((lineup, index) => [
    `// Lineup ${index + 1}: ${displayGrenade(lineup.grenade_type)} | ${lineup.throw_type.label} | round ${lineup.round ?? "?"}`,
    `// ${lineup.movement_instruction}`,
    lineup.setpos_command,
    lineup.setang_command,
  ].join("\n"));
  downloadBlob(`${source}-lineups.cfg`, `${blocks.join("\n\n")}\n`, "text/plain");
}

function displayGrenade(value) {
  return {
    flashbang: "Flashbang", smokegrenade: "Smoke", hegrenade: "HE grenade",
    molotov: "Molotov", incendiary: "Incendiary", decoy: "Decoy",
    tagrenade: "Tactical grenade", snowball: "Snowball",
  }[value] || value;
}

function setAnalysisType(type) {
  const previousType = appState.analysisType;
  if (type !== "flash") setMoveFlashMode(false);
  if (type !== "lineup") setLineupCameraMode(false);
  appState.analysisType = type;
  if (previousType === "lineup" && type !== "lineup") restoreActiveVisualization();
  for (const name of analysisTypes) {
    const button = $(`#${name}-tab`);
    const active = name === type;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }
  $("#vision-controls").hidden = type !== "vision";
  $("#flash-controls").hidden = type !== "flash";
  $("#lineup-controls").hidden = type !== "lineup";
  if (type === "flash") {
    renderer.setLineupVisualization(null);
    renderer.setPlayerMarker(null);
    configureFlashSource();
  } else if (type === "lineup") {
    setFlashCameraMode(false);
    renderer.setMarker(null);
    renderer.setPlayerMarker(null);
    renderer.clearFaceValues();
    populateLineups();
  } else {
    setFlashCameraMode(false);
    renderer.setMarker(null);
    renderer.setLineupVisualization(null);
    if (appState.replay?.type === "vision") applyReplayFrame();
  }
}

function configureTimeMode() {
  const instant = $("#time-mode").value === "instant";
  $("#end-time-field").hidden = instant;
  $("#end-slider").hidden = instant;
  $("#time-mode-help").textContent = instant
    ? "Analyze one snapshot at the nearest demo tick."
    : "Analyze start to end, then replay current or accumulated visibility.";
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
  if (manual) {
    ensureManualFlashPosition();
    previewManualFlash();
  } else {
    setMoveFlashMode(false);
    previewSelectedFlash();
  }
}

function previewSelectedFlash() {
  if (!appState.session || appState.analysisType !== "flash" || $("#flash-source").value !== "event") return;
  const flash = appState.session.flashes.find((item) => item.index === appState.selectedFlashIndex);
  const position = flash ? positionArray(flash.position) : null;
  renderer.setMarker(position);
  if (position) renderer.setSelected("analysis-marker");
}

function previewManualFlash() {
  if (appState.analysisType !== "flash" || $("#flash-source").value !== "manual") return;
  const position = manualPosition(false);
  renderer.setMarker(position);
  if (position) {
    renderer.setSelected("analysis-marker");
    if (appState.flashCamera) renderer.setFlashCamera(position);
  }
}

function currentFlashPosition() {
  if (!appState.session || appState.analysisType !== "flash") return null;
  if ($("#flash-source").value === "manual") return manualPosition(false);
  const flash = appState.session.flashes.find((item) => item.index === appState.selectedFlashIndex);
  return flash ? positionArray(flash.position) : null;
}

function focusSelectedFlash() {
  const position = currentFlashPosition();
  if (!position) return;
  renderer.setSelected("analysis-marker");
  renderer.focusPoint(position);
}

function setFlashCameraMode(enabled) {
  if (enabled) {
    setMoveFlashMode(false);
    const position = currentFlashPosition();
    if (!position) {
      showError(new Error("Select a recorded flash or create a placed flash before entering flash camera mode."));
      return;
    }
    if ($("#projection-button").textContent !== "Perspective") $("#projection-button").click();
    appState.flashCamera = true;
    renderer.setSelected("analysis-marker");
    renderer.setFlashCamera(position);
    viewport.classList.add("flash-camera-active");
    $("#projection-button").disabled = appState.flashCamera || appState.lineupCamera;
    $("#flash-camera-button").classList.add("active");
    $("#flash-camera-button").setAttribute("aria-pressed", "true");
    $("#flash-camera-button").textContent = "Exit flash camera";
    canvas.focus();
    return;
  }
  appState.flashCamera = false;
  renderer.setFlashCamera(null);
  viewport.classList.remove("flash-camera-active");
  $("#projection-button").disabled = appState.flashCamera || appState.lineupCamera;
  $("#flash-camera-button").classList.remove("active");
  $("#flash-camera-button").setAttribute("aria-pressed", "false");
  $("#flash-camera-button").textContent = "Enter flash camera";
}

function setMoveFlashMode(enabled) {
  const mover = appState.flashMover;
  if (enabled && (!appState.session || appState.analysisType !== "flash" || $("#flash-source").value !== "manual")) {
    showError(new Error("Move flash requires a loaded demo and Placed flash as the source."));
    return;
  }
  if (enabled === mover.enabled) return;

  mover.epoch++;
  mover.enabled = enabled;
  mover.moving = false;
  mover.finalizing = false;
  mover.keys.clear();
  mover.fast = false;
  mover.lastAppliedRevision = 0;
  mover.pendingPosition = null;
  clearTimeout(mover.debounceTimer);
  mover.debounceTimer = null;
  if (mover.animationFrame !== null) cancelAnimationFrame(mover.animationFrame);
  mover.animationFrame = null;
  const button = $("#move-flash-button");
  button.classList.toggle("active", enabled);
  button.setAttribute("aria-pressed", String(enabled));
  button.textContent = enabled ? "Stop moving flash" : "Move flash";
  viewport.classList.toggle("move-flash-active", enabled);
  renderer.setInteractionLocked(enabled);

  if (enabled) {
    setFlashCameraMode(false);
    ensureManualFlashPosition();
    renderer.focusPoint(appState.manualFlashPosition);
    setMoveFlashStatus("Use WASD to move, Q/E for height, and Shift for faster movement.");
    canvas.focus();
  } else {
    if (mover.previewVisible) restoreActiveVisualization();
    mover.previewVisible = false;
    setMoveFlashStatus("Enable movement, then use WASD to move and Q/E to change height. Hold Shift to move faster.");
  }
}

function setMoveFlashStatus(message) {
  $("#move-flash-status").textContent = message;
}

function restoreActiveVisualization() {
  if (appState.replay?.type === "vision") applyReplayFrame();
  else if (appState.replay?.type === "flash") renderer.setFaceValues(appState.replay.values, "flash");
  else renderer.clearFaceValues();
}

function beginFlashMovement(key, fast) {
  const mover = appState.flashMover;
  mover.keys.add(key);
  mover.fast = Boolean(fast);
  if (mover.animationFrame === null) {
    mover.lastFrameTime = 0;
    mover.animationFrame = requestAnimationFrame(moveFlashFrame);
  }
}

function endFlashMovementKey(key) {
  const mover = appState.flashMover;
  mover.keys.delete(key);
  if (!mover.keys.size) stopFlashMovement();
}

function stopFlashMovement() {
  const mover = appState.flashMover;
  mover.keys.clear();
  mover.fast = false;
  if (mover.animationFrame !== null) cancelAnimationFrame(mover.animationFrame);
  mover.animationFrame = null;
  if (!mover.enabled || !mover.moving || mover.finalizing) return;
  mover.moving = false;
  setMoveFlashStatus("Position ready. Click Analyze flash when you want the full-quality result.");
}

function moveFlashFrame(time) {
  const mover = appState.flashMover;
  mover.animationFrame = null;
  if (!mover.enabled || mover.finalizing || !mover.keys.size) {
    if (!mover.keys.size) stopFlashMovement();
    return;
  }
  const deltaSeconds = mover.lastFrameTime ? Math.min((time - mover.lastFrameTime) / 1000, 0.05) : 0;
  mover.lastFrameTime = time;
  const { forward, right } = renderer.getMovementBasis();
  let direction = [0, 0, 0];
  if (mover.keys.has("w")) direction = direction.map((value, index) => value + forward[index]);
  if (mover.keys.has("s")) direction = direction.map((value, index) => value - forward[index]);
  if (mover.keys.has("d")) direction = direction.map((value, index) => value + right[index]);
  if (mover.keys.has("a")) direction = direction.map((value, index) => value - right[index]);
  if (mover.keys.has("e")) direction[2] += 1;
  if (mover.keys.has("q")) direction[2] -= 1;
  const length = Math.hypot(...direction);
  if (length && deltaSeconds) {
    const speed = 300 * (mover.fast ? 3 : 1);
    const offset = direction.map((value) => value / length * speed * deltaSeconds);
    const position = manualPosition().map((value, index) => value + offset[index]);
    setManualPosition(position);
    renderer.translateCamera(offset);
    mover.moving = true;
    mover.revision++;
    queueFlashMovePreview(position, mover.epoch, mover.revision);
  }
  mover.animationFrame = requestAnimationFrame(moveFlashFrame);
}

function queueFlashMovePreview(position, epoch, revision) {
  const mover = appState.flashMover;
  mover.pendingPosition = { position: [...position], epoch, revision };
  if (mover.previewInFlight || mover.debounceTimer !== null) return;
  mover.debounceTimer = setTimeout(() => runFlashMovePreview(), 180);
  setMoveFlashStatus("Moving flash; preparing a quick coverage preview…");
}

async function runFlashMovePreview() {
  const mover = appState.flashMover;
  mover.debounceTimer = null;
  if (!mover.enabled || mover.finalizing || mover.previewInFlight || !mover.pendingPosition) return;
  const pending = mover.pendingPosition;
  mover.pendingPosition = null;
  if (pending.epoch !== mover.epoch) return;
  mover.previewInFlight = true;
  setMoveFlashStatus("Calculating quick preview (1 sample per triangle)…");
  try {
    const job = await apiJson("/api/preview/flash", {
      method: "POST",
      body: JSON.stringify(flashAnalysisRequest(pending.position, 1)),
    });
    const completed = await waitForJobQuiet(job.id);
    const response = await fetch(`/api/results/${completed.resultId}/data.bin`);
    if (!response.ok) throw new Error("Could not load the live flash preview.");
    const buffer = await response.arrayBuffer();
    if (
      mover.enabled && !mover.finalizing &&
      pending.epoch === mover.epoch && pending.revision > mover.lastAppliedRevision
    ) {
      mover.lastAppliedRevision = pending.revision;
      installFlashMovePreview(buffer, manualPosition());
    }
  } catch (error) {
    if (pending.epoch === mover.epoch && mover.enabled) showError(error);
  } finally {
    mover.previewInFlight = false;
    if (mover.enabled && !mover.finalizing && mover.pendingPosition) {
      mover.debounceTimer = setTimeout(() => runFlashMovePreview(), 180);
    }
  }
}

function installFlashMovePreview(buffer, position) {
  const header = parseHeader(buffer, "CSF1");
  if (buffer.byteLength !== 24 + header.faceCount) throw new Error("Flash preview is truncated.");
  renderer.setFaceValues(new Uint8Array(buffer, 24, header.faceCount), "flash");
  renderer.setMarker(position);
  appState.flashMover.previewVisible = true;
  setMoveFlashStatus(appState.flashMover.moving
    ? "Quick preview shown. Keep moving, or release the keys to hold this position."
    : "Quick preview ready. Click Analyze flash for the full-quality result.");
  $("#status-summary").textContent = `Moving flash preview · ${formatNumber(header.faceCount)} faces · 1 sample`;
}

function flashAnalysisRequest(position, samplesPerTriangle = Number($("#flash-samples").value)) {
  return {
    position,
    maxDistance: Number($("#flash-distance").value),
    falloffPower: Number($("#flash-falloff").value),
    samplesPerTriangle,
  };
}

function setManualPosition(position) {
  appState.manualFlashPosition = position.map(Number);
  renderer.setMarker(appState.manualFlashPosition);
  renderer.setSelected("analysis-marker");
}

function manualPosition(required = true) {
  const position = appState.manualFlashPosition;
  if (!position && required) throw new Error("Create the placed flash before analyzing it.");
  return position ? [...position] : null;
}

function ensureManualFlashPosition() {
  if (appState.manualFlashPosition) return appState.manualFlashPosition;
  const selected = appState.session?.flashes.find((item) => item.index === appState.selectedFlashIndex)
    || appState.session?.flashes[0];
  setManualPosition(selected ? positionArray(selected.position) : renderer.getSceneCenter());
  return appState.manualFlashPosition;
}

async function prepareFlashMoverForAnalysis() {
  const mover = appState.flashMover;
  stopFlashMovement();
  mover.finalizing = true;
  mover.revision++;
  mover.lastAppliedRevision = mover.revision;
  mover.pendingPosition = null;
  clearTimeout(mover.debounceTimer);
  mover.debounceTimer = null;
  if (mover.previewInFlight) {
    setLoading(true, "Analyzing flash coverage", "Finishing the quick preview", 0);
    while (mover.previewInFlight) await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

async function analyzeCurrentSelection(type) {
  if (!appState.session) return;
  const isVision = type === "vision";
  let completedFlashAnalysis = false;
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
      if (!manual && appState.selectedFlashIndex === null) throw new Error("Select a recorded flash to analyze.");
      request = manual
        ? flashAnalysisRequest(manualPosition())
        : {
          eventIndex: appState.selectedFlashIndex,
          maxDistance: Number($("#flash-distance").value),
          falloffPower: Number($("#flash-falloff").value),
          samplesPerTriangle: Number($("#flash-samples").value),
        };
    }
    if (!isVision) await prepareFlashMoverForAnalysis();
    setLoading(true, isVision ? "Analyzing player vision" : "Analyzing flash coverage", "Preparing rays", 0);
    const job = await apiJson(endpoint, { method: "POST", body: JSON.stringify(request) });
    const completed = await waitForJob(job.id);
    await refreshApplicationState(false);
    await loadResult(completed.resultId);
    completedFlashAnalysis = !isVision;
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
    if (!isVision) {
      appState.flashMover.finalizing = false;
      if (appState.flashMover.enabled) {
        setMoveFlashStatus(completedFlashAnalysis
          ? "Full-quality result ready. Move the flash again or analyze another position."
          : "Position ready. Click Analyze flash when you want the full-quality result.");
      }
    }
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

async function waitForJobQuiet(jobId) {
  while (true) {
    const job = await apiJson(`/api/jobs/${jobId}`);
    if (job.status === "complete") return job;
    if (job.status === "error") throw new Error(job.error || "The preview job failed.");
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
}

async function loadResult(resultId) {
  const metadata = appState.results.find((item) => item.id === resultId);
  if (!metadata) throw new Error("The completed result was not found.");
  const response = await fetch(`/api/results/${resultId}/data.bin`);
  if (!response.ok) throw new Error("Could not load the analysis result.");
  const buffer = await response.arrayBuffer();
  appState.activeResult = metadata;
  setAnalysisType(metadata.analysisType);
  $("#export-button").disabled = false;
  $("#result-section").hidden = false;
  renderResultDetails(metadata);
  if (metadata.analysisType === "vision") installVisionReplay(buffer, metadata);
  else installFlashResult(buffer, metadata);
  renderHistory();
}

function installVisionReplay(buffer, metadata) {
  const header = parseHeader(buffer, "CSV1");
  const ticksOffset = 24;
  const ticksBytes = header.frameCount * 4;
  const posesOffset = ticksOffset + ticksBytes;
  const posesBytes = header.frameCount * header.poseWidth * 4;
  const masksOffset = posesOffset + posesBytes;
  const masksBytes = header.frameCount * header.width;
  if (buffer.byteLength !== masksOffset + masksBytes * 2) throw new Error("Vision result is truncated.");
  appState.replay = {
    type: "vision", faceCount: header.faceCount, frameCount: header.frameCount, width: header.width,
    ticks: new Uint32Array(buffer, ticksOffset, header.frameCount),
    poseWidth: header.poseWidth,
    poses: header.poseWidth ? new Float32Array(buffer, posesOffset, header.frameCount * header.poseWidth) : null,
    instant: new Uint8Array(buffer, masksOffset, masksBytes),
    cumulative: new Uint8Array(buffer, masksOffset + masksBytes, masksBytes),
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
  renderer.setPlayerMarker(null);
  renderer.setFaceValues(appState.replay.values, "flash");
  renderer.setMarker(positionArray(metadata.flash.position));
  appState.flashMover.previewVisible = false;
  $("#status-summary").textContent = `${formatNumber(header.faceCount)} faces · ${metadata.backend} · ${formatNumber(metadata.testedRays)} rays`;
}

function parseHeader(buffer, expectedMagic) {
  if (buffer.byteLength < 24) throw new Error("Analysis result is truncated.");
  const bytes = new Uint8Array(buffer, 0, 4);
  const magic = String.fromCharCode(...bytes);
  const view = new DataView(buffer);
  const version = view.getUint32(4, true);
  const validVersion = expectedMagic === "CSV1" ? version === 1 || version === 2 : version === 1;
  if (magic !== expectedMagic || !validVersion) throw new Error("Unsupported analysis result format.");
  const poseWidth = version >= 2 ? view.getUint32(20, true) : 0;
  if (expectedMagic === "CSV1" && version >= 2 && poseWidth !== 5) throw new Error("Unsupported vision pose data.");
  return {
    version, faceCount: view.getUint32(8, true), frameCount: view.getUint32(12, true),
    width: view.getUint32(16, true), poseWidth,
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
  if (replay.poses && $("#player-marker-toggle").checked && appState.analysisType === "vision") {
    const poseOffset = appState.frame * replay.poseWidth;
    renderer.setPlayerMarker(
      [replay.poses[poseOffset], replay.poses[poseOffset + 1], replay.poses[poseOffset + 2]],
      replay.poses[poseOffset + 3], true,
    );
  } else {
    renderer.setPlayerMarker(null);
  }
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

function renderHistory() {
  const section = $("#history-section");
  if (!appState.session) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  const history = appState.resultHistory;
  $("#history-usage").textContent = history
    ? `${formatBytes(history.totalBytes)} / ${formatBytes(history.maxBytes)}`
    : formatBytes(appState.results.reduce((total, result) => total + (result.sizeBytes || 0), 0));
  const warning = $("#history-warning");
  warning.hidden = !history?.warning;
  warning.textContent = history?.warning || "";

  const activeResults = appState.results.filter((result) => !result.discarded);
  const discardedResults = appState.results.filter((result) => result.discarded);
  renderHistoryRows($("#history-list"), activeResults, false);
  renderHistoryRows($("#discarded-history-list"), discardedResults, true);
  $("#discarded-history-count").textContent = String(discardedResults.length);
}

function renderHistoryRows(list, results, discarded) {
  list.replaceChildren();
  if (!results.length) {
    const empty = document.createElement("div");
    empty.className = "empty-copy";
    empty.textContent = discarded ? "Nothing discarded." : "Completed analyses will appear here.";
    list.append(empty);
    return;
  }
  for (const result of [...results].reverse()) {
    const row = document.createElement("div");
    row.className = `history-row${appState.activeResult?.id === result.id ? " active" : ""}`;
    const main = document.createElement("div");
    main.className = "history-main";

    const copy = document.createElement("div");
    copy.className = "history-copy";
    const title = document.createElement("div");
    title.className = "history-title";
    title.textContent = historyResultTitle(result);
    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = `${historyResultMeta(result)} · ${formatBytes(result.sizeBytes)}`;
    copy.append(title, meta);
    main.append(copy);
    if (!discarded) {
      const pin = document.createElement("button");
      pin.className = `history-pin${result.pinned ? " active" : ""}`;
      pin.textContent = result.pinned ? "◆" : "◇";
      pin.title = result.pinned ? "Unpin result" : "Pin result";
      pin.setAttribute("aria-label", pin.title);
      pin.setAttribute("aria-pressed", String(result.pinned));
      pin.addEventListener("click", () => setResultPinned(result.id, !result.pinned));
      main.append(pin);
    }

    const actions = document.createElement("div");
    actions.className = "history-actions";
    const open = document.createElement("button");
    open.textContent = appState.activeResult?.id === result.id ? "Current" : "Open";
    open.disabled = appState.activeResult?.id === result.id;
    open.addEventListener("click", () => openHistoryResult(result.id));
    const exportButton = document.createElement("button");
    exportButton.textContent = "Export GLB";
    exportButton.addEventListener("click", () => exportHistoryResult(result));
    if (discarded) {
      const restore = document.createElement("button");
      restore.className = "history-restore";
      restore.textContent = "Restore";
      restore.addEventListener("click", () => setResultDiscarded(result.id, false));
      const remove = document.createElement("button");
      remove.className = "history-delete";
      remove.textContent = "Delete";
      remove.title = "Permanently delete result";
      remove.addEventListener("click", () => deleteHistoryResult(result.id, true));
      actions.append(open, exportButton, restore, remove);
    } else {
      const rename = document.createElement("button");
      rename.textContent = "Rename";
      rename.addEventListener("click", () => renameHistoryResult(result));
      const discard = document.createElement("button");
      discard.className = "history-discard";
      discard.textContent = "Discard";
      discard.addEventListener("click", () => setResultDiscarded(result.id, true));
      actions.append(open, exportButton, rename, discard);
    }
    row.append(main, actions);
    list.append(row);
  }
}

function historyResultTitle(result) {
  if (result.name) return result.name;
  if (result.analysisType === "vision") return `Vision · ${result.playerName || "Unknown player"}`;
  return `Flash · ${result.flash?.thrower || (result.source === "manual" ? "Manual placement" : "Unknown")}`;
}

function historyResultMeta(result) {
  if (result.analysisType === "vision") {
    const range = result.timeMode === "instant"
      ? formatTime(result.startSeconds)
      : `${formatTime(result.startSeconds)}–${formatTime(result.endSeconds)}`;
    return `R${result.roundNumber} · ${range} · ${formatNumber(result.frameCount)} frame${result.frameCount === 1 ? "" : "s"}`;
  }
  return result.flash?.round_number ? `R${result.flash.round_number} · ${result.source}` : result.source;
}

async function openHistoryResult(resultId) {
  setLoading(true, "Opening analysis", "Reading compact result");
  try {
    await loadResult(resultId);
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

function exportHistoryResult(result) {
  const parameters = new URLSearchParams();
  if (result.analysisType === "vision" && appState.activeResult?.id === result.id) {
    parameters.set("frame", String(appState.frame));
    parameters.set("mode", $("#replay-mode").value);
  }
  const query = parameters.size ? `?${parameters}` : "";
  download(`/api/results/${result.id}/export.glb${query}`);
}

async function setResultPinned(resultId, pinned) {
  try {
    await apiJson(`/api/results/${resultId}/pin`, {
      method: "POST", body: JSON.stringify({ pinned }),
    });
    await refreshApplicationState(false);
  } catch (error) {
    showError(error);
  }
}

async function setResultDiscarded(resultId, discarded) {
  try {
    await apiJson(`/api/results/${resultId}/discard`, {
      method: "POST", body: JSON.stringify({ discarded }),
    });
    await refreshApplicationState(false);
    if (appState.activeResult?.id === resultId) {
      appState.activeResult = appState.results.find((item) => item.id === resultId) || appState.activeResult;
      renderResultDetails(appState.activeResult);
      renderHistory();
    }
  } catch (error) {
    showError(error);
  }
}

async function deleteHistoryResult(resultId, permanent = false) {
  if (permanent && !window.confirm("Permanently delete this discarded analysis? This cannot be undone.")) return;
  try {
    await apiJson(`/api/results/${resultId}/delete`, {
      method: "POST", body: JSON.stringify({}),
    });
    const wasActive = appState.activeResult?.id === resultId;
    if (wasActive) clearActiveResult();
    await refreshApplicationState(false);
    const latestActive = [...appState.results].reverse().find((result) => !result.discarded);
    if (wasActive && latestActive) await loadResult(latestActive.id);
  } catch (error) {
    showError(error);
  }
}

async function renameHistoryResult(result) {
  const name = window.prompt("Result name", result.name || historyResultTitle(result));
  if (name === null || !name.trim()) return;
  try {
    await apiJson(`/api/results/${result.id}/rename`, {
      method: "POST", body: JSON.stringify({ name: name.trim() }),
    });
    await refreshApplicationState(false);
    if (appState.activeResult?.id === result.id) {
      appState.activeResult = appState.results.find((item) => item.id === result.id) || appState.activeResult;
      renderResultDetails(appState.activeResult);
    }
  } catch (error) {
    showError(error);
  }
}

function clearActiveResult() {
  stopPlayback();
  setLineupCameraMode(false);
  appState.activeResult = null;
  appState.replay = null;
  $("#result-section").hidden = true;
  $("#timeline").hidden = true;
  $("#export-button").disabled = true;
  renderer.clearFaceValues();
  renderer.setMarker(null);
  renderer.setPlayerMarker(null);
  renderer.setLineupVisualization(null);
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
  setMoveFlashMode(false);
  setFlashCameraMode(false);
  clearActiveResult();
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
  if (appState.analysisType === "lineup" && selectedLineup()) selectLineup(appState.selectedLineupIndex, false);
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

function downloadBlob(filename, content, contentType) {
  const url = URL.createObjectURL(new Blob([content], { type: contentType }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
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
      await refreshApplicationState(true, true);
      setLoading(false);
    } else if (config.modelUrl) {
      await loadUrl(config.modelUrl, config.modelName || "model.glb");
      setLoading(false);
    } else {
      await refreshApplicationState(false, true);
    }
  })
  .catch((error) => { setLoading(false); showError(error); });
