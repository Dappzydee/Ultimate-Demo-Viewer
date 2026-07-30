import { parseGlb } from "./glb.js";
import { boundsRadius } from "./math.js";
import { ViewerRenderer } from "./renderer.js";

const $ = (selector) => document.querySelector(selector);
const canvas = $("#canvas");
const viewport = $("#viewport");
const fileInput = $("#file-input");
const dropZone = $("#drop-zone");
const loading = $("#loading");
const toast = $("#toast");
let model = null;
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

const chooseFile = () => fileInput.click();
$("#open-button").addEventListener("click", chooseFile);
$("#empty-open-button").addEventListener("click", chooseFile);
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
  fileInput.value = "";
});

$("#frame-button").addEventListener("click", () => renderer.frameSelection());
$("#show-all-button").addEventListener("click", () => {
  if (!model) return;
  model.drawables.forEach((item) => { item.visible = true; });
  renderObjectList();
  renderer.requestRender();
});

bindToggle("#wireframe-button", (value) => renderer.setWireframe(value));
bindToggle("#grid-button", (value) => renderer.setGrid(value), true);
bindToggle("#axes-button", (value) => renderer.setAxes(value), true);
$("#shading-select").addEventListener("change", (event) => renderer.setShading(event.target.value));
$("#exposure").addEventListener("input", (event) => renderer.setExposure(event.target.value));
$("#projection-button").addEventListener("click", () => {
  const button = $("#projection-button");
  const orthographic = button.textContent === "Perspective";
  button.textContent = orthographic ? "Orthographic" : "Perspective";
  button.classList.toggle("active", orthographic);
  renderer.setProjection(orthographic ? "orthographic" : "perspective");
});

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, button")) return;
  const key = event.key.toLowerCase();
  if (key === "o" && event.ctrlKey) { event.preventDefault(); chooseFile(); }
  if (key === "f") { event.preventDefault(); renderer.frameSelection(); }
  if (key === "w" && event.shiftKey) { event.preventDefault(); clickToggle("#wireframe-button"); }
  if (key === "g") clickToggle("#grid-button");
  if (key === "x") clickToggle("#axes-button");
  if (key === "5") $("#projection-button").click();
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
  const file = [...event.dataTransfer.files].find((candidate) => candidate.name.toLowerCase().endsWith(".glb"));
  if (file) loadFile(file); else showError(new Error("Drop a binary glTF file with the .glb extension."));
});

async function loadFile(file) {
  if (!file.name.toLowerCase().endsWith(".glb")) return showError(new Error("Choose a .glb file."));
  setLoading(true, file.name, formatBytes(file.size));
  await nextPaint();
  try {
    await installModel(await file.arrayBuffer(), file.name);
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

async function loadUrl(url, name) {
  setLoading(true, name, "Reading GLB…");
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${name} (${response.status}).`);
    const total = Number(response.headers.get("content-length")) || 0;
    let buffer;
    if (response.body && total) {
      const output = new Uint8Array(total);
      const reader = response.body.getReader();
      let offset = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        output.set(value, offset);
        offset += value.length;
        $("#loading-detail").textContent = `${formatBytes(offset)} / ${formatBytes(total)}`;
      }
      buffer = output.buffer;
    } else {
      buffer = await response.arrayBuffer();
    }
    await nextPaint();
    await installModel(buffer, name);
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

async function installModel(buffer, name) {
  $("#loading-detail").textContent = "Parsing scene…";
  await nextPaint();
  const parsed = parseGlb(buffer, name);
  $("#loading-detail").textContent = "Uploading geometry to GPU…";
  await nextPaint();
  renderer.setModel(parsed);
  model = parsed;
  $("#model-name").textContent = name;
  $("#status-summary").textContent = `${formatNumber(model.triangleCount)} triangles · ${formatNumber(model.vertexCount)} vertices · ${formatBytes(model.sourceBytes)}`;
  dropZone.hidden = true;
  renderObjectList();
  updateSelection(null);
  canvas.focus();
}

function renderObjectList() {
  const list = $("#object-list");
  list.classList.remove("empty");
  list.replaceChildren();
  for (const item of model.drawables) {
    const row = document.createElement("div");
    row.className = `object-row${renderer.selectedId === item.id ? " selected" : ""}`;
    row.title = "Select; double-click to frame";
    const visibility = document.createElement("button");
    visibility.className = `visibility${item.visible ? "" : " off"}`;
    visibility.textContent = item.visible ? "◉" : "○";
    visibility.setAttribute("aria-label", `${item.visible ? "Hide" : "Show"} ${item.name}`);
    visibility.addEventListener("click", (event) => {
      event.stopPropagation();
      item.visible = !item.visible;
      renderObjectList();
      renderer.requestRender();
    });
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = item.name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = compactNumber(item.triangleCount);
    row.append(visibility, name, count);
    row.addEventListener("click", () => selectObject(item));
    row.addEventListener("dblclick", () => { selectObject(item); renderer.frameSelection(); });
    list.append(row);
  }
}

function selectObject(item) {
  renderer.setSelected(item.id);
  renderObjectList();
  updateSelection(item);
}

function updateSelection(item) {
  const values = item ? [
    item.name,
    formatNumber(item.triangleCount),
    formatNumber(item.vertexCount),
    `${formatLength(boundsRadius(item.bounds) * 2)} span`,
  ] : ["Scene", model ? formatNumber(model.triangleCount) : "—", model ? formatNumber(model.vertexCount) : "—", model ? `${formatLength(boundsRadius(model.bounds) * 2)} span` : "—"];
  $("#selection-details").querySelectorAll("dd").forEach((element, index) => { element.textContent = values[index]; });
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

function clickToggle(selector) {
  $(selector).click();
}

function setLoading(visible, title = "Loading model…", detail = "") {
  loading.hidden = !visible;
  $("#loading-title").textContent = title;
  $("#loading-detail").textContent = detail;
}

function showError(error) {
  console.error(error);
  toast.textContent = error instanceof Error ? error.message : String(error);
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 8000);
}

const formatNumber = (value) => new Intl.NumberFormat().format(value);
const compactNumber = (value) => new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
const formatLength = (value) => Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(1)}k units` : `${value.toFixed(1)} units`;
const formatBytes = (bytes) => {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
};
const nextPaint = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

fetch("/viewer-config.json")
  .then((response) => response.json())
  .then((config) => { if (config.modelUrl) loadUrl(config.modelUrl, config.modelName || "model.glb"); })
  .catch(() => {});
