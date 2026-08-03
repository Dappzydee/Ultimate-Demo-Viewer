import { boundsCenter, boundsRadius, lookAt, multiply, orthographic, perspective, v3 } from "./math.js";

const VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_position;
layout(location=1) in vec4 a_color;
layout(location=2) in float a_resultValue;
uniform mat4 u_viewProjection;
uniform mat4 u_model;
uniform vec4 u_baseColor;
uniform bool u_hasColor;
out vec4 v_color;
out vec3 v_worldPosition;
flat out float v_resultValue;
void main() {
  vec4 world = u_model * vec4(a_position, 1.0);
  v_worldPosition = world.xyz;
  v_color = (u_hasColor ? a_color : vec4(1.0)) * u_baseColor;
  v_resultValue = a_resultValue;
  gl_Position = u_viewProjection * world;
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec4 v_color;
in vec3 v_worldPosition;
flat in float v_resultValue;
uniform int u_shading;
uniform int u_resultMode;
uniform float u_exposure;
uniform bool u_selected;
uniform bool u_wirePass;
out vec4 outColor;
void main() {
  if (u_wirePass) {
    outColor = u_selected ? vec4(1.0, 0.72, 0.22, 0.95) : vec4(0.04, 0.055, 0.07, 0.82);
    return;
  }
  vec3 normal = normalize(cross(dFdx(v_worldPosition), dFdy(v_worldPosition)));
  if (!gl_FrontFacing) normal = -normal;
  vec3 color;
  if (u_shading == 2) {
    color = normal * 0.5 + 0.5;
  } else {
    if (u_resultMode == 1) {
      color = mix(vec3(0.627), vec3(1.0, 0.0, 0.0), step(0.5, v_resultValue));
    } else if (u_resultMode == 2) {
      color = v_resultValue <= 0.0
        ? vec3(0.627)
        : mix(vec3(0.39, 0.0, 0.0), vec3(1.0, 0.96, 0.31), v_resultValue);
    } else {
      color = v_color.rgb;
    }
    if (u_shading == 0) {
      vec3 light = normalize(vec3(0.38, -0.46, 0.80));
      float diffuse = max(dot(normal, light), 0.0);
      color *= 0.38 + diffuse * 0.72;
    }
  }
  if (u_selected) color = mix(color, vec3(1.0, 0.62, 0.18), 0.13);
  color = vec3(1.0) - exp(-color * u_exposure);
  color = pow(color, vec3(1.0 / 2.2));
  outColor = vec4(color, v_color.a);
}`;

const LINE_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_position;
layout(location=1) in vec4 a_color;
uniform mat4 u_viewProjection;
out vec4 v_color;
void main() { v_color = a_color; gl_Position = u_viewProjection * vec4(a_position, 1.0); }`;

const LINE_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec4 v_color;
out vec4 outColor;
void main() { outColor = v_color; }`;

const TYPE_ENUM = { 5120: 0x1400, 5121: 0x1401, 5122: 0x1402, 5123: 0x1403, 5125: 0x1405, 5126: 0x1406 };

function poseMatrix(position, yawDegrees = 0) {
  const angle = Number(yawDegrees) * Math.PI / 180;
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  const [x, y, z] = position.map(Number);
  return new Float32Array([
    cosine, sine, 0, 0,
    -sine, cosine, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}

function pitchMatrix(pitchDegrees, pivotHeight = 51) {
  const angle = Number(pitchDegrees) * Math.PI / 180;
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  return new Float32Array([
    cosine, 0, -sine, 0,
    0, 1, 0, 0,
    sine, 0, cosine, 0,
    -sine * pivotHeight, 0, pivotHeight * (1 - cosine), 1,
  ]);
}

export class ViewerRenderer {
  constructor(canvas, onStatus = () => {}) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { antialias: true, alpha: false, powerPreference: "high-performance" });
    if (!this.gl) throw new Error("WebGL 2 is required. Try a current Chrome, Edge, or Firefox browser.");
    this.onStatus = onStatus;
    this.program = createProgram(this.gl, VERTEX_SHADER, FRAGMENT_SHADER);
    this.lineProgram = createProgram(this.gl, LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER);
    this.model = null;
    this.resources = [];
    this.markerResource = null;
    this.playerMarkerResource = null;
    this.lineupMarkerResources = [];
    this.lineupAssetInstances = [];
    this.lineupLineResource = null;
    this.lineupBounds = null;
    this.overlayAssetResources = new Map();
    this.analysisAssetInstance = null;
    this.replayPlayerInstance = null;
    this.visionPreviewInstances = [];
    this.visionPathResource = null;
    this.visionSelectionResource = null;
    this.visionPathSource = null;
    this.selectedId = null;
    this.shading = 0;
    this.exposure = 1;
    this.wireframe = false;
    this.showGrid = true;
    this.showAxes = true;
    this.projection = "perspective";
    this.camera = { target: [0, 0, 0], yaw: Math.PI * 0.22, pitch: Math.PI * 0.24, distance: 10, orthoSize: 5 };
    this.flashCameraPosition = null;
    this.lineupCameraPosition = null;
    this.lineupCameraFov = Math.PI / 2;
    this.interactionLocked = false;
    this.sceneRadius = 10;
    this.lineResources = null;
    this.keys = new Set();
    this.lastFrame = performance.now();
    this.fpsSamples = [];
    this.needsRender = true;
    this.#installControls();
    new ResizeObserver(() => this.requestRender()).observe(canvas);
    requestAnimationFrame((time) => this.#frame(time));
  }

  setModel(model) {
    this.#disposeModel();
    this.model = model;
    this.resources = model.drawables.map((drawable) => this.#uploadDrawable(drawable));
    this.lineResources = this.#createReferenceLines(model.bounds);
    this.selectedId = null;
    this.frameBounds(model.bounds, true);
  }

  setOverlayAssets(models) {
    for (const resources of this.overlayAssetResources.values()) {
      for (const resource of resources) this.#disposeResource(resource);
    }
    this.overlayAssetResources.clear();
    for (const [name, model] of Object.entries(models || {})) {
      this.overlayAssetResources.set(
        name, model.drawables.map((drawable) => this.#uploadDrawable(drawable)),
      );
    }
    this.requestRender();
  }

  setVisionPreview(value) {
    this.visionPreviewInstances = [];
    if (!value) {
      if (this.visionPathResource) this.#disposeLineResource(this.visionPathResource);
      if (this.visionSelectionResource) this.#disposeLineResource(this.visionSelectionResource);
      this.visionPathResource = null;
      this.visionSelectionResource = null;
      this.visionPathSource = null;
      this.requestRender();
      return;
    }
    const path = value.path || [];
    if (path !== this.visionPathSource) {
      if (this.visionPathResource) this.#disposeLineResource(this.visionPathResource);
      const pathLines = [];
      for (let index = 1; index < path.length; index++) {
        const start = path[index - 1].position.map(Number);
        const end = path[index].position.map(Number);
        start[2] += 2; end[2] += 2;
        pathLines.push([start, end, [0.18, 0.48, 0.65, 0.52]]);
      }
      this.visionPathResource = pathLines.length ? this.#createLineResource(pathLines) : null;
      this.visionPathSource = path;
    }
    if (this.visionSelectionResource) this.#disposeLineResource(this.visionSelectionResource);
    const selectedLines = [];
    for (let index = 1; index < path.length; index++) {
      const selected = path[index].seconds >= value.start.seconds
        && path[index - 1].seconds <= value.end.seconds;
      if (!selected) continue;
      const start = path[index - 1].position.map(Number);
      const end = path[index].position.map(Number);
      start[2] += 2; end[2] += 2;
      selectedLines.push([start, end, [0.20, 0.92, 1.0, 1.0]]);
    }
    this.visionSelectionResource = selectedLines.length ? this.#createLineResource(selectedLines) : null;
    if (value.showModels !== false) {
      this.visionPreviewInstances.push({
        asset: "playerAim", id: "vision-start", position: value.start.position,
        yaw: value.start.yaw, pitch: value.start.pitch, tint: [0.55, 1.0, 1.0, 1.0],
      });
      if (!value.instant) this.visionPreviewInstances.push({
        asset: "playerAim", id: "vision-end", position: value.end.position,
        yaw: value.end.yaw, pitch: value.end.pitch, tint: [1.0, 0.63, 0.28, 1.0],
      });
    }
    this.requestRender();
  }

  setFaceValues(faceValues, mode) {
    const resource = this.resources.find((item) => item.drawable.triangleCount === faceValues.length);
    if (!resource) throw new Error(`Result has ${faceValues.length} faces, but the loaded map does not.`);
    const source = resource.drawable.indices?.array;
    const values = resource.resultArray;
    for (let face = 0; face < faceValues.length; face++) {
      const value = faceValues[face];
      if (source) {
        values[source[face * 3]] = value;
        values[source[face * 3 + 1]] = value;
        values[source[face * 3 + 2]] = value;
      } else {
        const base = face * 3;
        values[base] = value;
        values[base + 1] = value;
        values[base + 2] = value;
      }
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, resource.resultBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, values);
    resource.resultMode = mode === "flash" ? 2 : 1;
    this.requestRender();
  }

  clearFaceValues() {
    for (const resource of this.resources) {
      resource.resultArray.fill(0);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, resource.resultBuffer);
      this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, resource.resultArray);
      resource.resultMode = 0;
    }
    this.requestRender();
  }

  setMarker(position, radius = 24) {
    if (this.markerResource) this.#disposeResource(this.markerResource);
    this.markerResource = null;
    this.analysisAssetInstance = null;
    if (!position) {
      if (this.selectedId === "analysis-marker") this.selectedId = null;
      this.requestRender();
      return;
    }
    if (this.overlayAssetResources.has("flashbang")) {
      this.analysisAssetInstance = {
        asset: "flashbang", id: "analysis-marker", position: position.map(Number),
        yaw: 0, pitch: 0, scale: 2.2, tint: [1.0, 0.95, 0.35, 1.0],
      };
      this.requestRender();
      return;
    }
    const [x, y, z] = position;
    const points = {
      top: [x, y, z + radius], bottom: [x, y, z - radius],
      east: [x + radius, y, z], west: [x - radius, y, z],
      north: [x, y + radius, z], south: [x, y - radius, z],
    };
    const triangles = [
      [points.top, points.east, points.north], [points.top, points.north, points.west],
      [points.top, points.west, points.south], [points.top, points.south, points.east],
      [points.bottom, points.north, points.east], [points.bottom, points.west, points.north],
      [points.bottom, points.south, points.west], [points.bottom, points.east, points.south],
    ];
    const positions = new Float32Array(triangles.flat(2));
    const colors = new Uint8Array((positions.length / 3) * 4);
    for (let i = 0; i < colors.length; i += 4) colors.set([255, 238, 35, 255], i);
    const drawable = {
      id: "analysis-marker", name: "Analysis marker",
      position: { array: positions, count: positions.length / 3, components: 3, componentType: 5126, normalized: false },
      color: { array: colors, count: positions.length / 3, components: 4, componentType: 5121, normalized: true },
      indices: null,
      worldMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      bounds: { min: [x - radius, y - radius, z - radius], max: [x + radius, y + radius, z + radius] },
      triangleCount: 8, vertexCount: positions.length / 3,
      baseColor: [1, 1, 1, 1], doubleSided: true, visible: true,
    };
    this.markerResource = this.#uploadDrawable(drawable);
    this.requestRender();
  }

  setPlayerMarker(position, yawDegrees = 0, visible = true, pitchDegrees = 0) {
    this.replayPlayerInstance = null;
    if (!position || !visible) {
      if (this.playerMarkerResource) this.playerMarkerResource.drawable.visible = false;
      if (this.selectedId === "player-marker") this.selectedId = null;
      this.requestRender();
      return;
    }
    if (this.overlayAssetResources.has("playerAim")) {
      if (this.playerMarkerResource) this.playerMarkerResource.drawable.visible = false;
      this.replayPlayerInstance = {
        asset: "playerAim", id: "player-marker", position: position.map(Number),
        yaw: Number(yawDegrees), pitch: Number(pitchDegrees), tint: [0.45, 0.92, 1.0, 1.0],
      };
      this.requestRender();
      return;
    }
    if (!this.playerMarkerResource) {
      const top = [0, 0, 72], bottom = [0, 0, 0];
      const east = [14, 0, 36], west = [-14, 0, 36];
      const north = [0, 14, 36], south = [0, -14, 36];
      const tip = [42, 0, 54], left = [12, -9, 48], right = [12, 9, 48], crown = [12, 0, 65];
      const triangles = [
        [top, east, north], [top, north, west], [top, west, south], [top, south, east],
        [bottom, north, east], [bottom, west, north], [bottom, south, west], [bottom, east, south],
        [tip, left, right], [tip, right, crown], [tip, crown, left], [left, crown, right],
      ];
      const positions = new Float32Array(triangles.flat(2));
      const colors = new Uint8Array((positions.length / 3) * 4);
      for (let index = 0; index < colors.length; index += 4) colors.set([44, 211, 255, 255], index);
      const drawable = {
        id: "player-marker", name: "Player position",
        position: { array: positions, count: positions.length / 3, components: 3, componentType: 5126, normalized: false },
        color: { array: colors, count: positions.length / 3, components: 4, componentType: 5121, normalized: true },
        indices: null,
        worldMatrix: new Float32Array(16),
        bounds: { min: [-14, -14, 0], max: [42, 14, 72] },
        triangleCount: triangles.length, vertexCount: positions.length / 3,
        baseColor: [1, 1, 1, 1], doubleSided: true, visible: true,
      };
      this.playerMarkerResource = this.#uploadDrawable(drawable);
    }
    const angle = Number(yawDegrees) * Math.PI / 180;
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    const [x, y, z] = position.map(Number);
    this.playerMarkerResource.drawable.worldMatrix.set([
      cosine, sine, 0, 0,
      -sine, cosine, 0, 0,
      0, 0, 1, 0,
      x, y, z, 1,
    ]);
    this.playerMarkerResource.drawable.bounds = {
      min: [x - 42, y - 42, z], max: [x + 42, y + 42, z + 72],
    };
    this.playerMarkerResource.drawable.visible = true;
    this.requestRender();
  }

  setLineupVisualization(value) {
    for (const resource of this.lineupMarkerResources) this.#disposeResource(resource);
    this.lineupMarkerResources = [];
    this.lineupAssetInstances = [];
    if (this.lineupLineResource) this.#disposeLineResource(this.lineupLineResource);
    this.lineupLineResource = null;
    this.lineupBounds = null;
    if (!value) { this.requestRender(); return; }

    const fixed = Boolean(value.fixed);
    const reference = value.reference?.map(Number) || null;
    const pinPull = value.pinPull?.map(Number) || null;
    const release = value.release.map(Number);
    const detonation = value.detonation?.map(Number) || null;
    const hasPlayers = this.overlayAssetResources.has("playerHold")
      && this.overlayAssetResources.has("playerThrow");
    if (hasPlayers) {
      const holdPosition = fixed ? reference : pinPull;
      if (holdPosition) this.lineupAssetInstances.push({
        asset: "playerHold", id: fixed ? "lineup-reference" : "lineup-pin-pull",
        position: holdPosition, yaw: Number(value.holdYaw ?? value.releaseYaw ?? 0),
        pitch: Number(value.holdPitch ?? value.releasePitch ?? 0),
        tint: fixed ? [0.35, 0.82, 1.0, 1.0] : [0.72, 0.42, 1.0, 1.0],
      });
      this.lineupAssetInstances.push({
        asset: "playerThrow", id: "lineup-release", position: release,
        yaw: Number(value.releaseYaw ?? 0), pitch: Number(value.releasePitch ?? 0),
        tint: fixed ? [1.0, 0.58, 0.25, 1.0] : [0.35, 0.82, 1.0, 1.0],
      });
    } else {
      if (reference) this.lineupMarkerResources.push(this.#createMarkerResource(
        "lineup-reference", "Lineup reference", reference, 22, [42, 174, 255, 255],
      ));
      if (!fixed && pinPull) this.lineupMarkerResources.push(this.#createMarkerResource(
        "lineup-pin-pull", "Grenade pin pull", pinPull, 20, [169, 92, 255, 255],
      ));
      this.lineupMarkerResources.push(this.#createMarkerResource(
        "lineup-release", "Grenade release", release, 18,
        fixed ? [255, 137, 48, 255] : [42, 174, 255, 255],
      ));
    }
    if (detonation && value.grenadeType === "flashbang" && this.overlayAssetResources.has("flashbang")) {
      this.lineupAssetInstances.push({
        asset: "flashbang", id: "lineup-detonation", position: detonation,
        yaw: 0, pitch: 0, scale: 2.2, tint: [1.0, 0.22, 0.26, 1.0],
      });
    } else if (detonation) {
      this.lineupMarkerResources.push(this.#createMarkerResource(
        "lineup-detonation", "Grenade detonation", detonation, 24, [255, 64, 77, 255],
      ));
    }

    const lines = [];
    const path = (value.path || []).map((point) => point.map(Number));
    const route = path.length > 1 ? path : reference ? [reference, release] : [];
    for (let index = 1; index < route.length; index++) {
      lines.push([route[index - 1], route[index], [0.20, 0.88, 0.94, 0.95]]);
    }
    if (value.aim) {
      const origin = value.aim.origin.map(Number);
      const endpoint = value.aim.endpoint.map(Number);
      lines.push([origin, endpoint, [1.0, 0.82, 0.25, 1.0]]);
    }
    if (lines.length) this.lineupLineResource = this.#createLineResource(lines);
    const points = [
      release, ...(reference ? [reference] : []), ...(pinPull ? [pinPull] : []),
      ...(detonation ? [detonation] : []), ...path,
    ];
    if (value.aim) points.push(value.aim.origin.map(Number));
    this.lineupBounds = {
      min: [0, 1, 2].map((axis) => Math.min(...points.map((point) => point[axis])) - 28),
      max: [0, 1, 2].map((axis) => Math.max(...points.map((point) => point[axis])) + 28),
    };
    this.requestRender();
  }

  frameLineup() {
    if (this.lineupBounds) this.frameBounds(this.lineupBounds);
  }

  setLineupCamera(pose) {
    if (pose) {
      this.lineupCameraPosition = pose.position.map(Number);
      this.camera.yaw = Number(pose.yaw) * Math.PI / 180 + Math.PI;
      this.camera.pitch = Number(pose.pitch) * Math.PI / 180;
    } else if (this.lineupCameraPosition) {
      const previousPosition = this.lineupCameraPosition;
      this.lineupCameraPosition = null;
      this.focusPoint(previousPosition);
    }
    this.requestRender();
  }

  setLineupCameraFov(degrees) {
    const value = Math.max(40, Math.min(140, Number(degrees)));
    this.lineupCameraFov = value * Math.PI / 180;
    this.requestRender();
  }

  focusPoint(position, radius = 24) {
    if (!position) return;
    this.camera.target = position.map(Number);
    this.camera.distance = Math.max(radius * 4, this.sceneRadius * 0.025);
    this.camera.orthoSize = Math.max(radius * 2.5, this.sceneRadius * 0.015);
    this.requestRender();
  }

  translateCamera(offset) {
    this.camera.target = v3.add(this.camera.target, offset.map(Number));
    this.requestRender();
  }

  getMovementBasis() {
    const forward = this.#forwardDirection();
    const planarForward = v3.normalize([forward[0], forward[1], 0]);
    return {
      forward: planarForward,
      right: v3.normalize(v3.cross(planarForward, [0, 0, 1])),
    };
  }

  getSceneCenter() {
    return this.model ? boundsCenter(this.model.bounds) : [0, 0, 0];
  }

  setFlashCamera(position) {
    if (position) {
      this.flashCameraPosition = position.map(Number);
    } else if (this.flashCameraPosition) {
      const previousPosition = this.flashCameraPosition;
      this.flashCameraPosition = null;
      this.focusPoint(previousPosition);
    }
    this.requestRender();
  }

  setInteractionLocked(locked) {
    this.interactionLocked = Boolean(locked);
    if (this.interactionLocked) this.keys.clear();
  }

  getPickRay(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;
    const eye = this.#eyePosition();
    const forward = this.#forwardDirection();
    const right = v3.normalize(v3.cross(forward, [0, 0, 1]));
    const up = v3.normalize(v3.cross(right, forward));
    const aspect = Math.max(rect.width / Math.max(rect.height, 1), 0.01);
    if (this.projection === "orthographic") {
      const origin = v3.add(eye, v3.add(
        v3.scale(right, ndcX * this.camera.orthoSize * aspect),
        v3.scale(up, ndcY * this.camera.orthoSize),
      ));
      return { origin, direction: forward };
    }
    const scale = Math.tan(Math.PI / 6);
    return {
      origin: eye,
      direction: v3.normalize(v3.add(forward, v3.add(
        v3.scale(right, ndcX * scale * aspect), v3.scale(up, ndcY * scale),
      ))),
    };
  }

  frameBounds(bounds, resetAngle = false) {
    if (!bounds) return;
    const radius = Math.max(boundsRadius(bounds), 0.01);
    this.sceneRadius = Math.max(this.model ? boundsRadius(this.model.bounds) : radius, 0.01);
    this.camera.target = boundsCenter(bounds);
    this.camera.distance = radius / Math.tan(Math.PI / 7) * 1.25;
    this.camera.orthoSize = radius * 1.2;
    if (resetAngle) {
      this.camera.yaw = Math.PI * 0.22;
      this.camera.pitch = Math.PI * 0.24;
    }
    this.requestRender();
  }

  frameSelection() {
    const selected = this.selectedId === "analysis-marker" ? this.markerResource?.drawable
      : this.selectedId === "player-marker" ? this.playerMarkerResource?.drawable
        : this.model?.drawables.find((item) => item.id === this.selectedId);
    const instance = [
      this.analysisAssetInstance, this.replayPlayerInstance,
      ...this.visionPreviewInstances, ...this.lineupAssetInstances,
    ].find((item) => item?.id === this.selectedId);
    const instanceBounds = instance ? {
      min: [instance.position[0] - 42, instance.position[1] - 42, instance.position[2] - 12],
      max: [instance.position[0] + 42, instance.position[1] + 42, instance.position[2] + 78],
    } : null;
    this.frameBounds(selected?.bounds || instanceBounds || this.model?.bounds);
  }

  setSelected(id) { this.selectedId = id; this.requestRender(); }
  setShading(value) { this.shading = { lit: 0, flat: 1, normal: 2 }[value] ?? 0; this.requestRender(); }
  setExposure(value) { this.exposure = Number(value); this.requestRender(); }
  setWireframe(value) { this.wireframe = value; this.requestRender(); }
  setGrid(value) { this.showGrid = value; this.requestRender(); }
  setAxes(value) { this.showAxes = value; this.requestRender(); }
  setProjection(value) { this.projection = value; this.requestRender(); }
  requestRender() { this.needsRender = true; }

  #frame(time) {
    const delta = Math.min((time - this.lastFrame) / 1000, 0.1);
    this.lastFrame = time;
    const moving = this.#updateMovement(delta);
    if (this.needsRender || moving) this.#render(time);
    requestAnimationFrame((nextTime) => this.#frame(nextTime));
  }

  #render(time) {
    const gl = this.gl;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.055, 0.068, 0.084, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const eye = this.#eyePosition();
    const view = lookAt(eye, this.#viewTarget(), [0, 0, 1]);
    const aspect = width / height;
    const far = Math.max(this.sceneRadius * 20, this.camera.distance + this.sceneRadius * 5, 100);
    const near = Math.max(far / 100000, 0.01);
    const projection = this.projection === "orthographic"
      ? orthographic(this.camera.orthoSize, aspect, -far, far)
      : perspective(this.lineupCameraPosition ? this.lineupCameraFov : Math.PI / 3, aspect, near, far);
    const viewProjection = multiply(projection, view);

    if (this.lineResources && (this.showGrid || this.showAxes)) this.#drawReferenceLines(viewProjection);
    if (this.model) this.#drawModel(viewProjection);
    if (this.visionPathResource) this.#drawLineResource(this.visionPathResource, viewProjection);
    if (this.visionSelectionResource) this.#drawLineResource(this.visionSelectionResource, viewProjection);
    if (this.lineupLineResource) this.#drawLineupLines(viewProjection);
    this.needsRender = false;

    this.fpsSamples.push(time);
    while (this.fpsSamples[0] < time - 1000) this.fpsSamples.shift();
    this.onStatus({ eye, fps: Math.max(0, this.fpsSamples.length - 1) });
  }

  #drawModel(viewProjection) {
    const gl = this.gl;
    gl.useProgram(this.program);
    uniformMatrix(gl, this.program, "u_viewProjection", viewProjection);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_shading"), this.shading);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_exposure"), this.exposure);
    const drawResources = [...this.resources];
    if (this.markerResource && !this.flashCameraPosition) drawResources.push(this.markerResource);
    if (this.playerMarkerResource) drawResources.push(this.playerMarkerResource);
    drawResources.push(...this.lineupMarkerResources);
    for (const resource of drawResources) {
      const drawable = resource.drawable;
      if (!drawable.visible) continue;
      gl.bindVertexArray(resource.vao);
      uniformMatrix(gl, this.program, "u_model", drawable.worldMatrix);
      gl.uniform4fv(gl.getUniformLocation(this.program, "u_baseColor"), drawable.baseColor);
      gl.uniform1i(gl.getUniformLocation(this.program, "u_hasColor"), Boolean(drawable.color));
      gl.uniform1i(gl.getUniformLocation(this.program, "u_resultMode"), resource.resultMode);
      gl.uniform1i(gl.getUniformLocation(this.program, "u_selected"), drawable.id === this.selectedId);
      gl.uniform1i(gl.getUniformLocation(this.program, "u_wirePass"), false);
      if (drawable.doubleSided) gl.disable(gl.CULL_FACE); else { gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK); }
      if (resource.indexBuffer) gl.drawElements(gl.TRIANGLES, resource.indexCount, resource.indexType, 0);
      else gl.drawArrays(gl.TRIANGLES, 0, drawable.vertexCount);

      if (this.wireframe) {
        if (!resource.wireIndexBuffer) this.#createWireIndices(resource);
        gl.uniform1i(gl.getUniformLocation(this.program, "u_wirePass"), true);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, resource.wireIndexBuffer);
        gl.drawElements(gl.LINES, resource.wireIndexCount, gl.UNSIGNED_INT, 0);
        if (resource.indexBuffer) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, resource.indexBuffer);
      }
    }
    const overlayInstances = [
      ...this.visionPreviewInstances,
      ...(this.replayPlayerInstance ? [this.replayPlayerInstance] : []),
      ...(this.analysisAssetInstance && !this.flashCameraPosition ? [this.analysisAssetInstance] : []),
      ...this.lineupAssetInstances,
    ];
    for (const instance of overlayInstances) {
      const assetResources = this.overlayAssetResources.get(instance.asset) || [];
      const baseTransform = poseMatrix(instance.position, instance.yaw);
      const scale = Number(instance.scale || 1);
      if (scale !== 1) {
        for (const offset of [0, 1, 2, 4, 5, 6, 8, 9, 10]) baseTransform[offset] *= scale;
      }
      for (const resource of assetResources) {
        const drawable = resource.drawable;
        const poseTransform = drawable.nodeName?.startsWith("look_")
          ? multiply(baseTransform, pitchMatrix(instance.pitch || 0)) : baseTransform;
        const modelMatrix = multiply(poseTransform, drawable.worldMatrix);
        const tint = instance.tint || [1, 1, 1, 1];
        const baseColor = drawable.baseColor.map((value, index) => value * tint[index]);
        gl.bindVertexArray(resource.vao);
        uniformMatrix(gl, this.program, "u_model", modelMatrix);
        gl.uniform4fv(gl.getUniformLocation(this.program, "u_baseColor"), baseColor);
        gl.uniform1i(gl.getUniformLocation(this.program, "u_hasColor"), Boolean(drawable.color));
        gl.uniform1i(gl.getUniformLocation(this.program, "u_resultMode"), 0);
        gl.uniform1i(gl.getUniformLocation(this.program, "u_selected"), instance.id === this.selectedId);
        gl.uniform1i(gl.getUniformLocation(this.program, "u_wirePass"), false);
        gl.disable(gl.CULL_FACE);
        if (resource.indexBuffer) gl.drawElements(gl.TRIANGLES, resource.indexCount, resource.indexType, 0);
        else gl.drawArrays(gl.TRIANGLES, 0, drawable.vertexCount);
      }
    }
    gl.bindVertexArray(null);
  }

  #drawReferenceLines(viewProjection) {
    const gl = this.gl;
    gl.useProgram(this.lineProgram);
    uniformMatrix(gl, this.lineProgram, "u_viewProjection", viewProjection);
    gl.bindVertexArray(this.lineResources.vao);
    if (this.showGrid) gl.drawArrays(gl.LINES, 0, this.lineResources.gridVertices);
    if (this.showAxes) gl.drawArrays(gl.LINES, this.lineResources.gridVertices, 6);
    gl.bindVertexArray(null);
  }

  #drawLineupLines(viewProjection) {
    this.#drawLineResource(this.lineupLineResource, viewProjection);
  }

  #drawLineResource(resource, viewProjection) {
    const gl = this.gl;
    gl.useProgram(this.lineProgram);
    uniformMatrix(gl, this.lineProgram, "u_viewProjection", viewProjection);
    gl.bindVertexArray(resource.vao);
    gl.drawArrays(gl.LINES, 0, resource.vertexCount);
    gl.bindVertexArray(null);
  }

  #createMarkerResource(id, name, position, radius, color) {
    const [x, y, z] = position;
    const points = {
      top: [x, y, z + radius], bottom: [x, y, z - radius],
      east: [x + radius, y, z], west: [x - radius, y, z],
      north: [x, y + radius, z], south: [x, y - radius, z],
    };
    const triangles = [
      [points.top, points.east, points.north], [points.top, points.north, points.west],
      [points.top, points.west, points.south], [points.top, points.south, points.east],
      [points.bottom, points.north, points.east], [points.bottom, points.west, points.north],
      [points.bottom, points.south, points.west], [points.bottom, points.east, points.south],
    ];
    const positions = new Float32Array(triangles.flat(2));
    const colors = new Uint8Array((positions.length / 3) * 4);
    for (let index = 0; index < colors.length; index += 4) colors.set(color, index);
    return this.#uploadDrawable({
      id, name,
      position: { array: positions, count: positions.length / 3, components: 3, componentType: 5126, normalized: false },
      color: { array: colors, count: positions.length / 3, components: 4, componentType: 5121, normalized: true },
      indices: null,
      worldMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      bounds: { min: [x - radius, y - radius, z - radius], max: [x + radius, y + radius, z + radius] },
      triangleCount: 8, vertexCount: positions.length / 3,
      baseColor: [1, 1, 1, 1], doubleSided: true, visible: true,
    });
  }

  #createLineResource(lines) {
    const gl = this.gl;
    const positions = [];
    const colors = [];
    for (const [start, end, color] of lines) {
      positions.push(...start, ...end);
      colors.push(...color, ...color);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return { vao, positionBuffer, colorBuffer, vertexCount: positions.length / 3 };
  }

  #uploadDrawable(drawable) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, drawable.position.array, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, drawable.position.components, TYPE_ENUM[drawable.position.componentType], drawable.position.normalized, 0, 0);

    let colorBuffer = null;
    if (drawable.color) {
      colorBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, drawable.color.array, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, drawable.color.components, TYPE_ENUM[drawable.color.componentType], drawable.color.normalized, 0, 0);
    } else {
      gl.disableVertexAttribArray(1);
      gl.vertexAttrib4f(1, 1, 1, 1, 1);
    }

    let indexBuffer = null;
    let indexType = null;
    let indexCount = 0;
    if (drawable.indices) {
      indexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, drawable.indices.array, gl.STATIC_DRAW);
      indexType = TYPE_ENUM[drawable.indices.componentType];
      indexCount = drawable.indices.count;
    }
    const resultBuffer = gl.createBuffer();
    const resultArray = new Uint8Array(drawable.vertexCount);
    gl.bindBuffer(gl.ARRAY_BUFFER, resultBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, resultArray, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.bindVertexArray(null);
    return {
      drawable, vao, vertexBuffer, colorBuffer, indexBuffer, indexType, indexCount,
      resultBuffer, resultArray, resultMode: 0, wireIndexBuffer: null, wireIndexCount: 0,
    };
  }

  #createWireIndices(resource) {
    const gl = this.gl;
    const source = resource.drawable.indices?.array;
    const triangleCount = resource.drawable.triangleCount;
    const lines = new Uint32Array(triangleCount * 6);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
      const a = source ? source[triangle * 3] : triangle * 3;
      const b = source ? source[triangle * 3 + 1] : triangle * 3 + 1;
      const c = source ? source[triangle * 3 + 2] : triangle * 3 + 2;
      lines.set([a, b, b, c, c, a], triangle * 6);
    }
    resource.wireIndexBuffer = gl.createBuffer();
    gl.bindVertexArray(resource.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, resource.wireIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lines, gl.STATIC_DRAW);
    resource.wireIndexCount = lines.length;
    gl.bindVertexArray(null);
  }

  #createReferenceLines(bounds) {
    const gl = this.gl;
    const radius = Math.max(boundsRadius(bounds), 1);
    const magnitude = 10 ** Math.floor(Math.log10(radius));
    const step = magnitude / (radius / magnitude > 5 ? 1 : 2);
    const extent = Math.ceil(radius * 1.5 / step) * step;
    const center = boundsCenter(bounds);
    const originX = Math.round(center[0] / step) * step;
    const originY = Math.round(center[1] / step) * step;
    const positions = [];
    const colors = [];
    const lineColor = [0.27, 0.30, 0.34, 0.5];
    for (let i = -10; i <= 10; i++) {
      const x = originX + i * extent / 10;
      const y = originY + i * extent / 10;
      positions.push(x, originY - extent, 0, x, originY + extent, 0, originX - extent, y, 0, originX + extent, y, 0);
      for (let vertex = 0; vertex < 4; vertex++) colors.push(...lineColor);
    }
    const gridVertices = positions.length / 3;
    const axisLength = Math.max(radius * 0.2, step);
    positions.push(0, 0, 0, axisLength, 0, 0, 0, 0, 0, 0, axisLength, 0, 0, 0, 0, 0, 0, axisLength);
    colors.push(1, .2, .16, 1, 1, .2, .16, 1, .2, 1, .35, 1, .2, 1, .35, 1, .25, .5, 1, 1, .25, .5, 1, 1);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return { vao, positionBuffer, colorBuffer, gridVertices };
  }

  #disposeModel() {
    const gl = this.gl;
    for (const resource of this.resources) this.#disposeResource(resource);
    if (this.markerResource) this.#disposeResource(this.markerResource);
    if (this.playerMarkerResource) this.#disposeResource(this.playerMarkerResource);
    for (const resource of this.lineupMarkerResources) this.#disposeResource(resource);
    if (this.lineupLineResource) this.#disposeLineResource(this.lineupLineResource);
    if (this.visionPathResource) this.#disposeLineResource(this.visionPathResource);
    if (this.visionSelectionResource) this.#disposeLineResource(this.visionSelectionResource);
    this.markerResource = null;
    this.playerMarkerResource = null;
    this.lineupMarkerResources = [];
    this.lineupLineResource = null;
    this.visionPathResource = null;
    this.visionSelectionResource = null;
    this.visionPathSource = null;
    this.lineupBounds = null;
    this.analysisAssetInstance = null;
    this.replayPlayerInstance = null;
    this.visionPreviewInstances = [];
    this.lineupAssetInstances = [];
    if (this.lineResources) {
      gl.deleteVertexArray(this.lineResources.vao);
      gl.deleteBuffer(this.lineResources.positionBuffer);
      gl.deleteBuffer(this.lineResources.colorBuffer);
    }
    this.resources = [];
  }

  #disposeResource(resource) {
    const gl = this.gl;
    gl.deleteVertexArray(resource.vao);
    gl.deleteBuffer(resource.vertexBuffer);
    gl.deleteBuffer(resource.colorBuffer);
    gl.deleteBuffer(resource.indexBuffer);
    gl.deleteBuffer(resource.resultBuffer);
    gl.deleteBuffer(resource.wireIndexBuffer);
  }

  #disposeLineResource(resource) {
    const gl = this.gl;
    gl.deleteVertexArray(resource.vao);
    gl.deleteBuffer(resource.positionBuffer);
    gl.deleteBuffer(resource.colorBuffer);
  }

  #eyePosition() {
    if (this.lineupCameraPosition) return [...this.lineupCameraPosition];
    if (this.flashCameraPosition) return [...this.flashCameraPosition];
    const cp = Math.cos(this.camera.pitch);
    return v3.add(this.camera.target, [
      Math.cos(this.camera.yaw) * cp * this.camera.distance,
      Math.sin(this.camera.yaw) * cp * this.camera.distance,
      Math.sin(this.camera.pitch) * this.camera.distance,
    ]);
  }

  #forwardDirection() {
    const cp = Math.cos(this.camera.pitch);
    return [-Math.cos(this.camera.yaw) * cp, -Math.sin(this.camera.yaw) * cp, -Math.sin(this.camera.pitch)];
  }

  #viewTarget() {
    const fixedPosition = this.lineupCameraPosition || this.flashCameraPosition;
    if (!fixedPosition) return this.camera.target;
    return v3.add(fixedPosition, this.#forwardDirection());
  }

  #updateMovement(delta) {
    if (this.flashCameraPosition || this.lineupCameraPosition || !this.keys.size || !this.model) return false;
    const eye = this.#eyePosition();
    const forward = this.#forwardDirection();
    const horizontalForward = v3.normalize([forward[0], forward[1], 0]);
    const right = v3.normalize(v3.cross(horizontalForward, [0, 0, 1]));
    let move = [0, 0, 0];
    if (this.keys.has("w")) move = v3.add(move, horizontalForward);
    if (this.keys.has("s")) move = v3.sub(move, horizontalForward);
    if (this.keys.has("d")) move = v3.add(move, right);
    if (this.keys.has("a")) move = v3.sub(move, right);
    if (this.keys.has("e")) move[2] += 1;
    if (this.keys.has("q")) move[2] -= 1;
    if (v3.length(move) === 0) return false;
    const speed = Math.max(this.camera.distance * 0.7, this.sceneRadius * 0.08);
    this.camera.target = v3.add(this.camera.target, v3.scale(v3.normalize(move), speed * delta));
    this.needsRender = true;
    return true;
  }

  #installControls() {
    let drag = null;
    this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    this.canvas.addEventListener("pointerdown", (event) => {
      if (this.interactionLocked) return;
      this.canvas.focus();
      this.canvas.setPointerCapture(event.pointerId);
      const mode = (this.flashCameraPosition || this.lineupCameraPosition) ? "look" : event.button === 0 && !event.shiftKey ? "orbit" : "pan";
      drag = { x: event.clientX, y: event.clientY, mode };
      this.canvas.classList.add("dragging");
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (this.interactionLocked) {
        drag = null;
        this.canvas.classList.remove("dragging");
        return;
      }
      if (!drag) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      drag.x = event.clientX; drag.y = event.clientY;
      if (drag.mode === "orbit" || drag.mode === "look") {
        this.camera.yaw -= dx * 0.006;
        this.camera.pitch = Math.max(-Math.PI * .48, Math.min(Math.PI * .48, this.camera.pitch + dy * 0.006));
      } else {
        const eye = this.#eyePosition();
        const forward = v3.normalize(v3.sub(this.camera.target, eye));
        const right = v3.normalize(v3.cross(forward, [0, 0, 1]));
        const up = v3.normalize(v3.cross(right, forward));
        const scale = (this.projection === "orthographic" ? this.camera.orthoSize : this.camera.distance) * 0.0025;
        this.camera.target = v3.add(this.camera.target, v3.add(v3.scale(right, -dx * scale), v3.scale(up, dy * scale)));
      }
      this.requestRender();
    });
    const endDrag = () => { drag = null; this.canvas.classList.remove("dragging"); };
    this.canvas.addEventListener("pointerup", endDrag);
    this.canvas.addEventListener("pointercancel", endDrag);
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      if (this.flashCameraPosition || this.lineupCameraPosition) return;
      const factor = Math.exp(Math.sign(event.deltaY) * Math.min(Math.abs(event.deltaY), 200) * 0.0015);
      this.camera.distance = Math.max(this.sceneRadius * 0.0005, this.camera.distance * factor);
      this.camera.orthoSize = Math.max(this.sceneRadius * 0.0005, this.camera.orthoSize * factor);
      this.requestRender();
    }, { passive: false });
    this.canvas.addEventListener("keydown", (event) => {
      if (this.interactionLocked) return;
      if (["w", "a", "s", "d", "q", "e"].includes(event.key.toLowerCase()) && !(event.shiftKey && event.key.toLowerCase() === "w")) {
        this.keys.add(event.key.toLowerCase());
        event.preventDefault();
      }
    });
    this.canvas.addEventListener("keyup", (event) => this.keys.delete(event.key.toLowerCase()));
    this.canvas.addEventListener("blur", () => this.keys.clear());
  }
}

function createProgram(gl, vertexSource, fragmentSource) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

function uniformMatrix(gl, program, name, value) {
  gl.uniformMatrix4fv(gl.getUniformLocation(program, name), false, value);
}
