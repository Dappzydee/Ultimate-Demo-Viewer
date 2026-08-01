import { combineBounds, fromNode, identity, multiply, transformPoint } from "./math.js";

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };
const COMPONENT_TYPES = {
  5120: { ArrayType: Int8Array, bytes: 1 },
  5121: { ArrayType: Uint8Array, bytes: 1 },
  5122: { ArrayType: Int16Array, bytes: 2 },
  5123: { ArrayType: Uint16Array, bytes: 2 },
  5125: { ArrayType: Uint32Array, bytes: 4 },
  5126: { ArrayType: Float32Array, bytes: 4 },
};

export function parseGlb(arrayBuffer, filename = "model.glb") {
  if (arrayBuffer.byteLength < 20) throw new Error("The file is too small to be a GLB.");
  const header = new DataView(arrayBuffer, 0, 12);
  if (header.getUint32(0, true) !== GLB_MAGIC) throw new Error("Not a binary glTF (.glb) file.");
  if (header.getUint32(4, true) !== 2) throw new Error("Only glTF 2.0 GLB files are supported.");
  const declaredLength = header.getUint32(8, true);
  if (declaredLength > arrayBuffer.byteLength) throw new Error("The GLB is truncated.");

  let offset = 12;
  let json = null;
  const binaryChunks = [];
  while (offset + 8 <= declaredLength) {
    const chunkHeader = new DataView(arrayBuffer, offset, 8);
    const length = chunkHeader.getUint32(0, true);
    const type = chunkHeader.getUint32(4, true);
    offset += 8;
    if (offset + length > declaredLength) throw new Error("A GLB chunk extends past the file boundary.");
    if (type === JSON_CHUNK) {
      const text = new TextDecoder().decode(new Uint8Array(arrayBuffer, offset, length)).replace(/\0+$/g, "").trim();
      json = JSON.parse(text);
    } else if (type === BIN_CHUNK) {
      binaryChunks.push({ byteOffset: offset, byteLength: length });
    }
    offset += length;
  }
  if (!json) throw new Error("The GLB has no JSON scene chunk.");
  if (!binaryChunks.length) throw new Error("The GLB has no binary geometry chunk.");

  const context = { json, arrayBuffer, binaryChunks };
  const sceneIndex = json.scene ?? 0;
  const scene = json.scenes?.[sceneIndex];
  if (!scene) throw new Error("The GLB has no renderable scene.");

  const drawables = [];
  let sceneBounds = null;
  const visitNode = (nodeIndex, parentMatrix, inheritedName) => {
    const node = json.nodes?.[nodeIndex];
    if (!node) return;
    const worldMatrix = multiply(parentMatrix, fromNode(node));
    const nodeName = node.name || inheritedName || `Node ${nodeIndex}`;
    if (node.mesh !== undefined) {
      const mesh = json.meshes?.[node.mesh];
      if (!mesh) throw new Error(`Node ${nodeIndex} references a missing mesh.`);
      mesh.primitives.forEach((primitive, primitiveIndex) => {
        if ((primitive.mode ?? 4) !== 4) return;
        if (primitive.attributes?.POSITION === undefined) return;
        const position = readAccessor(context, primitive.attributes.POSITION);
        const color = primitive.attributes.COLOR_0 !== undefined ? readAccessor(context, primitive.attributes.COLOR_0) : null;
        const indices = primitive.indices !== undefined ? readAccessor(context, primitive.indices) : null;
        const material = json.materials?.[primitive.material] || {};
        const localBounds = accessorBounds(position);
        const bounds = transformBounds(localBounds, worldMatrix);
        const triangleCount = indices ? Math.floor(indices.count / 3) : Math.floor(position.count / 3);
        const name = mesh.primitives.length > 1 ? `${nodeName} · ${primitiveIndex + 1}` : (mesh.name || nodeName);
        const drawable = {
          id: drawables.length,
          name,
          nodeName,
          position,
          color,
          indices,
          worldMatrix,
          bounds,
          triangleCount,
          vertexCount: position.count,
          baseColor: material.pbrMetallicRoughness?.baseColorFactor || [1, 1, 1, 1],
          doubleSided: material.doubleSided ?? true,
          visible: true,
        };
        drawables.push(drawable);
        sceneBounds = combineBounds(sceneBounds, bounds);
      });
    }
    for (const child of node.children || []) visitNode(child, worldMatrix, nodeName);
  };
  for (const node of scene.nodes || []) visitNode(node, identity(), "Scene");
  if (!drawables.length || !sceneBounds) throw new Error("The GLB scene has no triangle meshes.");

  return {
    filename,
    drawables,
    bounds: sceneBounds,
    triangleCount: drawables.reduce((sum, item) => sum + item.triangleCount, 0),
    vertexCount: drawables.reduce((sum, item) => sum + item.vertexCount, 0),
    sourceBytes: arrayBuffer.byteLength,
  };
}

function readAccessor(context, index) {
  const accessor = context.json.accessors?.[index];
  if (!accessor) throw new Error(`Missing accessor ${index}.`);
  if (accessor.sparse) throw new Error("Sparse glTF accessors are not supported yet.");
  const type = COMPONENT_TYPES[accessor.componentType];
  const components = COMPONENTS[accessor.type];
  if (!type || !components) throw new Error(`Unsupported accessor ${index}.`);
  const view = context.json.bufferViews?.[accessor.bufferView];
  if (!view) throw new Error(`Accessor ${index} has no buffer view.`);
  const buffer = context.json.buffers?.[view.buffer ?? 0];
  if (buffer?.uri) throw new Error("External glTF buffers are not valid inside this GLB viewer.");
  const chunk = context.binaryChunks[view.buffer ?? 0] || context.binaryChunks[0];
  const byteOffset = chunk.byteOffset + (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const packedStride = type.bytes * components;
  const byteStride = view.byteStride || packedStride;
  let array;
  if (byteStride === packedStride && byteOffset % type.bytes === 0) {
    array = new type.ArrayType(context.arrayBuffer, byteOffset, accessor.count * components);
  } else {
    array = new type.ArrayType(accessor.count * components);
    const source = new DataView(context.arrayBuffer);
    const getter = { 5120: "getInt8", 5121: "getUint8", 5122: "getInt16", 5123: "getUint16", 5125: "getUint32", 5126: "getFloat32" }[accessor.componentType];
    for (let item = 0; item < accessor.count; item++) {
      for (let component = 0; component < components; component++) {
        array[item * components + component] = source[getter](byteOffset + item * byteStride + component * type.bytes, true);
      }
    }
  }
  return {
    array,
    count: accessor.count,
    components,
    componentType: accessor.componentType,
    normalized: accessor.normalized || false,
    min: accessor.min,
    max: accessor.max,
  };
}

function accessorBounds(accessor) {
  if (accessor.min && accessor.max) return { min: accessor.min.slice(0, 3), max: accessor.max.slice(0, 3) };
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < accessor.array.length; i += accessor.components) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], accessor.array[i + axis]);
      max[axis] = Math.max(max[axis], accessor.array[i + axis]);
    }
  }
  return { min, max };
}

function transformBounds(bounds, matrix) {
  const result = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const x of [bounds.min[0], bounds.max[0]]) for (const y of [bounds.min[1], bounds.max[1]]) for (const z of [bounds.min[2], bounds.max[2]]) {
    const point = transformPoint(matrix, [x, y, z]);
    for (let axis = 0; axis < 3; axis++) {
      result.min[axis] = Math.min(result.min[axis], point[axis]);
      result.max[axis] = Math.max(result.max[axis], point[axis]);
    }
  }
  return result;
}
