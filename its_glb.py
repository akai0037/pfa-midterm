"""
its_glb  -  minimal reader for .glb 3D files (binary glTF), numpy only.

Maya can't open .glb by itself, so the tool reads the file here and builds the
mesh with Maya's API. Handles what image-to-3D models output: triangle meshes
with positions, UVs and one embedded color texture.

read_glb(path) -> dict(points (N,3), uvs (N,2) or None, tris (M,3),
                       texture bytes or None, texture_ext ".png"/".jpg")
"""

import json
import struct

import numpy as np

_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
           5125: np.uint32, 5126: np.float32}
_COMPS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major
    m = np.eye(4)
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    r = _quat_to_mat(node.get("rotation", [0, 0, 0, 1]))
    m[:3, :3] = r * s
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, _version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a .glb file: %s" % path)
    gl, binary, off = None, b"", 12
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == 0x4E4F534A:
            gl = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binary = chunk
    if gl is None:
        raise ValueError("no glTF JSON in %s" % path)
    if "KHR_draco_mesh_compression" in gl.get("extensionsUsed", []):
        raise ValueError("Draco-compressed .glb files aren't supported")

    def view(i):
        bv = gl["bufferViews"][i]
        o = bv.get("byteOffset", 0)
        return binary[o:o + bv["byteLength"]], bv.get("byteStride")

    def accessor(i):
        a = gl["accessors"][i]
        dt = np.dtype(_DTYPES[a["componentType"]])
        n = _COMPS[a["type"]]
        raw, stride = view(a["bufferView"])
        o = a.get("byteOffset", 0)
        if stride and stride != dt.itemsize * n:
            arr = np.ndarray((a["count"], n), dtype=dt, buffer=raw, offset=o,
                             strides=(stride, dt.itemsize))
        else:
            arr = np.frombuffer(raw, dtype=dt, count=a["count"] * n, offset=o).reshape(a["count"], n)
        arr = arr.astype(np.float64)
        if a.get("normalized") and dt.kind in "iu":
            arr = arr / float(np.iinfo(dt).max)
        return arr

    # collect every mesh in the scene with its world transform
    items = []

    def walk(ni, parent):
        node = gl["nodes"][ni]
        m = parent @ _node_matrix(node)
        if "mesh" in node:
            items.append((node["mesh"], m))
        for c in node.get("children", []):
            walk(c, m)

    scene = gl.get("scenes", [{}])[gl.get("scene", 0)] if gl.get("scenes") else {}
    roots = scene.get("nodes", list(range(len(gl.get("nodes", [])))))
    for r in roots:
        walk(r, np.eye(4))
    if not items and gl.get("meshes"):
        items = [(i, np.eye(4)) for i in range(len(gl["meshes"]))]

    pts, uvs, tris, base = [], [], [], 0
    tex_index = None
    for mi, m in items:
        for prim in gl["meshes"][mi]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue  # triangles only
            attrs = prim["attributes"]
            p = accessor(attrs["POSITION"])[:, :3]
            p = (np.c_[p, np.ones(len(p))] @ m.T)[:, :3]
            uv = accessor(attrs["TEXCOORD_0"])[:, :2] if "TEXCOORD_0" in attrs else np.zeros((len(p), 2))
            idx = accessor(prim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in prim \
                else np.arange(len(p)).reshape(-1, 3)
            pts.append(p)
            uvs.append(uv)
            tris.append(idx + base)
            base += len(p)
            if tex_index is None and "material" in prim:
                mat = gl["materials"][prim["material"]]
                bc = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
                if bc is not None:
                    tex_index = bc["index"]
    if not pts:
        raise ValueError("no triangle mesh in %s" % path)

    tex, ext = None, ".png"
    if tex_index is not None:
        src = gl["textures"][tex_index].get("source")
        img = gl["images"][src] if src is not None else None
        if img and "bufferView" in img:
            tex = bytes(view(img["bufferView"])[0])
            ext = ".jpg" if "jpeg" in img.get("mimeType", "") else ".png"

    return {"points": np.concatenate(pts), "uvs": np.concatenate(uvs),
            "tris": np.concatenate(tris), "texture": tex, "texture_ext": ext}
