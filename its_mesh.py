"""
its_mesh  -  turn a cut-out object into a puffy 3D mesh ("inflate").

Part of the Illustration to 3D Scene tool. numpy + OpenCV only, no Maya.

How it works:
  1. The object's outline is traced from the cut-out's alpha and smoothed,
     so the edge is a clean curve instead of pixel stairs.
  2. Points are spread evenly along that outline and inside the shape, and
     joined into triangles (Delaunay). Triangles outside the shape are dropped.
     Delaunay can skip an outline edge, which leaves a chipped corner, so any
     missing outline edge gets an extra point and the step is repeated.
  3. Each inside point gets a height from how far it is from the edge, shaped
     like a quarter circle, so the middle bulges out and the edge comes down
     to zero, like a balloon in the shape of the drawing. The height map is
     blurred first, so there's no sharp ridge down the middle.
  4. The back is a mirror of the front. Front and back share the outline
     points, so the mesh is closed.

inflate(alpha) returns, in the cut-out's pixel space:
    verts  (N, 3)  x right, y down (pixels), z = puff in -1..1
    faces  (M, 3)  triangles, front faces point toward +z (the camera)
    uvs    (N, 2)  0..1, v up (Maya style)
"""

import numpy as np
import cv2

WORK = 512          # the mask is processed at this size (longest side)


def _smooth_mask(alpha, scale):
    """Alpha -> clean 0/255 mask at working size: no pixel stairs, no specks."""
    h, w = alpha.shape[:2]
    m = cv2.resize(alpha, (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
                   interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (0, 0), 2.0)            # rounds off jaggies
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return m


def _resample(contour, step):
    """Smooth a closed outline and put points on it every `step` pixels."""
    c = contour.reshape(-1, 2).astype(np.float64)
    if len(c) < 8:
        return c
    # smooth along the outline (wraps around, it's closed)
    k = max(1, int(step / 2))
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    padded = np.concatenate([c[-k:], c, c[:k]], 0)
    c = np.stack([np.convolve(padded[:, i], kernel, "valid") for i in range(2)], 1)
    # even spacing
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    total = seg.sum()
    n = max(8, int(total / step))
    t = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, total, n, endpoint=False)
    closed = np.vstack([c, c[:1]])
    x = np.interp(targets, t, closed[:, 0])
    y = np.interp(targets, t, closed[:, 1])
    return np.stack([x, y], 1)


def _triangulate(points, w, h):
    """Delaunay triangles of the points (indices into points)."""
    sub = cv2.Subdiv2D((-2, -2, w + 4, h + 4))
    lookup = {}
    for i, (x, y) in enumerate(points):
        key = (round(float(x), 3), round(float(y), 3))
        if key in lookup:
            continue
        lookup[key] = i
        sub.insert((float(x), float(y)))
    tris = []
    for t in sub.getTriangleList():
        ids = []
        for j in range(3):
            key = (round(float(t[2 * j]), 3), round(float(t[2 * j + 1]), 3))
            if key not in lookup:
                break
            ids.append(lookup[key])
        if len(ids) == 3:
            tris.append(ids)
    return np.array(tris, np.int64).reshape(-1, 3)


def _split_necks(pts, tris, is_rim, inner_edges):
    """Split every inside edge whose two ends are both on the rim."""
    necks = [tuple(e) for e in inner_edges if is_rim[e[0]] and is_rim[e[1]]]
    if not necks:
        return pts, tris, is_rim
    # split at most one edge per triangle in each pass (the rest wait for the next pass)
    owner = {}
    for k, t in enumerate(tris):
        for j in range(3):
            a, b = t[j], t[(j + 1) % 3]
            owner.setdefault((min(a, b), max(a, b)), []).append(k)
    busy, chosen = set(), []
    for e in necks:
        ks = owner.get(e, [])
        if not any(k in busy for k in ks):
            busy.update(ks)
            chosen.append(e)
    necks = chosen
    mid = {}
    new_pts = [pts]
    n = len(pts)
    for a, b in necks:
        mid[(a, b)] = n
        new_pts.append(((pts[a] + pts[b]) / 2.0)[None])
        n += 1
    out = []
    for t in tris:
        for j in range(3):
            a, b, c = t[j], t[(j + 1) % 3], t[(j + 2) % 3]
            m = mid.get((min(a, b), max(a, b)))
            if m is not None:
                out += [[a, m, c], [m, b, c]]    # same winding as t
                break
        else:
            out.append(list(t))
    tris = np.array(out, np.int64)
    pts = np.vstack(new_pts)
    is_rim = np.concatenate([is_rim, np.zeros(len(necks), bool)])
    # a triangle can have two neck edges: repeat until none are left
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return _split_necks(pts, tris, is_rim, uniq[counts == 2])


def inflate(alpha, detail=60):
    """detail = roughly how many points across the object's longest side."""
    h, w = alpha.shape[:2]
    scale = float(WORK) / max(h, w)
    mask = _smooth_mask(alpha, scale)
    mh, mw = mask.shape
    if cv2.countNonZero(mask) < 20:
        return None

    step = max(3.0, max(mh, mw) / float(detail))

    # 1. outlines: outer edges and holes, smoothed, a point every `step` pixels
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    rings, shape = [], np.zeros_like(mask)
    for c, info in zip(contours, hier[0]):
        if cv2.contourArea(c) <= step * step:
            continue
        ring = _resample(c, step)
        rings.append(ring)
        # the inside test uses these smoothed outlines, so it agrees with the rim
        cv2.fillPoly(shape, [np.round(ring * 16).astype(np.int32)], 0 if info[3] >= 0 else 255,
                     lineType=cv2.LINE_8, shift=4)
    if not rings:
        return None
    mask = shape

    # 2. inside points on a staggered grid, a full step away from the edge
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    inner = []
    row = 0
    y = step / 2.0
    while y < mh:
        x = step / 2.0 + (step / 2.0 if row % 2 else 0.0)
        while x < mw:
            if dist[int(y), int(x)] > 0.9 * step:
                inner.append((x, y))
            x += step
        y += step * 0.866
        row += 1
    inner = np.array(inner, np.float64).reshape(-1, 2)

    # 3. triangles. Delaunay doesn't promise to keep the outline's edges, and where
    # one goes missing the edge gets a chipped corner. So: triangulate, check every
    # outline edge, put an extra point in the middle of the missing ones, repeat.
    def inside(p):
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, mw - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, mh - 1)
        return mask[yi, xi] > 0

    for _ in range(8):
        rim = np.vstack(rings)
        pts = np.vstack([rim, inner])
        tris = _triangulate(pts, mw, mh)
        if len(tris) == 0:
            return None
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        tris = tris[inside((a + b + c) / 3.0)]
        have = set(map(tuple, np.sort(np.vstack(
            [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1).tolist()))
        fixed, start, missing = [], 0, 0
        for ring in rings:
            n = len(ring)
            out = []
            for k in range(n):
                out.append(ring[k])
                i, j = start + k, start + (k + 1) % n
                if (min(i, j), max(i, j)) not in have:
                    out.append((ring[k] + ring[(k + 1) % n]) / 2.0)
                    missing += 1
            fixed.append(np.array(out))
            start += n
        if not missing:
            break
        rings = fixed
    if len(tris) == 0:
        return None

    # point order faces +z once y is flipped to up: make every triangle wind the same way
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = cross > 0     # same winding as the old grid mesh (negative in y-down pixels)
    tris[flip] = tris[flip][:, [0, 2, 1]]

    # drop points no triangle uses (renumber)
    used = np.zeros(len(pts), bool)
    used[tris.reshape(-1)] = True
    new_id = -np.ones(len(pts), np.int64)
    new_id[used] = np.arange(used.sum())
    pts = pts[used]
    tris = new_id[tris]

    # the mesh's open edge = edges used by only one triangle. Points on it are the
    # rim: they sit at z = 0 and are shared by front and back, which closes the mesh.
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    is_rim = np.zeros(len(pts), bool)
    is_rim[uniq[counts == 1].reshape(-1)] = True
    # an inside edge between two rim points (a thin neck) would be shared by front
    # and back too, which breaks the mesh: split it with a new point in the middle
    pts, tris, is_rim = _split_necks(pts, tris, is_rim, uniq[counts == 2])

    # 4. height: blurred distance from the edge, quarter-circle profile
    soft = cv2.GaussianBlur(dist, (0, 0), max(1.0, step * 0.8))
    soft = soft / max(float(soft.max()), 1e-6)
    xi = np.clip(pts[:, 0].astype(int), 0, mw - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, mh - 1)
    t = np.clip(soft[yi, xi], 0.0, 1.0)
    z = np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0))
    z[is_rim] = 0.0

    # 5. front + mirrored back, sharing the rim points
    n = len(pts)
    back_id = np.arange(n)
    off = np.nonzero(~is_rim)[0]
    back_id[off] = n + np.arange(len(off))
    verts_px = np.vstack([pts, pts[off]]) / scale       # back to cut-out pixels
    zs = np.concatenate([z, -z[off]])
    verts = np.column_stack([verts_px, zs])
    uv = np.column_stack([verts_px[:, 0] / w, 1.0 - verts_px[:, 1] / h])
    back = back_id[tris][:, [0, 2, 1]]                  # reversed: faces -z
    faces = np.vstack([tris, back])
    return verts, faces, uv


def is_closed(faces):
    """Every edge used by exactly two faces -> watertight."""
    edges = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            e = tuple(sorted((int(f[i]), int(f[(i + 1) % k]))))
            edges[e] = edges.get(e, 0) + 1
    return all(n == 2 for n in edges.values())
