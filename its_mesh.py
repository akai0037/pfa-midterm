"""
its_mesh  -  turn a cut-out object into a puffy 3D mesh ("inflate").

Part of the Illustration to 3D Scene tool. numpy + OpenCV only, no Maya.

The mesh is all quads, laid out the way you'd model it by hand:
  1. The object's outline is traced from the cut-out's alpha and smoothed.
  2. Inside it, a regular grid of square cells (only cells that sit fully
     inside, a little away from the edge).
  3. One ring of quads joins the edge of that grid to the outline: every
     grid point on the border gets a matching point on the outline. The
     border points are rounded off a little first, so the grid ends in a
     smooth line instead of stairs.
  4. Each point gets a height from how far it is from the edge, shaped like a
     quarter circle, so the middle bulges out and the edge comes down to zero,
     like a balloon in the shape of the drawing.
  5. The back is a mirror of the front. Front and back share the outline
     points, so the mesh is closed.

inflate(alpha) returns, in the cut-out's pixel space:
    verts  (N, 3)  x right, y down (pixels), z = puff in -1..1
    faces  (M, 4)  quads, front faces point toward +z (the camera)
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
    return np.where(m > 127, 255, 0).astype(np.uint8)


def _outline(contour, smooth):
    """Smoothed closed outline with a point every pixel or so.
    Returns the points (N, 2), the distance along the outline at each, the length."""
    c = contour.reshape(-1, 2).astype(np.float64)
    k = max(1, int(smooth))
    if len(c) > 2 * k + 2:
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        padded = np.concatenate([c[-k:], c, c[:k]], 0)
        c = np.stack([np.convolve(padded[:, i], kernel, "valid") for i in range(2)], 1)
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    return c, t[:-1], t[-1]


def _point_at(ring, t, length, s):
    """Point on a closed outline at distance s along it."""
    s = np.mod(s, length)
    closed = np.vstack([ring, ring[:1]])
    tt = np.append(t, length)
    return np.stack([np.interp(s, tt, closed[:, 0]), np.interp(s, tt, closed[:, 1])], -1)


def _pick_cells(inside):
    """Cells to fill with quads. Where two cells touch only at a corner, the
    mesh would pinch to a single point there, so one more cell is added."""
    cells = inside.copy()
    for _ in range(10):
        c = np.pad(cells, 1)
        tl, tr, bl, br = c[:-1, :-1], c[:-1, 1:], c[1:, :-1], c[1:, 1:]
        pinch = (tl & br & ~tr & ~bl) | (tr & bl & ~tl & ~br)
        if not pinch.any():
            break
        # pinch[r, k] is the grid corner shared by cells (r-1, k-1), (r-1, k), (r, k-1), (r, k)
        for r, k in zip(*np.nonzero(pinch)):
            if tl[r, k]:            # top-left + bottom-right -> add top-right
                cells[r - 1, k] = True
            else:                   # top-right + bottom-left -> add top-left
                cells[r - 1, k - 1] = True
    return cells


def _quad_part(shape, rings, s, strict=True):
    """Quads for one piece of the shape (a 0/255 mask) and its outlines.
    Returns (points, quads, is_rim), or None if the grid doesn't fit (strict=False
    accepts a few twisted border quads instead of giving up)."""
    mh, mw = shape.shape
    dist = cv2.distanceTransform(shape, cv2.DIST_L2, 5)

    # grid cells whose middle is at least about half a cell inside the outline
    gw, gh = int(np.ceil(mw / s)) + 1, int(np.ceil(mh / s)) + 1
    cy, cx = np.mgrid[0:gh, 0:gw]
    px = np.clip(((cx + 0.5) * s).astype(int), 0, mw - 1)
    py = np.clip(((cy + 0.5) * s).astype(int), 0, mh - 1)
    inside = dist[py, px] > 0.55 * s
    cells = _pick_cells(inside)
    if cells.sum() < 1:
        return None               # too thin for this cell size

    # grid corners used by the cells -> vertex ids
    used = np.zeros((gh + 1, gw + 1), bool)
    r, c = np.nonzero(cells)
    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        used[r + dr, c + dc] = True
    vid = -np.ones(used.shape, np.int64)
    vr, vc = np.nonzero(used)
    vid[vr, vc] = np.arange(len(vr))
    pts = np.stack([vc * s, vr * s], 1).astype(np.float64)

    # cell quads in the order bottom-left, bottom-right, top-right, top-left
    # (y is down here, so this faces the camera once y is flipped up in Maya)
    quads = np.stack([vid[r + 1, c], vid[r + 1, c + 1], vid[r, c + 1], vid[r, c]], 1)

    # border of the grid: directed edges with no partner going the other way,
    # chained into closed loops
    edges = set()
    for q in quads:
        for k in range(4):
            edges.add((int(q[k]), int(q[(k + 1) % 4])))
    nxt = {a: b for (a, b) in edges if (b, a) not in edges}
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        loops.append(loop)
    # the grid must have one border loop per outline (outer edge + one per hole);
    # if a thin part split the grid in two, the caller tries smaller cells
    if len(loops) != len(rings):
        return None

    # round off the stairs: each border point moves halfway toward the middle of its
    # two neighbors along the border (the inside of the grid stays a perfect grid)
    for _ in range(3):
        for loop in loops:
            L = np.array(loop)
            P = pts[L]
            pts[L] = 0.5 * P + 0.25 * (np.roll(P, 1, 0) + np.roll(P, -1, 0))

    # each border loop gets a twin loop on the outline, one outline point per
    # grid point, and a quad joins every border edge to its twin
    rim_pts, strip = [], []
    n_grid = len(pts)
    for loop in loops:
        L = np.array(loop)
        P = pts[L]
        # which outline this loop follows, and the nearest spot on it for every point
        best = None
        for (ring, t, length) in rings:
            d = np.linalg.norm(P[:, None, :] - ring[None, ::2, :], axis=2)
            j = d.argmin(1)
            score = d[np.arange(len(P)), j].sum()
            if best is None or score < best[0]:
                best = (score, ring, t, length, t[::2][j])
        _, ring, t, length, pos = best
        # walk along the outline in the loop's direction, never backwards,
        # go round exactly once, and even out the spacing a little
        steps = np.mod(np.diff(np.append(pos, pos[0])), length)
        sign = 1.0
        if np.median(steps) > length / 2:            # the loop runs the other way round
            steps = np.mod(-np.diff(np.append(pos, pos[0])), length)
            sign = -1.0
        steps = np.where(steps > length / 2, 0.0, steps)
        steps = np.maximum(steps, 0.05 * s)
        steps *= length / steps.sum()
        for _ in range(3):
            steps = 0.5 * steps + 0.25 * (np.roll(steps, 1) + np.roll(steps, -1))
        n = len(L)
        # where a joining quad comes out twisted (its outline side runs backwards),
        # even out the spacing around it more, and try again
        for _ in range(20):
            along = pos[0] + sign * np.concatenate([[0.0], np.cumsum(steps)[:-1]])
            on_curve = _point_at(ring, t, length, along)
            quad = np.stack([P[(np.arange(n) + 1) % n], P, on_curve,
                             on_curve[(np.arange(n) + 1) % n]], 1)
            x, y = quad[..., 0], quad[..., 1]
            # (signed area; cell quads are negative, a clearly positive one is twisted)
            twisted = (x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1) > 0.04 * s * s
            if not twisted.any():
                break
            spread = twisted | np.roll(twisted, 1) | np.roll(twisted, -1)
            spread = spread | np.roll(spread, 1) | np.roll(spread, -1)
            smoothed = (np.roll(steps, 1) + steps + np.roll(steps, -1)) / 3.0
            steps = np.where(spread, smoothed, steps)
        if twisted.any() and strict:
            return None           # this cell size doesn't fit the shape; try smaller cells
        base = n_grid + sum(len(x) for x in rim_pts)
        rim_pts.append(on_curve)
        for k in range(n):
            a, b = L[k], L[(k + 1) % n]              # grid border edge a -> b
            a2, b2 = base + k, base + (k + 1) % n    # its twin on the outline
            strip.append([b, a, a2, b2])             # uses a-b the other way round, like a neighbor cell

    pts = np.vstack([pts] + rim_pts)
    faces = np.vstack([quads, np.array(strip, np.int64)])
    return pts, faces, np.arange(len(pts)) >= n_grid



def inflate(alpha, detail=40):
    """detail = roughly how many quads across the object's longest side."""
    h, w = alpha.shape[:2]
    scale = float(WORK) / max(h, w)
    mask = _smooth_mask(alpha, scale)
    mh, mw = mask.shape
    if cv2.countNonZero(mask) < 20:
        return None
    s = max(4.0, max(mh, mw) / float(detail))          # cell size (working pixels)

    # 1. smoothed outlines (outer edges and holes) -> a mask that matches them exactly
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    shape = np.zeros_like(mask)
    for c, info in zip(contours, hier[0]):
        if cv2.contourArea(c) < 4:
            continue
        ring, _, _ = _outline(c, s / 3.0)
        cv2.fillPoly(shape, [np.round(ring * 16).astype(np.int32)], 0 if info[3] >= 0 else 255,
                     lineType=cv2.LINE_8, shift=4)
    dist = cv2.distanceTransform(shape, cv2.DIST_L2, 5)

    # 2 + 3. quads for each separate piece; a small or thin piece gets smaller quads
    n_pieces, labels = cv2.connectedComponents(shape, connectivity=8)
    all_pts, all_faces, all_rim = [], [], []
    count = 0
    for k in range(1, n_pieces):
        piece = np.where(labels == k, 255, 0).astype(np.uint8)
        if cv2.countNonZero(piece) < 12:
            continue
        cs, hs = cv2.findContours(piece, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        rings = [_outline(c, 1) for c in cs if len(c) >= 4]
        if not rings:
            continue
        res = None
        for size in (s, s / 1.5, s / 2.0, s / 3.0, s / 4.0, s / 6.0):
            res = _quad_part(piece, rings, max(1.5, size))
            if res is not None:
                break
        if res is None:
            res = _quad_part(piece, rings, s, strict=False)
        if res is None:
            continue              # thinner than the smallest cells: left out
        p, f, rim = res
        all_pts.append(p)
        all_faces.append(f + count)
        all_rim.append(rim)
        count += len(p)
    if not all_pts:
        return None
    pts = np.vstack(all_pts)
    faces = np.vstack(all_faces)
    is_rim = np.concatenate(all_rim)

    # 4. height: blurred distance from the edge, quarter-circle profile
    soft = cv2.GaussianBlur(dist, (0, 0), max(1.0, s * 0.8))
    soft = soft / max(float(soft.max()), 1e-6)
    xi = np.clip(pts[:, 0].astype(int), 0, mw - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, mh - 1)
    tt = np.clip(soft[yi, xi], 0.0, 1.0)
    z = np.sqrt(np.clip(1.0 - (1.0 - tt) ** 2, 0.0, 1.0))
    z[is_rim] = 0.0

    # 5. front + mirrored back, sharing the outline points
    n = len(pts)
    back_id = np.arange(n)
    off = np.nonzero(~is_rim)[0]
    back_id[off] = n + np.arange(len(off))
    verts_px = np.vstack([pts, pts[off]]) / scale       # back to cut-out pixels
    zs = np.concatenate([z, -z[off]])
    verts = np.column_stack([verts_px, zs])
    uv = np.column_stack([verts_px[:, 0] / w, 1.0 - verts_px[:, 1] / h])
    back = back_id[faces][:, ::-1]                      # reversed: faces -z
    return verts, np.vstack([faces, back]), uv


def is_closed(faces):
    """Every edge used by exactly two faces -> watertight."""
    edges = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            e = tuple(sorted((int(f[i]), int(f[(i + 1) % k]))))
            edges[e] = edges.get(e, 0) + 1
    return all(n == 2 for n in edges.values())
