"""
its_mesh  -  turn a cut-out object into a puffy 3D mesh ("inflate").

Part of the Illustration to 3D Scene tool. numpy + OpenCV only, no Maya.

The object's silhouette (alpha channel of the cut-out PNG) is put on a grid.
Each point gets a height from how far it is from the edge, shaped like a
quarter circle, so the middle bulges out and the edges come down to zero,
like blowing up a balloon in the shape of the drawing. The back is a mirror
of the front, and front and back share the edge vertices, so the mesh is
closed.

inflate(alpha) returns, in the cut-out's pixel space:
    verts  (N, 3)  x right, y down (pixels), z = puff in -1..1
    quads  (M, 4)  vertex indices, front faces point toward +z (the camera)
    uvs    (N, 2)  0..1, v up (Maya style)
"""

import numpy as np
import cv2


def inflate(alpha, grid=90):
    h, w = alpha.shape[:2]
    s = float(grid) / max(h, w)
    gw, gh = max(2, int(round(w * s))), max(2, int(round(h * s)))
    small = cv2.resize(alpha, (gw, gh), interpolation=cv2.INTER_AREA)
    cells = small > 127
    if cells.sum() < 3:
        return None

    # distance from the edge -> rounded profile (quarter circle)
    pad = np.pad(cells.astype(np.uint8), 1)
    dt = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[1:-1, 1:-1]
    dt = dt / max(dt.max(), 1e-6)
    prof = np.sqrt(np.clip(1.0 - (1.0 - dt) ** 2, 0, 1)) * cells

    # corners of the cells: used if touching any inside cell,
    # height = average of the 4 cells if all inside, else 0 (the rim)
    c = np.pad(cells, 1)
    p = np.pad(prof, 1)
    tl, tr, bl, br = c[:-1, :-1], c[:-1, 1:], c[1:, :-1], c[1:, 1:]
    used = tl | tr | bl | br
    inner = tl & tr & bl & br
    height = np.where(inner, (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:]) / 4.0, 0.0)

    rows, cols = np.nonzero(used)
    front_id = -np.ones(used.shape, np.int64)
    front_id[rows, cols] = np.arange(len(rows))
    xs = cols / float(gw) * w
    ys = rows / float(gh) * h
    zs = height[rows, cols]
    verts = [np.stack([xs, ys, zs], 1)]
    uvs = [np.stack([cols / float(gw), 1.0 - rows / float(gh)], 1)]

    # back vertices only where the surface is off the rim
    back_id = front_id.copy()
    puffy = (used & (height > 0))
    br_rows, br_cols = np.nonzero(puffy)
    back_id[br_rows, br_cols] = len(rows) + np.arange(len(br_rows))
    verts.append(np.stack([br_cols / float(gw) * w, br_rows / float(gh) * h,
                           -height[br_rows, br_cols]], 1))
    uvs.append(np.stack([br_cols / float(gw), 1.0 - br_rows / float(gh)], 1))

    # faces: corner (r, c) is the top-left of cell (r, c)
    r, cc = np.nonzero(cells)
    a = (r, cc)              # top-left
    b = (r, cc + 1)          # top-right
    d2 = (r + 1, cc + 1)     # bottom-right
    d = (r + 1, cc)          # bottom-left
    # y points down in pixels, so this order faces +z once y is flipped to up
    front = np.stack([front_id[d], front_id[d2], front_id[b], front_id[a]], 1)
    back = np.stack([back_id[a], back_id[b], back_id[d2], back_id[d]], 1)
    quads = np.concatenate([front, back], 0)

    return np.concatenate(verts, 0), quads, np.concatenate(uvs, 0)


def is_closed(quads):
    """Every edge used by exactly two faces -> watertight."""
    edges = {}
    for q in quads:
        for i in range(4):
            e = tuple(sorted((int(q[i]), int(q[(i + 1) % 4]))))
            edges[e] = edges.get(e, 0) + 1
    return all(n == 2 for n in edges.values())
