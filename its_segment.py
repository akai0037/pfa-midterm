"""
its_segment  -  cut objects out of an illustration and fill the background.

Part of the Illustration to 3D Scene tool. Plain Python (numpy + OpenCV), no
Maya needed, so it can also run outside Maya.

For every box:
  * GrabCut uses the box as a hint ("the object is somewhere in here") and
    separates the object from what's around it.
  * Optional brush strokes on the box correct it: keep (always the object),
    remove (never the object) and fill (hidden behind something; painted in).
  * Holes inside the object are filled (e.g. a house's windows stay).
  * Pixels that belong to a NEARER object are removed, so a far object
    doesn't carry a copy of whatever stands in front of it.
  * Result: <image>_its/cut_XX_<name>.png  (box-sized crop with alpha)

Then every cut-out area is painted over (inpainted) in a copy of the drawing:
  <image>_its/background.png
so when the camera moves you don't see a second copy of the objects.

Command line:
    python its_segment.py drawing.png drawing_boxes.json
"""

import os
import re
import sys
import json

import numpy as np
import cv2


def _read(path):
    """cv2.imread that also works with non-ASCII Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError("Can't read image: %s" % path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # composite transparent drawings over white
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    return img


def _write(path, img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError("Can't write %s" % path)
    buf.tofile(path)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name or "object") or "object"


def _fill_holes(mask):
    """Fill closed holes inside a 0/255 mask."""
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    flood[1:-1, 1:-1] = mask
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood[1:-1, 1:-1])
    return cv2.bitwise_or(mask, holes)


def _clean(mask):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    # drop little specks, keep the real parts
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= max(30, 0.12 * areas.max())
        mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    return _fill_holes(mask)


def stroke_mask(shape, strokes, mode):
    """0/255 mask of the brush strokes of one kind ("keep", "remove" or "fill")."""
    m = np.zeros(shape[:2], np.uint8)
    for st in strokes or []:
        if st.get("mode") != mode:
            continue
        r = max(1, int(round(st.get("r", 10))))
        pts = [(int(round(px)), int(round(py))) for px, py in st.get("pts", [])]
        for a, b in zip(pts, pts[1:]):
            cv2.line(m, a, b, 255, 2 * r)
        for p in pts:                       # round ends and joints
            cv2.circle(m, p, r, 255, -1)
    return m


def cut_object(img, box, iterations=5):
    """Mask of the object inside box (x, y, w, h), plus the "fill hidden" area.

    The box can carry brush strokes (box["strokes"]) that correct GrabCut:
      keep   - always part of the object (things GrabCut missed)
      remove - never part of the object (things that touch it but aren't it)
      fill   - part of the object that's hidden behind something else;
               it gets painted in from the object's own colors later
    Returns (mask, fill) as full-image 0/255 masks; mask doesn't include fill.
    """
    H, W = img.shape[:2]
    x = int(max(0, min(box["x"], W - 1)))
    y = int(max(0, min(box["y"], H - 1)))
    w = int(max(2, min(box["w"], W - x)))
    h = int(max(2, min(box["h"], H - y)))
    strokes = box.get("strokes") or []
    keep = stroke_mask(img.shape, strokes, "keep")
    remove = stroke_mask(img.shape, strokes, "remove")
    fill = stroke_mask(img.shape, strokes, "fill")
    inside_full = np.zeros((H, W), np.uint8)
    inside_full[y:y + h, x:x + w] = 255
    keep = cv2.bitwise_and(keep, inside_full)
    fill = cv2.bitwise_and(fill, inside_full)
    remove = cv2.bitwise_and(remove, cv2.bitwise_not(fill))   # where they overlap, "hidden part" wins

    # work on the box plus some margin (faster, and gives GrabCut background samples)
    pad = int(0.15 * max(w, h)) + 10
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1]

    gc = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    rx, ry = x - x0, y - y0
    gc[ry:ry + h, rx:rx + w] = cv2.GC_PR_FGD
    if not (gc == cv2.GC_BGD).any():
        # box covers the whole picture: treat its rim as probably-background
        gc[:2, :] = gc[-2:, :] = cv2.GC_PR_BGD
        gc[:, :2] = gc[:, -2:] = cv2.GC_PR_BGD
    # the user's strokes are certain: they teach GrabCut what is / isn't the object.
    # A "fill" area is covered by something else, so what shows there now isn't the
    # object: GrabCut shouldn't learn its colors, and it all gets painted in later
    gc[remove[y0:y1, x0:x1] > 0] = cv2.GC_BGD
    gc[fill[y0:y1, x0:x1] > 0] = cv2.GC_PR_BGD
    gc[keep[y0:y1, x0:x1] > 0] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
        fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg = np.zeros(crop.shape[:2], np.uint8)

    # only keep what's inside the box (the card only covers the box)
    fg = _clean(cv2.bitwise_and(fg, inside_full[y0:y1, x0:x1]))

    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = fg
    # GrabCut found (almost) nothing -> fall back to the whole box
    if cv2.countNonZero(full) < 0.03 * w * h and not keep.any():
        full[y:y + h, x:x + w] = 255
    # strokes win over everything (also over hole filling in _clean)
    full = cv2.bitwise_or(full, keep)
    full = cv2.bitwise_and(full, cv2.bitwise_not(remove))
    full = cv2.bitwise_and(full, cv2.bitwise_not(cv2.bitwise_and(fill, cv2.bitwise_not(keep))))

    # a "fill" stroke can just be the outline of the hidden part: whatever it closes
    # off together with the visible object counts as hidden too
    if fill.any():
        both = cv2.bitwise_or(full, fill)
        enclosed = cv2.bitwise_and(_fill_holes(both), cv2.bitwise_not(both))
        enclosed = cv2.bitwise_and(enclosed, inside_full)
        fill = cv2.bitwise_or(fill, enclosed)
        fill = cv2.bitwise_and(fill, cv2.bitwise_not(full))   # only what's really hidden
    return full, fill


def main_color(rgb, known, near, k=4):
    """The object's main color next to the hidden part: the visible object pixels
    around it are sorted into k color groups, and the biggest group wins. So outlines,
    windows and highlights don't leak in; on the UFO this is its pink."""
    pix = rgb[known & near].reshape(-1, 3).astype(np.float32)
    if len(pix) < k * 10:
        pix = rgb[known].reshape(-1, 3).astype(np.float32)
    if len(pix) < k:
        return np.median(pix, 0) if len(pix) else np.array([200, 200, 200], np.float32)
    if len(pix) > 20000:                    # plenty for picking a color, and faster
        pix = pix[np.random.RandomState(0).choice(len(pix), 20000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)                       # same answer every time
    _, labels, centers = cv2.kmeans(pix, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centers[np.bincount(labels.ravel(), minlength=k).argmax()]


def complete_hidden(rgb, known, fill):
    """Paint the hidden part of an object (fill) in the object's main color, with
    the seam blended a few pixels into the neighbors so it doesn't look cut out."""
    if not fill.any() or not known.any():
        return rgb
    # look at the visible object within ~30 px of the hidden part, a few pixels in
    # from its edge (the anti-aliased edge still has a bit of what was next to it)
    inner = cv2.erode(known.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if inner.sum() > 50:
        known = inner
    near = cv2.dilate(fill.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    color = main_color(rgb, known, near)

    out = rgb.copy()
    out[fill] = np.clip(color, 0, 255).astype(np.uint8)
    # soften only a thin seam where the painted part meets the visible object
    seam = cv2.dilate(fill.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    blurred = cv2.GaussianBlur(out, (5, 5), 0)
    edge = seam & ~cv2.erode(fill.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out[edge] = blurred[edge]
    return out


def _inpaint(img, hole):
    """Paint over the hole. Big pictures are filled at lower res, then blended in."""
    H, W = img.shape[:2]
    scale = min(1.0, 900.0 / max(H, W))
    if scale < 1.0:
        small = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        small_hole = cv2.resize(hole, small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        filled = cv2.inpaint(small, small_hole, 7, cv2.INPAINT_TELEA)
        filled = cv2.resize(filled, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        filled = cv2.inpaint(img, hole, 7, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(hole, (9, 9), 0).astype(np.float32)[:, :, None] / 255.0
    return (filled * soft + img * (1 - soft)).astype(np.uint8)


def segment(image_path, boxes, out_dir=None, progress=None):
    """Cut out every box. Returns a manifest dict (also saved as manifest.json)."""
    img = _read(image_path)
    H, W = img.shape[:2]
    if out_dir is None:
        out_dir = os.path.splitext(image_path)[0] + "_its"
    os.makedirs(out_dir, exist_ok=True)

    # nearest first: nearer objects claim their pixels first
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].get("dist", 0))
    taken = np.zeros((H, W), np.uint8)
    union = np.zeros((H, W), np.uint8)
    results = {}
    for n, i in enumerate(order):
        b = boxes[i]
        if progress:
            progress(n, len(boxes), b.get("name", ""))
        m, fill = cut_object(img, b)
        # the background loses the object AND its hidden part (that area often still
        # shows bits of the object); the background fill paints both over
        union = cv2.bitwise_or(union, cv2.bitwise_or(m, fill))
        own = cv2.bitwise_and(m, cv2.bitwise_not(taken))
        # a nearer object sitting fully inside this one (a window on a UFO) would punch
        # a hole through it. That part is hidden, not missing: keep it, painted in
        covered = cv2.bitwise_and(cv2.bitwise_and(_fill_holes(own), cv2.bitwise_not(own)), taken)
        fill = cv2.bitwise_or(fill, covered)
        own = cv2.bitwise_or(own, fill)            # ...but belongs to this object's card
        taken = cv2.bitwise_or(taken, m)

        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        rgb = img[y:y + h, x:x + w].copy()
        f = fill[y:y + h, x:x + w] > 0
        if f.any():
            rgb = complete_hidden(rgb, (own[y:y + h, x:x + w] > 0) & ~f, f)
        alpha = cv2.GaussianBlur(own, (3, 3), 0)[y:y + h, x:x + w]
        rgba = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        fname = "cut_%02d_%s.png" % (i, _safe(b.get("name")))
        _write(os.path.join(out_dir, fname), rgba)
        results[i] = {"index": i, "name": b.get("name"), "file": fname,
                      "coverage": round(cv2.countNonZero(own) / float(max(1, w * h)), 3)}

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hole = cv2.dilate(union, k, iterations=2)
    _write(os.path.join(out_dir, "background.png"), _inpaint(img, hole))
    _write(os.path.join(out_dir, "background_hole.png"), cv2.dilate(hole, k, iterations=1))
    _write(os.path.join(out_dir, "masks_preview.png"), union)

    manifest = {"image": os.path.basename(image_path), "width": W, "height": H,
                "background": "background.png",
                "cutouts": [results[i] for i in range(len(boxes))]}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["dir"] = out_dir
    return manifest


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[2], "r", encoding="utf-8") as f:
        data = json.load(f)
    m = segment(sys.argv[1], data["boxes"])
    print("Saved %d cutouts + background to %s" % (len(m["cutouts"]), m["dir"]))
