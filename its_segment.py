"""
its_segment  -  cut objects out of an illustration and fill the background.

Part of the Illustration to 3D Scene tool. Plain Python (numpy + OpenCV), no
Maya needed, so it can also run outside Maya.

For every box:
  * GrabCut uses the box as a hint ("the object is somewhere in here") and
    separates the object from what's around it.
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


def cut_object(img, box, iterations=5):
    """Full-image 0/255 mask of the object inside box (x, y, w, h)."""
    H, W = img.shape[:2]
    x = int(max(0, min(box["x"], W - 1)))
    y = int(max(0, min(box["y"], H - 1)))
    w = int(max(2, min(box["w"], W - x)))
    h = int(max(2, min(box["h"], H - y)))

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

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
        fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg = np.zeros(crop.shape[:2], np.uint8)

    # only keep what's inside the box (the card only covers the box)
    inside = np.zeros_like(fg)
    inside[ry:ry + h, rx:rx + w] = 255
    fg = _clean(cv2.bitwise_and(fg, inside))

    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = fg
    # GrabCut found (almost) nothing -> fall back to the whole box
    if cv2.countNonZero(full) < 0.03 * w * h:
        full[y:y + h, x:x + w] = 255
    return full


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
        m = cut_object(img, b)
        union = cv2.bitwise_or(union, m)
        own = cv2.bitwise_and(m, cv2.bitwise_not(taken))
        taken = cv2.bitwise_or(taken, m)

        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        alpha = cv2.GaussianBlur(own, (3, 3), 0)[y:y + h, x:x + w]
        rgba = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        fname = "cut_%02d_%s.png" % (i, _safe(b.get("name")))
        _write(os.path.join(out_dir, fname), rgba)
        results[i] = {"index": i, "name": b.get("name"), "file": fname,
                      "coverage": round(cv2.countNonZero(own) / float(max(1, w * h)), 3)}

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hole = cv2.dilate(union, k, iterations=2)
    _write(os.path.join(out_dir, "background.png"), _inpaint(img, hole))
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
