"""
its_lama  -  AI background fill with LaMa ("Resolution-robust Large Mask Inpainting").

Part of the Illustration to 3D Scene tool. Where objects were cut out of the
drawing, LaMa paints in background that fits the picture (sky, gradients, the
shapes around the hole), so objects look like they came straight out of it.

Runs OUTSIDE Maya (the tool starts it with mayapy as a separate process) on the
computer's CPU. The model (Carve/LaMa-ONNX on Hugging Face, Apache-2.0, ~200 MB)
is downloaded once.

    mayapy its_lama.py <image> <hole mask> <out.png> --model <lama.onnx> [--site <dir>;<dir>]
    mayapy its_lama.py --download --model <lama.onnx>

Prints progress lines; the last line is "DONE <path>" or "ERROR <message>".
"""

import os
import sys

MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
SIZE = 512          # the model works on 512 x 512 pictures


def log(msg):
    print(msg, flush=True)


def download(path):
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    log("Downloading the LaMa model (~200 MB, one time) ...")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, step = 0, max(1, total // 10)
        nxt = step
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done >= nxt:
                log("  %d%%" % (done * 100 // total))
                nxt += step
    if total and os.path.getsize(tmp) != total:
        raise IOError("download incomplete")
    os.replace(tmp, path)


def _read(path, flags):
    import numpy as np
    import cv2
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if img is None:
        raise IOError("can't read %s" % path)
    return img


def inpaint(img, hole, session):
    """Fill hole (0/255) in img (BGR uint8). Works on a square crop around the hole,
    so small holes keep detail; only the hole itself is replaced."""
    import numpy as np
    import cv2
    H, W = hole.shape[:2]
    ys, xs = np.nonzero(hole)
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0) * 1.6) + 32       # the hole plus context around it
    side = min(side, max(H, W))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0 = int(min(max(cx - side // 2, 0), max(W - side, 0)))
    sy0 = int(min(max(cy - side // 2, 0), max(H - side, 0)))
    sx1, sy1 = min(W, sx0 + side), min(H, sy0 + side)
    crop = img[sy0:sy1, sx0:sx1]
    m = hole[sy0:sy1, sx0:sx1]
    ch, cw = crop.shape[:2]

    small = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    small_m = (cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
    rgb = small[:, :, ::-1].astype(np.float32) / 255.0
    feeds = {}
    for inp in session.get_inputs():
        channels = inp.shape[1] if len(inp.shape) > 1 and isinstance(inp.shape[1], int) else None
        if channels == 1 or "mask" in inp.name.lower():
            feeds[inp.name] = small_m[None, None]
        else:
            feeds[inp.name] = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    out = session.run(None, feeds)[0][0].transpose(1, 2, 0)
    if out.max() <= 1.5:                                    # some exports give 0..1
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)[:, :, ::-1]
    out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_CUBIC)

    soft = cv2.GaussianBlur(m, (7, 7), 0).astype(np.float32)[:, :, None] / 255.0
    result = img.copy()
    result[sy0:sy1, sx0:sx1] = (out * soft + crop * (1 - soft)).astype(np.uint8)
    return result


def main():
    args = sys.argv[1:]
    model = None
    if "--site" in args:
        i = args.index("--site")
        for folder in args[i + 1].split(os.pathsep):
            sys.path.insert(0, folder)
        del args[i:i + 2]
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i:i + 2]
    if not model:
        log("ERROR no --model path")
        return 1
    try:
        if not os.path.exists(model):
            download(model)
        if "--download" in args:
            log("DONE " + model)
            return 0
        src, hole_path, out = args[0], args[1], args[2]
        import numpy as np  # noqa: F401
        import cv2
        import onnxruntime as ort
        log("Filling the background with LaMa ...")
        session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        img = _read(src, cv2.IMREAD_COLOR)
        hole = _read(hole_path, cv2.IMREAD_GRAYSCALE)
        if hole.shape[:2] != img.shape[:2]:
            hole = cv2.resize(hole, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        result = inpaint(img, hole, session)
        ok, buf = cv2.imencode(".png", result)
        buf.tofile(out)
        log("DONE " + out)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        text = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
        log("ERROR %s: %s" % (type(e).__name__, text))
        return 1


if __name__ == "__main__":
    sys.exit(main())
