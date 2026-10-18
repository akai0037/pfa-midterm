"""
its_ai3d  -  make a real 3D model from one cut-out object with a free online
image-to-3D model on Hugging Face (TRELLIS, falling back to Hunyuan3D-2).

Runs OUTSIDE Maya (the tool starts it with mayapy as a separate process), so a
slow or failed request never freezes or crashes Maya.

    mayapy its_ai3d.py <cutout.png> <out.glb> [--site <folder>;<folder>...]

Prints progress lines; the last line is "DONE <path>" or "ERROR <message>".

Free GPU time on Hugging Face is limited per day. Logged-in users get more:
make a free account, create a token (huggingface.co/settings/tokens), and
either set the HF_TOKEN environment variable or save the token in
~/.cache/huggingface/token.
"""

import os
import sys
import shutil
import inspect
import tempfile

SPACES = ["trellis-community/TRELLIS", "tencent/Hunyuan3D-2"]

# endpoints to try, in order, for each kind of Space
GENERATE = ["/generate_and_extract_glb", "/shape_generation", "/generation_all", "/image_to_3d"]


def log(msg):
    print(msg, flush=True)


def square_png(src, dst, margin=0.08):
    """Center the cut-out on a white square canvas."""
    import numpy as np
    import cv2
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = img.shape[:2]
    side = int(max(h, w) * (1 + 2 * margin))
    # the object on plain white: what image-to-3D models are trained on
    # (a transparent background can be read as part of a flat picture)
    canvas = np.full((side, side, 3), 255, np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    a = img[:, :, 3:4].astype(np.float32) / 255.0
    canvas[y:y + h, x:x + w] = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    ok, buf = cv2.imencode(".png", canvas)
    buf.tofile(dst)
    return dst


def find_glb(result):
    """Dig a .glb file path out of whatever a Space returned."""
    if isinstance(result, str):
        return result if result.lower().endswith((".glb", ".gltf")) and os.path.exists(result) else None
    if isinstance(result, dict):
        for v in result.values():
            p = find_glb(v)
            if p:
                return p
    if isinstance(result, (list, tuple)):
        for v in result:
            p = find_glb(v)
            if p:
                return p
    return None


def make_client(space):
    from gradio_client import Client
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    params = inspect.signature(Client.__init__).parameters
    kw = {}
    if token:
        kw["hf_token" if "hf_token" in params else "token"] = token
    if "verbose" in params:
        kw["verbose"] = False
    log("Connecting to %s%s ..." % (space, " (logged in)" if token else " (anonymous)"))
    return Client(space, **kw)


def endpoints(client):
    info = client.view_api(return_format="dict", print_info=False)
    return info.get("named_endpoints", {})


def fill_args(params, image):
    """The main image input gets our picture; extra image inputs (multi-view front/back/...)
    and galleries stay empty; everything else keeps the Space's own defaults."""
    from gradio_client import handle_file
    names = [p.get("parameter_name") for p in params]
    image_params = [p.get("parameter_name") for p in params
                    if (p.get("component") or "").lower() == "image"]
    main = next((n for n in ("image", "image_prompt", "input_image") if n in names),
                image_params[0] if image_params else None)
    kwargs = {}
    for p in params:
        name = p.get("parameter_name")
        comp = (p.get("component") or "").lower()
        if name == main:
            kwargs[name] = handle_file(image)
        elif comp == "gallery":
            kwargs[name] = []
        elif not p.get("parameter_has_default", False):
            kwargs[name] = None
    return kwargs


def run_space(space, image):
    client = make_client(space)
    eps = endpoints(client)

    # some Spaces (TRELLIS) set up a per-user folder when the page loads
    if "/start_session" in eps:
        try:
            client.predict(api_name="/start_session")
        except Exception as e:
            log("  (start_session: %s)" % str(e).splitlines()[0][:200] if str(e) else "")

    # TRELLIS cleans up the picture first (crop / background) in a separate step
    if "/preprocess_image" in eps:
        log("Preparing the picture ...")
        try:
            out = client.predict(**fill_args(eps["/preprocess_image"]["parameters"], image),
                                 api_name="/preprocess_image")
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, dict):
                out = out.get("path") or out.get("value")
            if isinstance(out, str) and os.path.exists(out):
                image = out
        except Exception as e:
            log("  (preprocess skipped: %s)" % str(e).splitlines()[0][:200])

    problems = []
    for api in GENERATE:
        if api not in eps:
            continue
        log("Generating 3D with %s %s (usually 1-3 minutes) ..." % (space, api))
        try:
            result = client.predict(**fill_args(eps[api]["parameters"], image), api_name=api)
            glb = find_glb(result)
            if not glb and api == "/image_to_3d" and "/extract_glb" in eps:   # older TRELLIS
                log("Extracting the mesh ...")
                glb = find_glb(client.predict(**fill_args(eps["/extract_glb"]["parameters"], image),
                                              api_name="/extract_glb"))
            if glb:
                return glb
            problems.append("%s returned no model" % api)
        except Exception as e:
            text = str(e).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__
            log("  %s: %s" % (api, text))
            problems.append(text)   # keep going: a lighter endpoint may still fit the quota
    if problems:
        raise RuntimeError(problems[0])
    raise RuntimeError("%s has no image-to-3D endpoint this script understands" % space)


def main():
    args = sys.argv[1:]
    if "--site" in args:
        i = args.index("--site")
        for folder in args[i + 1].split(os.pathsep):
            sys.path.insert(0, folder)
        del args[i:i + 2]
    if len(args) < 2:
        log(__doc__)
        return 1
    src, out = args[0], args[1]
    try:
        import gradio_client  # noqa: F401
    except ImportError:
        log("ERROR gradio_client is not installed (click the install button in the tool)")
        return 1

    tmp = os.path.join(tempfile.mkdtemp(prefix="its_ai_"), "object.png")
    square_png(src, tmp)
    errors = []
    for space in SPACES:
        try:
            glb = run_space(space, tmp)
            shutil.copyfile(glb, out)
            log("DONE " + out)
            return 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            text = str(e).strip()
            msg = "%s: %s" % (type(e).__name__, text.splitlines()[0][:300]) if text else type(e).__name__
            log("  %s failed: %s" % (space, msg))
            errors.append("%s: %s" % (space.split("/")[-1], msg))
    log("ERROR " + " | ".join(errors))
    return 1


if __name__ == "__main__":
    sys.exit(main())
