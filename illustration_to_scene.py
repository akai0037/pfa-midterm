"""
Illustration to 3D Scene  -  a Maya 2026 tool
Programming for Animators midterm (Tong / akai0037)

Turns a 2D illustration into a simple 2.5D scene you can animate:
  1. Load an illustration.
  2. Drag boxes around the objects in it (or load boxes from a JSON file).
  3. Give each box a name and a distance from the camera.
  4. Build: a camera that matches the drawing, one proxy card per box placed
     at its distance, and the illustration projected onto the cards, so the
     camera view looks like your drawing. Move the camera a little and you
     get parallax.

Install: drag drag_into_maya.py (in the same folder) into Maya's viewport.
Or open it from the Script Editor (Python tab), with this folder on sys.path:
    import illustration_to_scene
    illustration_to_scene.show()
"""

import os
import re
import sys
import json
import traceback
import subprocess

import maya.cmds as cmds
import maya.OpenMayaUI as omui

# OpenCV (for cutting objects out) gets installed under here, outside Maya's own folder.
# Every install goes into its own new sub-folder (cv_<time>), so a copy that Maya has
# already loaded (and Windows has locked) never has to be overwritten.
SITE_ROOT = os.path.join(cmds.internalVar(userAppDir=True), "its_site-packages")


def _pick_site_dir():
    """Newest folder that has OpenCV in it; older ones are cleaned up when possible."""
    import glob
    import shutil
    found = sorted((d for d in glob.glob(os.path.join(SITE_ROOT, "cv_*"))
                    if os.path.isdir(os.path.join(d, "cv2"))), reverse=True)
    for old in found[1:]:
        shutil.rmtree(old, ignore_errors=True)     # skipped silently if Windows has it locked
    if found:
        return found[0]
    return SITE_ROOT                                # first version of the tool installed here


SITE_DIR = _pick_site_dir()
if os.path.isdir(SITE_DIR) and SITE_DIR not in sys.path:
    sys.path.insert(0, SITE_DIR)
_cv_error = ""

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from shiboken6 import wrapInstance
except ImportError:  # older Maya
    from PySide2 import QtCore, QtGui, QtWidgets
    from shiboken2 import wrapInstance


TOOL_TITLE = "Illustration to 3D Scene"
ROOT_GRP = "illustrationScene_grp"
NODE_PREFIX = "ITS_"            # every shading node the tool makes starts with this
FILM_APERTURE = 1.41732          # inches (36 mm, Maya's default film back width)
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"

PROXY_TYPES = ["Textured cards (2.5D)", "Colored cubes (layout check)",
               "Inflated 3D (local)", "AI 3D models (use generated)"]
P_CARDS, P_CUBES, P_INFLATE, P_AI = range(4)
DEFAULT_THICK = 20.0   # cube thickness, % of the box's shorter side

# one color per box, shared by the cubes and the Preview Cutouts tint
CUBE_COLORS = [(255, 90, 90), (90, 220, 120), (90, 160, 255), (255, 200, 60),
               (220, 110, 255), (60, 220, 220), (255, 140, 60), (180, 230, 90)]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)


def safe_name(name):
    """Turn whatever the user typed into a legal Maya node name."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip()) or "object"
    if name[0].isdigit():
        name = "obj_" + name
    return name


def boxes_json_path(image_path):
    return os.path.splitext(image_path)[0] + "_boxes.json"


# --- OpenCV / cut-outs -------------------------------------------------------

def mayapy_path():
    loc = os.environ.get("MAYA_LOCATION", os.path.dirname(os.path.dirname(sys.executable)))
    return os.path.join(loc, "bin", "mayapy.exe" if os.name == "nt" else "mayapy")


def opencv_ok():
    """True if OpenCV (and numpy) load in Maya. On failure the reason is kept in _cv_error."""
    global _cv_error
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
        import its_segment  # noqa: F401
        _cv_error = ""
        return True
    except Exception as e:
        _cv_error = "%s: %s" % (type(e).__name__, str(e).strip().splitlines()[0] if str(e).strip() else "")
        _forget_cv2()
        return False


def _forget_cv2():
    """Undo a failed OpenCV import so a different copy can be loaded in the same session.
    (A failed import leaves a 'loading' flag behind, and the next try reports 'recursion'.)"""
    for name in list(sys.modules):
        if name == "cv2" or name.startswith("cv2."):
            del sys.modules[name]
    for flag in ("OpenCV_LOADER", "OpenCV_LOADER_DEBUG"):
        if hasattr(sys, flag):
            delattr(sys, flag)


def _opencv_packages():
    """Pick an OpenCV build that matches the numpy this Maya ships with."""
    try:
        import numpy
        major = int(numpy.__version__.split(".")[0])
    except Exception:
        return ["numpy", "opencv-python-headless"]          # no numpy at all: bring both
    if major >= 2:
        return ["opencv-python-headless"]                    # newest, built for numpy 2
    return ["opencv-python-headless==4.10.0.84"]            # last build that also works with numpy 1


def install_opencv():
    """One-time install of OpenCV for Maya's Python, into a new folder under
    Documents/maya/its_site-packages. Nothing in the Maya install folder changes.
    Returns (ok, message)."""
    global SITE_DIR
    import time
    target = os.path.join(SITE_ROOT, "cv_%d" % int(time.time()))
    os.makedirs(target, exist_ok=True)
    cmd = [mayapy_path(), "-m", "pip", "install", "--no-deps", "--target", target] + _opencv_packages()
    print("[Illustration to 3D] " + " ".join(cmd))
    flags = 0x08000000 if os.name == "nt" else 0   # no console window on Windows
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    print(res.stdout[-2000:])
    print(res.stderr[-2000:])
    if res.returncode != 0:
        lines = [l for l in (res.stderr or res.stdout).splitlines() if l.strip()]
        return False, "pip failed: " + (lines[-1] if lines else "unknown error")

    # use the new folder first, forget any copy that failed to load before
    for p in (SITE_DIR, SITE_ROOT):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, target)
    SITE_DIR = target
    _forget_cv2()
    if opencv_ok():
        return True, "OpenCV installed."
    if "recursion" in _cv_error or "DLL" in _cv_error or "_ARRAY_API" in _cv_error:
        return False, ("OpenCV was downloaded, but Maya still has an older copy loaded. "
                       "Restart Maya and the cut-outs will use the new one.\n\n(%s)" % _cv_error)
    return False, "OpenCV was downloaded but doesn't load in this Maya:\n%s" % _cv_error


# --- AI image-to-3D (runs outside Maya) ---------------------------------------

AI_DIR = os.path.join(cmds.internalVar(userAppDir=True), "its_ai_packages")


def ai_ok():
    return os.path.isdir(os.path.join(AI_DIR, "gradio_client"))


def install_ai():
    """One-time install of gradio_client (talks to Hugging Face) into its own folder.
    Only the separate AI process uses it, never Maya itself."""
    os.makedirs(AI_DIR, exist_ok=True)
    cmd = [mayapy_path(), "-m", "pip", "install", "--upgrade", "--target", AI_DIR, "gradio_client"]
    print("[Illustration to 3D] " + " ".join(cmd))
    flags = 0x08000000 if os.name == "nt" else 0
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    print(res.stdout[-2000:])
    print(res.stderr[-2000:])
    return res.returncode == 0 and ai_ok()


# --- optionVar settings (Maya remembers these between sessions) -------------

def _pref(key, default):
    full = "its_" + key
    if not cmds.optionVar(exists=full):
        return default
    return cmds.optionVar(q=full)


def _set_pref(key, value):
    full = "its_" + key
    if isinstance(value, bool) or isinstance(value, int):
        cmds.optionVar(iv=(full, int(value)))
    elif isinstance(value, float):
        cmds.optionVar(fv=(full, value))
    else:
        cmds.optionVar(sv=(full, str(value)))


# ----------------------------------------------------------------------------
# scene building (pure Maya, no UI - can be called from scripts too)
# ----------------------------------------------------------------------------

class SceneBuilder(object):
    """Builds the 2.5D scene from an image size, boxes and settings.

    Camera sits at the origin looking down -Z. A pixel (px, py) seen at
    distance d lands at:
        x = (px / W - 0.5) * 2 * half_w(d)
        y = (0.5 - py / H) * 2 * half_h(d)
        z = -d
    where half_w(d) = d * (film width in mm / 2) / focal length.
    So a card built from those corners exactly covers its box in the camera.
    """

    def __init__(self, image_path, width, height, boxes,
                 focal=35.0, proxy_type=0, background=True, drift=True,
                 far=40.0, cutouts=None, ai_projection=False):
        self.cutouts = cutouts   # manifest from its_segment.segment(), or None
        self.ai_projection = ai_projection   # paint AI models with the drawing even if they have a texture
        self.image_path = image_path.replace("\\", "/")
        self.W = float(width)
        self.H = float(height)
        self.boxes = boxes
        self.focal = float(focal)
        self.proxy_type = proxy_type
        self.background = background
        self.drift = drift
        self.far = float(far)
        self.created = []

    # --- camera math --------------------------------------------------------
    def half_w(self, d):
        return d * (FILM_APERTURE * 25.4 / 2.0) / self.focal

    def half_h(self, d):
        return self.half_w(d) * self.H / self.W

    def to_world(self, px, py, d):
        x = (px / self.W - 0.5) * 2.0 * self.half_w(d)
        y = (0.5 - py / self.H) * 2.0 * self.half_h(d)
        return x, y, -d

    # --- shading ------------------------------------------------------------
    def textured_shader(self, tag, x, y, w, h, path=None, alpha=False):
        """Lambert that shows only the (x, y, w, h) pixel region of the image.
        With path= it shows that whole picture instead (x=y=0, w=W, h=H).
        alpha=True uses the picture's transparency (cut-out objects)."""
        # unlit surface shader: shows the drawing's colors exactly, and its alpha
        mtl = cmds.shadingNode("surfaceShader", asShader=True, name=NODE_PREFIX + tag + "_mtl")
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                       name=NODE_PREFIX + tag + "_SG")
        cmds.connectAttr(mtl + ".outColor", sg + ".surfaceShader", force=True)

        fnode = cmds.shadingNode("file", asTexture=True, isColorManaged=True,
                                 name=NODE_PREFIX + tag + "_file")
        p2d = cmds.shadingNode("place2dTexture", asUtility=True,
                               name=NODE_PREFIX + tag + "_p2d")
        for attr in ("coverage", "translateFrame", "rotateFrame", "mirrorU", "mirrorV",
                     "stagger", "wrapU", "wrapV", "repeatUV", "offset", "rotateUV",
                     "noiseUV", "vertexUvOne", "vertexUvTwo", "vertexUvThree",
                     "vertexCameraOne"):
            cmds.connectAttr(p2d + "." + attr, fnode + "." + attr, force=True)
        cmds.connectAttr(p2d + ".outUV", fnode + ".uvCoord", force=True)
        cmds.connectAttr(p2d + ".outUvFilterSize", fnode + ".uvFilterSize", force=True)
        cmds.setAttr(fnode + ".fileTextureName", (path or self.image_path).replace("\\", "/"),
                     type="string")

        if path is None:
            # crop: card UV (0..1) -> image region
            cmds.setAttr(p2d + ".repeatUV", w / self.W, h / self.H, type="double2")
            cmds.setAttr(p2d + ".offset", x / self.W, 1.0 - (y + h) / self.H, type="double2")

        cmds.connectAttr(fnode + ".outColor", mtl + ".outColor", force=True)
        if alpha:
            cmds.connectAttr(fnode + ".outTransparency", mtl + ".outTransparency", force=True)
        return sg

    def cube_shader(self, i, tag):
        """Each cube gets its own color (same colors as Preview Cutouts)."""
        r, g, b = CUBE_COLORS[i % len(CUBE_COLORS)]
        mtl = cmds.shadingNode("lambert", asShader=True, name=NODE_PREFIX + tag + "_cube_mtl")
        cmds.setAttr(mtl + ".color", r / 255.0, g / 255.0, b / 255.0, type="double3")
        cmds.setAttr(mtl + ".transparency", 0.3, 0.3, 0.3, type="double3")
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                       name=NODE_PREFIX + tag + "_cube_SG")
        cmds.connectAttr(mtl + ".outColor", sg + ".surfaceShader", force=True)
        return sg

    # --- build --------------------------------------------------------------
    def make_camera(self, parent):
        cam, shape = cmds.camera()
        cam = cmds.rename(cam, "illustration_cam")
        shape = cmds.listRelatives(cam, shapes=True)[0]
        cmds.setAttr(shape + ".focalLength", self.focal)
        cmds.setAttr(shape + ".horizontalFilmAperture", FILM_APERTURE)
        cmds.setAttr(shape + ".verticalFilmAperture", FILM_APERTURE * self.H / self.W)
        cmds.setAttr(shape + ".filmFit", 1)  # horizontal
        cmds.setAttr(shape + ".nearClipPlane", 0.1)
        cmds.setAttr(shape + ".farClipPlane", self.far * 5.0)
        cmds.setAttr(shape + ".displayResolution", 1)
        cmds.setAttr(shape + ".displayGateMask", 1)
        cmds.setAttr(shape + ".overscan", 1.15)

        # render resolution matches the drawing's aspect ratio
        rw = int(min(self.W, 1920))
        rh = max(1, int(round(rw * self.H / self.W)))
        cmds.setAttr("defaultResolution.width", rw)
        cmds.setAttr("defaultResolution.height", rh)
        cmds.setAttr("defaultResolution.pixelAspect", 1.0)
        cmds.setAttr("defaultResolution.deviceAspectRatio", float(rw) / rh)

        cam = cmds.parent(cam, parent)[0]
        return cam, shape

    def make_card(self, name, x, y, w, h, d, sg, parent):
        x0, y0, _ = self.to_world(x, y, d)
        x1, y1, _ = self.to_world(x + w, y + h, d)
        cw, ch = abs(x1 - x0), abs(y1 - y0)
        card = cmds.polyPlane(name=name, width=cw, height=ch,
                              subdivisionsX=1, subdivisionsY=1,
                              axis=(0, 0, 1), createUVs=1, constructionHistory=False)[0]
        # make sure UVs fill exactly 0..1 so the texture crop lines up
        cmds.polyNormalizeUV(card, normalizeType=1, preserveAspectRatio=False,
                             constructionHistory=False)
        cmds.xform(card, translation=((x0 + x1) / 2.0, (y0 + y1) / 2.0, -d))
        cmds.sets(card, edit=True, forceElement=sg)
        return cmds.parent(card, parent)[0]

    def make_cube(self, name, x, y, w, h, d, sg, parent, index=0, thick=DEFAULT_THICK):
        x0, y0, _ = self.to_world(x, y, d)
        x1, y1, _ = self.to_world(x + w, y + h, d)
        cw, ch = abs(x1 - x0), abs(y1 - y0)
        # depth 1 + scaleZ, so the thickness can be changed later without rebuilding
        cube = cmds.polyCube(name=name, width=cw, height=ch, depth=1.0,
                             constructionHistory=False)[0]
        cmds.xform(cube, translation=((x0 + x1) / 2.0, (y0 + y1) / 2.0, -d))
        cmds.addAttr(cube, longName="itsIndex", attributeType="long")
        cmds.setAttr(cube + ".itsIndex", index)
        cmds.addAttr(cube, longName="itsFront", attributeType="double")
        cmds.setAttr(cube + ".itsFront", -d)
        cmds.addAttr(cube, longName="itsMinSide", attributeType="double")
        cmds.setAttr(cube + ".itsMinSide", min(cw, ch))
        apply_thickness(cube, thick)
        cmds.sets(cube, edit=True, forceElement=sg)
        return cmds.parent(cube, parent)[0]

    def make_inflated(self, name, b, cut_path, sg, parent, index=0, thick=DEFAULT_THICK):
        """Puffy mesh in the shape of the cut-out object (see its_mesh.py)."""
        import numpy as np
        import cv2
        import its_mesh
        import maya.api.OpenMaya as om2

        rgba = cv2.imdecode(np.fromfile(cut_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
            return None
        res = its_mesh.inflate(rgba[:, :, 3])
        if res is None:
            return None
        verts, quads, uvs = res
        ph, pw = rgba.shape[:2]
        d = b["dist"]
        x0, y0, _ = self.to_world(b["x"], b["y"], d)
        x1, y1, _ = self.to_world(b["x"] + b["w"], b["y"] + b["h"], d)
        wx = x0 + verts[:, 0] / pw * (x1 - x0)
        wy = y0 + verts[:, 1] / ph * (y1 - y0)
        pts = om2.MFloatPointArray()
        pts.setLength(len(verts))
        for k in range(len(verts)):
            pts[k] = om2.MFloatPoint(float(wx[k]), float(wy[k]), float(verts[k, 2]))
        counts = om2.MIntArray([4] * len(quads))
        connects = om2.MIntArray(quads.reshape(-1).astype(int).tolist())
        fn = om2.MFnMesh()
        obj = fn.create(pts, counts, connects,
                        uvs[:, 0].astype(float).tolist(), uvs[:, 1].astype(float).tolist())
        fn.assignUVs(counts, connects)
        node = cmds.rename(om2.MFnDagNode(obj).fullPathName(), name)
        cmds.polySoftEdge(node, angle=180, constructionHistory=False)
        cmds.sets(node, edit=True, forceElement=sg)
        cmds.addAttr(node, longName="itsIndex", attributeType="long")
        cmds.setAttr(node + ".itsIndex", index)
        cmds.addAttr(node, longName="itsFront", attributeType="double")
        cmds.setAttr(node + ".itsFront", -d)
        cmds.addAttr(node, longName="itsMinSide", attributeType="double")
        cmds.setAttr(node + ".itsMinSide", min(abs(x1 - x0), abs(y1 - y0)))
        cmds.addAttr(node, longName="itsKind", dataType="string")
        cmds.setAttr(node + ".itsKind", "inflate", type="string")
        apply_thickness(node, thick)
        return cmds.parent(node, parent)[0]

    def cut_path(self, i):
        if not self.cutouts:
            return None
        return os.path.join(self.cutouts["dir"], self.cutouts["cutouts"][i]["file"])

    def make_proxy(self, i, b, parent):
        """One object, in whatever form the Proxies setting asks for."""
        name = safe_name(b["name"]) + "_proxy"
        tag = "%02d_%s" % (i, safe_name(b["name"]))
        thick = b.get("thick", DEFAULT_THICK)
        cut = self.cut_path(i)

        if self.proxy_type == P_CUBES:
            sg = self.cube_shader(i, tag)
            return self.make_cube(name, b["x"], b["y"], b["w"], b["h"], b["dist"], sg, parent,
                                  index=i, thick=thick)

        if self.proxy_type == P_AI and cut:
            glb = ai_model_path(cut)
            if os.path.exists(glb):
                try:
                    node = self.make_ai_model(name, tag, b, glb, parent, index=i)
                    if node:
                        return node
                except Exception:
                    traceback.print_exc()
                    print("[Illustration to 3D] AI model for '%s' failed to load, "
                          "using an inflated mesh instead." % b["name"])

        if self.proxy_type in (P_INFLATE, P_AI) and cut:
            sg = self.textured_shader(tag, 0, 0, self.W, self.H, path=cut, alpha=True)
            node = self.make_inflated(name, b, cut, sg, parent, index=i, thick=thick)
            if node:
                return node

        # textured card (also the fallback)
        if cut:
            sg = self.textured_shader(tag + "_card", 0, 0, self.W, self.H, path=cut, alpha=True)
        else:
            sg = self.textured_shader(tag + "_card", b["x"], b["y"], b["w"], b["h"])
        return self.make_card(name, b["x"], b["y"], b["w"], b["h"], b["dist"], sg, parent)

    def make_ai_model(self, name, tag, b, glb, parent, index=0):
        """Load an AI-made .glb and fit it into the box: same width/height as the
        object in the drawing, centered on it, front at the box's distance."""
        import numpy as np
        import importlib
        import its_glb
        importlib.reload(its_glb)
        import maya.api.OpenMaya as om2

        g = its_glb.read_glb(glb)
        p = g["points"]
        lo, hi = p.min(0), p.max(0)
        size = np.maximum(hi - lo, 1e-6)
        d = b["dist"]
        x0, y0, _ = self.to_world(b["x"], b["y"], d)
        x1, y1, _ = self.to_world(b["x"] + b["w"], b["y"] + b["h"], d)
        cw, ch = abs(x1 - x0), abs(y1 - y0)
        s = min(cw / size[0], ch / size[1])
        center = (lo + hi) / 2.0
        q = (p - center) * s
        q[:, 0] += (x0 + x1) / 2.0
        q[:, 1] += (y0 + y1) / 2.0
        q[:, 2] += -d - size[2] * s / 2.0      # front of the model at the box's distance

        cut = self.cut_path(index)
        project = bool((not g["texture"] or self.ai_projection) and cut and os.path.exists(cut))
        if project:
            # shape-only model (or projection asked for): project the drawing onto it
            # straight from the front, so from the camera it looks like the drawing
            u = (q[:, 0] - x0) / (x1 - x0)
            v = (q[:, 1] - y1) / (y0 - y1)
        else:
            u, v = g["uvs"][:, 0], 1.0 - g["uvs"][:, 1]       # glTF v is flipped

        pts = om2.MFloatPointArray()
        pts.setLength(len(q))
        for k in range(len(q)):
            pts[k] = om2.MFloatPoint(float(q[k, 0]), float(q[k, 1]), float(q[k, 2]))
        tris = g["tris"].astype(np.int64)
        # drop degenerate triangles (Maya would silently skip them and the UVs wouldn't line up)
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
        counts = om2.MIntArray([3] * len(tris))
        connects = om2.MIntArray(tris.reshape(-1).tolist())
        fn = om2.MFnMesh()
        obj = fn.create(pts, counts, connects, u.astype(float).tolist(), v.astype(float).tolist())
        # one UV per vertex: ask Maya which faces it actually kept and reuse their vertex ids
        real_counts, real_connects = fn.getVertices()
        fn.assignUVs(real_counts, real_connects)
        node = cmds.rename(om2.MFnDagNode(obj).fullPathName(), name)
        cmds.polySoftEdge(node, angle=60, constructionHistory=False)
        cmds.xform(node, centerPivots=True)

        # its own texture (lit, so the 3D shape reads)
        mtl = cmds.shadingNode("lambert", asShader=True, name=NODE_PREFIX + tag + "_ai_mtl")
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                       name=NODE_PREFIX + tag + "_ai_SG")
        cmds.connectAttr(mtl + ".outColor", sg + ".surfaceShader", force=True)
        tex_path = None
        if project:
            tex_path = bleed_texture(cut)
        elif g["texture"]:
            tex_path = os.path.splitext(glb)[0] + "_tex" + g["texture_ext"]
            with open(tex_path, "wb") as f:
                f.write(g["texture"])
        if tex_path:
            fnode = cmds.shadingNode("file", asTexture=True, isColorManaged=True,
                                     name=NODE_PREFIX + tag + "_ai_file")
            p2d = cmds.shadingNode("place2dTexture", asUtility=True,
                                   name=NODE_PREFIX + tag + "_ai_p2d")
            cmds.connectAttr(p2d + ".outUV", fnode + ".uvCoord", force=True)
            cmds.connectAttr(p2d + ".outUvFilterSize", fnode + ".uvFilterSize", force=True)
            cmds.setAttr(fnode + ".fileTextureName", tex_path.replace("\\", "/"), type="string")
            cmds.connectAttr(fnode + ".outColor", mtl + ".color", force=True)
        cmds.sets(node, edit=True, forceElement=sg)
        cmds.addAttr(node, longName="itsKind", dataType="string")
        cmds.setAttr(node + ".itsKind", "ai", type="string")
        return cmds.parent(node, parent)[0]

    def build(self):
        delete_scene()
        root = cmds.group(empty=True, name=ROOT_GRP)
        cam, shape = self.make_camera(root)
        proxies = cmds.group(empty=True, name="proxies_grp", parent=root)

        made = []
        # far boxes first so the outliner reads back-to-front
        order = sorted(range(len(self.boxes)), key=lambda i: -self.boxes[i]["dist"])
        for i in order:
            b = self.boxes[i]
            node = self.make_proxy(i, b, proxies)
            cmds.addAttr(node, longName="itsBox", dataType="string")
            cmds.setAttr(node + ".itsBox", json.dumps(b), type="string")
            made.append(node)

        if self.background:
            d = self.far * 1.1
            if self.cutouts:
                bg = os.path.join(self.cutouts["dir"], self.cutouts["background"])
                sg = self.textured_shader("background", 0, 0, self.W, self.H, path=bg)
            else:
                sg = self.textured_shader("background", 0, 0, self.W, self.H)
            self.make_card("background_card", 0, 0, self.W, self.H, d, sg, root)

        if self.cutouts:
            # sort see-through cards properly in the viewport
            try:
                cmds.setAttr("hardwareRenderingGlobals.transparencyAlgorithm", 3)  # depth peeling
            except Exception:
                pass

        if self.drift:
            amount = self.half_w(min(b["dist"] for b in self.boxes) if self.boxes else 5.0) * 0.08
            cmds.setKeyframe(cam, attribute="translateX", time=1, value=-amount)
            cmds.setKeyframe(cam, attribute="translateX", time=60, value=0.0)
            cmds.setKeyframe(cam, attribute="translateX", time=120, value=amount)
            cmds.playbackOptions(minTime=1, maxTime=120)
            cmds.currentTime(60)

        # remember how it was built, so the scene can be read back later
        cmds.addAttr(root, longName="illustrationScene", dataType="string")
        cmds.setAttr(root + ".illustrationScene", json.dumps({
            "image": self.image_path, "width": self.W, "height": self.H,
            "focal": self.focal, "boxes": self.boxes}), type="string")

        look_through(cam)
        return root, cam, made


def bleed_texture(cut_path):
    """Copy of a cut-out where the see-through area is filled with the object's own
    edge colors. Projected onto a 3D model, the sides then get the object's colors
    instead of stripes of whatever was around it in the drawing."""
    import numpy as np
    import cv2
    out = os.path.splitext(cut_path)[0] + "_bleed.png"
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(cut_path):
        return out
    img = cv2.imdecode(np.fromfile(cut_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim < 3 or img.shape[2] < 4:
        return cut_path
    rgb = np.ascontiguousarray(img[:, :, :3])
    hole = (img[:, :, 3] < 128).astype(np.uint8) * 255
    if hole.all() or not hole.any():
        return cut_path
    # nearest-color fill: grow the object's edge colors outwards
    _, labels = cv2.distanceTransformWithLabels(hole, cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    inside = hole == 0
    ys, xs = np.nonzero(inside)
    lut = np.zeros((labels.max() + 1, 3), np.uint8)
    lut[labels[ys, xs]] = rgb[ys, xs]
    filled = np.where(inside[:, :, None], rgb, lut[labels])
    ok, buf = cv2.imencode(".png", filled)
    buf.tofile(out)
    return out


def ai_model_path(cut_path):
    """Where the AI 3D model for a cut-out is stored: cut_03_tree.png -> ai_03_tree.glb"""
    folder, fname = os.path.split(cut_path)
    return os.path.join(folder, "ai_" + os.path.splitext(fname)[0][4:] + ".glb")


def apply_thickness(cube, percent):
    """Thickness = percent of the box's shorter side.
    Cubes: the front face stays at the box's distance and the cube grows backwards.
    Inflated meshes: they puff out evenly to the front and back of that distance."""
    depth = max(0.001, cmds.getAttr(cube + ".itsMinSide") * percent / 100.0)
    kind = cmds.getAttr(cube + ".itsKind") if cmds.attributeQuery("itsKind", node=cube, exists=True) else "cube"
    if kind == "inflate":   # mesh z runs -1..1
        cmds.setAttr(cube + ".scaleZ", depth / 2.0)
        cmds.setAttr(cube + ".translateZ", cmds.getAttr(cube + ".itsFront"))
    else:
        cmds.setAttr(cube + ".scaleZ", depth)
        cmds.setAttr(cube + ".translateZ", cmds.getAttr(cube + ".itsFront") - depth / 2.0)


def set_cube_thickness(index, percent):
    """Live update of an already-built cube. Returns True if a cube was changed."""
    if not cmds.objExists(ROOT_GRP):
        return False
    done = False
    for n in cmds.listRelatives(ROOT_GRP, allDescendents=True, type="transform", fullPath=True) or []:
        if cmds.attributeQuery("itsIndex", node=n, exists=True) and \
                cmds.getAttr(n + ".itsIndex") == index:
            apply_thickness(n, percent)
            done = True
    return done


def delete_scene():
    """Remove what the tool built (group + its shading nodes)."""
    if cmds.objExists(ROOT_GRP):
        cmds.delete(ROOT_GRP)
    leftovers = cmds.ls(NODE_PREFIX + "*")
    if leftovers:
        cmds.delete(leftovers)


def look_through(cam):
    panel = cmds.getPanel(withFocus=True)
    if not panel or cmds.getPanel(typeOf=panel) != "modelPanel":
        visible = [p for p in (cmds.getPanel(visiblePanels=True) or [])
                   if cmds.getPanel(typeOf=p) == "modelPanel"]
        panel = visible[0] if visible else None
    if panel:
        cmds.lookThru(panel, cam)
        cmds.modelEditor(panel, edit=True, displayTextures=True,
                         displayAppearance="smoothShaded")


# ----------------------------------------------------------------------------
# image panel where you draw boxes
# ----------------------------------------------------------------------------

class BoxCanvas(QtWidgets.QGraphicsView):
    boxDrawn = QtCore.Signal(QtCore.QRectF)
    boxClicked = QtCore.Signal(int)
    boxEdited = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setBackgroundBrush(QtGui.QColor(42, 42, 46))
        self.setMinimumSize(480, 360)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.pix_item = None
        self.overlay = []
        self.cut_items = []
        self._selected = -1
        self._edit = None          # (box index, mode, start point, original box)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._start = None
        self._temp = None
        self._boxes = []
        self.show_placeholder()

    def show_placeholder(self):
        self._scene.clear()
        self.pix_item, self.overlay, self._temp, self.cut_items = None, [], None, []
        t = self._scene.addText("Click  Load Image  to start.\n\n"
                                "Then drag on the picture to draw a box\n"
                                "around each object.")
        t.setDefaultTextColor(QtGui.QColor(170, 170, 170))
        self._scene.setSceneRect(t.boundingRect().adjusted(-20, -20, 20, 20))

    def set_image(self, path):
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            return None
        self._scene.clear()
        self.overlay, self._temp, self.cut_items = [], None, []
        self.pix_item = self._scene.addPixmap(pm)
        self._scene.setSceneRect(QtCore.QRectF(pm.rect()))
        self.fit()
        return pm.width(), pm.height()

    def fit(self):
        if self.pix_item:
            self.fitInView(self.pix_item, QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    # --- mouse -----------------------------------------------------------
    def _img_pos(self, event):
        p = event.position().toPoint() if hasattr(event, "position") else event.pos()
        sp = self.mapToScene(p)
        r = self.pix_item.boundingRect()
        return QtCore.QPointF(min(max(sp.x(), 0), r.width()), min(max(sp.y(), 0), r.height()))

    # How the mouse works on the picture:
    #   * drag on empty space (or inside a box that is NOT selected) -> draw a new box
    #   * click a box                          -> select it (yellow, with corner handles)
    #   * drag inside the selected box         -> move it
    #   * drag an edge / corner of the selected box -> resize it
    #   * arrow keys move the selected box 1 px (Shift: 10 px)
    MIN_BOX = 6
    CURSORS = {"tl": QtCore.Qt.SizeFDiagCursor, "br": QtCore.Qt.SizeFDiagCursor,
               "tr": QtCore.Qt.SizeBDiagCursor, "bl": QtCore.Qt.SizeBDiagCursor,
               "l": QtCore.Qt.SizeHorCursor, "r": QtCore.Qt.SizeHorCursor,
               "t": QtCore.Qt.SizeVerCursor, "b": QtCore.Qt.SizeVerCursor,
               "move": QtCore.Qt.SizeAllCursor}

    def _tol(self):
        """8 screen pixels, measured in image pixels."""
        scale = self.transform().m11() or 1.0
        return 8.0 / scale

    def _hit(self, i, p):
        """Which part of box i is under point p: corner, edge, 'move' or None."""
        if not (0 <= i < len(self._boxes)):
            return None
        b, t = self._boxes[i], self._tol()
        x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        if not (x0 - t <= p.x() <= x1 + t and y0 - t <= p.y() <= y1 + t):
            return None
        near_l, near_r = abs(p.x() - x0) <= t, abs(p.x() - x1) <= t
        near_t, near_b = abs(p.y() - y0) <= t, abs(p.y() - y1) <= t
        if near_t and near_l: return "tl"
        if near_t and near_r: return "tr"
        if near_b and near_l: return "bl"
        if near_b and near_r: return "br"
        if near_l: return "l"
        if near_r: return "r"
        if near_t: return "t"
        if near_b: return "b"
        return "move"

    def _box_at(self, p):
        for i in reversed(range(len(self._boxes))):
            b = self._boxes[i]
            if b["x"] <= p.x() <= b["x"] + b["w"] and b["y"] <= p.y() <= b["y"] + b["h"]:
                return i
        return None

    def _apply_edit(self, p):
        i, mode, start, o = self._edit
        b = self._boxes[i]
        W = self.pix_item.boundingRect().width()
        H = self.pix_item.boundingRect().height()
        dx, dy = p.x() - start.x(), p.y() - start.y()
        if mode == "move":
            b["x"] = int(round(min(max(o["x"] + dx, 0), W - o["w"])))
            b["y"] = int(round(min(max(o["y"] + dy, 0), H - o["h"])))
            return
        x0, y0 = o["x"], o["y"]
        x1, y1 = o["x"] + o["w"], o["y"] + o["h"]
        if "l" in mode: x0 = min(max(x0 + dx, 0), x1 - self.MIN_BOX)
        if "r" in mode: x1 = max(min(x1 + dx, W), x0 + self.MIN_BOX)
        if "t" in mode: y0 = min(max(y0 + dy, 0), y1 - self.MIN_BOX)
        if "b" in mode: y1 = max(min(y1 + dy, H), y0 + self.MIN_BOX)
        b["x"], b["y"] = int(round(x0)), int(round(y0))
        b["w"], b["h"] = int(round(x1 - x0)), int(round(y1 - y0))

    def mousePressEvent(self, event):
        if self.pix_item is None or event.button() != QtCore.Qt.LeftButton:
            return super().mousePressEvent(event)
        self.setFocus()
        p = self._img_pos(event)
        mode = self._hit(self._selected, p)
        if mode:   # grab the selected box
            self._edit = (self._selected, mode, p, dict(self._boxes[self._selected]))
            self.viewport().setCursor(self.CURSORS[mode])
            return
        self._start = p
        pen = QtGui.QPen(QtGui.QColor(255, 210, 60), 2, QtCore.Qt.DashLine)
        pen.setCosmetic(True)
        self._temp = self._scene.addRect(QtCore.QRectF(p, p), pen)

    def mouseMoveEvent(self, event):
        if self.pix_item is None:
            return super().mouseMoveEvent(event)
        p = self._img_pos(event)
        if self._edit is not None:
            self._apply_edit(p)
            self.draw_boxes(self._boxes, self._selected)
        elif self._temp is not None:
            self._temp.setRect(QtCore.QRectF(self._start, p).normalized())
        else:   # just hovering: show what a drag would do
            mode = self._hit(self._selected, p)
            self.viewport().setCursor(self.CURSORS.get(mode, QtCore.Qt.CrossCursor))
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._edit is not None:
            i, mode, start, o = self._edit
            self._edit = None
            if any(self._boxes[i][k] != o[k] for k in ("x", "y", "w", "h")):
                self.boxEdited.emit(i)
            return
        if self._temp is None:
            return super().mouseReleaseEvent(event)
        rect = self._temp.rect()
        self._scene.removeItem(self._temp)
        self._temp = None
        if rect.width() >= self.MIN_BOX and rect.height() >= self.MIN_BOX:
            self.boxDrawn.emit(rect)
        else:  # a click: select the box under the mouse (top-most)
            i = self._box_at(self._start)
            if i is not None:
                self.boxClicked.emit(i)

    def keyPressEvent(self, event):
        arrows = {QtCore.Qt.Key_Left: (-1, 0), QtCore.Qt.Key_Right: (1, 0),
                  QtCore.Qt.Key_Up: (0, -1), QtCore.Qt.Key_Down: (0, 1)}
        if event.key() in arrows and 0 <= self._selected < len(self._boxes) and self.pix_item:
            step = 10 if event.modifiers() & QtCore.Qt.ShiftModifier else 1
            dx, dy = arrows[event.key()]
            b = self._boxes[self._selected]
            W = self.pix_item.boundingRect().width()
            H = self.pix_item.boundingRect().height()
            b["x"] = int(min(max(b["x"] + dx * step, 0), W - b["w"]))
            b["y"] = int(min(max(b["y"] + dy * step, 0), H - b["h"]))
            self.draw_boxes(self._boxes, self._selected)
            self.boxEdited.emit(self._selected)
            return
        event.ignore()   # let the window handle Delete etc.

    # --- cut-out preview ----------------------------------------------------
    CUT_COLORS = CUBE_COLORS

    def clear_cutouts(self):
        for item in self.cut_items:
            self._scene.removeItem(item)
        self.cut_items = []
        if self.pix_item:
            self.pix_item.setOpacity(1.0)

    def show_cutouts(self, manifest, boxes):
        """Tint each cut-out object in its own color over a dimmed drawing."""
        self.clear_cutouts()
        if self.pix_item is None:
            return
        self.pix_item.setOpacity(0.35)
        for n, c in enumerate(manifest["cutouts"]):
            b = boxes[c["index"]]
            img = QtGui.QImage(os.path.join(manifest["dir"], c["file"]))
            if img.isNull():
                continue
            img = img.convertToFormat(QtGui.QImage.Format_ARGB32_Premultiplied)
            # the object itself at full strength ...
            obj = self._scene.addPixmap(QtGui.QPixmap.fromImage(img))
            obj.setPos(b["x"], b["y"])
            obj.setZValue(0.4)
            # ... plus a colored glow of its silhouette
            tint = QtGui.QImage(img.size(), QtGui.QImage.Format_ARGB32_Premultiplied)
            tint.fill(QtGui.QColor(*self.CUT_COLORS[c["index"] % len(self.CUT_COLORS)]))
            p = QtGui.QPainter(tint)
            p.setCompositionMode(QtGui.QPainter.CompositionMode_DestinationIn)
            p.drawImage(0, 0, img)
            p.end()
            glow = self._scene.addPixmap(QtGui.QPixmap.fromImage(tint))
            glow.setPos(b["x"], b["y"])
            glow.setOpacity(0.35)
            glow.setZValue(0.5)
            self.cut_items += [obj, glow]

    # --- drawing boxes ----------------------------------------------------
    def draw_boxes(self, boxes, selected=-1):
        self._boxes = boxes
        self._selected = selected
        for item in self.overlay:
            self._scene.removeItem(item)
        self.overlay = []
        if self.pix_item is None:
            return
        size = max(12, int(self.pix_item.boundingRect().width() / 55))
        font = QtGui.QFont()
        font.setPixelSize(size)
        font.setBold(True)
        for i, b in enumerate(boxes):
            sel = (i == selected)
            color = QtGui.QColor(255, 255, 255) if sel else \
                QtGui.QColor(*CUBE_COLORS[i % len(CUBE_COLORS)])
            pen = QtGui.QPen(color, 3 if sel else 2)
            pen.setCosmetic(True)
            fill = QtGui.QColor(color)
            fill.setAlpha(45 if sel else 25)
            r = self._scene.addRect(QtCore.QRectF(b["x"], b["y"], b["w"], b["h"]), pen, QtGui.QBrush(fill))
            label = self._scene.addSimpleText("%d  %s" % (i + 1, b["name"]), font)
            label.setBrush(QtGui.QBrush(color))
            label.setPos(b["x"] + 4, b["y"] + 2)
            bg = self._scene.addRect(label.sceneBoundingRect().adjusted(-3, -1, 3, 1),
                                     QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtGui.QColor(0, 0, 0, 150)))
            label.setZValue(2)
            bg.setZValue(1)
            self.overlay += [r, bg, label]
            if sel:   # corner + edge handles on the selected box
                hs = self._tol() * 1.1
                hpen = QtGui.QPen(QtGui.QColor(40, 40, 40), 1)
                hpen.setCosmetic(True)
                x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
                xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                for hx, hy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1),
                               (xm, y0), (xm, y1), (x0, ym), (x1, ym)):
                    h = self._scene.addRect(QtCore.QRectF(hx - hs / 2, hy - hs / 2, hs, hs),
                                            hpen, QtGui.QBrush(QtGui.QColor(255, 230, 120)))
                    h.setZValue(3)
                    self.overlay.append(h)


# ----------------------------------------------------------------------------
# main window
# ----------------------------------------------------------------------------

class IllustrationToSceneUI(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("illustrationToSceneWin")
        self.setWindowTitle(TOOL_TITLE)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinMaxButtonsHint)
        self.resize(1150, 700)

        self.image_path = ""
        self.img_w = self.img_h = 0
        self.boxes = []          # dicts: name, x, y, w, h, dist
        self.manifest = None     # cut-out results for the current boxes
        self._syncing = False

        self._build_ui()
        self._load_prefs()
        self.status("Load an illustration to start.")

    # --- layout ------------------------------------------------------------
    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Image...")
        self.btn_load_boxes = QtWidgets.QPushButton("Load Boxes...")
        self.btn_save_boxes = QtWidgets.QPushButton("Save Boxes")
        self.btn_clear = QtWidgets.QPushButton("Clear Boxes")
        self.lbl_image = QtWidgets.QLabel("(no image)")
        self.lbl_image.setStyleSheet("color: #aaa;")
        for b in (self.btn_load, self.btn_load_boxes, self.btn_save_boxes, self.btn_clear):
            top.addWidget(b)
        top.addSpacing(10)
        top.addWidget(self.lbl_image, 1)
        main.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.canvas = BoxCanvas()
        split.addWidget(self.canvas)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)

        # objects
        obj_box = QtWidgets.QGroupBox("Objects  (double-click a name to rename)")
        ol = QtWidgets.QVBoxLayout(obj_box)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Distance", "Thickness %"])
        self.table.horizontalHeaderItem(2).setToolTip(
            "Cube thickness, as % of the box's shorter side.\n"
            "Only used by Colored cubes. Changes built cubes right away.")
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        ol.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Delete Box")
        self.btn_auto = QtWidgets.QPushButton("Auto Distance")
        self.btn_auto.setToolTip("Guess distances from the drawing: boxes lower in the\n"
                                 "picture are closer, boxes higher up are farther.")
        row.addWidget(self.btn_delete)
        row.addWidget(self.btn_auto)
        ol.addLayout(row)
        self.btn_cut = QtWidgets.QPushButton()
        self.btn_cut.setToolTip("Find the object inside each box and cut it out of the drawing\n"
                                "(GrabCut). Each object gets its own color.")
        ol.addWidget(self.btn_cut)
        self.btn_ai = QtWidgets.QPushButton("AI 3D Model for Selected Box (online)")
        self.btn_ai.setToolTip(
            "Sends the selected object's cut-out to a free image-to-3D model on Hugging Face\n"
            "(TRELLIS, or Hunyuan3D-2 if that fails) and saves the 3D model next to the cut-outs.\n"
            "Takes 1-3 minutes; Maya keeps working meanwhile. Free GPU time is limited per day.\n"
            "Then set Proxies to 'AI 3D models (use generated)' and Build.")
        self.btn_token = QtWidgets.QPushButton("HF Token...")
        self.btn_token.setToolTip("Save your free Hugging Face token so the AI gets more\n"
                                  "free GPU time per day (huggingface.co/settings/tokens).")
        ai_row = QtWidgets.QHBoxLayout()
        ai_row.addWidget(self.btn_ai, 1)
        ai_row.addWidget(self.btn_token)
        ol.addLayout(ai_row)
        self.ai_proc = None
        rl.addWidget(obj_box, 1)

        # settings
        set_box = QtWidgets.QGroupBox("Scene settings")
        fl = QtWidgets.QFormLayout(set_box)
        self.sp_focal = QtWidgets.QDoubleSpinBox()
        self.sp_focal.setRange(10, 200)
        self.sp_focal.setSuffix(" mm")
        self.sp_near = QtWidgets.QDoubleSpinBox()
        self.sp_near.setRange(0.5, 500)
        self.sp_far = QtWidgets.QDoubleSpinBox()
        self.sp_far.setRange(1, 2000)
        self.cb_proxy = QtWidgets.QComboBox()
        self.cb_proxy.addItems(PROXY_TYPES)
        self.chk_bg = QtWidgets.QCheckBox("Background card with the full drawing")
        self.chk_drift = QtWidgets.QCheckBox("Camera drift (parallax test, frames 1-120)")
        self.chk_cut = QtWidgets.QCheckBox("Cut objects out of their boxes + fill the background")
        fl.addRow("Focal length", self.sp_focal)
        fl.addRow("Nearest distance", self.sp_near)
        fl.addRow("Farthest distance", self.sp_far)
        fl.addRow("Proxies", self.cb_proxy)
        fl.addRow(self.chk_cut)
        fl.addRow(self.chk_bg)
        fl.addRow(self.chk_drift)
        rl.addWidget(set_box)

        self.btn_build = QtWidgets.QPushButton("Build Scene")
        self.btn_build.setMinimumHeight(40)
        self.btn_build.setStyleSheet("QPushButton{background:#3d7a4f;color:white;font-weight:bold;}")
        self.btn_remove = QtWidgets.QPushButton("Delete Built Scene")
        rl.addWidget(self.btn_build)
        rl.addWidget(self.btn_remove)

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        main.addWidget(split, 1)

        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setWordWrap(True)
        main.addWidget(self.lbl_status)

        # signals
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_load_boxes.clicked.connect(self.load_boxes_dialog)
        self.btn_save_boxes.clicked.connect(self.save_boxes)
        self.btn_clear.clicked.connect(self.clear_boxes)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_auto.clicked.connect(self.auto_distance_all)
        self.btn_cut.clicked.connect(self.preview_cutouts)
        self.btn_ai.clicked.connect(self.make_ai_selected)
        self.btn_token.clicked.connect(self.set_hf_token)
        self._update_cut_button()
        self.btn_build.clicked.connect(self.build_scene)
        self.btn_remove.clicked.connect(self.remove_scene)
        self.canvas.boxDrawn.connect(self.add_box)
        self.canvas.boxClicked.connect(self.select_row)
        self.canvas.boxEdited.connect(self._on_box_edited)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection)

    # --- status ---------------------------------------------------------------
    def status(self, text, error=False):
        color = "#ff7070" if error else "#9fd89f"
        self.lbl_status.setText('<span style="color:%s">%s</span>' % (color, text))
        print("[Illustration to 3D] " + text)

    # --- prefs ----------------------------------------------------------------
    def _load_prefs(self):
        self.sp_focal.setValue(float(_pref("focal", 35.0)))
        self.sp_near.setValue(float(_pref("near", 6.0)))
        self.sp_far.setValue(float(_pref("far", 40.0)))
        self.cb_proxy.setCurrentIndex(int(_pref("proxy", 0)))
        self.chk_bg.setChecked(bool(_pref("bg", 1)))
        self.chk_drift.setChecked(bool(_pref("drift", 1)))
        self.chk_cut.setChecked(bool(_pref("cut", 1)))

    def _save_prefs(self):
        _set_pref("cut", self.chk_cut.isChecked())
        _set_pref("focal", float(self.sp_focal.value()))
        _set_pref("near", float(self.sp_near.value()))
        _set_pref("far", float(self.sp_far.value()))
        _set_pref("proxy", self.cb_proxy.currentIndex())
        _set_pref("bg", self.chk_bg.isChecked())
        _set_pref("drift", self.chk_drift.isChecked())

    def closeEvent(self, event):
        self._save_prefs()
        super().closeEvent(event)

    # --- image ----------------------------------------------------------------
    def load_image_dialog(self):
        start = str(_pref("lastDir", ""))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load illustration", start, IMAGE_FILTER)
        if path:
            self.load_image(path)

    def load_image(self, path):
        size = self.canvas.set_image(path)
        if not size:
            self.status("Could not open image: %s" % path, error=True)
            return False
        self.image_path = path.replace("\\", "/")
        self.img_w, self.img_h = size
        self.boxes = []
        _set_pref("lastDir", os.path.dirname(path))
        self.lbl_image.setText("%s   (%d x %d)" % (os.path.basename(path), self.img_w, self.img_h))
        sidecar = boxes_json_path(path)
        if os.path.exists(sidecar):
            self.load_boxes(sidecar)
        else:
            self.refresh()
            self.status("Image loaded. Drag on the picture to draw a box around each object.")
        return True

    # --- boxes ----------------------------------------------------------------
    def auto_distance(self, b):
        near, far = self.sp_near.value(), max(self.sp_far.value(), self.sp_near.value() + 1)
        t = min(max((b["y"] + b["h"]) / float(self.img_h), 0.0), 1.0)  # 0 top .. 1 bottom
        return round(far - t * (far - near), 1)

    def add_box(self, rect):
        b = {"name": "object%d" % (len(self.boxes) + 1),
             "x": int(rect.x()), "y": int(rect.y()),
             "w": int(rect.width()), "h": int(rect.height())}
        b["dist"] = self.auto_distance(b)
        self.boxes.append(b)
        self.refresh(selected=len(self.boxes) - 1)
        item = self.table.item(len(self.boxes) - 1, 0)
        self.table.editItem(item)   # type the name right away
        self.status("Box %d added. Type a name, then press Enter." % len(self.boxes))

    def delete_selected(self):
        r = self.table.currentRow()
        if 0 <= r < len(self.boxes):
            del self.boxes[r]
            self.refresh(selected=min(r, len(self.boxes) - 1))
            self.status("Box deleted.")

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace) and \
                self.table.state() != QtWidgets.QAbstractItemView.EditingState:
            self.delete_selected()
            return
        super().keyPressEvent(event)

    def clear_boxes(self):
        self.boxes = []
        self.refresh()
        self.status("All boxes cleared.")

    def auto_distance_all(self):
        if not self.img_h:
            return
        for b in self.boxes:
            b["dist"] = self.auto_distance(b)
        self.refresh(selected=self.table.currentRow())
        self.status("Distances guessed from where each box sits in the drawing.")

    def select_row(self, i):
        self.table.selectRow(i)
        self.status("Box %d selected: drag inside it to move, drag an edge or corner to "
                    "resize, arrow keys to nudge (Shift = 10 px)." % (i + 1))

    def _on_box_edited(self, i):
        self._invalidate_cutouts()   # the object inside changed
        self.canvas.draw_boxes(self.boxes, i)
        b = self.boxes[i]
        self.status("Box %d '%s': x %d, y %d, %d x %d" % (i + 1, b["name"], b["x"], b["y"], b["w"], b["h"]))

    def save_boxes(self):
        if not self.image_path:
            self.status("Load an image first.", error=True)
            return
        path = boxes_json_path(self.image_path)
        data = {"image": os.path.basename(self.image_path),
                "width": self.img_w, "height": self.img_h, "boxes": self.boxes}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.status("Saved %d boxes to %s" % (len(self.boxes), path))

    def load_boxes_dialog(self):
        start = boxes_json_path(self.image_path) if self.image_path else str(_pref("lastDir", ""))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load boxes", start, "Boxes (*.json)")
        if path:
            self.load_boxes(path)

    def load_boxes(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.status("Could not read %s: %s" % (path, e), error=True)
            return
        if not self.image_path and data.get("image"):
            img = data["image"]
            if not os.path.isabs(img):
                img = os.path.join(os.path.dirname(path), img)
            if not self.load_image(img):
                return
        sx = float(self.img_w) / data.get("width", self.img_w)
        sy = float(self.img_h) / data.get("height", self.img_h)
        boxes = []
        for i, b in enumerate(data.get("boxes", [])):
            nb = {"name": str(b.get("name") or b.get("label") or "object%d" % (i + 1)),
                  "x": int(b["x"] * sx), "y": int(b["y"] * sy),
                  "w": int(b["w"] * sx), "h": int(b["h"] * sy)}
            nb["dist"] = float(b["dist"]) if "dist" in b else self.auto_distance(nb)
            if "thick" in b:
                nb["thick"] = float(b["thick"])
            boxes.append(nb)
        self.boxes = boxes
        self.refresh()
        self.status("Loaded %d boxes from %s" % (len(boxes), os.path.basename(path)))

    # --- cut-outs ---------------------------------------------------------------
    def _update_cut_button(self):
        if opencv_ok():
            self.btn_cut.setText("Preview Cutouts")
            self.btn_cut.setToolTip("Find the object inside each box and cut it out of the drawing\n"
                                    "(GrabCut). Each object gets its own color.")
        else:
            self.btn_cut.setText("Install OpenCV for cut-outs (one time, ~40 MB)")
            if _cv_error:
                self.btn_cut.setToolTip("OpenCV doesn't load yet: " + _cv_error)

    def _invalidate_cutouts(self):
        if self.manifest is not None:
            self.manifest = None
            self.canvas.clear_cutouts()

    def run_cutouts(self):
        """Cut every box out of the drawing. Returns the manifest or None."""
        import importlib
        import its_segment
        importlib.reload(its_segment)

        def progress(n, total, name):
            self.status("Cutting out %d / %d: %s ..." % (n + 1, total, name))
            QtWidgets.QApplication.processEvents()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.manifest = its_segment.segment(self.image_path, self.boxes, progress=progress)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        return self.manifest

    def preview_cutouts(self):
        if not opencv_ok():
            ok = QtWidgets.QMessageBox.question(
                self, "Install OpenCV",
                "Cutting objects out needs OpenCV.\n\nInstall it now for Maya's Python? "
                "(downloads ~40 MB into Documents/maya/its_site-packages, takes about a minute)")
            if ok != QtWidgets.QMessageBox.Yes:
                return
            self.status("Installing OpenCV... Maya will freeze for a minute.")
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                done, msg = install_opencv()
            except Exception as e:
                traceback.print_exc()
                done, msg = False, "%s: %s" % (type(e).__name__, e)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self._update_cut_button()
            if not done:
                self.status("OpenCV not ready: " + msg.splitlines()[0], error=True)
                QtWidgets.QMessageBox.warning(
                    self, "OpenCV not ready",
                    msg + "\n\nThe rest of the tool still works without it: Build Scene "
                          "uses whole-box cards instead of cut-outs.")
                return
            self.status(msg)
        if not self.image_path or not self.boxes:
            self.status("Load an image and draw some boxes first.", error=True)
            return
        try:
            m = self.run_cutouts()
        except Exception as e:
            traceback.print_exc()
            self.status("Cut-out failed: %s" % e, error=True)
            return
        self.canvas.show_cutouts(m, self.boxes)
        self.status("Cut out %d objects (colored). Files are in %s. "
                    "Draw or change a box to go back to box view." % (len(m["cutouts"]), m["dir"]))

    # --- AI 3D -----------------------------------------------------------------
    def make_ai_selected(self):
        if self.ai_proc is not None:
            self.status("An AI model is still being made - wait for it to finish.", error=True)
            return
        r = self.table.currentRow()
        if not self.image_path or not (0 <= r < len(self.boxes)):
            self.status("Select a box first (click it on the picture).", error=True)
            return
        if not opencv_ok():
            self.status("Install OpenCV first (button above) - the AI needs the cut-out.", error=True)
            return
        if not ai_ok():
            ok = QtWidgets.QMessageBox.question(
                self, "Install AI helper",
                "Making AI 3D models needs a small helper (gradio_client, ~10 MB) that talks to "
                "Hugging Face.\n\nInstall it now? (into Documents/maya/its_ai_packages, ~30 seconds)")
            if ok != QtWidgets.QMessageBox.Yes:
                return
            self.status("Installing the AI helper... Maya will freeze for a moment.")
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                done = install_ai()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            if not done:
                self.status("AI helper install failed - see the Script Editor.", error=True)
                return
        try:
            m = self.manifest or self.run_cutouts()
        except Exception as e:
            traceback.print_exc()
            self.status("Cut-out failed: %s" % e, error=True)
            return
        cut = os.path.join(m["dir"], m["cutouts"][r]["file"])
        self._ai_target = (r, self.boxes[r]["name"], ai_model_path(cut))
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "its_ai3d.py")

        self.ai_proc = QtCore.QProcess(self)
        self.ai_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.ai_proc.readyReadStandardOutput.connect(self._ai_output)
        self.ai_proc.finished.connect(self._ai_finished)
        self._ai_last = ""
        self.ai_proc.start(mayapy_path(), ["-u", script, cut, self._ai_target[2],
                                           "--site", AI_DIR + os.pathsep + SITE_DIR])
        self.btn_ai.setEnabled(False)
        self.btn_ai.setText("Making AI 3D model for '%s'..." % self.boxes[r]["name"])
        self.status("Started the AI 3D model for '%s'. This takes 1-3 minutes; you can keep "
                    "working." % self.boxes[r]["name"])

    def set_hf_token(self):
        """Save / remove the Hugging Face token in its standard place
        (~/.cache/huggingface/token), where the AI helper picks it up."""
        path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
        have = os.path.exists(path)
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Hugging Face token",
            "Paste your Hugging Face token (free account; huggingface.co/settings/tokens,\n"
            "type 'Read' is enough). Logged-in users get more free GPU time per day.\n\n"
            "It's saved only on this computer, in ~/.cache/huggingface/token.\n"
            "%s" % ("A token is already saved. Leave empty and press OK to remove it."
                    if have else "No token saved yet."),
            QtWidgets.QLineEdit.Password)
        if not ok:
            return
        text = text.strip()
        if text:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.status("Hugging Face token saved. AI 3D requests now run as your account.")
        elif have:
            os.remove(path)
            self.status("Hugging Face token removed.")

    def _ai_output(self):
        text = bytes(self.ai_proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            print("[AI 3D] " + line)
            self._ai_last = line
            if not line.startswith(("DONE", "ERROR")):
                self.status("AI 3D: " + line)

    def _ai_finished(self, *args):
        self._ai_output()
        r, name, glb = self._ai_target
        self.ai_proc.deleteLater()
        self.ai_proc = None
        self.btn_ai.setEnabled(True)
        self.btn_ai.setText("AI 3D Model for Selected Box (online)")
        if os.path.exists(glb) and self._ai_last.startswith("DONE"):
            hint = "" if self.cb_proxy.currentIndex() == P_AI else \
                " Set Proxies to 'AI 3D models (use generated)' and click Build Scene to use it."
            self.status("AI 3D model for '%s' is ready.%s" % (name, hint))
        else:
            msg = self._ai_last[6:] if self._ai_last.startswith("ERROR") else self._ai_last
            self.status("AI 3D model for '%s' failed: %s  (the box will use an inflated "
                        "mesh instead)" % (name, msg[:300]), error=True)

    # --- table <-> data -------------------------------------------------------
    def refresh(self, selected=-1):
        self._invalidate_cutouts()
        self._syncing = True
        self.table.setRowCount(len(self.boxes))
        for i, b in enumerate(self.boxes):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(b["name"]))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.5, 2000)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setValue(b["dist"])
            spin.valueChanged.connect(lambda v, i=i: self._set_dist(i, v))
            self.table.setCellWidget(i, 1, spin)
            thick = QtWidgets.QDoubleSpinBox()
            thick.setRange(1, 300)
            thick.setDecimals(0)
            thick.setSingleStep(5)
            thick.setSuffix(" %")
            thick.setValue(b.get("thick", DEFAULT_THICK))
            thick.valueChanged.connect(lambda v, i=i: self._set_thick(i, v))
            self.table.setCellWidget(i, 2, thick)
        self._syncing = False
        if 0 <= selected < len(self.boxes):
            self.table.selectRow(selected)
        self.canvas.draw_boxes(self.boxes, selected)

    def _set_dist(self, i, v):
        if 0 <= i < len(self.boxes):
            self.boxes[i]["dist"] = float(v)
            self._invalidate_cutouts()   # nearer/farther order changed

    def _set_thick(self, i, v):
        if 0 <= i < len(self.boxes):
            self.boxes[i]["thick"] = float(v)
            if set_cube_thickness(i, float(v)):
                self.status("Cube %d thickness: %d%% (updated in the scene)" % (i + 1, v))

    def _on_item_changed(self, item):
        if self._syncing or item.column() != 0:
            return
        r = item.row()
        if 0 <= r < len(self.boxes):
            self.boxes[r]["name"] = item.text().strip() or "object%d" % (r + 1)
            self.canvas.draw_boxes(self.boxes, r)

    def _on_selection(self):
        if not self._syncing:
            self.canvas.draw_boxes(self.boxes, self.table.currentRow())

    # --- build --------------------------------------------------------------
    def build_scene(self):
        if not self.image_path:
            self.status("Load an illustration first.", error=True)
            return
        self._save_prefs()
        cutouts = None
        proxy = self.cb_proxy.currentIndex()
        needs_cut = proxy in (P_INFLATE, P_AI) or (proxy == P_CARDS and self.chk_cut.isChecked())
        if needs_cut and self.boxes:
            if opencv_ok():
                try:
                    cutouts = self.manifest or self.run_cutouts()
                except Exception as e:
                    traceback.print_exc()
                    self.status("Cut-out failed (%s), building with plain boxes." % e, error=True)
            else:
                self.status("OpenCV not installed - building with plain boxes. "
                            "Click the Install OpenCV button to get cut-outs.", error=True)
        builder = SceneBuilder(self.image_path, self.img_w, self.img_h, self.boxes,
                               focal=self.sp_focal.value(),
                               proxy_type=self.cb_proxy.currentIndex(),
                               background=self.chk_bg.isChecked(),
                               drift=self.chk_drift.isChecked(),
                               far=max(self.sp_far.value(), max([b["dist"] for b in self.boxes] or [1])),
                               cutouts=cutouts)
        cmds.undoInfo(openChunk=True, chunkName="illustrationScene_build")
        try:
            root, cam, made = builder.build()
            self.status("Built %d proxies and camera '%s'. The viewport now looks through it. "
                        "Ctrl+Z undoes the whole build." % (len(made), cam))
        except Exception as e:
            traceback.print_exc()
            self.status("Build failed: %s (see Script Editor)" % e, error=True)
        finally:
            cmds.undoInfo(closeChunk=True)

    def remove_scene(self):
        cmds.undoInfo(openChunk=True, chunkName="illustrationScene_delete")
        try:
            delete_scene()
        finally:
            cmds.undoInfo(closeChunk=True)
        self.status("Built scene deleted (Ctrl+Z brings it back).")


# ----------------------------------------------------------------------------

_window = None


def show():
    global _window
    # close any open copy, even one from an older version of this module
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if widget.objectName() == "illustrationToSceneWin":
            try:
                widget.close()
                widget.deleteLater()
            except Exception:
                pass
    try:
        parent = maya_main_window()
    except Exception:
        parent = None
    _window = IllustrationToSceneUI(parent)
    _window.show()
    return _window


if __name__ == "__main__":
    show()
