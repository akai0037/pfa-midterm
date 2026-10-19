"""
Drag this file into Maya's viewport to install the Illustration to 3D Scene tool.

It doesn't copy anything: it tells Maya where this folder is, adds a button to
your current shelf, and opens the tool. Keep the folder where it is (if you
move it, drag this file in again).
"""

import os
import sys

TOOL_MODULES = ("illustration_to_scene", "its_segment", "its_mesh", "its_glb", "its_ai3d")


def _repo_folder():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:   # some Maya versions run dropped files without __file__
        import maya.cmds as cmds
        picked = cmds.fileDialog2(fileMode=3, caption="Pick the downloaded pfa-midterm folder")
        return picked[0] if picked else None


def install(repo=None, add_shelf_button=True, open_tool=True):
    import maya.cmds as cmds
    import maya.mel as mel

    if sys.version_info[0] < 3:
        cmds.confirmDialog(title="Illustration to 3D Scene",
                           message="This tool needs Maya 2022 or newer (Python 3).",
                           button=["OK"])
        return False
    repo = repo or _repo_folder()
    if not repo or not os.path.exists(os.path.join(repo, "illustration_to_scene.py")):
        cmds.confirmDialog(title="Illustration to 3D Scene",
                           message="Couldn't find illustration_to_scene.py next to this file.\n"
                                   "Drag drag_into_maya.py from inside the downloaded folder.",
                           button=["OK"])
        return False
    repo = repo.replace("\\", "/")

    # use this folder's copy of the tool (also replaces an older one already loaded)
    if repo in sys.path:
        sys.path.remove(repo)
    sys.path.insert(0, repo)
    for m in TOOL_MODULES:
        sys.modules.pop(m, None)

    if add_shelf_button and not cmds.about(batch=True):
        shelf_top = mel.eval("$tmp = $gShelfTopLevel")
        shelf = cmds.tabLayout(shelf_top, query=True, selectTab=True)
        for b in cmds.shelfLayout(shelf, query=True, childArray=True) or []:
            if cmds.objectTypeUI(b) == "shelfButton" and \
                    cmds.shelfButton(b, query=True, docTag=True) == "illustrationToScene":
                cmds.deleteUI(b)
        command = ("import sys\n"
                   "p = r'%s'\n"
                   "if p not in sys.path: sys.path.insert(0, p)\n"
                   "import illustration_to_scene\n"
                   "illustration_to_scene.show()\n" % repo)
        cmds.shelfButton(parent=shelf, command=command, sourceType="python",
                         docTag="illustrationToScene", label="Illustration to 3D",
                         annotation="Illustration to 3D Scene", image1="pythonFamily.png",
                         imageOverlayLabel="2D>3D", overlayLabelColor=(1, 1, 1),
                         overlayLabelBackColor=(0.2, 0.45, 0.3, 0.9))
        print("[Illustration to 3D] Added a '2D>3D' button to the '%s' shelf." % shelf)

    print("[Illustration to 3D] Installed from %s" % repo)
    if open_tool and not cmds.about(batch=True):
        import illustration_to_scene
        illustration_to_scene.show()
    return True


def onMayaDroppedPythonFile(*args):
    """Maya calls this when the file is dragged into the viewport."""
    install()


if __name__ == "__main__":
    install()
