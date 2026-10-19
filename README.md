# Illustration to 3D Scene

Programming for Animators midterm, Tong Lin (akai0037).

A Maya tool that turns a 2D illustration into a 3D scene you can animate. You draw a box around each object in the drawing. The tool cuts each object out, paints over the hole it leaves in the background, and builds a camera that matches the drawing with every object at its own distance from the camera. Seen through the camera, the scene looks exactly like the drawing. Move the camera and near objects slide more than far ones (parallax). The objects can be flat cards (2.5D), rounded "inflated" 3D shapes, or real 3D models made by an AI.

**App:** Autodesk Maya (Python tool with its own window). Tested on **Maya 2026, Windows**. Maya 2025 should work too. Not standalone.

---

## Install (2 minutes)

1. **Get the code.** On this GitHub page click the green **Code** button → **Download ZIP**, then unzip it. (Or `git clone` the repo.) Put the folder somewhere it can stay, e.g. `Documents`. Don't move it later; if you do, repeat step 3.
2. **Open Maya.**
3. **Drag `drag_into_maya.py` from the folder into Maya's viewport** (the big 3D view).
   - A **2D>3D** button is added to your current shelf.
   - The **Illustration to 3D Scene** window opens.
4. **One-time OpenCV install.** In the window, under the object list, click **Install OpenCV for cut-outs**, then **Yes**. Maya freezes for 10-60 seconds while it downloads OpenCV (~40 MB, needs internet) into `Documents/maya/its_site-packages`. Nothing in the Maya program folder changes. When it's done the button says **Preview Cutouts**.
   - If a message says to **restart Maya**, do that and click the button again.
   - If it still fails, the message shows the reason. The rest of the tool works without OpenCV (Build Scene uses whole-box cards).

Next time, just click the **2D>3D** shelf button.

> If dragging doesn't work: open **Windows → General Editors → Script Editor**, click the **Python** tab, paste this (change the path to your folder), and press **Ctrl+Enter**:
> ```python
> import sys
> sys.path.insert(0, r"C:/Users/YOURNAME/Documents/pfa-midterm-main")
> import illustration_to_scene
> illustration_to_scene.show()
> ```

---

## Try it with the sample (5 minutes)

Start from a new empty scene (**File → New Scene**). The tool builds into the open scene.

1. Click **Load Image...** and open `examples/sample_illustration.png` from the folder. Six boxes (sun, mountains, house, tree, rock, bush) load automatically, with their distances already set.
2. Click **Preview Cutouts**. After a few seconds each object is tinted its own color: that's what was cut out of its box.
3. Click **Build Scene**. The viewport now looks through `illustration_cam` and shows the drawing.
4. Press **Play** in the timeline (frames 1-120). The camera drifts sideways and the layers move at different speeds.
5. Under **Scene settings → Proxies**, try the other options, clicking **Build Scene** after each one:
   - **Inflated 3D (local)**: every object becomes a rounded 3D shape. To see the depth, switch the viewport to the normal camera (viewport menu **Panels → Perspective → persp**) and orbit around (Alt + left-drag).
   - **AI 3D models (use generated)**: the tree uses a real AI-made 3D model (included in the folder); the other objects use inflated shapes.
   - **Colored cubes (layout check)**: one colored box per object, to check where everything sits.
6. Press **Ctrl+Z** once to undo a whole build, or click **Delete Built Scene**.

---

## Use it on your own drawing

1. **Load Image...** → any PNG or JPG.
2. **Drag on the picture** to draw a box around an object. Type a name and press Enter.
3. **Edit a box:** click it (it turns white with little square handles), then
   - drag inside it to move it, drag an edge or corner to resize it
   - arrow keys nudge it 1 px (Shift + arrow = 10 px)
   - **Delete Box** or the Delete key removes it
   - to draw a new box inside a big box, start dragging inside a box that is *not* selected
4. **Object list** (right side):
   - **Name**: double-click to rename. It becomes the Maya object's name.
   - **Distance**: how far the object is from the camera. Smaller = closer = moves more when the camera moves. **Auto Distance** guesses from the drawing: lower in the picture = closer.
   - **Thickness %**: how thick cubes and inflated shapes are (% of the box's shorter side). Changes built objects right away.
5. **Preview Cutouts** shows what gets cut out of each box. If a box grabs too much, make it tighter.
6. **Save Boxes** writes `<image name>_boxes.json` next to the image. Next time you load the image, the boxes come back.
7. **Scene settings:**
   - **Focal length**: the camera lens. The drawing always lines up through the camera; a wider lens (smaller number) gives stronger parallax.
   - **Nearest / Farthest distance**: the range Auto Distance uses. The background card goes just behind Farthest.
   - **Proxies**: textured cards, colored cubes, inflated 3D, or AI 3D models (see above).
   - **Cut objects out of their boxes**: off = rectangular cards with the whole box.
   - **Background card**: the drawing (with objects painted out) behind everything.
   - **Camera drift**: adds the sideways camera move on frames 1-120. Frame 60 lines up exactly with the drawing.
8. **Build Scene**. Building again replaces the old build. Ctrl+Z undoes it.

The tool remembers its settings. Everything it makes is saved next to your image, in a folder called `<image name>_its`.

### AI 3D models (optional, needs internet)

1. Select a box and click **AI 3D Model for Selected Box (online)**. The first time, it installs a small helper (~10 MB, into `Documents/maya/its_ai_packages`).
2. The object is sent to a free image-to-3D model on Hugging Face (**TRELLIS**, or **Hunyuan3D-2** if that fails). It takes 1-3 minutes in the background; you can keep working.
3. When the status line says it's ready, set **Proxies** to **AI 3D models (use generated)** and click **Build Scene**. The model is fitted into its box and the drawing is painted onto its front.

Free GPU time on Hugging Face is limited: without an account it runs out after about one model a day. With a free Hugging Face account you get more: create a token at huggingface.co/settings/tokens (type "Read") and paste it with the **HF Token...** button (saved only on your computer). If the AI fails, the object simply uses an inflated shape.

---

## What works now (Oct 19 checkpoint)

- Window with image panel; draw, select, move, resize and nudge boxes; object list with name, distance and thickness; save/load boxes as JSON
- Camera matched to the drawing (aspect ratio, film back, render resolution)
- **Cut-outs:** inside each box, OpenCV GrabCut separates the object from what's around it. Nearer objects win where boxes overlap.
- **Background fill:** the cut-out areas are painted over (OpenCV inpaint), so nothing shows up twice when the camera moves
- Each object placed at its distance and scaled so it covers its box exactly from the camera
- Four kinds of proxies: textured cards (2.5D), colored cubes, inflated 3D meshes, AI 3D models (TRELLIS / Hunyuan3D-2 online, loaded from .glb)
- Camera drift animation to show parallax
- Whole build is one undo step; settings remembered between sessions
- Slow or failing parts never crash Maya: the AI runs in a separate process, and anything that fails falls back to a simpler proxy

## What doesn't work yet

- **Cut-outs get confused when an object and its surroundings have similar colors.** In the sample, the mountains' cut-out keeps a bit of sky. GrabCut only looks at colors; SAM 2 understands shapes and should fix this.
- **Background fill is blurry** where a big object was removed. Fine for small camera moves.
- **No automatic detection yet**: you draw every box.
- **Distances are guessed** (from height in the picture) or typed in. They don't read the drawing's real depth yet.
- **AI 3D depends on free online GPU time** (see above). The Hunyuan3D-2 textured mode currently errors on their server, so the tool uses its shape-only mode and paints the drawing on. AI models of flat cartoon drawings come out fairly flat, and the back of any 3D shape is a guess.
- Cut-outs take a few seconds and Maya freezes while they run.
- Only tested on Windows.

## Plan for Oct 26

Following Isaac's feedback (depth, not more features):

1. **Better masks:** swap GrabCut for SAM 2 (box → mask). Only the mask step changes.
2. **Depth from the drawing:** run Depth Anything outside Maya and use the depth map to set each object's distance.
3. **Auto boxes:** Grounding DINO finds objects by name ("tree, house, rock") and writes the same `_boxes.json` the tool already loads. Hand-drawn boxes stay as the fallback.
4. A finished demo shot from one of my own illustrations.

---

## Peer run

_Classmate: (name)_

1. What they did:
2. What happened:
3. What confused them or broke:

**What I'll fix:**

---

## Files

- `drag_into_maya.py`: installer, drag it into Maya's viewport
- `illustration_to_scene.py`: the tool (window + scene builder)
- `its_segment.py`: cuts objects out of their boxes and fills the background (numpy + OpenCV)
- `its_mesh.py`: turns a cut-out into an inflated 3D mesh
- `its_ai3d.py`: sends a cut-out to TRELLIS / Hunyuan3D-2 and saves the .glb (runs outside Maya)
- `its_glb.py`: reads .glb 3D files (Maya can't open them by itself)
- `examples/`: sample drawing, its boxes, and an AI model of its tree
- `PITCH.md`: the original pitch and the feedback that changed the plan
