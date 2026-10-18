# Midterm pitch (original, Oct 2026)

I want to build a Maya tool that turns 2D illustrations into editable, animated 3D scenes.

When making illustrations or animations, I spend time imagining 3D space, modeling objects, adjusting their positions and sizes, and matching them to the camera. I want this tool to turn my sketches into simple 3D reference scenes and set up backgrounds faster, so I can focus more on drawing and animating main characters.

About the UI, I need buttons for Load Image, Detect, Preview, Build; an image panel where detected objects appear as boxes; and an object list with a model dropdown, position slider, and animation settings for each object. (UI sketch: see the original pitch repo, akai0037/pfa-Midtermpitch.)

It's more than one script because it previews the scene with proxy cubes before building, so user can check the layout first. Every detected object can be corrected in the list before anything is built. A full build can be undone in one step, and the tool remembers my last settings. And detection runs separately from Maya, so if it fails, Maya doesn't crash, and I can still add boxes manually.

Oct 13: The UI window opens in Maya. I can load an image, draw boxes by hand, and build placeholder cubes in the scene with a camera that matches the illustration.

Oct 20: Detect finds objects automatically, and each object can be matched to a model from my library and placed with the position slider.

Oct 26: Animation settings, Preview, undo, saved settings, error handling, and a finished demo scene.

The smallest version might be that the user import their own model library, use the boxes generated from the image or draw their own, choose and adjust a model for each box, add simple animations, and build the scene.

The riskiest part might be recognizing objects in the image and then generating a model or finding the best match in the model library. Illustrations are stylized or not clear, so the tool may not identify objects or choose models that really fit. If this doesn't work well, I'll pick models from the dropdown and draw boxes by hand instead.

## Feedback (Isaac) and what changed

- Instead of matching a model library, project the illustration onto simple proxies from the matched camera → a 2.5D shot with parallax. (Image-to-3D like Hunyuan3D / TRELLIS is an option for rough meshes.)
- Grounding DINO + SAM 2 for detection and cutouts, no training.
- Depth Anything (outside Maya) for placement, replacing the position slider.
- Get hand-drawn boxes and the camera match solid first.
