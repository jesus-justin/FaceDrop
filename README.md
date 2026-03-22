# FaceDrop

FaceDrop is now a pure browser-side AR filter app powered by `face-api.js`.

No Flask backend, no OpenCV server pipeline, and no Python dependencies are required for runtime.

## Architecture

- Single file app: `index.html`
- AI models loaded from CDN via `face-api.js`
- Webcam frames processed directly in browser
- Two stacked canvases:
	- base canvas for mirrored camera feed
	- overlay canvas for PNG filter rendering
- PNG transparency is handled natively by Canvas

## How It Works

1. Loads TinyFaceDetector and Landmark68 models from:
	 `https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights`
2. Opens webcam with `getUserMedia`
3. Every animation frame:
	 - draws video frame to canvas
	 - detects faces + landmarks with `face-api.js`
	 - draws uploaded filter over each detected face
4. On filter upload:
	 - loads PNG in browser memory
	 - runs face detection on the filter image itself
	 - aligns the filter image so detected filter-face matches detected user-face

## Run

Option 1:
- Double-click `index.html` to open directly in Chrome

Option 2 (recommended):

```bash
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

## UI Features

- Loading indicator:
	- `Loading AI models...`
	- `Models ready ✅`
- Live face counter:
	- `Faces detected: N`
- Drag-and-drop PNG upload
- Filter preview thumbnail
- Works with transparent and non-transparent PNG files
