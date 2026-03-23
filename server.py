"""
FaceDrop — Real-time face swap using InsightFace inswapper_128
Run: python server.py
Then open: http://localhost:5500
"""

import os, base64, io, threading
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import insightface
from insightface.model_zoo import get_model

try:
    from insightface.utils import face_align
except ImportError:
    face_align = None

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ── Global state ─────────────────────────────────────────────────────────────
face_detector = None
face_swapper  = None
source_face_embedding = None
source_face_landmark = None
models_loaded = False
swap_available = False
lock = threading.Lock()

# ── Boot: load models ─────────────────────────────────────────────────────────
def load_models():
    global face_detector, face_swapper, models_loaded, swap_available
    print("[FaceDrop] Loading InsightFace models...")
    
    try:
        # Load face detection model
        print("[FaceDrop] Loading face detector...")
        face_detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        models_loaded = True
        print("[FaceDrop] Face detector loaded ✅")
    except Exception as e:
        print(f"[FaceDrop] Error loading face detector: {str(e)}")
        return
    
    # Try to load face swapper model
    swapper_path = os.path.join(
        os.path.expanduser('~'), '.insightface', 'models', 'inswapper_128.onnx'
    )
    
    if os.path.exists(swapper_path):
        try:
            print("[FaceDrop] Loading face swapper...")
            face_swapper = get_model(swapper_path, download=False, download_zip=False)
            swap_available = True
            print("[FaceDrop] Face swapper loaded ✅")
        except Exception as e:
            print(f"[FaceDrop] Error loading face swapper: {str(e)}")
    else:
        print(f"[FaceDrop] ⚠️  Face swapper model not found at: {swapper_path}")
        print("[FaceDrop] Instructions for manual setup:")
        print("[FaceDrop] 1. Download inswapper_128.onnx from:")
        print("[FaceDrop]    https://huggingface.co/deepinsight/inswapper")
        print("[FaceDrop] 2. Create directory: ~/.insightface/models/")
        print("[FaceDrop] 3. Place the .onnx file there")
        print("[FaceDrop] The web interface will start but face swapping won't work until the model is installed.")
    
    if models_loaded:
        print("[FaceDrop] Core models ready ✅")
    else:
        print("[FaceDrop] ❌ Failed to load required models")

# ── Helper: decode base64 image → cv2 BGR array ──────────────────────────────
def b64_to_cv2(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_bytes = base64.b64decode(b64_str)
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# ── Helper: cv2 BGR array → base64 JPEG string ───────────────────────────────
def cv2_to_b64(img, quality=90):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')

# ── Route: serve index.html ───────────────────────────────────────────────────
@app.route('/')
def index():
    return app.send_static_file('index.html')

# ── Route: upload source face image ──────────────────────────────────────────
@app.route('/upload', methods=['POST'])
def upload():
    global source_face_embedding, source_face_landmark
    
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    if not swap_available:
        return jsonify({'error': 'Face swap model not installed. Please download inswapper_128.onnx and place it in ~/.insightface/models/'}), 400
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img = b64_to_cv2(data['image'])
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    try:
        # Detect faces in the uploaded image
        bboxes, landmarks = face_detector.detect(img)
        if bboxes is None or len(bboxes) == 0:
            return jsonify({'error': 'No face detected in uploaded image. Please use a clear front-facing photo.'}), 400

        # Pick the largest face
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        largest_idx = np.argmax(areas)
        
        with lock:
            source_face_landmark = landmarks[largest_idx]

        confidence = min(bboxes[largest_idx, 4] if bboxes.shape[1] > 4 else 0.95, 0.99)
        return jsonify({
            'status': 'ok',
            'message': f'Source face detected ✅ (confidence: {confidence:.2f})'
        })
    except Exception as e:
        return jsonify({'error': f'Face detection error: {str(e)}'}), 500

# ── Route: process webcam frame ───────────────────────────────────────────────
@app.route('/swap', methods=['POST'])
def swap():
    global source_face_landmark

    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame'}), 400

    frame = b64_to_cv2(data['frame'])
    if frame is None:
        return jsonify({'frame': data['frame']}), 200

    if not models_loaded:
        return jsonify({'frame': cv2_to_b64(frame), 'faces': 0, 'error': 'Models not loaded'})

    if not swap_available:
        return jsonify({'frame': cv2_to_b64(frame), 'faces': 0, 'error': 'Face swap model not available'})

    with lock:
        src_landmark = source_face_landmark

    if src_landmark is None:
        return jsonify({'frame': cv2_to_b64(frame), 'faces': 0})

    try:
        # Detect faces in the live frame
        bboxes, landmarks = face_detector.detect(frame)
        if bboxes is None or len(bboxes) == 0:
            return jsonify({'frame': cv2_to_b64(frame), 'faces': 0})

        # Swap each detected face
        result = frame.copy()
        for i in range(len(landmarks)):
            try:
                tgt_landmark = landmarks[i]
                # Simple face swap using landmarks
                result = face_swapper.get(result, tgt_landmark, src_landmark)
            except Exception as e:
                print(f"[FaceDrop] Swap error for face {i}: {str(e)}")
                continue

        return jsonify({
            'frame': cv2_to_b64(result, quality=88),
            'faces': len(landmarks)
        })
    except Exception as e:
        print(f"[FaceDrop] Frame processing error: {str(e)}")
        return jsonify({'frame': cv2_to_b64(frame), 'faces': 0})

# ── Route: status check ───────────────────────────────────────────────────────
@app.route('/status')
def status():
    return jsonify({
        'models_loaded': models_loaded,
        'swap_available': swap_available,
        'source_face': source_face_landmark is not None
    })

# ── Boot ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_models()
    status_msg = "✅ Face swap ready!" if swap_available else "⚠️  Web UI ready (face swap model needed)"
    print(f"\n[FaceDrop] {status_msg}")
    print("[FaceDrop] Server running at http://localhost:5500")
    print("[FaceDrop] Opening browser in 2 seconds...\n")
    
    # Auto-open browser
    import time
    import webbrowser
    time.sleep(1)
    webbrowser.open('http://localhost:5500')
    
    app.run(port=5500, debug=False, threaded=True)
