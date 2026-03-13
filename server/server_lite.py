"""
MangaVoice Lite — Lightweight ONNX-only server
Runs on http://127.0.0.1:5055
Same API as the full server so the extension works without changes.

Stack (all ONNX, no PyTorch/PaddlePaddle/Transformers):
  - RT-DETR ONNX:    Bubble detection (161 MB model)
  - Apple Vision:    macOS native OCR (primary, uses Neural Engine)
  - Tesseract:       Fallback OCR (system package)
  - Kokoro ONNX:     TTS with 54 voices (~310 MB model)
  - MangaCNN ONNX:   Junk/SFX text classifier (~2 MB model)

Requirements:
  pip install flask flask-cors opencv-python-headless numpy Pillow
  pip install piper-tts onnxruntime
  pip install symspellpy wordninja pytesseract
  brew install tesseract  (or apt/choco on Linux/Windows)
"""

import base64
import concurrent.futures
import io
import logging
import os
import re
import sys
import threading
import time
import uuid
import wave

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────
PORT = 5055
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IDLE_SHUTDOWN_MINUTES = 10

# Piper TTS (fast, lightweight)
PIPER_MODEL = os.path.join(MODELS_DIR, 'piper', 'en_US-lessac-medium.onnx')

# RT-DETR bubble detector
DETECTOR_MODEL = os.path.join(MODELS_DIR, 'detector.onnx')

# MangaCNN junk classifier
CLASSIFIER_MODEL = os.path.join(MODELS_DIR, 'manga_cnn.onnx')


# Florence-2 ONNX (high quality OCR, used as background quality pass)
FLORENCE_DIR = os.path.join(MODELS_DIR, 'florence2')
USE_FLORENCE = os.environ.get('MVR_USE_FLORENCE', '1') != '0'  # enabled by default (background quality mode)

# ONNX thread caps
ONNX_THREADS = int(os.environ.get('MVR_ONNX_THREAD_CAP', '4'))

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='\033[36m[MVR-lite]\033[0m %(message)s',
)
log = logging.getLogger('mvr-lite')

_apple_vision_available = False
if sys.platform == 'darwin':
    try:
        from Foundation import NSData
        from Vision import (
            VNRecognizeTextRequest,
            VNImageRequestHandler,
            VNRequestTextRecognitionLevelAccurate,
        )
    except ImportError:
        pass
    else:
        _apple_vision_available = True

# ─── LanguageTool grammar checker (starts Java server in background) ─────────
_grammar_tool = None
_grammar_ready = False
def _init_grammar():
    global _grammar_tool, _grammar_ready
    try:
        import language_tool_python
        _grammar_tool = language_tool_python.LanguageTool('en-US')
        _grammar_ready = True
        log.info('LanguageTool grammar checker ready')
    except Exception as e:
        log.warning(f'LanguageTool not available: {e}')
threading.Thread(target=_init_grammar, daemon=True).start()

# ─── App ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max request
CORS(app, origins='*', allow_private_network=True)  # safe: server only listens on 127.0.0.1

# ─── State ───────────────────────────────────────────────────────────────────
_shutting_down = False
_models_ready = False
_last_activity = time.time()

# Model holders
_piper_voice = None  # Piper TTS
_piper_lock = threading.Lock()
_detector_session = None
_classifier_session = None
_florence2 = None  # Florence-2 ONNX OCR engine


# Background quality pass state
_quality_results = {}  # request_id -> {bubbles: [...], ready: bool}
_quality_lock = threading.Lock()
_quality_executor = None  # initialized after model load

# TTS audio cache (text + voice + speed -> WAV bytes)
_tts_cache = {}
_tts_cache_lock = threading.Lock()

# Page audio cache (request_id -> {audio_bytes, timestamps, ready})
_page_audio_cache = {}
_page_audio_lock = threading.Lock()

# Protected proper nouns (loaded eagerly so _clean_text works immediately)
_PROTECTED_NOUNS_FILE = os.path.join(BASE_DIR, 'protected_nouns.txt')

def _load_protected_nouns():
    """Load protected nouns at module level so they're available before model loading."""
    path = _PROTECTED_NOUNS_FILE
    if not os.path.isfile(path):
        path = os.path.join(BASE_DIR, 'protected_vocab.txt')
    if os.path.isfile(path):
        with open(path) as f:
            return {w.strip().upper() for w in f if w.strip() and not w.startswith('#')}
    return set()

_PROTECTED_NOUNS = _load_protected_nouns()

# RT-DETR constants
_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DET_CLASSES = {0: 'bubble', 1: 'text_bubble', 2: 'text_free'}


# =============================================================================
# Model loading
# =============================================================================

def _ort_session(model_path, use_coreml=False):
    """Create an ONNX Runtime session with thread caps and optional CoreML/ANE."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    if ONNX_THREADS > 0:
        opts.inter_op_num_threads = ONNX_THREADS
        opts.intra_op_num_threads = ONNX_THREADS
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    providers = ['CPUExecutionProvider']
    if use_coreml and 'CoreMLExecutionProvider' in ort.get_available_providers():
        providers = [
            ('CoreMLExecutionProvider', {
                'MLComputeUnits': 'ALL',  # Use ANE + GPU + CPU
                'AllowLowPrecisionAccumulationOnGPU': '1',
            }),
            'CPUExecutionProvider',
        ]
        log.info(f'  CoreML/ANE enabled for {os.path.basename(model_path)}')

    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def _load_florence2(models_dir):
    """Load Florence-2 ONNX models for OCR.
    Note: CoreML EP fails with q4 quantized models (dynamic resize error),
    so we use CPU with 4 threads for background quality pass. ~3s per image."""
    from tokenizers import Tokenizer

    return {
        'vision': _ort_session(os.path.join(models_dir, 'vision_encoder_q4.onnx')),
        'encoder': _ort_session(os.path.join(models_dir, 'encoder_model_q4.onnx')),
        'embed': _ort_session(os.path.join(models_dir, 'embed_tokens_q4.onnx')),
        'decoder': _ort_session(os.path.join(models_dir, 'decoder_model_q4.onnx')),
        'tokenizer': Tokenizer.from_file(os.path.join(models_dir, 'tokenizer.json')),
    }


def _florence2_ocr(img_bgr, max_tokens=100):
    """Run Florence-2 ONNX OCR on an image crop. Returns text."""
    if _florence2 is None:
        return ''

    # 1. Preprocess: resize to 768x768, ImageNet normalize
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (768, 768), interpolation=cv2.INTER_CUBIC)
    pv = resized.astype(np.float32) / 255.0
    pv = (pv - _DET_MEAN) / _DET_STD  # reuse ImageNet mean/std
    pv = np.expand_dims(pv.transpose(2, 0, 1), 0).astype(np.float32)

    # 2. Vision encoder
    img_feat = _florence2['vision'].run(None, {'pixel_values': pv})[0]

    # 3. Tokenize prompt "What is the text in the image?"
    prompt = "What is the text in the image?"
    enc = _florence2['tokenizer'].encode(prompt)
    prompt_ids = np.array([enc.ids], dtype=np.int64)
    prompt_emb = _florence2['embed'].run(None, {'input_ids': prompt_ids})[0]

    # 4. Concat image features + prompt → encoder
    enc_input = np.concatenate([img_feat, prompt_emb], axis=1).astype(np.float32)
    enc_mask = np.ones((1, enc_input.shape[1]), dtype=np.int64)
    enc_hidden = _florence2['encoder'].run(None, {
        'inputs_embeds': enc_input, 'attention_mask': enc_mask,
    })[0]

    # 5. Autoregressive decode (greedy, no KV cache — simple for short text)
    decoder_ids = [2]  # decoder_start_token_id
    for _ in range(max_tokens):
        d_ids = np.array([decoder_ids], dtype=np.int64)
        d_emb = _florence2['embed'].run(None, {'input_ids': d_ids})[0]
        out = _florence2['decoder'].run(None, {
            'encoder_hidden_states': enc_hidden,
            'encoder_attention_mask': enc_mask,
            'inputs_embeds': d_emb,
        })
        next_id = int(np.argmax(out[0][0, -1, :]))
        if next_id == 2:  # eos_token_id
            break
        decoder_ids.append(next_id)

    # 6. Decode
    text = _florence2['tokenizer'].decode(decoder_ids[1:], skip_special_tokens=True)
    return text.strip()


def _load_all_models():
    """Load all models in background. Server accepts requests immediately (503 until ready)."""
    global _piper_voice, _detector_session, _classifier_session, _florence2, _models_ready

    t_start = time.time()

    # 1. Kokoro TTS
    # 1. Piper TTS
    if os.path.isfile(PIPER_MODEL):
        try:
            from piper import PiperVoice
            t0 = time.time()
            _piper_voice = PiperVoice.load(PIPER_MODEL)
            log.info(f'Piper TTS loaded ({int((time.time()-t0)*1000)}ms)')
        except Exception as e:
            log.warning(f'Piper TTS failed: {e}')
    else:
        log.info('Piper model not found, fast TTS unavailable')

    # 2. RT-DETR bubble detector
    if os.path.isfile(DETECTOR_MODEL):
        try:
            t0 = time.time()
            _detector_session = _ort_session(DETECTOR_MODEL)
            log.info(f'RT-DETR detector loaded ({int((time.time()-t0)*1000)}ms)')
        except Exception as e:
            log.warning(f'RT-DETR detector failed: {e}')
    else:
        log.warning(f'Detector model not found at {DETECTOR_MODEL}')

    # 3. MangaCNN classifier
    if os.path.isfile(CLASSIFIER_MODEL):
        try:
            t0 = time.time()
            _classifier_session = _ort_session(CLASSIFIER_MODEL)
            log.info(f'MangaCNN classifier loaded ({int((time.time()-t0)*1000)}ms)')
        except Exception as e:
            log.warning(f'MangaCNN classifier failed: {e}')

    # 4. Florence-2 ONNX (background quality pass — uses CoreML/ANE for speed)
    if USE_FLORENCE and os.path.isdir(FLORENCE_DIR):
        try:
            t0 = time.time()
            _florence2 = _load_florence2(FLORENCE_DIR)
            log.info(f'Florence-2 ONNX loaded ({int((time.time()-t0)*1000)}ms, background quality mode)')
        except Exception as e:
            log.warning(f'Florence-2 ONNX failed: {e}')
    elif os.path.isdir(FLORENCE_DIR):
        log.info('Florence-2 available but disabled (set MVR_USE_FLORENCE=1 to enable)')
    else:
        log.info('Florence-2 models not found — background quality pass disabled')

    # 5. Protected nouns already loaded at module level
    global _quality_executor
    log.info(f'Protected nouns: {len(_PROTECTED_NOUNS)} loaded')

    # Initialize background quality executor (1 thread — Florence-2 is heavy)
    if _florence2 is not None:
        _quality_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        log.info('Background quality pass enabled (Florence-2 ONNX)')

    total_ms = int((time.time() - t_start) * 1000)
    log.info(f'All models loaded in {total_ms}ms')
    _models_ready = True


# =============================================================================
# Bubble Detection (RT-DETR ONNX)
# =============================================================================

def _detect_bubbles(img_bgr, conf_threshold=0.4):
    """Run RT-DETR bubble detection. Returns list of {x, y, w, h, score, class}."""
    if _detector_session is None:
        return []

    orig_h, orig_w = img_bgr.shape[:2]

    # Preprocess: resize to 640x640, ImageNet normalize
    resized = cv2.resize(img_bgr, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - _DET_MEAN) / _DET_STD).transpose(2, 0, 1)
    blob = np.expand_dims(blob, 0).astype(np.float32)
    orig_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)

    labels, boxes, scores = _detector_session.run(None, {
        'images': blob,
        'orig_target_sizes': orig_sizes,
    })

    # Collect detections
    detections = []
    for i in range(len(labels[0])):
        score = float(scores[0][i])
        if score < conf_threshold:
            continue
        label = int(labels[0][i])
        box = boxes[0][i]
        cls = _DET_CLASSES.get(label, 'unknown')
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(orig_w, int(box[2]))
        y2 = min(orig_h, int(box[3]))
        if x2 > x1 and y2 > y1:
            detections.append({
                'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                'score': score, 'class': cls,
            })

    # NMS: remove overlapping boxes
    detections.sort(key=lambda d: d['score'], reverse=True)
    keep = []
    for d in detections:
        overlap = False
        for k in keep:
            iou = _box_iou(d, k)
            if iou > 0.3:
                overlap = True
                break
        if not overlap:
            keep.append(d)

    return keep


def _box_iou(a, b):
    """Compute IoU between two boxes {x, y, w, h}."""
    ax1, ay1, ax2, ay2 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
    bx1, by1, bx2, by2 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# =============================================================================
# OCR: Apple Vision + Tesseract fallback
# =============================================================================

def _enhance_ocr_crop(crop_bgr):
    """Enhance a crop for OCR: upscale small text, sharpen, add white padding.
    Adapted from the full server's _enhance_ocr_crop() but without AI upscaler."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]

    # 1. Upscale so text lines are big enough for PP-OCR det model.
    #    Manga bubbles have stacked single words — each line can be tiny.
    #    Target: longest side at least 300px so individual text lines are ~40px+.
    max_dim = max(h, w)
    if max_dim < 300:
        scale = 300 / max_dim
        crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)
    elif max_dim < 400:
        crop_bgr = cv2.resize(crop_bgr, None, fx=1.5, fy=1.5,
                              interpolation=cv2.INTER_LINEAR)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE contrast enhancement (helps with gradient/textured bubble backgrounds)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Mild unsharp mask sharpening to crisp text edges
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)

    # 5. Add 15px white padding border (helps OCR at edges)
    pad = 15
    padded = cv2.copyMakeBorder(sharpened, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=255)

    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)



def _extract_bubble_mask(img, det):
    """Return a refined box and interior mask using fast OpenCV ops."""
    orig_box = (det['x'], det['y'], det['w'], det['h'])

    pad = int(max(10, min(60, max(det['w'], det['h']) * 0.12)))
    x1 = max(0, det['x'] - pad)
    y1 = max(0, det['y'] - pad)
    x2 = min(img.shape[1], det['x'] + det['w'] + pad)
    y2 = min(img.shape[0], det['y'] + det['h'] + pad)

    h_roi = max(0, y2 - y1)
    w_roi = max(0, x2 - x1)
    if h_roi == 0 or w_roi == 0:
        fallback_mask = np.full((max(1, h_roi), max(1, w_roi)), 255, dtype=np.uint8)
        return orig_box, fallback_mask, False

    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    center_x = int(np.clip(det['x'] + det['w'] / 2 - x1, 0, w_roi - 1))
    center_y = int(np.clip(det['y'] + det['h'] / 2 - y1, 0, h_roi - 1))

    # Sample a donut ring at 30-40% from center to avoid landing on text.
    # 8 patches at N/NE/E/SE/S/SW/W/NW positions, take median of medians.
    ring_r_y = max(5, int(h_roi * 0.35))
    ring_r_x = max(5, int(w_roi * 0.35))
    pr = max(2, min(8, int(min(h_roi, w_roi) * 0.04)))  # small patch radius
    offsets = [
        (0, -ring_r_y), (ring_r_x, -ring_r_y),   # N, NE
        (ring_r_x, 0), (ring_r_x, ring_r_y),       # E, SE
        (0, ring_r_y), (-ring_r_x, ring_r_y),       # S, SW
        (-ring_r_x, 0), (-ring_r_x, -ring_r_y),     # W, NW
    ]
    patch_medians = []
    for dx, dy in offsets:
        px = int(np.clip(center_x + dx, pr, w_roi - pr - 1))
        py = int(np.clip(center_y + dy, pr, h_roi - pr - 1))
        p = gray[py - pr:py + pr + 1, px - pr:px + pr + 1]
        if p.size:
            patch_medians.append(float(np.median(p)))
    interior_intensity = float(np.median(patch_medians)) if patch_medians else 230.0

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    is_dark_bubble = interior_intensity < 128
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if is_dark_bubble:
        # ── BLACK/DARK BUBBLE PATH ──
        # Dark bubbles don't have neighbor bleed (dark bg isolates text).
        # Skip CC masking — just use the full RT-DETR box. The crop will be
        # inverted before OCR so Apple Vision sees black-on-white.
        mask = np.full((h_roi, w_roi), 255, dtype=np.uint8)
        return orig_box, mask, True

    # ── WHITE/LIGHT BUBBLE PATH ──
    # Interior is white, text is black. Find the white connected component.
    thresh_val = int(max(0, min(255, interior_intensity - 30)))
    _, bubble_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bubble_mask = cv2.dilate(bubble_mask, kernel, iterations=1)
    bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(bubble_mask, connectivity=8)
    if num_labels <= 1:
        fallback_mask = np.full((h_roi, w_roi), 255, dtype=np.uint8)
        return orig_box, fallback_mask, False

    if not (0 <= center_y < labels.shape[0] and 0 <= center_x < labels.shape[1]):
        center_label = 0
    else:
        center_label = labels[center_y, center_x]
    if center_label == 0:
        fallback_mask = np.full((h_roi, w_roi), 255, dtype=np.uint8)
        return orig_box, fallback_mask, False

    mask = np.where(labels == center_label, 255, 0).astype(np.uint8)

    # Dilate the mask to cover text inside the bubble.
    # CC finds white pixels, dilation covers black text regions.
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, fill_kernel, iterations=2)
    # Then close to smooth the edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, fill_kernel, iterations=1)

    if cv2.countNonZero(mask) == 0:
        fallback_mask = np.full((h_roi, w_roi), 255, dtype=np.uint8)
        return orig_box, fallback_mask, False

    points = cv2.findNonZero(mask)
    if points is None:
        fallback_mask = np.full((h_roi, w_roi), 255, dtype=np.uint8)
        return orig_box, fallback_mask, False

    bx, by, bw, bh = cv2.boundingRect(points)
    det_area = max(1, det['w'] * det['h'])
    if bw * bh < det_area * 0.20:
        fallback_mask = np.full((h_roi, w_roi), 255, dtype=np.uint8)
        return orig_box, fallback_mask, False

    refined_x = x1 + bx
    refined_y = y1 + by
    refined_x2 = refined_x + bw
    refined_y2 = refined_y + bh

    refined_x = max(det['x'], refined_x)
    refined_y = max(det['y'], refined_y)
    refined_x2 = min(det['x'] + det['w'], refined_x2)
    refined_y2 = min(det['y'] + det['h'], refined_y2)

    refined_w = max(1, int(refined_x2 - refined_x))
    refined_h = max(1, int(refined_y2 - refined_y))

    return (int(refined_x), int(refined_y), refined_w, refined_h), mask, False


def _apple_vision_ocr(crop_bgr):
    """Run Apple Vision text recognition on a bubble crop image."""
    if not _apple_vision_available:
        return ''

    _MIN_CONFIDENCE = 0.3  # Filter noise from bubble edges

    lines = []
    try:
        # Only upscale very tiny crops — Apple Vision handles most sizes fine
        h, w = crop_bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim < 150:
            scale = 150 / max_dim
            crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LINEAR)

        # JPEG is ~5-10x faster to encode than PNG, Apple Vision handles it fine
        success, encoded = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            return ''
        img_bytes = encoded.tobytes()
        if not img_bytes:
            return ''

        data = NSData.dataWithBytes_length_(img_bytes, len(img_bytes))
        handler = VNImageRequestHandler.alloc().initWithData_options_(data, None)

        def completion_handler(request, error):
            if error:
                log.info(f'  [OCR:Apple Vision] handler error: {error}')
                return
            for obs in request.results() or []:
                candidates = obs.topCandidates_(1)
                if not candidates:
                    continue
                candidate = candidates[0]
                conf = float(candidate.confidence())
                text_line = candidate.string().strip()
                if not text_line:
                    continue
                # Filter low-confidence single chars (edge noise like "f", "e")
                if len(text_line) <= 1 and conf < _MIN_CONFIDENCE:
                    log.info(f'  [OCR:Apple Vision] skipped low-conf char: "{text_line}" ({conf:.2f})')
                    continue
                box = obs.boundingBox()
                origin = getattr(box, 'origin', None)
                size = getattr(box, 'size', None)
                x = float(origin.x) if origin and hasattr(origin, 'x') else 0.0
                y = float(origin.y) if origin and hasattr(origin, 'y') else 0.0
                w = float(size.width) if size and hasattr(size, 'width') else 1.0
                h = float(size.height) if size and hasattr(size, 'height') else 0.1
                # Filter lines at crop edges (bleed from neighboring bubbles)
                # Vision coords: x,y are 0-1 normalized, origin is bottom-left
                cx = x + w / 2  # center x
                cy = y + h / 2  # center y
                _EDGE = 0.12  # 12% margin — catches neighbor bleed at crop edges
                near_edge = cx < _EDGE or cx > (1 - _EDGE) or cy < _EDGE or cy > (1 - _EDGE)
                if near_edge:
                    # Short fragments (1-3 chars) near edge = likely neighbor bleed
                    # Longer text (4+ chars) near edge = likely real dialogue extending to bubble edge
                    alpha_len = len(re.sub(r'[^a-zA-Z]', '', text_line))
                    if alpha_len <= 3:
                        log.info(f'  [OCR:Apple Vision] skipped edge fragment: "{text_line}" (cx={cx:.2f}, cy={cy:.2f})')
                        continue
                    else:
                        log.info(f'  [OCR:Apple Vision] kept edge text (long): "{text_line}" (cx={cx:.2f}, cy={cy:.2f})')
                lines.append((y, text_line))

        request = VNRecognizeTextRequest.alloc().initWithCompletionHandler_(completion_handler)
        request.setRecognitionLanguages_(('en-US',))
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        success, error = handler.performRequests_error_([request], None)
        if not success:
            log.info(f'  [OCR:Apple Vision] handler failed: {error}')

        if not lines:
            log.info('  [OCR:Apple Vision] no lines detected')
            return ''

        sorted_lines = sorted(lines, key=lambda item: -item[0])
        text = ' '.join(line for _, line in sorted_lines)
        log.info(f'  [OCR:Apple Vision] {len(sorted_lines)} lines -> "{text[:80]}"')
        return text.strip()
    except Exception as exc:
        log.info(f'  [OCR:Apple Vision] exception: {exc}')
        return ''


def _ocr_score(text):
    """Heuristic quality score for OCR output. Higher = better."""
    if not text or not text.strip():
        return -1
    score = 0
    words = text.split()
    alpha_chars = re.sub(r'[^a-zA-Z]', '', text)
    digit_chars = re.sub(r'[^0-9]', '', text)

    # Pure numbers are valid (manga has price/amount bubbles like "1,200,000")
    if digit_chars and not alpha_chars:
        return max(1, len(digit_chars))

    # Prefer more alphabetic content (not just longer strings)
    score += len(alpha_chars) * 1.0
    # Count digits as half-value content (numbers mixed with text)
    score += len(digit_chars) * 0.5
    # Prefer reasonable word count (proper spacing)
    score += len(words) * 1.5
    # Penalize very long words (likely merged, >15 chars)
    for w in words:
        core = re.sub(r'[^a-zA-Z]', '', w)
        if len(core) > 15:
            score -= 8
    # Penalize real junk chars (not punctuation — those are normal)
    ok_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-\"…:;()[]")
    junk = sum(1 for c in text if c not in ok_chars)
    score -= junk * 5
    # Penalize very short results — but not if they contain digits
    # Reduced penalty for short text with proper punctuation (e.g. "NO!", "HM.", "OH?")
    if len(alpha_chars) < 3 and not digit_chars:
        has_punct = bool(re.search(r'[.!?]', text))
        score -= 8 if has_punct else 20
    return score


def _ocr_crop(crop_bgr):
    """Extract text from a bubble crop using Apple Vision, falling back to Tesseract."""
    # Apple Vision works best on raw crops (CLAHE/sharpening adds noise)
    text = _apple_vision_ocr(crop_bgr)

    # Only enhance for Tesseract fallback (it needs the preprocessing)
    enhanced = None

    if not text.strip():
        try:
            import pytesseract
            enhanced = _enhance_ocr_crop(crop_bgr)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            text = pytesseract.image_to_string(
                binary, lang='eng', config='--oem 1 --psm 6'
            ).strip()
            log.info(f'  [OCR:Tesseract fallback] "{text[:80]}"')
        except Exception as e:
            log.info(f'  [OCR:Tesseract fallback] error: {e}')

    return text.strip()


def _classify_crop(crop_bgr):
    """Classify a crop as dialogue (True) or junk (False) using MangaCNN."""
    if _classifier_session is None:
        return True  # No classifier = let everything through

    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128)).astype(np.float32) / 255.0
        blob = resized.reshape(1, 1, 128, 128)
        inp_name = _classifier_session.get_inputs()[0].name
        output = _classifier_session.run(None, {inp_name: blob})
        score = float(output[0][0][0])
        # Positive = dialogue, negative = junk (raw logit, not sigmoid)
        # Very lenient — better to let junk through than miss real dialogue.
        # OCR score filter downstream catches actual junk.
        is_dialogue = score > -5.0
        if not is_dialogue:
            log.info(f'  Classifier score={score:.1f} (rejected)')
        return is_dialogue
    except Exception:
        return True  # On error, let it through


# =============================================================================
# Text cleanup (lightweight, pure Python)
# =============================================================================

# Optional: word splitting for merged words (common in OCR)
_wordninja = None
try:
    import wordninja as _wordninja
except ImportError:
    pass

# Optional: spell checking
_symspell = None
try:
    import importlib.resources
    from symspellpy import SymSpell
    _symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = str(importlib.resources.files('symspellpy').joinpath('frequency_dictionary_en_82_765.txt'))
    _symspell.load_dictionary(dict_path, term_index=0, count_index=1)
except Exception:
    pass

# SFX patterns (repeated characters like "WAAAA", "EEEK", "AAAAH!!!")
_SFX_PATTERN = re.compile(r'^[A-Z]{0,3}(.)\1{2,}[A-Z]{0,3}[!?.…]*$', re.IGNORECASE)
_DOTS_ONLY = re.compile(r'^[.\s…]+$')
_PUNCT_ONLY = re.compile(r'^[^a-zA-Z0-9]*$')
# Short interjections/exclamations that aren't real dialogue
_INTERJECTIONS = {
    'HUH', 'HMM', 'HM', 'RAH', 'BAH', 'MEH', 'TSK', 'PST', 'SHH', 'SSH',
    'GAH', 'PAH', 'NAH', 'HAH', 'EEK', 'EEP', 'OOH', 'AAH', 'UGH', 'GRR',
    'ARG', 'ERR', 'URK', 'ACK', 'TCH', 'TUT', 'PFF', 'BRR', 'ZZZ', 'MMM',
    'HEH', 'FEH', 'GUH', 'NNG', 'NGH', 'KUH', 'BNNT', 'THWP', 'FWSH',
    'MURMUR', 'GASP', 'GULP', 'SNIFF', 'SOB', 'PANT', 'GRUNT', 'GROAN',
    'MOAN', 'SHRIEK', 'SCREECH', 'SQUEAL', 'YELP', 'WHINE', 'WHIMPER',
    'CHORTLE', 'SNICKER', 'GIGGLE', 'CACKLE', 'TITTER', 'SNORE', 'WHEEZE',
    'COUGH', 'HICCUP', 'BURP', 'RETCH', 'SPLUTTER', 'STAMMER', 'MUTTER',
    'MUMBLE', 'WHISPER', 'MURMUR', 'RUSTLE', 'SHUFFLE', 'CLATTER', 'RATTLE',
    'JINGLE', 'TINKLE', 'CREAK', 'SQUEAK', 'SCREECH', 'SIZZLE', 'FIZZLE',
    'DRIP', 'PLOP', 'SPLISH', 'SPLASH', 'SLURP', 'CHOMP', 'MUNCH', 'CRUNCH',
    'CLAP', 'STOMP', 'THWACK', 'SMACK', 'WHACK', 'BONK', 'CLUNK', 'THUNK',
    'DING', 'DONG', 'BONG', 'GONG', 'BUZZ', 'HUM', 'WHIR', 'PURR', 'CHIRP',
    'TWEET', 'HOWL', 'BARK', 'WOOF', 'MEOW', 'RIBBIT', 'QUACK',
}


def _split_merged_words(text):
    """Fix partially merged words like 'DOSOME' -> 'DO SOME', 'NOWWE'LL' -> 'NOW WE'LL'.
    Respects protected proper nouns to avoid splitting names like POCHITA."""
    if not _wordninja:
        return text
    result = []
    for word in text.split():
        core = re.sub(r'^[^a-zA-Z\']*|[^a-zA-Z\']*$', '', word)
        prefix = word[:word.index(core[0])] if core and core[0] in word else ''
        suffix = word[word.rindex(core[-1]) + 1:] if core and core[-1] in word else ''

        # Strip apostrophes for length/alpha check but keep them for splitting
        alpha_only = core.replace("'", "")
        if len(alpha_only) >= 3 and alpha_only.isalpha():
            # Skip protected proper nouns (character names, etc.)
            if alpha_only.upper() in _PROTECTED_NOUNS:
                result.append(word)
                continue

            parts = _wordninja.split(core.lower())
            if len(parts) > 1:
                # Verify it's actually merged (not a real dictionary word)
                is_real_word = False
                if _symspell:
                    suggestions = _symspell.lookup(alpha_only.lower(), 2, max_edit_distance=0)
                    is_real_word = len(suggestions) > 0
                if not is_real_word:
                    # Check if any split part is a protected noun — if so, don't split
                    parts_upper = [p.replace("'", "").upper() for p in parts]
                    if any(p in _PROTECTED_NOUNS for p in parts_upper):
                        result.append(word)
                        continue

                    rejoined = ' '.join(parts)
                    if alpha_only.isupper():
                        rejoined = rejoined.upper()
                    result.append(prefix + rejoined + suffix)
                    continue
            elif len(parts) == 1 and len(alpha_only) >= 6 and _symspell:
                # Wordninja didn't split, but word is long and not in dictionary
                # Try brute-force: find a split point where both halves are real words
                exact = _symspell.lookup(alpha_only.lower(), 2, max_edit_distance=0)
                if not exact:
                    best_split = None
                    for pos in range(2, len(alpha_only) - 1):
                        left = alpha_only[:pos].lower()
                        right = alpha_only[pos:].lower()
                        l_match = _symspell.lookup(left, 2, max_edit_distance=0)
                        r_match = _symspell.lookup(right, 2, max_edit_distance=0)
                        if l_match and r_match:
                            # Prefer splits where both words are common
                            score = min(l_match[0].count, r_match[0].count)
                            if best_split is None or score > best_split[2]:
                                best_split = (left, right, score)
                    if best_split:
                        rejoined = best_split[0] + ' ' + best_split[1]
                        if alpha_only.isupper():
                            rejoined = rejoined.upper()
                        result.append(prefix + rejoined + suffix)
                        continue
        result.append(word)
    return ' '.join(result)


def _clean_text(text):
    """Clean OCR output for TTS."""
    if not text:
        return ''

    # Strip common OCR artifacts
    text = text.strip()
    text = re.sub(r'</s>|<s>|<pad>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Reject any Japanese/Chinese/Korean characters
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]', text):
        return ''

    # Filter watermark text (scanlation sites)
    _lower = text.lower()
    if any(w in _lower for w in ['scans.com', 'read this only at', 'kirascans',
                                  'mangadex', 'scan this', 'visit our website',
                                  'read at our', 'only at our']):
        return ''
    # Filter hashtag URLs
    if text.startswith('#') and '.' in text:
        return ''

    # Replace non-Latin alphabetic chars with closest ASCII (e.g. Cyrillic Н→H, Е→E)
    # Vision sometimes detects Cyrillic lookalikes
    _CYRILLIC_MAP = str.maketrans('АВСЕНІКМОРТХаеорсух', 'ABCEHIKMOPTXaeopcyx')
    text = text.translate(_CYRILLIC_MAP)
    # Strip any remaining non-ASCII letters (but keep punctuation and numbers)
    text = re.sub(r'[^\x00-\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Apostrophe glue: OCR often splits contractions with spaces
    # "DON ' T" -> "DON'T", "I ' M" -> "I'M", "WE ' LL" -> "WE'LL"
    text = re.sub(r"(\w) ' (M|S|T|D|RE|VE|LL)\b", r"\1'\2", text, flags=re.IGNORECASE)
    # "Y ' KNOW" -> "Y'KNOW", "' CAUSE" -> "'CAUSE"
    text = re.sub(r"\bY ' KNOW\b", "Y'KNOW", text, flags=re.IGNORECASE)
    text = re.sub(r"\b' CAUSE\b", "'CAUSE", text, flags=re.IGNORECASE)

    # Fix missing apostrophes in contractions (Apple Vision sometimes drops them)
    # Only safe ones where the word without apostrophe isn't a real English word
    text = re.sub(r"\bWONT\b", "WON'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDONT\b", "DON'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCANT\b", "CAN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bISNT\b", "ISN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDIDNT\b", "DIDN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWASNT\b", "WASN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHASNT\b", "HASN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWERENT\b", "WEREN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCOULDNT\b", "COULDN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWOULDNT\b", "WOULDN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSHOULDNT\b", "SHOULDN'T", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTHATS\b", "THAT'S", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWHATS\b", "WHAT'S", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTHEYRE\b", "THEY'RE", text, flags=re.IGNORECASE)
    text = re.sub(r"\bYOURE\b", "YOU'RE", text, flags=re.IGNORECASE)
    text = re.sub(r"\bYOULL\b", "YOU'LL", text, flags=re.IGNORECASE)
    text = re.sub(r"\bYOUVE\b", "YOU'VE", text, flags=re.IGNORECASE)
    # These need context guards (IVE, ILL, ITS, HES, SHES, WERE are real words)
    text = re.sub(r"\bIVE\b(?=\s+(?:GOT|BEEN|HAD|DONE|SEEN|MADE|THOUGHT|NEVER|ALWAYS))", "I'VE", text, flags=re.IGNORECASE)

    # Remove stray symbols between words (OCR neighbor bleed): "to = become" -> "to become"
    text = re.sub(r'\s+[=|/\\<>@#$%^&*~`]+\s+', ' ', text)
    # Remove stray single punctuation tokens: "the . prince" -> "the prince"
    text = re.sub(r'(?<=\w)\s+[.,;:]\s+(?=\w)', ' ', text)
    # Remove stray single letters between words (neighbor bleed)
    # Keeps "I" only at sentence start or before common verbs, keeps "A" before nouns
    # Removes: "have I justice" -> "have justice", "the s our" -> "the our"
    text = re.sub(r'(?<=[a-zA-Z])\s+[B-HJ-Zb-hj-z]\s+(?=[a-zA-Z])', ' ', text)
    # Strip leading garbage: single letter/fragment + comma at start
    # "L, Perhaps" -> "Perhaps", "N, We" -> "We"
    text = re.sub(r'^[A-Z],\s+', '', text)
    # Strip leading apostrophe fragments: "'en army" -> "army"
    text = re.sub(r"^'[a-z]{1,3}\s+", '', text)
    # Strip leading stray short tokens (1-2 chars) followed by comma/space
    text = re.sub(r'^[A-Za-z]{1,2}[,;]\s+', '', text)

    # Normalize repeated letter emphasis: "SOOO" -> "SOO", "NOOO" -> "NOO"
    # Cap any letter run to max 2 (keeps "OOH", "OOF" intact)
    text = re.sub(r'([A-Za-z])\1{2,}', r'\1\1', text)

    if not text:
        return ''

    # Reject text that's mostly non-ASCII (garbled OCR output)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if len(text) > 0 and ascii_chars / len(text) < 0.6:
        return ''

    # Fix leading dashes/dots from OCR ("- GUESS" -> "GUESS", ".GOOD" -> "GOOD")
    text = re.sub(r'^[-.\s]+', '', text).strip()

    # Strip leading garbage digits from OCR noise (e.g. "040120 THAT BRINGS" -> "THAT BRINGS")
    # Only if followed by a real word
    text = re.sub(r'^[\d\s,.]+(?=[A-Z]{2})', '', text).strip()

    # Fix hyphenated line breaks: "BAKU- SAN" -> "BAKU-SAN"
    text = re.sub(r'([A-Z]+)-\s+([A-Z])', r'\1-\2', text)
    # Fix "THRILL- ING" -> "THRILLING" (hyphen at syllable break)
    text = re.sub(r'(\w)-\s+([a-z])', r'\1\2', text)
    # Fix mid-word hyphens from vertical text OCR: "SOME-THING" -> "SOMETHING"
    # Only join if both halves are short (likely one word split by line break)
    def _fix_mid_hyphen(m):
        left, right = m.group(1), m.group(2)
        if len(left) <= 6 and len(right) <= 6:
            return left + right
        return m.group(0)
    text = re.sub(r'\b([A-Z]{2,6})-([A-Z]{2,6})\b', _fix_mid_hyphen, text)

    # Split merged contractions: "I'MNOT" -> "I'M NOT", "DON'TKNOW" -> "DON'T KNOW"
    text = re.sub(r"(\w'(?:M|S|T|RE|VE|LL|D))([A-Z])", r"\1 \2", text)

    # Skip dots-only or punctuation-only
    if _DOTS_ONLY.match(text) or _PUNCT_ONLY.match(text):
        return ''

    # Skip SFX (repeated chars like "WAAAA", "GRRRR")
    if _SFX_PATTERN.match(text):
        return ''

    # Skip single-word SFX common in manga
    sfx_words = {'CRUNCH', 'SIGH', 'BANG', 'CRASH', 'THUD', 'WHOOSH', 'SLASH',
                 'THUMP', 'CRACK', 'SNAP', 'BOOM', 'CLANG', 'SPLAT', 'WHAM',
                 'SWOOSH', 'RUMBLE', 'SHATTER', 'GROWL', 'ROAR', 'HISS',
                 'CLAP', 'STOMP', 'THWACK', 'SMACK', 'WHACK', 'BONK', 'CLUNK',
                 'THUNK', 'DING', 'DONG', 'BONG', 'GONG', 'BUZZ', 'DRIP',
                 'PLOP', 'SPLISH', 'SPLASH', 'SLURP', 'CHOMP', 'MUNCH',
                 'SIZZLE', 'FIZZLE', 'CREAK', 'SQUEAK', 'RATTLE', 'CLATTER',
                 'SPOTS', 'SNIP', 'SWIPE', 'SLICE', 'STAB', 'GRIP', 'GRAB',
                 'SHOVE', 'YANK', 'FLING', 'TOSS', 'DODGE', 'DASH', 'LUNGE'}
    stripped_upper = re.sub(r'[!?.…\s]+$', '', text.strip().upper())
    if stripped_upper in sfx_words:
        return ''

    # Skip interjections/exclamations (HUH?!, HMM, RAH, MURMUR, BNNT, etc.)
    if stripped_upper in _INTERJECTIONS:
        return ''

    # Skip text that is just a word repeated with punctuation (e.g. "HA HA HA", "HEH HEH")
    words_upper = text.strip().upper().split()
    unique_words = set(w.strip('!?.,…~') for w in words_upper)
    if len(words_upper) >= 2 and len(unique_words) == 1 and list(unique_words)[0] in _INTERJECTIONS:
        return ''

    # Skip gibberish repeated syllables (e.g. "GABA BA BA BA", "GEGYA GYA GYA", "AA HYA HYA")
    if len(words_upper) >= 3:
        cleaned_words = [w.strip('!?.,…~') for w in words_upper]
        # Check if most words are the same (repeated SFX/laughter)
        from collections import Counter
        counts = Counter(cleaned_words)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count >= len(cleaned_words) * 0.6 and len(most_common_word) <= 4:
            return ''

    # Skip very short garbage (but keep pure numbers like "1,200,000")
    alpha = re.sub(r'[^a-zA-Z]', '', text)
    digits = re.sub(r'[^0-9]', '', text)
    if len(alpha) < 2 and not digits:
        return ''

    # Skip garbage pure-digit strings (e.g. "2212222" from misread SFX)
    # Real numbers in manga have formatting (commas, periods) or are short
    if digits and not alpha:
        has_formatting = bool(re.search(r'[,.]', text))
        if not has_formatting and len(digits) > 4:
            return ''

    # Skip short single-word garbage (e.g. "NAt", "Fk", "Bx") — not real dialogue
    # Allow common short words: I, A, OK, NO, OH, HI, GO, SO, UP, IT, etc.
    _VALID_SHORT = {'I', 'A', 'OK', 'NO', 'OH', 'HI', 'GO', 'SO', 'UP', 'IT', 'IS',
                    'IN', 'ON', 'AT', 'TO', 'DO', 'IF', 'OR', 'AN', 'AS', 'BE', 'BY',
                    'HE', 'ME', 'MY', 'OF', 'WE', 'US', 'AM', 'HA', 'AH', 'YO', 'YA',
                    'HM', 'UM', 'OW', 'AW', 'EH', 'OI', 'OX', 'AX', 'YES', 'NOT',
                    'NOW', 'BUT', 'AND', 'THE', 'FOR', 'HAS', 'HAD', 'HIS', 'HER',
                    'HOW', 'WHO', 'WHY', 'LET', 'GET', 'GOT', 'RUN', 'DIE', 'YOU',
                    'ALL', 'OUT', 'CAN', 'MAN', 'OLD', 'NEW', 'BIG', 'ONE', 'TWO',
                    'TEN', 'SIR', 'WAR', 'GOD', 'END', 'WIN', 'SEE', 'SAY', 'ASK',
                    'TRY', 'CUT', 'HIT', 'PUT', 'SET', 'SIT', 'LIE', 'RAN', 'SAT',
                    'EAT', 'ATE', 'MOM', 'DAD', 'SON', 'BOY', 'AGO', 'YET', 'OUR',
                    'HEY', 'WOW', 'NAH', 'YEP', 'YUP', 'NOP', 'GAH', 'BAH'}
    if len(words_upper) == 1 and len(alpha) <= 3 and not digits:
        if stripped_upper not in _VALID_SHORT:
            return ''

    # Skip short text (1-2 words, <6 alpha chars) that is mostly punctuation/exclamation
    if len(alpha) <= 5 and len(words_upper) <= 2 and not digits:
        punct_count = sum(1 for c in text if c in '!?.…')
        if punct_count >= len(alpha):
            return ''

    text = _fix_common_digit_misreads(text)

    # Fix common Apple Vision misreads on handwritten manga fonts
    # "I'W" → "I'LL" (W misread for LL in handwritten style)
    text = re.sub(r"\bI'W\b", "I'LL", text, flags=re.IGNORECASE)
    # "YOU'O" → "YOU'D" (O misread for D)
    text = re.sub(r"\bYOU'O\b", "YOU'D", text, flags=re.IGNORECASE)
    # "FRAI" → "FRAU" (I misread for U in names — common in this manga)
    text = re.sub(r"\bFRAI\b", "FRAU", text, flags=re.IGNORECASE)
    # "SURRENTER" → "SURRENDER" (common Apple Vision misread on handwritten text)
    text = re.sub(r"\bSURRENTER\b", "SURRENDER", text, flags=re.IGNORECASE)
    # "SOME- THING?" → "SOMETHING?" (hyphenated line break with space)
    text = re.sub(r'(\w+)-\s+(\w+)\?', r'\1\2?', text)

    # Reject garbage: mostly random short tokens (misread dots, Japanese, noise)
    # e.g. "rth bp Sry i xe Hy Wy it i", "s 4 Ry of SHA"
    final_words = text.split()
    if len(final_words) >= 3:
        short = sum(1 for w in final_words if len(re.sub(r'[^a-zA-Z]', '', w)) <= 2)
        if short / len(final_words) >= 0.6:
            return ''

    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    return text


_DIGIT_MISREAD_PATTERNS = (
    (r'\b10\b', 'TO'),
    (r'\b1\b', 'I'),
    (r'\b0\b', 'O'),
)

# Words near a digit that mean it is a real number, not an OCR misread
_NUMERIC_CONTEXT_WORDS = {
    'MINUTES', 'MINUTE', 'MIN', 'MINS',
    'HOURS', 'HOUR', 'HRS',
    'SECONDS', 'SECOND', 'SECS', 'SEC',
    'DAYS', 'DAY',
    'WEEKS', 'WEEK',
    'MONTHS', 'MONTH',
    'YEARS', 'YEAR', 'YRS',
    'TIMES', 'TIME',
    'PERCENT', '%',
    'MILES', 'MILE',
    'METERS', 'METER',
    'FEET', 'FOOT',
    'INCHES', 'INCH',
    'POUNDS', 'POUND', 'LBS',
    'CHAPTER', 'CHAPTERS', 'CH',
    'VOLUME', 'VOLUMES', 'VOL',
    'LEVEL', 'LEVELS', 'LV', 'LVL',
    'PAGE', 'PAGES',
    'EPISODE', 'EPISODES', 'EP',
    'NUMBER', 'NO',
    'STAGE', 'STAGES',
    'RANK', 'RANKS',
    'GRADE', 'GRADES',
    'FLOOR', 'FLOORS',
    'ROUND', 'ROUNDS',
    'STEP', 'STEPS',
    'POINT', 'POINTS', 'PTS',
    'KILL', 'KILLS',
    'AGE',
    'KM', 'KG', 'CM', 'MM',
    'ST', 'ND', 'RD', 'TH',
}


def _looks_like_manga_caps(text):
    alpha_chars = [c for c in text if c.isalpha()]
    return bool(alpha_chars) and all(c.isupper() for c in alpha_chars)


def _has_numeric_context(text, match):
    """Return True if the digit is near other numbers or measurement words."""
    start, end = match.start(), match.end()
    if start > 0 and text[start - 1].isdigit():
        return True
    if end < len(text) and text[end].isdigit():
        return True
    after_m = re.match(r'\s+(\S+)', text[end:])
    if after_m:
        w = after_m.group(1).strip('.,!?;:').upper()
        if w in _NUMERIC_CONTEXT_WORDS:
            return True
    before_m = re.search(r'(\S+)\s+$', text[:start])
    if before_m:
        w = before_m.group(1).strip('.,!?;:').upper()
        if w in _NUMERIC_CONTEXT_WORDS:
            return True
    return False


def _fix_common_digit_misreads(text):
    if not text or not _looks_like_manga_caps(text):
        return text
    digit_groups = re.findall(r'\d+', text)
    if len(digit_groups) != 1 or len(digit_groups[0]) > 2:
        return text
    for pattern, replacement in _DIGIT_MISREAD_PATTERNS:
        m = re.search(pattern, text)
        if m and not _has_numeric_context(text, m):
            text = re.sub(pattern, replacement, text)
    return text


def _grammar_fix(text):
    """Apply grammar correction via LanguageTool if available.
    Only applies punctuation/grammar rules — spelling rules are disabled
    because they mangle manga names and Japanese terms."""
    if not text or not _grammar_ready or not _grammar_tool:
        return text
    try:
        import language_tool_python
        matches = _grammar_tool.check(text)
        # Filter out spelling rules — they destroy manga names/honorifics
        # Keep only grammar, punctuation, and whitespace rules
        _BLOCKED_CATEGORIES = {'TYPOS', 'SPELLING', 'MORFOLOGIK_RULE_EN_US',
                               'HUNSPELL_RULE', 'CONFUSED_WORDS'}
        filtered = [m for m in matches
                    if m.ruleId not in _BLOCKED_CATEGORIES
                    and m.category not in _BLOCKED_CATEGORIES
                    and 'SPELLER' not in m.ruleId
                    and 'MORFOLOGIK' not in m.ruleId
                    and 'SPELL' not in (m.category or '')]
        if filtered:
            corrected = language_tool_python.utils.correct(text, filtered)
            if corrected and corrected != text:
                log.info(f'  [Grammar] "{text[:50]}" -> "{corrected[:50]}"')
                return corrected
    except Exception:
        pass
    return text


# =============================================================================
# Number-to-words for TTS (so "6000" is spoken as "six thousand")
# =============================================================================

_ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
         'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
         'sixteen', 'seventeen', 'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
         'eighty', 'ninety']

def _num_to_words(n):
    """Convert integer to English words. Handles 0 to 999,999,999."""
    if n == 0:
        return 'zero'
    if n < 0:
        return 'negative ' + _num_to_words(-n)

    parts = []
    if n >= 1_000_000:
        parts.append(_num_to_words(n // 1_000_000) + ' million')
        n %= 1_000_000
    if n >= 1000:
        parts.append(_num_to_words(n // 1000) + ' thousand')
        n %= 1000
    if n >= 100:
        parts.append(_ones_word(n // 100) + ' hundred')
        n %= 100
    if n >= 20:
        w = _TENS[n // 10]
        if n % 10:
            w += '-' + _ONES[n % 10]
        parts.append(w)
    elif n > 0:
        parts.append(_ONES[n])
    return ' '.join(parts)

def _ones_word(n):
    return _ONES[n] if n < 20 else _TENS[n // 10] + ('-' + _ONES[n % 10] if n % 10 else '')

# Ordinal suffixes
_ORDINALS = {'1st': 'first', '2nd': 'second', '3rd': 'third'}

def _prepare_tts_text(text):
    """Preprocess text for TTS: fix number artifacts, normalize case, expand numbers."""
    # Fix O/0 confusion before anything else: "60O00" -> "60000"
    def _fix_digit_o_tts(m):
        token = m.group(0)
        if any(c.isdigit() for c in token) and 'O' in token:
            return token.replace('O', '0')
        return token
    text = re.sub(r'[0-9O][0-9O,.:]+[0-9O]', _fix_digit_o_tts, text)
    # "OOO" -> "000" when likely a number
    text = re.sub(r'\bOOO+\b', lambda m: '0' * len(m.group(0)), text)
    # Period as comma in numbers: "8.04020" -> "804020"
    text = re.sub(r'(\d)\.(\d{3,})', r'\1\2', text)
    # Mid-word hyphens: "SOME-THING" -> "SOMETHING"
    text = re.sub(r'\b([A-Za-z]{2,6})-([A-Za-z]{2,6})\b',
                  lambda m: m.group(1) + m.group(2) if len(m.group(1)) <= 6 and len(m.group(2)) <= 6 else m.group(0),
                  text)

    # Check if text is mostly uppercase BEFORE number conversion
    # (number words are lowercase and would dilute the ratio)
    alpha = [c for c in text if c.isalpha()]
    is_shouting = alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.5

    # Normalize ALL CAPS to sentence case first — Kokoro speaks this more naturally
    if is_shouting:
        text = text[0].upper() + text[1:].lower() if text else text
        # Re-capitalize "I" as a standalone word
        text = re.sub(r'\bi\b', 'I', text)
        # Capitalize after sentence-ending punctuation
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

    # Ordinals: 1st, 2nd, 3rd, 4th, 21st, etc.
    def _ordinal_replace(m):
        num = int(m.group(1))
        key = m.group(0).lower()
        if key in _ORDINALS:
            return _ORDINALS[key]
        word = _num_to_words(num)
        if word.endswith('y'):
            word = word[:-1] + 'ieth'
        elif word.endswith('ve'):
            word = word[:-2] + 'fth'
        elif word.endswith('t') and not word.endswith('ht'):
            word = word + 'h'
        elif word.endswith('e') and not word.endswith('le'):
            word = word[:-1] + 'th'
        elif word.endswith('n') and word.endswith('teen'):
            word = word + 'th'
        else:
            word = word + 'th'
        return word

    text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', _ordinal_replace, text, flags=re.IGNORECASE)

    # Strip commas from numbers: "100,000" -> "100000", "38,40000" -> "3840000"
    text = re.sub(r'(\d),(\d)', r'\1\2', text)

    # Standalone numbers: "6000" -> "six thousand"
    def _number_replace(m):
        num_str = m.group(0)
        try:
            n = int(num_str)
            if n > 999_999_999:
                return num_str  # too large, leave as-is
            return _num_to_words(n)
        except ValueError:
            return num_str

    text = re.sub(r'\b\d+\b', _number_replace, text)

    # Normalize punctuation runs for cleaner TTS
    # "!!!" -> "!", "???" -> "?", "..." -> "."
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{2,}', '.', text)
    # Mixed punctuation: "?!" -> "?", "!?" -> "!"
    text = re.sub(r'[!?]{2,}', lambda m: m.group(0)[0], text)

    return text


# =============================================================================
# Image helpers
# =============================================================================

def _decode_image(image_data):
    """Decode base64 data URL or raw base64 to OpenCV image."""
    if image_data.startswith('data:'):
        image_data = image_data.split(',', 1)[1]
    img_bytes = base64.b64decode(image_data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# =============================================================================
# Kokoro TTS
# =============================================================================



def _tts_cache_key(text, voice, speed):
    return (text, voice, round(float(speed), 4))


def _get_cached_tts(text, voice, speed):
    key = _tts_cache_key(text, voice, speed)
    with _tts_cache_lock:
        return _tts_cache.get(key, None)


def _add_tts_cache(text, voice, speed, audio_bytes):
    key = _tts_cache_key(text, voice, speed)
    with _tts_cache_lock:
        _tts_cache[key] = audio_bytes
        # Evict oldest entries if cache gets too large
        if len(_tts_cache) > 100:
            oldest_keys = list(_tts_cache.keys())[:-100]
            for k in oldest_keys:
                del _tts_cache[k]



def _piper_generate(text, speed=1.0):
    """Generate WAV audio using Piper TTS. Returns bytes."""
    from piper.voice import SynthesisConfig
    syn_config = SynthesisConfig()
    syn_config.length_scale = 1.0 / max(0.5, speed)
    buf = io.BytesIO()
    wf = wave.open(buf, 'wb')
    with _piper_lock:
        _piper_voice.synthesize_wav(text, wf, syn_config=syn_config)
    try:
        wf.close()
    except Exception:
        pass
    return buf.getvalue()


def _generate_page_audio(audio_id, bubbles, voice='_piper', speed=1.0):
    """Generate per-bubble WAV clips with Piper TTS. ~150ms per bubble."""
    if not bubbles or _piper_voice is None:
        with _page_audio_lock:
            _page_audio_cache[audio_id] = {'clips': [], 'total': 0, 'done': True}
        return

    total = len(bubbles)
    with _page_audio_lock:
        _page_audio_cache[audio_id] = {'clips': [], 'total': total, 'done': False}

    for i in range(total):
        if _shutting_down:
            break
        text = _prepare_tts_text(bubbles[i].get('text', '').strip())
        if not text:
            with _page_audio_lock:
                _page_audio_cache[audio_id]['clips'].append(b'')
            continue
        cached = _get_cached_tts(text, voice, speed)
        if cached:
            log.info(f'[PageAudio] Bubble {i}/{total}: "{text[:40]}" (cache hit)')
            with _page_audio_lock:
                _page_audio_cache[audio_id]['clips'].append(cached)
            continue
        try:
            audio_bytes = _piper_generate(text, speed)
            _add_tts_cache(text, voice, speed, audio_bytes)
            log.info(f'[PageAudio] Bubble {i}/{total}: "{text[:40]}" ({len(audio_bytes)} bytes)')
            with _page_audio_lock:
                _page_audio_cache[audio_id]['clips'].append(audio_bytes)
        except Exception as e:
            log.info(f'[PageAudio] Error on bubble {i}: {e}')
            with _page_audio_lock:
                _page_audio_cache[audio_id]['clips'].append(b'')

    with _page_audio_lock:
        _page_audio_cache[audio_id]['done'] = True
    log.info(f'[PageAudio] All {total} bubbles generated for {audio_id}')

    # Clean old entries (keep last 5)
    with _page_audio_lock:
        if len(_page_audio_cache) > 5:
            oldest = list(_page_audio_cache.keys())[:-5]
            for k in oldest:
                del _page_audio_cache[k]


# =============================================================================
# Background quality pass (Florence-2)
# =============================================================================


def _run_quality_pass(request_id, img_bgr, bubble_data, dpr, actual_crop_left, actual_crop_top, reading_dir):
    """Run Florence-2 only on LOW-QUALITY bubbles (short text, low score, suspicious).
    Skips bubbles where primary OCR already produced good text.
    Typically re-OCRs 2-5 bubbles instead of all 13 → 12-30s instead of 78s."""
    if _florence2 is None:
        return

    t_start = time.time()
    total = len(bubble_data)
    log.info(f'[Quality] Evaluating {total} bubbles for re-OCR (id={request_id[:8]})')

    improved_bubbles = []
    crops_rerun = 0
    skipped = 0

    for i, (det, crop_img, rapid_text, rapid_score) in enumerate(bubble_data):
        # Decide if this bubble needs Florence-2 re-OCR
        needs_reocr = False
        words = rapid_text.split() if rapid_text else []

        if not rapid_text:
            needs_reocr = True  # primary OCR got nothing
        elif rapid_score < 5:
            needs_reocr = True  # Low quality score
        elif len(words) <= 2 and len(rapid_text) < 10:
            needs_reocr = True  # Very short text (might be truncated)
        elif any(len(w.replace("'", "")) > 8 for w in words):
            needs_reocr = True  # Likely merged words (lowered from 12)
        elif len(words) > 0 and sum(len(w) for w in words) / len(words) > 6.5:
            needs_reocr = True  # Average word length too high (space-sparse)

        if not needs_reocr:
            skipped += 1
            improved_bubbles.append(None)  # Keep primary OCR result
            continue

        try:
            florence_text = _florence2_ocr(crop_img, max_tokens=60)
            if florence_text:
                cleaned = _clean_text(florence_text)
                # Guard: only accept if Florence-2 text is strictly longer
                # and doesn't introduce doubled letters that weren't in the original
                accept = False
                if cleaned and len(cleaned) > len(rapid_text or '') and _ocr_score(cleaned) > rapid_score:
                    accept = True
                    # Reject if Florence-2 introduced double letters not in original
                    if rapid_text:
                        doubled = re.findall(r'(.)\1', cleaned.lower())
                        orig_doubled = set(re.findall(r'(.)\1', rapid_text.lower()))
                        for pair in doubled:
                            if pair not in orig_doubled:
                                accept = False
                                log.info(f'[Quality] Bubble {i} rejected: new doubled letter "{pair}{pair}" not in original')
                                break
                if accept:
                    improved_bubbles.append({
                        'text': cleaned,
                        'conf': round(det['score'], 3),
                        'left': round((actual_crop_left + det['x']) / dpr, 1),
                        'top': round((actual_crop_top + det['y']) / dpr, 1),
                        'width': round(det['w'] / dpr, 1),
                        'height': round(det['h'] / dpr, 1),
                        'source': 'florence2',
                    })
                    crops_rerun += 1
                    log.info(f'[Quality] Bubble {i} upgraded: "{rapid_text[:30]}" → "{cleaned[:30]}"')
                    continue
        except Exception as e:
            log.info(f'[Quality] Florence-2 crop {i} error: {e}')

        improved_bubbles.append(None)

    # Sort same as fast pass
    row_height = max(1, img_bgr.shape[0] // 4)
    if reading_dir == 'rtl':
        valid = [(b, i) for i, b in enumerate(improved_bubbles) if b is not None]
        valid.sort(key=lambda x: (x[0]['top'] * dpr // row_height, -x[0]['left']))
        improved_bubbles = [b for b, _ in valid]
    else:
        valid = [(b, i) for i, b in enumerate(improved_bubbles) if b is not None]
        valid.sort(key=lambda x: (x[0]['top'] * dpr // row_height, x[0]['left']))
        improved_bubbles = [b for b, _ in valid]

    elapsed = int((time.time() - t_start) * 1000)
    log.info(f'[Quality] Done: {len(improved_bubbles)} bubbles, {crops_rerun} re-OCRd ({elapsed}ms) (id={request_id[:8]})')

    with _quality_lock:
        _quality_results[request_id] = {
            'bubbles': improved_bubbles,
            'ready': True,
            'timing_ms': elapsed,
        }
        # Keep only last 5 results to avoid memory leak
        if len(_quality_results) > 5:
            oldest = list(_quality_results.keys())[:-5]
            for k in oldest:
                del _quality_results[k]


# =============================================================================
# Idle shutdown
# =============================================================================

def _idle_monitor():
    global _shutting_down
    while not _shutting_down:
        time.sleep(30)
        idle = time.time() - _last_activity
        if idle > IDLE_SHUTDOWN_MINUTES * 60:
            log.info(f'Idle for {IDLE_SHUTDOWN_MINUTES}min, shutting down')
            _shutting_down = True
            sys.exit(0)


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    return jsonify({
        'name': 'MangaVoice Lite',
        'status': 'ok',
        'endpoints': ['/health', '/process', '/tts', '/tts/status', '/heartbeat', '/shutdown'],
        'engine': 'RT-DETR + Apple Vision/Tesseract + Kokoro (ONNX-only)',
    })


@app.route('/health', methods=['GET'])
def health():
    status = 'ok' if _models_ready else 'loading'
    return jsonify({
        'status': status,
        'detector': _detector_session is not None,
        'ocr': 'Apple Vision' if _apple_vision_available else 'Tesseract',
        'florence': _florence2 is not None,
        'quality_pass': _florence2 is not None,
        'tts': _piper_voice is not None,
    })


@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    global _last_activity
    _last_activity = time.time()
    return jsonify({'ok': True})


@app.route('/process', methods=['POST'])
def process():
    global _last_activity
    _last_activity = time.time()

    if _shutting_down:
        return jsonify({'error': 'Server is shutting down'}), 503
    if not _models_ready:
        return jsonify({'error': 'Server is starting up, models still loading...'}), 503

    t_start = time.time()
    data = request.get_json(force=True, silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image field'}), 400

    dpr = data.get('dpr', 2)
    if not isinstance(dpr, (int, float)) or dpr < 1 or dpr > 5:
        dpr = 2
    reading_dir = data.get('readingDirection', 'rtl')

    # Decode image
    t0 = time.time()
    img = _decode_image(data['image'])
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400
    decode_ms = int((time.time() - t0) * 1000)
    img_h, img_w = img.shape[:2]
    log.info(f'Image: {img_w}x{img_h}, dpr={dpr} (decode {decode_ms}ms)')

    # Crop if provided
    crop = data.get('cropRect')
    actual_crop_left = 0
    actual_crop_top = 0
    if crop and crop.get('width') and crop.get('height'):
        cx = int(crop['left'] * dpr)
        cy = int(crop['top'] * dpr)
        cw = int(crop['width'] * dpr)
        ch = int(crop['height'] * dpr)
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        cw = min(cw, img_w - cx)
        ch = min(ch, img_h - cy)
        if cw > img_w * 0.15 and ch > img_h * 0.15:
            img = img[cy:cy + ch, cx:cx + cw]
            actual_crop_left = cx
            actual_crop_top = cy

    cropped_h, cropped_w = img.shape[:2]

    # Step 1: Detect bubbles with RT-DETR
    t_det = time.time()
    detections = _detect_bubbles(img)
    det_ms = int((time.time() - t_det) * 1000)

    # Use text_bubble regions for OCR. Also use bubble regions that contain
    # no text_bubble inside them (those are standalone speech bubbles).
    text_regions = [d for d in detections if d['class'] == 'text_bubble']
    bubble_regions = [d for d in detections if d['class'] == 'bubble']

    # Add bubble regions that don't overlap with any text_bubble
    for bub in bubble_regions:
        has_text_inside = False
        for tr in text_regions:
            # Check if text_bubble is inside this bubble
            if (tr['x'] >= bub['x'] - 10 and tr['y'] >= bub['y'] - 10 and
                tr['x'] + tr['w'] <= bub['x'] + bub['w'] + 10 and
                tr['y'] + tr['h'] <= bub['y'] + bub['h'] + 10):
                has_text_inside = True
                break
        if not has_text_inside:
            text_regions.append(bub)

    log.info(f'Detection: {len(detections)} total, {len(text_regions)} OCR targets '
             f'({len([d for d in detections if d["class"] == "text_bubble"])} text_bubble, '
             f'{len(bubble_regions)} bubble) ({det_ms}ms)')
    for d in detections:
        log.info(f'  [{d["class"]}] ({d["x"]},{d["y"]}) {d["w"]}x{d["h"]} conf={d.get("score",0):.2f}')

    # Step 2: OCR (parallel — each bubble independently)
    t_ocr = time.time()

    # Prepare crops: use gradient-based mask to isolate bubble interior
    ocr_tasks = []
    for det in text_regions:
        # Extract bubble mask using gradient-driven region growing
        refined_box, bubble_mask, is_dark = _extract_bubble_mask(img, det)
        rx, ry, rw, rh = refined_box
        det['_rx'], det['_ry'], det['_rw'], det['_rh'] = rx, ry, rw, rh

        # Crop the refined box region from the image
        pad_x = max(5, int(rw * 0.05))
        pad_y = max(5, int(rh * 0.05))
        x1 = max(0, rx - pad_x)
        y1 = max(0, ry - pad_y)
        x2 = min(cropped_w, rx + rw + pad_x)
        y2 = min(cropped_h, ry + rh + pad_y)
        crop_img = img[y1:y2, x1:x2].copy()
        if crop_img.size == 0:
            continue

        # Apply the bubble mask — white-out everything outside the bubble
        # The mask is in the padded ROI coordinate space from _extract_bubble_mask
        mask_pad = int(max(10, min(60, max(det['w'], det['h']) * 0.12)))
        mask_x1 = max(0, det['x'] - mask_pad)
        mask_y1 = max(0, det['y'] - mask_pad)
        # Map crop region onto mask coordinates
        off_y = y1 - mask_y1
        off_x = x1 - mask_x1
        ch, cw = crop_img.shape[:2]
        mh, mw = bubble_mask.shape[:2]
        # Extract the overlapping mask region for our crop
        src_y1 = max(0, off_y)
        src_x1 = max(0, off_x)
        src_y2 = min(mh, off_y + ch)
        src_x2 = min(mw, off_x + cw)
        dst_y1 = src_y1 - off_y
        dst_x1 = src_x1 - off_x
        dst_y2 = src_y2 - off_y
        dst_x2 = src_x2 - off_x
        # Start with background fill, then copy only masked pixels
        # Dark bubbles: black background (0) to preserve white text
        # Light bubbles: white background (255) to preserve black text
        bg_val = 0 if is_dark else 255
        masked_crop = np.full_like(crop_img, bg_val)
        if src_y2 > src_y1 and src_x2 > src_x1:
            mask_slice = bubble_mask[src_y1:src_y2, src_x1:src_x2]
            crop_slice = crop_img[dst_y1:dst_y2, dst_x1:dst_x2]
            if len(crop_img.shape) == 3:
                mask_3ch = mask_slice[:, :, np.newaxis] > 0
                masked_crop[dst_y1:dst_y2, dst_x1:dst_x2] = np.where(
                    mask_3ch, crop_slice, bg_val)
            else:
                masked_crop[dst_y1:dst_y2, dst_x1:dst_x2] = np.where(
                    mask_slice > 0, crop_slice, bg_val)
        if is_dark:
            log.info(f'  Dark bubble detected at ({rx},{ry})')
            det['_is_dark'] = True
        ocr_tasks.append((det, masked_crop))

    # For quality pass: collect (det, crop, text, score) for each bubble
    # Use a dict keyed by index for thread-safe concurrent access
    _quality_data_by_idx = {}
    _quality_data_lock = threading.Lock()

    def _process_one_bubble(det, crop_img, idx=0):
        """Classify + OCR + clean a single bubble crop. Returns dict or None."""
        if os.environ.get('MVR_DEBUG_CROPS'):
            try:
                os.makedirs('/tmp/mvr_crops', exist_ok=True)
                cv2.imwrite(f'/tmp/mvr_crops/crop_{idx:02d}_raw.png', crop_img)
            except Exception:
                pass
        if not _classify_crop(crop_img):
            log.info(f'  Classifier rejected crop at ({det["x"]},{det["y"]})')
            with _quality_data_lock:
                _quality_data_by_idx[idx] = (det, crop_img, '', -1)
            return None
        # Dark bubbles: invert to black-on-white before OCR
        ocr_input = crop_img
        if det.get('_is_dark'):
            ocr_input = cv2.bitwise_not(crop_img)
            if os.environ.get('MVR_DEBUG_CROPS'):
                try:
                    cv2.imwrite(f'/tmp/mvr_crops/crop_{idx:02d}_inverted.png', ocr_input)
                except Exception:
                    pass
        text = _ocr_crop(ocr_input)
        text = _clean_text(text)
        text = _grammar_fix(text)
        if not text:
            with _quality_data_lock:
                _quality_data_by_idx[idx] = (det, crop_img, '', -1)
            return None
        score = _ocr_score(text)
        with _quality_data_lock:
            _quality_data_by_idx[idx] = (det, crop_img, text, score)
        if len(text) > 4 and score < 0:
            log.info(f'  Low OCR score ({score:.0f}), rejected: "{text[:40]}"')
            return None
        # Use contour-refined box for highlight coordinates (set during crop prep)
        rx = det.get('_rx', det['x'])
        ry = det.get('_ry', det['y'])
        rw = det.get('_rw', det['w'])
        rh = det.get('_rh', det['h'])
        return {
            'text': text,
            'conf': round(det['score'], 3),
            'left': round((actual_crop_left + rx) / dpr, 1),
            'top': round((actual_crop_top + ry) / dpr, 1),
            'width': round(rw / dpr, 1),
            'height': round(rh / dpr, 1),
        }

    # Run OCR in parallel (ONNX Runtime sessions are thread-safe)
    bubbles = []
    max_workers = min(4, len(ocr_tasks)) if ocr_tasks else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one_bubble, det, crop, i): (i, det)
            for i, (det, crop) in enumerate(ocr_tasks)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                idx, det = futures[future]
                log.error(f'  Bubble {idx} processing failed: {e}')
                result = None
            if result is not None:
                bubbles.append(result)

    ocr_ms = int((time.time() - t_ocr) * 1000)

    # Sort reading order
    row_height = max(1, cropped_h // 4)
    if reading_dir == 'rtl':
        bubbles.sort(key=lambda b: (b['top'] * dpr // row_height, -b['left']))
    else:
        bubbles.sort(key=lambda b: (b['top'] * dpr // row_height, b['left']))

    total_ms = int((time.time() - t_start) * 1000)
    log.info(f'Found {len(bubbles)} bubbles ({det_ms}ms detect, {ocr_ms}ms OCR, {total_ms}ms total)')
    for i, b in enumerate(bubbles):
        log.info(f'  [{i+1}] {b["text"][:60]}')

    voice = data.get('voice', 'af_heart')
    speed = data.get('speed', 1.0)

    # Generate one-shot page audio in background
    audio_id = str(uuid.uuid4())[:8]
    use_server_tts = (voice != '_browser') and (_piper_voice is not None)
    if use_server_tts and bubbles:
        with _page_audio_lock:
            _page_audio_cache[audio_id] = {'audio': b'', 'timestamps': [], 'ready': False}
        threading.Thread(
            target=_generate_page_audio,
            args=(audio_id, bubbles, voice, float(speed)),
            daemon=True
        ).start()
    else:
        audio_id = None

    # Kick off background Florence-2 quality pass (if available)
    request_id = str(uuid.uuid4())
    quality_pending = False
    _quality_data = [_quality_data_by_idx[i] for i in sorted(_quality_data_by_idx)]
    if _quality_executor is not None and _florence2 is not None and len(_quality_data) > 0:
        with _quality_lock:
            _quality_results[request_id] = {'bubbles': [], 'ready': False}
        _quality_executor.submit(
            _run_quality_pass, request_id, img, _quality_data,
            dpr, actual_crop_left, actual_crop_top, reading_dir
        )
        quality_pending = True
        log.info(f'Background quality pass queued (id={request_id[:8]})')

    return jsonify({
        'bubbles': bubbles,
        'audioId': audio_id,
        'actualCropLeft': round(actual_crop_left / dpr, 1),
        'actualCropTop': round(actual_crop_top / dpr, 1),
        'requestId': request_id if quality_pending else None,
        'qualityPending': quality_pending,
        'timing': {
            'total_ms': total_ms,
            'decode_ms': decode_ms,
            'detect_ms': det_ms,
            'ocr_ms': ocr_ms,
            'boxes_detected': len(text_regions),
            'bubbles_returned': len(bubbles),
        },
    })


@app.route('/process/audio', methods=['GET'])
def process_audio():
    """Serve individual bubble audio. ?id=X&bubble=N returns WAV for bubble N.
    If bubble N isn't ready yet, returns 202 with progress info."""
    global _last_activity
    _last_activity = time.time()

    audio_id = request.args.get('id')
    if not audio_id:
        return jsonify({'error': 'Missing id parameter'}), 400

    bubble_idx = request.args.get('bubble')
    if bubble_idx is None:
        return jsonify({'error': 'Missing bubble parameter'}), 400
    try:
        bubble_idx = int(bubble_idx)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid bubble parameter'}), 400

    with _page_audio_lock:
        entry = _page_audio_cache.get(audio_id)
        if entry is None:
            return jsonify({'error': 'Unknown audio id'}), 404
        clips = list(entry.get('clips', []))
        total = entry.get('total', 0)
        done = entry.get('done', False)

    ready_count = len(clips)

    if bubble_idx >= total:
        return jsonify({'error': 'Bubble index out of range'}), 400

    if bubble_idx >= ready_count:
        return jsonify({'ready': False, 'generated': ready_count, 'total': total}), 202

    clip = clips[bubble_idx]
    if not clip:
        # Empty clip (bubble had no text)
        return jsonify({'ready': True, 'empty': True})

    return Response(clip, mimetype='audio/wav')


@app.route('/process/quality', methods=['GET'])
def process_quality():
    """Poll for background Florence-2 quality pass results."""
    request_id = request.args.get('id')
    if not request_id:
        return jsonify({'error': 'Missing id parameter'}), 400

    with _quality_lock:
        result = _quality_results.get(request_id)

    if result is None:
        return jsonify({'ready': False})

    if not result['ready']:
        return jsonify({'ready': False})

    return jsonify({
        'ready': True,
        'bubbles': result['bubbles'],
        'timing_ms': result.get('timing_ms', 0),
    })


@app.route('/tts', methods=['POST'])
def tts_endpoint():
    global _last_activity
    _last_activity = time.time()

    if _shutting_down:
        return jsonify({'error': 'Server is shutting down'}), 503

    data = request.get_json(force=True, silent=True) or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    voice = data.get('voice', 'af_heart')
    speed = float(data.get('speed', 1.0))

    # Expand numbers for natural speech: "6000" -> "six thousand"
    text = _prepare_tts_text(text)

    log.info(f'[TTS] "{text[:60]}" voice={voice} speed={speed}')
    cached = _get_cached_tts(text, voice, speed)
    if cached is not None:
        log.info('[TTS] Cache hit')
        return Response(cached, mimetype='audio/wav')

    t0 = time.time()
    try:
        if _piper_voice is None:
            raise RuntimeError('Piper TTS not loaded')
        audio = _piper_generate(text, speed)
    except Exception as e:
        log.error(f'[TTS] Error: {e}')
        return jsonify({'error': 'TTS synthesis failed'}), 500
    ms = int((time.time() - t0) * 1000)
    log.info(f'[TTS] {len(audio)} bytes, {ms}ms')
    return Response(audio, mimetype='audio/wav')


@app.route('/tts/status', methods=['GET'])
def tts_status():
    return jsonify({
        'available': _piper_voice is not None,
        'engine': 'piper',
        'voices': [],
        'loaded': ['piper'] if _piper_voice is not None else [],
        'piper': _piper_voice is not None,
    })


# ─── Free Text Detection (separate from bubble pipeline) ─────────────────────

def _apple_vision_fullpage_ocr(img_bgr):
    """Run Apple Vision OCR on the full page image.
    Returns list of dicts: {text, x, y, w, h} in pixel coordinates.
    Unlike _apple_vision_ocr() which works on crops, this returns bounding boxes.
    """
    if not _apple_vision_available:
        return []

    results = []
    img_h, img_w = img_bgr.shape[:2]

    try:
        success, encoded = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            return []
        img_bytes = encoded.tobytes()
        if not img_bytes:
            return []

        data = NSData.dataWithBytes_length_(img_bytes, len(img_bytes))
        handler = VNImageRequestHandler.alloc().initWithData_options_(data, None)

        def completion_handler(req, error):
            if error:
                log.info(f'  [FreeText:Vision] handler error: {error}')
                return
            for obs in req.results() or []:
                candidates = obs.topCandidates_(1)
                if not candidates:
                    continue
                candidate = candidates[0]
                conf = float(candidate.confidence())
                text_line = candidate.string().strip()
                if not text_line or conf < 0.3:
                    continue
                # Vision coords: normalized 0-1, origin bottom-left
                box = obs.boundingBox()
                origin = getattr(box, 'origin', None)
                size = getattr(box, 'size', None)
                bx = float(origin.x) if origin else 0.0
                by = float(origin.y) if origin else 0.0
                bw = float(size.width) if size else 0.0
                bh = float(size.height) if size else 0.0
                # Convert to pixel coords (flip Y since Vision is bottom-left origin)
                px = int(bx * img_w)
                py = int((1.0 - by - bh) * img_h)
                pw = int(bw * img_w)
                ph = int(bh * img_h)
                results.append({
                    'text': text_line,
                    'conf': conf,
                    'x': px, 'y': py, 'w': pw, 'h': ph,
                })

        vn_request = VNRecognizeTextRequest.alloc().initWithCompletionHandler_(completion_handler)
        vn_request.setRecognitionLanguages_(('en-US',))
        vn_request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        vn_request.setUsesLanguageCorrection_(True)

        success, error = handler.performRequests_error_([vn_request], None)
        if not success:
            log.info(f'  [FreeText:Vision] failed: {error}')

    except Exception as exc:
        log.info(f'  [FreeText:Vision] exception: {exc}')

    return results


def _group_nearby_text(regions, img_h):
    """Group text lines that are close together into single blocks.
    Lines that are vertically near each other and horizontally overlapping
    get merged into one text region with combined text.
    """
    if not regions:
        return []

    # Sort by vertical position (top to bottom)
    sorted_regions = sorted(regions, key=lambda r: r['y'])

    # Union-find grouping: merge regions that are close
    groups = list(range(len(sorted_regions)))

    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            groups[ri] = rj

    for i in range(len(sorted_regions)):
        ri = sorted_regions[i]
        for j in range(i + 1, len(sorted_regions)):
            rj = sorted_regions[j]
            # Vertical gap: distance between bottom of one and top of the other
            gap_y = rj['y'] - (ri['y'] + ri['h'])
            if gap_y > ri['h'] * 1.5:
                # Too far apart vertically, stop checking (sorted by y)
                break
            # Horizontal overlap check: do they overlap or are very close?
            x1_i, x2_i = ri['x'], ri['x'] + ri['w']
            x1_j, x2_j = rj['x'], rj['x'] + rj['w']
            overlap_x = min(x2_i, x2_j) - max(x1_i, x1_j)
            min_w = min(ri['w'], rj['w'])
            # Merge if horizontally overlapping by at least 30% of the smaller width
            # or if centers are close
            cx_i = ri['x'] + ri['w'] / 2
            cx_j = rj['x'] + rj['w'] / 2
            centers_close = abs(cx_i - cx_j) < max(ri['w'], rj['w']) * 0.7
            if overlap_x > min_w * 0.3 or centers_close:
                union(i, j)

    # Collect groups
    from collections import defaultdict
    group_map = defaultdict(list)
    for i in range(len(sorted_regions)):
        group_map[find(i)].append(sorted_regions[i])

    # Merge each group into a single region
    merged = []
    for members in group_map.values():
        # Sort members top to bottom for correct text order
        members.sort(key=lambda r: r['y'])
        # Combine text (top to bottom reading order)
        combined_text = ' '.join(m['text'] for m in members)
        # Bounding box: union of all member boxes
        x1 = min(m['x'] for m in members)
        y1 = min(m['y'] for m in members)
        x2 = max(m['x'] + m['w'] for m in members)
        y2 = max(m['y'] + m['h'] for m in members)
        avg_conf = sum(m['conf'] for m in members) / len(members)
        merged.append({
            'text': combined_text,
            'conf': avg_conf,
            'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
        })

    return merged


def _boxes_overlap(a, b, margin=10):
    """Check if two boxes overlap (with margin). Each is {x, y, w, h}."""
    ax1, ay1, ax2, ay2 = a['x'] - margin, a['y'] - margin, a['x'] + a['w'] + margin, a['y'] + a['h'] + margin
    bx1, by1, bx2, by2 = b['x'] - margin, b['y'] - margin, b['x'] + b['w'] + margin, b['y'] + b['h'] + margin
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _is_freetext_junk(text):
    """Filter out SFX, credits, watermarks, and other non-dialogue free text."""
    if not text or not text.strip():
        return True

    stripped = text.strip()

    # Too short (single char)
    if len(stripped) <= 1:
        return True

    # Garbage from Japanese OCR misread: mostly short random tokens
    # e.g. "2i ne YF ts zee", "s 4 Ry of SHA"
    words = stripped.split()
    if len(words) >= 3:
        short_words = sum(1 for w in words if len(w) <= 2)
        if short_words / len(words) >= 0.5:
            return True

    # Lowercase-heavy short fragments are usually OCR noise, not manga dialogue
    # Real manga text is mostly uppercase
    alpha = re.sub(r'[^a-zA-Z]', '', stripped)
    if len(alpha) >= 3:
        upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if upper_ratio < 0.4 and len(words) >= 2:
            return True

    # Credits/watermarks: starts with @, contains URL-like patterns
    if stripped.startswith('@') or stripped.startswith('#'):
        return True
    lower = stripped.lower()
    if any(w in lower for w in ['.com', '.net', '.org', 'scans', 'mangadex',
                                 'translated by', 'typeset by', 'proofread',
                                 'scanlat', 'chapter', 'vol.', 'volume']):
        return True

    # Any Japanese/Chinese/Korean characters = reject
    # Hiragana U+3040-309F, Katakana U+30A0-30FF, CJK U+4E00-9FFF, CJK Ext U+3400-4DBF
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]', stripped):
        return True

    # Mostly non-ASCII
    ascii_chars = sum(1 for c in stripped if ord(c) < 128)
    if len(stripped) > 0 and ascii_chars / len(stripped) < 0.5:
        return True

    # SFX: single word, all caps, matches known SFX
    upper = re.sub(r'[!?.…\s]+$', '', stripped.upper())
    sfx_words = {'CRUNCH', 'SIGH', 'BANG', 'CRASH', 'THUD', 'WHOOSH', 'SLASH',
                 'THUMP', 'CRACK', 'SNAP', 'BOOM', 'CLANG', 'SPLAT', 'WHAM',
                 'SWOOSH', 'RUMBLE', 'SHATTER', 'GROWL', 'ROAR', 'HISS',
                 'CLAP', 'STOMP', 'THWACK', 'SMACK', 'WHACK', 'BONK', 'CLUNK',
                 'THUNK', 'DING', 'DONG', 'BONG', 'GONG', 'BUZZ', 'DRIP',
                 'PLOP', 'SPLISH', 'SPLASH', 'SLURP', 'CHOMP', 'MUNCH',
                 'SIZZLE', 'FIZZLE', 'CREAK', 'SQUEAK', 'RATTLE', 'CLATTER',
                 'CLENCH', 'SNIP', 'SWIPE', 'SLICE', 'STAB', 'GRIP', 'GRAB',
                 'SHOVE', 'YANK', 'FLING', 'TOSS', 'DODGE', 'DASH', 'LUNGE',
                 'FLUTTER', 'TREMBLE', 'SHIVER', 'TWITCH', 'JOLT', 'GASP',
                 'PANT', 'WHEEZE', 'COUGH', 'SNORE', 'YAWN', 'STRETCH',
                 'KNOCK', 'TAP', 'CLICK', 'BEEP', 'RING', 'CHIME', 'RUSTLE',
                 'SHUFFLE', 'STUMBLE', 'TUMBLE', 'SLIDE', 'SKID', 'SCREECH',
                 'VROOM', 'HONK', 'SPLASH', 'DRIZZLE', 'THUNDER', 'ZAP',
                 'FLASH', 'GLOW', 'SHINE', 'SPARKLE', 'GLINT', 'GLEAM',
                 'STARE', 'GLARE', 'PEEK', 'TURN', 'SPIN', 'TWIST', 'GRAB',
                 'PULL', 'PUSH', 'LIFT', 'DROP', 'CATCH', 'THROW', 'SWING',
                 'PUNCH', 'KICK', 'STEP', 'WALK', 'RUN', 'JUMP', 'LEAP',
                 'ROLL', 'CRAWL', 'CLIMB', 'FALL', 'LAND', 'STOP', 'HALT',
                 'FREEZE', 'SHUDDER', 'FLINCH', 'WINCE', 'CRINGE', 'CLING',
                 'PAT', 'RUB', 'SQUEEZE', 'HUG', 'KISS', 'BITE', 'LICK',
                 'CHEW', 'SPIT', 'SWALLOW', 'GULP', 'SIP', 'POUR', 'SPILL',
                 'DROOL', 'BLUSH', 'GRIN', 'SMIRK', 'FROWN', 'SCOWL', 'SNEER',
                 'NOD', 'SHAKE', 'WAVE', 'POINT', 'REACH', 'TOUCH', 'POKE',
                 'PROD', 'NUDGE', 'BUMP', 'SLAM', 'SHUT', 'OPEN', 'CLOSE',
                 'LOCK', 'BREAK', 'TEAR', 'RIP', 'CUT', 'SCRATCH', 'SCRAPE',
                 'DIG', 'PEEL', 'FOLD', 'WRAP', 'TIE', 'BIND', 'SQUEEZE'}
    if upper in sfx_words:
        return True

    # SFX pattern: repeated chars like "WAAAA", "GRRRR"
    if _SFX_PATTERN.match(stripped):
        return True

    # Pure numbers without context
    alpha = re.sub(r'[^a-zA-Z]', '', stripped)
    if not alpha:
        return True

    return False


@app.route('/process/freetext', methods=['POST'])
def process_freetext():
    """Detect free-floating text outside speech bubbles.
    Completely separate from the bubble pipeline.
    Expects: {image, bubbleBoxes, dpr, readingDirection}
    """
    global _last_activity
    _last_activity = time.time()

    if _shutting_down:
        return jsonify({'error': 'Server is shutting down'}), 503

    if not _apple_vision_available:
        return jsonify({'error': 'Apple Vision not available'}), 503

    t_start = time.time()
    data = request.get_json(force=True, silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image field'}), 400

    dpr = data.get('dpr', 2)
    if not isinstance(dpr, (int, float)) or dpr < 1 or dpr > 5:
        dpr = 2
    reading_dir = data.get('readingDirection', 'rtl')
    bubble_boxes = data.get('bubbleBoxes', [])

    # Decode image
    img = _decode_image(data['image'])
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    # Crop if provided (same as /process)
    crop = data.get('cropRect')
    actual_crop_left = 0
    actual_crop_top = 0
    if crop and crop.get('width') and crop.get('height'):
        img_h, img_w = img.shape[:2]
        cx = int(crop['left'] * dpr)
        cy = int(crop['top'] * dpr)
        cw = int(crop['width'] * dpr)
        ch = int(crop['height'] * dpr)
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        cw = min(cw, img_w - cx)
        ch = min(ch, img_h - cy)
        if cw > img_w * 0.15 and ch > img_h * 0.15:
            img = img[cy:cy + ch, cx:cx + cw]
            actual_crop_left = cx
            actual_crop_top = cy

    cropped_h, cropped_w = img.shape[:2]
    log.info(f'[FreeText] Image: {cropped_w}x{cropped_h}, {len(bubble_boxes)} bubble boxes to exclude')

    # Run full-page Apple Vision OCR
    t_ocr = time.time()
    raw_regions = _apple_vision_fullpage_ocr(img)
    # Group nearby lines into blocks (e.g. "I HAVE" + "TO PROTECT" + "HER" -> one block)
    all_text_regions = _group_nearby_text(raw_regions, cropped_h)
    ocr_ms = int((time.time() - t_ocr) * 1000)
    log.info(f'[FreeText] Vision found {len(raw_regions)} lines, grouped into {len(all_text_regions)} blocks ({ocr_ms}ms)')

    # Convert bubble boxes from CSS coords back to pixel coords for comparison
    pixel_bubbles = []
    for bb in bubble_boxes:
        pixel_bubbles.append({
            'x': int(bb.get('left', 0) * dpr) - actual_crop_left,
            'y': int(bb.get('top', 0) * dpr) - actual_crop_top,
            'w': int(bb.get('width', 0) * dpr),
            'h': int(bb.get('height', 0) * dpr),
        })

    # Filter: remove text that overlaps with existing bubbles
    free_texts = []
    for region in all_text_regions:
        overlaps = False
        for pb in pixel_bubbles:
            if _boxes_overlap(region, pb, margin=20):
                overlaps = True
                break
        if overlaps:
            log.info(f'  [FreeText] skipped (in bubble): "{region["text"][:50]}"')
            continue

        # Filter junk using the initial fullpage OCR text
        if _is_freetext_junk(region['text']):
            log.info(f'  [FreeText] skipped (junk): "{region["text"][:50]}"')
            continue

        # Re-OCR: crop the region from the image and run Apple Vision on just the crop
        # This gives much better results than fullpage OCR (less noise from artwork)
        pad = max(10, int(max(region['w'], region['h']) * 0.1))
        cx1 = max(0, region['x'] - pad)
        cy1 = max(0, region['y'] - pad)
        cx2 = min(cropped_w, region['x'] + region['w'] + pad)
        cy2 = min(cropped_h, region['y'] + region['h'] + pad)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # Check if dark background (white text on dark) and invert if needed
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_val = int(np.mean(gray))
        if mean_val < 128:
            crop = cv2.bitwise_not(crop)
            log.info(f'  [FreeText] dark region at ({region["x"]},{region["y"]}), inverted')

        reocr_text = _apple_vision_ocr(crop)
        if not reocr_text:
            log.info(f'  [FreeText] re-OCR empty for "{region["text"][:50]}"')
            # Fall back to the fullpage OCR text
            reocr_text = region['text']

        log.info(f'  [FreeText] re-OCR: "{reocr_text[:60]}" (was: "{region["text"][:40]}")')

        # Clean the re-OCR text
        cleaned = _clean_text(reocr_text)
        cleaned = _grammar_fix(cleaned)
        if not cleaned:
            log.info(f'  [FreeText] skipped (empty after clean): "{reocr_text[:50]}"')
            continue

        # Score check: reject low-quality OCR (same as bubble pipeline)
        score = _ocr_score(cleaned)
        if score < 0:
            log.info(f'  [FreeText] skipped (low score {score:.0f}): "{cleaned[:50]}"')
            continue

        # Extra check: reject if junk after cleaning (Japanese remnants etc.)
        if _is_freetext_junk(cleaned):
            log.info(f'  [FreeText] skipped (junk after clean): "{cleaned[:50]}"')
            continue

        free_texts.append({
            'text': cleaned,
            'conf': round(region['conf'], 3),
            'left': round((actual_crop_left + region['x']) / dpr, 1),
            'top': round((actual_crop_top + region['y']) / dpr, 1),
            'width': round(region['w'] / dpr, 1),
            'height': round(region['h'] / dpr, 1),
            'freeText': True,
        })

    # Sort by reading order
    row_height = max(1, cropped_h // 4)
    if reading_dir == 'rtl':
        free_texts.sort(key=lambda b: (b['top'] * dpr // row_height, -b['left']))
    else:
        free_texts.sort(key=lambda b: (b['top'] * dpr // row_height, b['left']))

    total_ms = int((time.time() - t_start) * 1000)
    log.info(f'[FreeText] Found {len(free_texts)} free text regions ({total_ms}ms total)')
    for i, ft in enumerate(free_texts):
        log.info(f'  [{i+1}] "{ft["text"][:60]}"')

    return jsonify({
        'freeTexts': free_texts,
        'timing': {
            'total_ms': total_ms,
            'ocr_ms': ocr_ms,
            'regions_found': len(all_text_regions),
            'regions_returned': len(free_texts),
        },
    })


@app.route('/ext-log', methods=['POST'])
def ext_log():
    data = request.get_json(silent=True)
    msg = data.get('msg', '') if data else ''
    if msg:
        log.info(f'[EXT] {msg}')
    return jsonify({'ok': True})


@app.route('/shutdown', methods=['POST'])
def shutdown_server():
    if request.remote_addr not in ('127.0.0.1', '::1'):
        return jsonify({'error': 'forbidden'}), 403
    global _shutting_down
    log.info('Shutdown requested')
    _shutting_down = True
    threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()
    return jsonify({'ok': True})


@app.route('/showcase/status', methods=['GET'])
def showcase_status():
    """Model status for popup UI pills."""
    return jsonify({
        'models': {
            'florence2': {'status': _florence2 is not None},
            'apple_vision': {'status': _apple_vision_available},
            'bubble_detector': {'status': _detector_session is not None},
            'piper_tts': {'status': _piper_voice is not None},
        }
    })




# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Windows Unicode fix
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    log.info('=' * 50)
    log.info('MangaVoice Lite starting (ONNX-only)...')
    log.info(f'  Models dir: {MODELS_DIR}')

    # Check Tesseract
    try:
        import pytesseract
        ver = pytesseract.get_tesseract_version()
        log.info(f'  Tesseract: {ver}')
    except Exception:
        log.warning('  Tesseract not available')

    # Load all models in background
    def _load():
        try:
            _load_all_models()
        except Exception as e:
            log.error(f'Model loading failed: {e}')
    threading.Thread(target=_load, daemon=True).start()

    # Start idle monitor
    threading.Thread(target=_idle_monitor, daemon=True).start()

    log.info(f'  Server starting on http://127.0.0.1:{PORT}')
    log.info(f'  (models loading in background...)')
    log.info('=' * 50)
    app.run(host='127.0.0.1', port=PORT, debug=False, threaded=True)
