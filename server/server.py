"""
Manga Voice Reader — 100% Local OCR + TTS Server (no cloud APIs)
Runs on http://127.0.0.1:5055
Endpoint: POST /process
"""

import base64
import io
import json
import logging
import os

# ─── Thread caps: reduce peak CPU by limiting parallel threads per library ────
# Must be set BEFORE importing numpy/torch/onnxruntime/paddle
# Read from MVR_CPU_THREAD_CAP env var (default 2). Set to 0 to disable caps.
_tcap = os.environ.get('MVR_CPU_THREAD_CAP', '2')
if _tcap and _tcap != '0':
    for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[_k] = _tcap
    os.environ['FLAGS_num_threads'] = _tcap        # PaddlePaddle CPU threads

os.environ['FLAGS_use_mkldnn'] = '0'           # Fix PaddlePaddle oneDNN crash on Windows
os.environ['FLAGS_enable_pir_in_executor'] = '0'  # Disable PIR mode (crashes on Windows)
import re
import subprocess
import threading
import time
import math

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
from PIL import Image
import pytesseract
from comic_text_segmenter import ComicTextSegmenter

# ─── Configuration (centralized in config.py) ────────────────────────────────
from config import (
    PORT, MAX_DETECT_WIDTH, MAX_BOXES, IOU_MERGE_THRESHOLD, PROXIMITY_MERGE_RATIO,
    MIN_BOX_AREA, MIN_BOX_SIDE, MAX_ASPECT_RATIO, MIN_DET_SCORE,
    CROP_PAD_MIN, CROP_PAD_PCT, TESSERACT_CONFIG, ROW_Y_THRESHOLD_RATIO,
    USE_LOCAL_RECOGNIZER, COLLECT_TRAINING_DATA, SAVE_DEBUG_CROPS, DEBUG_VIEW,
    SAVE_DEBUG_IMAGES, PAGE_CACHE_MAX, IDLE_SHUTDOWN_MINUTES,
    CNN_REJECTION_CONFIDENCE, CNN_RESCUE_CONFIDENCE, DIALOGUE_SCORE_GATE,
    GIBBERISH_THRESHOLD, GIBBERISH_MIN_CHECKED, SYMBOL_JUNK_LIMIT,
    JUNK_SYMBOL_PATTERN, OCR_CROP_PAD_PCT, OCR_CROP_PAD_MIN,
    VERTICAL_MERGE_HEIGHT_FACTOR, VERTICAL_MERGE_WIDTH_FACTOR,
    MAX_CONCURRENT_PROCESS, MAX_CONCURRENT_TTS,
    MAX_IMAGE_BYTES, MAX_IMAGE_DIMENSION, MAX_DPR, VALID_READING_DIRECTIONS,
    DEBUG_FRAMES_DIR, MAX_DEBUG_FRAMES, DEBUG_MAX_AGE_SECONDS,
    EXT_LOG_MAX_BYTES, EXT_LOG_MAX_ROTATED,
    PRE_SFX_MIN_SIDE, PRE_SFX_MAX_ASPECT, PRE_SFX_MAX_ROTATION,
    PRE_SFX_MAX_AREA_RATIO, PRE_SFX_MIN_AREA_RATIO,
)

from config import LOW_BOX_THRESHOLD

# ─── Import from extracted modules ────────────────────────────────────────────
import ocr_engines
import tts_engine
from config import PIPER_MODEL
from ocr_engines import (
    load_detector, load_tesseract, load_manga_cnn, classify_crop_manga_cnn,
    load_local_recognizer, decode_base64_image, tesseract_ocr_crops,
    load_super_res, ai_upscale_2x, florence_ocr,
)
from text_processing import (
    load_manga_corrections, cleanup_text, reconstruct_ocr_text,
    format_text_for_speech, apply_manga_corrections, _clean_noise_fragments,
    _symspell,
)
from bubble_detection import (
    preprocess_for_detection, run_detection, detect_with_fallbacks,
    merge_box_lists, merge_overlapping, merge_same_line, merge_vertical_stack,
    filter_boxes, filter_dialogue_boxes, pre_filter_sfx_geometric,
    mask_based_bubble_grouping, scale_boxes, cap_boxes,
    sort_reading_order, BubbleDetector, save_debug_frame,
)

# Instantiate bubble detector (class imported from bubble_detection module)
bubble_detector = BubbleDetector()

# ─── Page cache (avoid re-processing identical screenshots) ──────────────
import hashlib
_page_cache = {}
_PAGE_CACHE_MAX = PAGE_CACHE_MAX

logging.basicConfig(level=logging.INFO, format='[MVR] %(message)s')
log = logging.getLogger('mvr')

# ─── Rate limiting (semaphore-based, no external deps) ───────────────────────
_process_semaphore = threading.Semaphore(MAX_CONCURRENT_PROCESS)
_tts_semaphore = threading.Semaphore(MAX_CONCURRENT_TTS)
_shutting_down = False
_last_shutdown_time = 0

# ─── Debug frame rotation ────────────────────────────────────────────────────
def _rotate_debug_frames():
    """Remove old debug frames to prevent disk bloat."""
    try:
        frames_dir = DEBUG_FRAMES_DIR
        if not os.path.isdir(frames_dir):
            return
        files = []
        for f in os.listdir(frames_dir):
            fp = os.path.join(frames_dir, f)
            if os.path.isfile(fp) and f.startswith('debug_') and f.endswith('.png'):
                files.append((os.path.getmtime(fp), fp))
        files.sort(reverse=True)
        now = time.time()
        for i, (mtime, fp) in enumerate(files):
            if i >= MAX_DEBUG_FRAMES or (now - mtime) > DEBUG_MAX_AGE_SECONDS:
                try:
                    os.remove(fp)
                except OSError:
                    pass
    except Exception:
        pass

# ─── Extension log rotation ──────────────────────────────────────────────────
def _rotate_ext_log(log_path):
    """Rotate extension debug log when it exceeds size limit."""
    try:
        if not os.path.isfile(log_path):
            return
        if os.path.getsize(log_path) < EXT_LOG_MAX_BYTES:
            return
        # Rotate: .log -> .log.1, .log.1 -> .log.2, etc.
        for i in range(EXT_LOG_MAX_ROTATED, 0, -1):
            src = f'{log_path}.{i}' if i > 0 else log_path
            dst = f'{log_path}.{i + 1}'
            if i == EXT_LOG_MAX_ROTATED:
                if os.path.isfile(f'{log_path}.{i}'):
                    os.remove(f'{log_path}.{i}')
            elif os.path.isfile(src):
                os.rename(src, dst)
        os.rename(log_path, f'{log_path}.1')
    except Exception:
        pass

# ─── Pipeline event tracker (for live dashboard) ─────────────────────────────
_pipeline_events = []
_pipeline_lock = threading.Lock()
_pipeline_run_id = 0
_pipeline_status = 'idle'  # idle, processing, done

def _pe(stage, title, detail='', image_key='', timing_ms=0, data=None):
    """Push a pipeline event for the live dashboard."""
    with _pipeline_lock:
        _pipeline_events.append({
            'id': len(_pipeline_events),
            'run': _pipeline_run_id,
            'ts': time.time(),
            'stage': stage,
            'title': title,
            'detail': detail,
            'image': image_key,
            'ms': timing_ms,
            'data': data or {},
        })

def _pe_reset():
    """Reset pipeline events for a new run."""
    global _pipeline_events, _pipeline_run_id, _pipeline_status
    with _pipeline_lock:
        _pipeline_events = []
        _pipeline_run_id += 1
        _pipeline_status = 'processing'

# ─── App ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

@app.errorhandler(500)
def _handle_500(e):
    import traceback
    tb = traceback.format_exc()
    log.error(f'500 ERROR: {e}\n{tb}')
    return jsonify({'error': str(e), 'traceback': tb}), 500

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Extension keep-alive — resets idle timer without processing."""
    return jsonify({'ok': True})

@app.route('/process', methods=['POST'])
def process():
    # Reject if shutting down
    if _shutting_down:
        return jsonify({'error': 'Server is shutting down'}), 503

    # Rate limiting: reject if too many concurrent requests
    if not _process_semaphore.acquire(blocking=False):
        log.warning('Rate limit: too many concurrent /process requests')
        return jsonify({'error': 'Server busy, try again shortly'}), 429

    try:
        return _do_process()
    finally:
        _process_semaphore.release()

def _do_process():
    t_start = time.time()
    data = request.json

    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image field'}), 400

    # Input validation
    image_data = data['image']
    if isinstance(image_data, str) and len(image_data) > MAX_IMAGE_BYTES:
        return jsonify({'error': f'Image too large (>{MAX_IMAGE_BYTES // 1024 // 1024}MB)'}), 400

    dpr = data.get('dpr', 2)
    if not isinstance(dpr, (int, float)) or dpr < 1 or dpr > MAX_DPR:
        dpr = 2

    reading_dir = data.get('readingDirection', 'rtl')
    if reading_dir not in VALID_READING_DIRECTIONS:
        reading_dir = 'rtl'

    # Rotate debug frames before writing new ones
    _rotate_debug_frames()

    _pe_reset()
    log.info('=' * 60)
    log.info('STAGE 1: Screenshot decode')
    _pe(1, 'Screenshot Received', 'Decoding base64 image from Chrome extension...')
    t0 = time.time()
    img = decode_base64_image(image_data)
    if img is None:
        log.error('  FAILED: could not decode image')
        _pe(1, 'FAILED', 'Could not decode image')
        return jsonify({'error': 'Failed to decode image'}), 400
    decode_ms = int((time.time() - t0) * 1000)

    img_h, img_w = img.shape[:2]
    if img_w > MAX_IMAGE_DIMENSION or img_h > MAX_IMAGE_DIMENSION:
        return jsonify({'error': f'Image too large ({img_w}x{img_h}, max {MAX_IMAGE_DIMENSION})'}), 400
    page_title = data.get('pageTitle', '')
    page_url = data.get('pageUrl', '')
    log.info(f'  Full screenshot size: {img_w}x{img_h} px, dpr={dpr} ({decode_ms}ms)')
    _pe(1, 'Screenshot Decoded', f'{img_w}x{img_h} px, dpr={dpr}', 'debug_1_full_screenshot.png', decode_ms,
        {'width': img_w, 'height': img_h, 'dpr': dpr, 'page': page_title or page_url})
    if page_title:
        log.info(f'  Page: {page_title}')

    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_frames')
    os.makedirs(debug_dir, exist_ok=True)

    log.info('STAGE 2: Crop to manga area')
    crop = data.get('cropRect')
    use_crop = False
    if crop and crop.get('width') and crop.get('height'):
        log.info(f'  cropRect from extension: left={crop["left"]}, top={crop["top"]}, '
                 f'width={crop["width"]}, height={crop["height"]}')
        cx = int(crop['left'] * dpr)
        cy = int(crop['top'] * dpr)
        cw = int(crop['width'] * dpr)
        ch = int(crop['height'] * dpr)
        log.info(f'  After dpr*{dpr}: cx={cx}, cy={cy}, cw={cw}, ch={ch}')
        # Clamp to image bounds
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        cw = min(cw, img_w - cx)
        ch = min(ch, img_h - cy)
        log.info(f'  After clamp: cx={cx}, cy={cy}, cw={cw}, ch={ch}')

        # Sanity check: reject suspiciously small crops
        # A valid manga crop should be at least 15% of the screenshot in both dimensions
        min_crop_w = img_w * 0.15
        min_crop_h = img_h * 0.15
        min_crop_area = img_w * img_h * 0.05  # at least 5% of screenshot area
        crop_area = cw * ch

        if cw < min_crop_w or ch < min_crop_h or crop_area < min_crop_area:
            log.warning(f'  cropRect too small! {cw}x{ch} (area={crop_area}) < '
                        f'min {min_crop_w:.0f}x{min_crop_h:.0f} (area={min_crop_area:.0f})')
            log.warning(f'  FALLING BACK to full screenshot')
            use_crop = False
        else:
            use_crop = True

    if use_crop:
        cropped = img[cy:cy + ch, cx:cx + cw]
        actual_crop_left = cx
        actual_crop_top = cy
    else:
        if not (crop and crop.get('width')):
            log.info('  No cropRect provided, using full screenshot')
        cropped = img
        actual_crop_left = 0
        actual_crop_top = 0
        cx, cy = 0, 0

    crop_h, crop_w = cropped.shape[:2]
    log.info(f'  Cropped image size: {crop_w}x{crop_h} px')
    _pe(2, 'Manga Area Cropped', f'{crop_w}x{crop_h} px (from {img_w}x{img_h})', 'debug_2_cropped_manga.png', 0,
        {'crop_w': crop_w, 'crop_h': crop_h, 'used_crop': use_crop})

    # Debug image saves moved to background for speed

    # Auto-save to test_images/ for OCR testing (only when debug enabled)
    if SAVE_DEBUG_IMAGES:
        test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
        os.makedirs(test_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        test_path = os.path.join(test_dir, f'page_{ts}.png')
        cv2.imwrite(test_path, cropped)
        log.info(f'  Auto-saved test image: {test_path}')

    # ── Run AI bubble detector AND PaddleOCR detection IN PARALLEL ──
    log.info('STAGE 3: Bubble detection (AI) + PaddleOCR detection (parallel)')
    _pe(3, 'Bubble Detection', 'Running RT-DETR-v2 + PaddleOCR in parallel...')
    bubble_det_results = None
    bubble_det_ms = 0
    seg_ms = 0
    text_mask = None
    used_bubble_detector = False
    boxes = []
    det_ms = 0
    used_inversion = False

    # Prepare detection image (may need downscale)
    det_scale = 1.0
    if crop_w > MAX_DETECT_WIDTH:
        det_scale = MAX_DETECT_WIDTH / crop_w
        det_w = MAX_DETECT_WIDTH
        det_h = int(crop_h * det_scale)
        det_img = cv2.resize(cropped, (det_w, det_h))
        log.info(f'  Downscaled for detection: {crop_w}x{crop_h} -> {det_w}x{det_h}')
    else:
        det_img = cropped

    # Run bubble detection and PaddleOCR detection in parallel using threads
    import concurrent.futures
    _parallel_bubble_result = [None]
    _parallel_paddle_result = [None]

    def _run_bubble_det():
        if bubble_detector.available:
            t_bd = time.time()
            result = bubble_detector.detect(cropped, conf_threshold=0.3)
            ms = int((time.time() - t_bd) * 1000)
            _parallel_bubble_result[0] = (result, ms)

    def _run_paddle_det():
        t_det = time.time()
        result = run_detection(det_img, ocr_engines.ocr_engine)
        ms = int((time.time() - t_det) * 1000)
        _parallel_paddle_result[0] = (result, ms)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut_bubble = executor.submit(_run_bubble_det)
        fut_paddle = executor.submit(_run_paddle_det)
        concurrent.futures.wait([fut_bubble, fut_paddle])

    # Collect bubble detection results
    if _parallel_bubble_result[0]:
        bubble_det_results, bubble_det_ms = _parallel_bubble_result[0]
        if bubble_det_results:
            n_text = len(bubble_det_results['text_bubble'])
            n_bubble = len(bubble_det_results.get('bubble', []))
            n_free = len(bubble_det_results.get('text_free', []))
            log.info(f'  AI bubble detector: {n_text} text_bubble, {n_bubble} bubble, {n_free} text_free ({bubble_det_ms}ms)')
            _pe(3, 'Bubbles Found', f'{n_text} text bubbles, {n_bubble} outlines, {n_free} SFX', '', bubble_det_ms,
                {'text_bubble': n_text, 'bubble': n_bubble, 'text_free': n_free,
                 'detections': [{'class': d.get('class',''), 'x': d['x'], 'y': d['y'], 'w': d['w'], 'h': d['h'], 'score': round(d['score'],2)}
                                for cls in ['text_bubble','bubble','text_free'] for d in bubble_det_results.get(cls, [])]})

    # Collect PaddleOCR detection results
    if _parallel_paddle_result[0]:
        boxes, det_ms = _parallel_paddle_result[0]
        if det_scale != 1.0:
            boxes = scale_boxes(boxes, 1.0 / det_scale)
        mode = 'parallel'
        log.info(f'  PaddleOCR detection: {len(boxes)} raw boxes ({det_ms}ms)')
        _pe('3b', 'Text Detection (PaddleOCR)', f'{len(boxes)} text regions found', '', det_ms,
            {'raw_boxes': len(boxes), 'mode': mode})

    # If no bubble detector results, we may need full fallback detection
    if not bubble_det_results or len(bubble_det_results.get('text_bubble', [])) < 1:
        if not _parallel_paddle_result[0]:
            boxes, det_ms, used_inversion = detect_with_fallbacks(det_img, ocr_engines.ocr_engine)
            if det_scale != 1.0:
                boxes = scale_boxes(boxes, 1.0 / det_scale)
            log.info(f'  PaddleOCR full fallback: {len(boxes)} raw boxes ({det_ms}ms)')
            _pe('3b', 'Text Detection (PaddleOCR)', f'{len(boxes)} text regions found (full detection)', '', det_ms,
                {'raw_boxes': len(boxes), 'mode': 'full-fallback'})

    log.info('STAGE 4a: Pre-filter SFX (geometry)')
    t_group = time.time()
    raw_boxes_snapshot = [dict(b) for b in boxes]

    boxes, pre_sfx_rejected = pre_filter_sfx_geometric(boxes, cropped.shape)
    dialogue_boxes = [dict(b) for b in boxes]

    if bubble_det_results and len(bubble_det_results.get('text_bubble', [])) >= 1:
        for cls in ['bubble', 'text_bubble', 'text_free']:
            for j, det in enumerate(bubble_det_results[cls]):
                log.info(f'    [{cls}] score={det["score"]:.2f} x={det["x"]} y={det["y"]} {det["w"]}x{det["h"]}')

    # ── Comic-translate approach: use bubble outlines to filter text ──
    # The detector gives us 3 classes:
    #   bubble (class 0) = speech bubble outlines (the white shapes)
    #   text_bubble (class 1) = text regions inside bubbles
    #   text_free (class 2) = text outside bubbles (titles, watermarks, SFX)
    #
    # Strategy: collect all text regions (text_bubble + text_free), then keep ONLY
    # those that are inside or overlap a bubble outline. This kills watermarks,
    # titles, Japanese text, author credits — anything not in a speech bubble.
    # If no bubble outlines found, fall back to using text_bubble regions directly.

    n_bubble_det = len(bubble_det_results.get('text_bubble', [])) if bubble_det_results else 0
    n_bubble_outlines = len(bubble_det_results.get('bubble', [])) if bubble_det_results else 0
    n_paddle_text = len([b for b in dialogue_boxes if b.get('text', '').strip()]) if dialogue_boxes else 0

    def _rect_inside_or_overlap(text_r, bubble_r, margin=0.15):
        """Check if text region is inside or overlaps a bubble outline.
        Returns True if text center is inside bubble, or if >30% of text area overlaps bubble."""
        tx1, ty1 = text_r['x'], text_r['y']
        tx2, ty2 = tx1 + text_r['w'], ty1 + text_r['h']
        # Add margin to bubble (bubbles don't always perfectly contain text)
        mw = int(bubble_r['w'] * margin)
        mh = int(bubble_r['h'] * margin)
        bx1, by1 = bubble_r['x'] - mw, bubble_r['y'] - mh
        bx2, by2 = bubble_r['x'] + bubble_r['w'] + mw, bubble_r['y'] + bubble_r['h'] + mh
        # Center check
        cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            return True
        # Overlap check
        ix = max(0, min(tx2, bx2) - max(tx1, bx1))
        iy = max(0, min(ty2, by2) - max(ty1, by1))
        inter = ix * iy
        text_area = max(1, text_r['w'] * text_r['h'])
        if inter / text_area > 0.3:
            return True
        return False

    n_bubble_with_text = 0  # bubble outlines that contain OCR text
    n_total_det = n_bubble_det + n_bubble_outlines
    bubble_det_sufficient = False  # set True if detector gives enough results

    if bubble_det_results and (n_bubble_det > 0 or n_bubble_outlines > 0):
        # The detector classifies text into:
        #   text_bubble = dialogue (inside bubbles, narration boxes, or floating speech)
        #   text_free = non-dialogue (titles, watermarks, SFX, credits, Japanese text)
        # Trust the model: keep text_bubble, drop text_free. The model was trained
        # on 11k comic images to learn this difference.
        #
        # Additionally, if bubble outlines exist, also keep any text_free that happens
        # to be inside a bubble outline (the model sometimes misclassifies).
        bubble_outlines = list(bubble_det_results.get('bubble', []))

        kept = []
        dropped = []

        # Always keep text_bubble
        for det in bubble_det_results.get('text_bubble', []):
            kept.append(det)

        # For text_free: keep if inside a bubble outline OR if MangaCNN says it's dialogue
        # This rescues narration boxes and floating text that aren't in speech bubbles
        for det in bubble_det_results.get('text_free', []):
            inside_bubble = any(_rect_inside_or_overlap(det, bo) for bo in bubble_outlines)
            if inside_bubble:
                kept.append(det)
                log.info(f'    rescued text_free inside bubble: ({det["x"]},{det["y"]}) {det["w"]}x{det["h"]}')
            else:
                # Use MangaCNN to check if this text_free region has readable English text
                try:
                    tx1 = max(0, det['x'] - 5)
                    ty1 = max(0, det['y'] - 5)
                    tx2 = min(crop_w, det['x'] + det['w'] + 5)
                    ty2 = min(crop_h, det['y'] + det['h'] + 5)
                    text_crop = cropped[ty1:ty2, tx1:tx2]
                    is_dialogue, cnn_conf = classify_crop_manga_cnn(text_crop)
                    if is_dialogue and cnn_conf > CNN_RESCUE_CONFIDENCE:
                        kept.append(det)
                        log.info(f'    rescued text_free by MangaCNN: ({det["x"]},{det["y"]}) {det["w"]}x{det["h"]} dialogue={cnn_conf*100:.0f}%')
                    else:
                        dropped.append(det)
                except Exception:
                    dropped.append(det)

        log.info(f'  Bubble detector: {len(kept)} text_bubble kept, {len(dropped)} text_free dropped, {len(bubble_outlines)} outlines')
        for d in dropped:
            log.info(f'    dropped text_free: ({d["x"]},{d["y"]}) {d["w"]}x{d["h"]} score={d["score"]:.2f}')

        n_bubble_with_text = len(kept)
        if len(kept) >= 1:
            bubble_det_sufficient = True
            boxes = []
            for det in kept:
                boxes.append({
                    'x': det['x'], 'y': det['y'],
                    'w': det['w'], 'h': det['h'],
                    'score': det['score'],
                    'shape': 'oval', 'raw_indices': [],
                    'source': 'bubble_detector',
                })

            # Merge overlapping boxes
            merged_changed = True
            while merged_changed:
                merged_changed = False
                new_boxes = []
                used_idx = set()
                for i2 in range(len(boxes)):
                    if i2 in used_idx:
                        continue
                    bx = boxes[i2]
                    for j2 in range(i2 + 1, len(boxes)):
                        if j2 in used_idx:
                            continue
                        other = boxes[j2]
                        ix1 = max(bx['x'], other['x'])
                        iy1 = max(bx['y'], other['y'])
                        ix2 = min(bx['x'] + bx['w'], other['x'] + other['w'])
                        iy2 = min(bx['y'] + bx['h'], other['y'] + other['h'])
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        area_a = bx['w'] * bx['h']
                        area_b = other['w'] * other['h']
                        frac_a = inter / area_a if area_a > 0 else 0
                        frac_b = inter / area_b if area_b > 0 else 0
                        if frac_a > 0.2 or frac_b > 0.2:
                            nx1 = min(bx['x'], other['x'])
                            ny1 = min(bx['y'], other['y'])
                            nx2 = max(bx['x'] + bx['w'], other['x'] + other['w'])
                            ny2 = max(bx['y'] + bx['h'], other['y'] + other['h'])
                            bx = {
                                'x': nx1, 'y': ny1, 'w': nx2 - nx1, 'h': ny2 - ny1,
                                'score': max(bx.get('score', 0), other.get('score', 0)),
                                'shape': 'oval', 'raw_indices': [],
                                'source': 'bubble_detector',
                            }
                            used_idx.add(j2)
                            merged_changed = True
                    new_boxes.append(bx)
                    used_idx.add(i2)
                boxes = new_boxes

            used_bubble_detector = True
            log.info(f'  Using BUBBLE DETECTOR: {len(boxes)} speech regions (filtered by bubble outlines)')
            merged_boxes_snapshot = [dict(b) for b in boxes]
    else:
        # ── Fallback: text segmentation + mask-based grouping ──
        if n_total_det >= 1 and not bubble_det_sufficient:
            log.info(f'  Bubble detector found only {n_total_det} ({n_bubble_det} text_bubble + {n_bubble_with_text} bubble w/text) but PaddleOCR has {n_paddle_text} text boxes — falling back to full grouping')
        log.info('STAGE 4b-fallback: Text segmentation')
        text_mask = None
        seg_ms = 0
        if text_segmenter.available:
            t_seg = time.time()
            text_mask = text_segmenter.get_text_mask(cropped)
            seg_ms = int((time.time() - t_seg) * 1000)
            if text_mask is not None:
                log.info(f'  Text mask generated ({seg_ms}ms, coverage: {np.mean(text_mask > 127)*100:.1f}%)')
                if SAVE_DEBUG_IMAGES:
                    cv2.imwrite(os.path.join(debug_dir, 'debug_4_text_mask.png'), text_mask)
            else:
                log.warning(f'  Text mask generation failed ({seg_ms}ms)')
        else:
            log.info('  Text segmenter not available')

        log.info('STAGE 4c: Bubble grouping (mask-based fallback)')
        mask_bubbles = None
        dilated_debug = None

        if text_mask is not None:
            result = mask_based_bubble_grouping(text_mask, dialogue_boxes, cropped.shape)
            if result is not None:
                mask_bubbles, dilated_debug = result

        if mask_bubbles is not None and len(mask_bubbles) > 0:
            log.info(f'  Using MASK-BASED bubble grouping ({len(mask_bubbles)} bubbles)')
            boxes = mask_bubbles
            merged_boxes_snapshot = [dict(b) for b in boxes]
            if SAVE_DEBUG_IMAGES and dilated_debug is not None:
                cv2.imwrite(os.path.join(debug_dir, 'debug_4_dilated_mask.png'), dilated_debug)
        else:
            # Fallback: geometric merge (if mask not available)
            log.info('  Falling back to geometric merge')
            boxes = merge_overlapping(boxes, IOU_MERGE_THRESHOLD)
            boxes = merge_same_line(boxes)
            boxes = merge_vertical_stack(boxes)
            merged_boxes_snapshot = [dict(b) for b in boxes]

    rejected_geo = list(pre_sfx_rejected)

    # ── Final column-merge: merge vertically-stacked, horizontally-aligned boxes ──
    # ONLY for orphan/fallback boxes (narration split into pieces).
    # NEVER merge two bubble_detector boxes — those are separate speech bubbles.
    _BUBBLE_DET_SOURCES = {'bubble_detector', 'bubble_detector_bubble_class'}
    col_merged = True
    while col_merged:
        col_merged = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used:
                continue
            bx = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                other = boxes[j]
                # NEVER merge two bubble-detector boxes (they are separate bubbles)
                src_a = bx.get('source', '')
                src_b = other.get('source', '')
                if src_a in _BUBBLE_DET_SOURCES and src_b in _BUBBLE_DET_SOURCES:
                    continue
                # Horizontal overlap check — must be well aligned
                ox1 = max(bx['x'], other['x'])
                ox2 = min(bx['x'] + bx['w'], other['x'] + other['w'])
                h_overlap = max(0, ox2 - ox1)
                min_w = min(bx['w'], other['w'])
                if min_w <= 0 or h_overlap < min_w * 0.5:
                    continue
                # Width ratio: don't merge a narrow box with a wide box
                max_w = max(bx['w'], other['w'])
                if max_w > min_w * 2.5:
                    continue
                # Vertical gap: must be VERY close (within 25% of smaller box height
                # or 20px, whichever is larger)
                by_bot = bx['y'] + bx['h']
                oy_bot = other['y'] + other['h']
                v_gap = max(0, max(bx['y'], other['y']) - min(by_bot, oy_bot))
                min_h = min(bx['h'], other['h'])
                max_gap = max(20, min_h * 0.25)
                if v_gap > max_gap:
                    continue
                # Max merged height cap: result must not exceed 20% of image height
                ny1 = min(bx['y'], other['y'])
                ny2 = max(by_bot, oy_bot)
                if (ny2 - ny1) > crop_h * 0.20:
                    continue
                # Merge: union bounding box
                nx1 = min(bx['x'], other['x'])
                nx2 = max(bx['x'] + bx['w'], other['x'] + other['w'])
                bx = {
                    'x': nx1, 'y': ny1, 'w': nx2 - nx1, 'h': ny2 - ny1,
                    'score': max(bx.get('score', 0), other.get('score', 0)),
                    'shape': bx.get('shape', 'rect'),
                    'raw_indices': bx.get('raw_indices', []) + other.get('raw_indices', []),
                    'source': 'column_merged',
                }
                used.add(j)
                col_merged = True
            new_boxes.append(bx)
            used.add(i)
        boxes = new_boxes
    log.info(f'  After column-merge: {len(boxes)} boxes')

    for i, b in enumerate(boxes):
        shape_str = f' [{b.get("shape", "?")}]' if 'shape' in b else ''
        log.info(f'  merged[{i}]{shape_str} x={b["x"]} y={b["y"]} {b["w"]}x{b["h"]}')

    pre_filter_count = len(boxes)
    boxes, rejected_geo_new = filter_boxes(boxes)
    rejected_geo += rejected_geo_new
    boxes = cap_boxes(boxes, MAX_BOXES)
    group_ms = int((time.time() - t_group) * 1000)
    log.info(f'  Grouping complete: {len(boxes)} bubbles ({group_ms}ms)')
    _pe(4, 'Bubbles Grouped', f'{len(boxes)} speech bubbles after grouping + filtering', '', group_ms,
        {'bubbles': len(boxes), 'raw_detected': len(raw_boxes_snapshot),
         'sources': {b.get('source','unknown') for b in boxes}.__len__() if boxes else 0,
         'bubble_boxes': [{'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h'], 'source': b.get('source','')} for b in boxes]})

    if len(boxes) == 0:
        log.info(f'  >>> NO BOXES after filtering. {raw_box_count} raw -> 0 final')
        log.info(f'  >>> RESULT: "No dialogue found" — failure at DETECTION/FILTER stage')
        if DEBUG_VIEW:
            save_debug_frame(cropped, raw_boxes_snapshot, merged_boxes_snapshot,
                             rejected_geo, boxes, [], [], {})
        log.info('=' * 60)
        return jsonify({
            'bubbles': [],
            'actualCropLeft': actual_crop_left / dpr,
            'actualCropTop': actual_crop_top / dpr,
            'timing': {
                'total_ms': int((time.time() - t_start) * 1000),
                'decode_ms': decode_ms,
                'detect_ms': det_ms,
                'merge_ms': group_ms,
                'ocr_ms': 0,
                'boxes_detected': 0,
                'bubbles_returned': 0,
                'used_inversion': used_inversion,
            }
        })

    def _word_level_hybrid(winner_text, paddle_text, tess_text, symspell):
        """Word-level OCR hybrid: replace non-dictionary words in the winner
        with dictionary words from the other engine if they match closely.

        This permanently fixes character-level misreads (L→b, U→li) without
        needing manga-specific dictionaries — works on any comic/manga."""
        winner_words = winner_text.split()
        if len(winner_words) < 1:
            return winner_text

        # Build word pool from the other engine
        other_words = set()
        for src in [paddle_text, tess_text]:
            for w in src.split():
                core = re.sub(r'[^a-zA-Z]', '', w).lower()
                if core and len(core) >= 2:
                    other_words.add((core, w))  # (lookup_key, original_with_case)

        result = []
        for w in winner_words:
            core = re.sub(r'[^a-zA-Z]', '', w).lower()
            # Skip short words, punctuation-only, or already valid words
            if not core or len(core) <= 2:
                result.append(w)
                continue

            # Check if this word is in the dictionary
            lookup = symspell.lookup(core, max_edit_distance=0, verbosity=0)
            if lookup:
                result.append(w)  # It's a real word, keep it
                continue

            # This word is NOT in the dictionary — look for a replacement
            # from the other engine that IS a dictionary word and is similar
            best_replacement = None
            best_dist = 3
            for other_core, other_orig in other_words:
                if abs(len(core) - len(other_core)) > 2:
                    continue
                # Check if the other word is a real dictionary word
                other_lookup = symspell.lookup(other_core, max_edit_distance=0, verbosity=0)
                if not other_lookup:
                    continue
                # Check edit distance between the two
                dist = _edit_distance(core, other_core)
                if dist <= 2 and dist < best_dist:
                    best_dist = dist
                    best_replacement = other_orig

            if best_replacement:
                # Preserve original punctuation
                prefix = ''
                suffix = ''
                for c in w:
                    if c.isalpha():
                        break
                    prefix += c
                for c in reversed(w):
                    if c.isalpha():
                        break
                    suffix = c + suffix
                # Match case
                repl_core = re.sub(r'[^a-zA-Z]', '', best_replacement)
                if core == core.upper() and len(core) > 1:
                    repl_core = repl_core.upper()
                elif core and core[0] == core[0].upper():
                    repl_core = repl_core.capitalize()
                result.append(prefix + repl_core + suffix)
                log.debug(f'    Hybrid OCR: "{w}" -> "{prefix + repl_core + suffix}" (from other engine)')
            else:
                # DON'T spell-correct capitalized words — they're likely proper nouns
                # (character names, place names) that aren't in the dictionary.
                # Spell-correcting "Luffy" → "buffy" would be wrong.
                alpha_part = re.sub(r'[^a-zA-Z]', '', w)
                is_proper = alpha_part and alpha_part[0].isupper() and len(alpha_part) >= 3
                if is_proper:
                    result.append(w)  # Keep proper nouns unchanged
                else:
                    result.append(w)  # Keep non-dictionary words too — no blind correction

        return ' '.join(result)

    def _edit_distance(s1, s2):
        """Fast Levenshtein distance for short strings."""
        if len(s1) < len(s2):
            return _edit_distance(s2, s1)
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if c1 == c2 else 1)))
            prev = curr
        return prev[-1]

    log.info('STAGE 5: Multi-OCR')
    _pe(5, 'OCR Recognition', f'Reading text from {len(boxes)} bubbles (PaddleOCR + Tesseract competition)...')

    # Common English words for scoring OCR quality
    _COMMON_WORDS = {
        'i', 'a', 'the', 'to', 'and', 'of', 'is', 'in', 'it', 'you', 'that', 'he',
        'she', 'was', 'for', 'on', 'are', 'with', 'his', 'they', 'be', 'at', 'one',
        'have', 'this', 'from', 'or', 'had', 'by', 'but', 'not', 'what', 'all', 'were',
        'we', 'when', 'your', 'can', 'said', 'there', 'her', 'an', 'will', 'my', 'do',
        'if', 'me', 'no', 'him', 'just', 'so', 'how', 'up', 'out', 'them', 'then',
        'about', 'would', 'been', 'now', 'like', 'did', 'get', 'has', 'more', 'than',
        'its', "it's", "don't", "didn't", "won't", "can't", "i'm", "i'll", "i've",
        'even', 'though', 'still', 'also', 'too', 'very', 'much', 'well', 'why',
    }

    def _score_ocr_text(text):
        """Score OCR quality by counting recognizable English words and penalizing noise.
        Returns a score that accounts for BOTH quality and quantity of text.
        More real words = higher score. Garbage words = penalties."""
        if not text.strip():
            return -10
        words = re.sub(r'[^a-zA-Z\s\']', ' ', text.lower()).split()
        if not words:
            return -5

        good_words = 0
        bad_words = 0
        common_words = 0
        for w in words:
            if w in _COMMON_WORDS:
                common_words += 1
                good_words += 1
            elif len(w) >= 3 and w.isalpha():
                good_words += 1
            elif len(w) == 1 and w not in ('i', 'a'):
                bad_words += 1
            elif not w.isalpha() and len(w) <= 2:
                bad_words += 1

        total = len(words)
        # Quality ratio: what fraction of words are good?
        quality = good_words / max(1, total)
        # Base score: good words contribute positively, bad words subtract
        score = (common_words * 2) + (good_words - common_words) - (bad_words * 2)
        # Length bonus: more real words = better (scaled by quality)
        # This prevents short garbage from beating long real text
        score += int(good_words * quality * 1.5)
        # Penalize if most words are joined (avg word length > 10)
        avg_word_len = len(text.replace(' ', '')) / max(1, total)
        if avg_word_len > 10:
            score -= 3
        # Penalize inconsistent casing: "WHAT THE. HELL ARE You DOING" has mixed case
        # Real manga text is usually ALL CAPS or consistently cased
        raw_words = text.split()
        case_types = set()
        for rw in raw_words:
            alpha = re.sub(r'[^a-zA-Z]', '', rw)
            if not alpha:
                continue
            if alpha.isupper():
                case_types.add('upper')
            elif alpha.islower():
                case_types.add('lower')
            elif alpha[0].isupper() and alpha[1:].islower() if len(alpha) > 1 else True:
                case_types.add('title')
            else:
                case_types.add('mixed')
        if len(case_types) >= 3:
            score -= 4  # heavily mixed case = likely Tesseract garbage
        # Penalize stray periods/dots mid-text (Tesseract artifact: "WHAT THE. HELL")
        stray_dots = len(re.findall(r'\w+\.\s+[A-Z]', text))
        score -= stray_dots * 2
        # Penalize leading numbers/noise: "1.. diet" pattern
        if re.match(r'^[\d\W]{2,}', text):
            score -= 3
        # Penalize very short results (1-2 words) unless they're common words
        if total <= 2 and common_words == 0:
            score -= 2
        return score

    def _tesseract_on_crop(img_region, box_x=0, box_y=0, box_w=0, box_h=0):
        """Run Tesseract on a crop. Uses text mask if available for clean input."""
        if img_region is None or img_region.size == 0:
            return ''

        # If we have a text mask, use it to create a super-clean crop
        if text_mask is not None and box_w > 0:
            pil_img = text_segmenter.clean_crop_for_ocr(
                cropped, text_mask, box_x, box_y, box_w, box_h)
        else:
            # Fallback: 2x LANCZOS upscale + adaptive binarization + white padding
            pil_crop = Image.fromarray(cv2.cvtColor(img_region, cv2.COLOR_BGR2RGB))
            w_c, h_c = pil_crop.size
            scale = max(2.0, 80.0 / max(1, h_c))
            new_w, new_h = int(w_c * scale), int(h_c * scale)
            pil_crop = pil_crop.resize((new_w, new_h), Image.LANCZOS)
            padded = Image.new('RGB', (new_w + 40, new_h + 40), (255, 255, 255))
            padded.paste(pil_crop, (20, 20))
            gray = cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=21, C=10
            )
            if np.mean(binary) < 127:
                binary = cv2.bitwise_not(binary)
            pil_img = Image.fromarray(binary)

        try:
            text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6').strip()
            return text.replace('|', 'I')
        except Exception:
            return ''

    def _paddle_text_for_box(merged_b, raw_boxes):
        """Collect PaddleOCR recognized text for a merged box."""
        mx1, my1 = merged_b['x'], merged_b['y']
        mx2, my2 = mx1 + merged_b['w'], my1 + merged_b['h']
        lines = []
        for raw_b in raw_boxes:
            rx_c = raw_b['x'] + raw_b['w'] / 2
            ry_c = raw_b['y'] + raw_b['h'] / 2
            if mx1 <= rx_c <= mx2 and my1 <= ry_c <= my2:
                t = raw_b.get('text', '').strip()
                if t:
                    lines.append((raw_b['y'], t))
        lines.sort(key=lambda x: x[0])
        return ' '.join(t for _, t in lines)

    def _mask_bubble_white(crop_img):
        """Mask out dark manga artwork areas, keeping only white bubble interiors.
        Used when bubble detector provides the crop box (which may extend into artwork)."""
        if crop_img is None or crop_img.size == 0:
            return crop_img
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # Threshold: pixels brighter than 170 are likely bubble interior
        _, white_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        # Close operation: fills small gaps (text characters) within white regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bubble_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        # Dilate to include bubble edges
        bubble_mask = cv2.dilate(bubble_mask, kernel, iterations=1)
        # Replace dark non-bubble areas with white
        result = crop_img.copy()
        result[bubble_mask == 0] = [255, 255, 255]
        return result

    def _enhance_ocr_crop(crop_img):
        """Enhance a crop for OCR: upscale small text, sharpen, contrast boost.
        Returns enhanced BGR image suitable for manga-ocr or Tesseract."""
        if crop_img is None or crop_img.size == 0:
            return crop_img

        h, w = crop_img.shape[:2]

        # 1. AI upscale for small/medium crops — sharper text for OCR
        if h < 80:
            scale = max(2.0, 80 / h)
            crop_img = ai_upscale_2x(crop_img)
            # If still too small after 2x, scale up more
            if crop_img.shape[0] < 80:
                extra = 80 / crop_img.shape[0]
                crop_img = cv2.resize(crop_img, None, fx=extra, fy=extra,
                                      interpolation=cv2.INTER_CUBIC)
        elif h < 200:
            crop_img = ai_upscale_2x(crop_img)

        # 2. Convert to grayscale for processing
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE contrast enhancement (adaptive, works on varied backgrounds)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Mild unsharp mask sharpening to crisp text edges
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)

        # 5. Add white padding border (helps OCR at edges)
        pad = 15
        padded = cv2.copyMakeBorder(sharpened, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=255)

        return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    # Generous crop padding for OCR — ensures full characters aren't clipped
    # Values from config.py: OCR_CROP_PAD_PCT, OCR_CROP_PAD_MIN

    ocr_texts = []
    win_counts = {'PADDLE': 0, 'TESS': 0}
    for i, merged_b in enumerate(boxes):
        # Crop with generous padding from original full-resolution image
        pad_x = max(OCR_CROP_PAD_MIN, int(merged_b['w'] * OCR_CROP_PAD_PCT))
        pad_y = max(OCR_CROP_PAD_MIN, int(merged_b['h'] * OCR_CROP_PAD_PCT))
        x1 = max(0, merged_b['x'] - pad_x)
        y1 = max(0, merged_b['y'] - pad_y)
        x2 = min(crop_w, merged_b['x'] + merged_b['w'] + pad_x)
        y2 = min(crop_h, merged_b['y'] + merged_b['h'] + pad_y)
        crop_img = cropped[y1:y2, x1:x2]

        # MangaCNN classifier: skip junk crops (SFX, Japanese text, dots, etc.)
        is_dialogue, cnn_conf = classify_crop_manga_cnn(crop_img)
        if not is_dialogue and cnn_conf > CNN_REJECTION_CONFIDENCE:
            log.info(f'  crop[{i}] SKIPPED by MangaCNN: junk ({cnn_conf*100:.0f}% confident)')
            ocr_texts.append('')
            continue

        # When using bubble detector, mask out dark artwork areas first
        if used_bubble_detector:
            crop_img = _mask_bubble_white(crop_img)

        enhanced_crop = None

        log.info(f'  crop[{i}] padded: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px (pad {pad_x},{pad_y}) | MangaCNN: dialogue ({cnn_conf*100:.0f}%)')

        # ── OCR Recognition ──
        # Primary: Florence-2 (reads bubble crops as whole images, no line-splitting)
        # Fallback: PaddleOCR raw box text mapping
        florence_text = ''
        paddle_text = _paddle_text_for_box(merged_b, raw_boxes_snapshot)

        if ocr_engines.florence_available:
            # Convert crop to PIL for Florence-2
            pil_crop = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            t_fl = time.time()
            florence_text = florence_ocr(pil_crop)
            fl_ms = int((time.time() - t_fl) * 1000)
            log.info(f'    FLORENCE: "{florence_text[:70]}" ({fl_ms}ms)')

        florence_score = _score_ocr_text(florence_text) if florence_text else -10
        paddle_score = _score_ocr_text(paddle_text)

        # Florence-2 is primary when available
        if ocr_engines.florence_available and florence_score >= 0:
            winner, combined = 'FLORENCE', florence_text
        else:
            winner, combined = 'PADDLE', paddle_text

        # Fallback: if Florence is empty, use PaddleOCR
        if not combined.strip() and paddle_text.strip():
            winner, combined = 'PADDLE', paddle_text
        elif not combined.strip() and florence_text.strip():
            winner, combined = 'FLORENCE', florence_text

        win_counts[winner] = win_counts.get(winner, 0) + 1
        ocr_texts.append(combined)

        log.info(f'  box[{i}] ({merged_b["w"]}x{merged_b["h"]}) [{winner}] florence={florence_score:+d} paddle={paddle_score:+d}')
        log.info(f'    PADDLE:   "{paddle_text[:70]}"')
        log.info(f'    FLORENCE: "{florence_text[:70]}"')
        log.info(f'    WINNER:   "{combined[:70]}"')
        _pe(5, f'Bubble {i+1} OCR', f'[{winner}] "{combined[:80]}"',
            f'debug_crop_{i}_raw.png' if i < 8 else '', 0,
            {'index': i, 'winner': winner, 'text': combined,
             'florence': {'text': florence_text, 'score': florence_score},
             'paddle': {'text': paddle_text, 'score': paddle_score},
             'size': f'{merged_b["w"]}x{merged_b["h"]}'})

        # Save debug crops only for first 3 boxes (was 8 — too slow)
        if SAVE_DEBUG_CROPS and i < 3:
            cv2.imwrite(os.path.join(debug_dir, f'debug_crop_{i}_raw.png'), crop_img)

    ocr_ms = int((time.time() - t0) * 1000)
    filled = sum(1 for t in ocr_texts if t.strip())
    win_str = ', '.join(f'{k} won {v}' for k, v in win_counts.items() if v > 0)
    log.info(f'  Multi-OCR: {filled}/{len(ocr_texts)} have text, {win_str} ({ocr_ms}ms)')
    _pe(5, 'OCR Complete', f'{filled}/{len(ocr_texts)} bubbles have text. {win_str}', '', ocr_ms,
        {'filled': filled, 'total': len(ocr_texts), 'winners': dict(win_counts)})

    log.info('STAGE 7: Dialogue vs SFX filter')
    _pe(7, 'Filtering', 'Removing SFX, non-English text, and garbage...')
    # Build OCR text map keyed by box id (for debug frame labeling)
    ocr_texts_map = {}
    for i, b in enumerate(boxes):
        ocr_texts_map[id(b)] = ocr_texts[i] if i < len(ocr_texts) else ''
    boxes, ocr_texts, rejected_sfx = filter_dialogue_boxes(boxes, ocr_texts, cropped.shape)
    # Map kept box texts too
    for i, b in enumerate(boxes):
        ocr_texts_map[id(b)] = ocr_texts[i] if i < len(ocr_texts) else ''

    log.info('STAGE 8: Text cleanup and formatting')
    bubbles = []
    ocr_with_text = 0
    filtered_out = 0

    for i, b in enumerate(boxes):
        raw_text = ocr_texts[i] if i < len(ocr_texts) else ''
        if raw_text.strip():
            ocr_with_text += 1

        # Early reject: if raw OCR text is mostly non-Latin (Japanese/Chinese/Korean),
        # skip the entire bubble — it's not English dialogue
        raw_flat = raw_text.replace('\n', ' ').strip()
        if raw_flat:
            raw_chars = raw_flat
            non_latin = sum(1 for c in raw_chars if ord(c) > 0x024F and c not in ' \n\t.,!?\'"-…~')
            if len(raw_chars) > 0 and non_latin / len(raw_chars) > 0.1:
                log.info(f'    DROPPED: non-English raw text ({non_latin}/{len(raw_chars)} non-Latin chars)')
                filtered_out += 1
                continue

        after_cleanup = cleanup_text(raw_text)
        after_reconstruct = reconstruct_ocr_text(after_cleanup)
        # Re-clean noise fragments that word splitting may have created
        after_reconstruct = _clean_noise_fragments(after_reconstruct)
        after_reconstruct = re.sub(r'\s+', ' ', after_reconstruct).strip()
        after_corrections = apply_manga_corrections(after_reconstruct)
        # Spellcheck disabled — too many false positives on manga names/SFX
        # (e.g. "Gunko" → "Gunk", character names mangled)
        after_spellfix = after_corrections
        after_speech = format_text_for_speech(after_spellfix)

        log.info(f'  bubble[{i}] text pipeline:')
        log.info(f'    [1] raw OCR:       "{raw_text.replace(chr(10)," ").strip()[:120]}"')
        if after_cleanup != raw_text.replace('\n', ' ').strip():
            log.info(f'    [2] cleanup:       "{after_cleanup[:120]}"')
        if after_reconstruct != after_cleanup:
            log.info(f'    [3] reconstruct:   "{after_reconstruct[:120]}"')
        if after_corrections != after_reconstruct:
            log.info(f'    [4] corrections:   "{after_corrections[:120]}"')
        log.info(f'    [5] final speech:  "{after_speech[:120]}"')

        if not after_speech.strip():
            reason = 'empty after cleanup' if raw_text.strip() else 'OCR returned nothing'
            log.info(f'    DROPPED: {reason}')
            if raw_text.strip():
                filtered_out += 1
            continue

        # Post-cleanup quality gate: drop text that's too short/garbled to be real dialogue
        # But keep valid short manga dialogue like "Ah!", "No!", "Fiancee...", single words
        speech_alpha = re.sub(r'[^a-zA-Z]', '', after_speech)
        speech_words = [w for w in after_speech.split() if len(w) >= 2 and any(c.isalpha() for c in w)]
        # Only drop truly empty/garbage: less than 2 alpha chars
        if len(speech_alpha) < 2:
            log.info(f'    DROPPED: too short/garbled after cleanup ({len(speech_alpha)} alpha, {len(speech_words)} words)')
            filtered_out += 1
            continue

        # Gibberish detector: multiple checks for garbled OCR output
        # Check 0: Short text where ALL words are non-dictionary (catches "Rrr Tii.", "Ii Rrr Tii")
        # Use exact match (distance 0) to be strict — "rrr" shouldn't match "err"
        if len(speech_words) >= 2 and len(speech_words) <= 4 and _symspell:
            all_unknown = True
            for w in speech_words:
                core = re.sub(r'[^a-zA-Z]', '', w).lower()
                if len(core) >= 2:
                    results = _symspell.lookup(core, max_edit_distance=0, verbosity=0)
                    if results:
                        all_unknown = False
                        break
                elif core in ('a', 'i'):
                    all_unknown = False
                    break
            if all_unknown:
                log.info(f'    DROPPED: all words unknown in short text')
                filtered_out += 1
                continue

        # Check 0b: Very short text (≤5 words) with repeated letters (OCR reading noise)
        # Catches patterns like "Rrr", "Tii", "usu" — repeated consonant/vowel patterns
        # But NOT real words like "too", "see", "all" which also have few unique chars
        if len(speech_words) <= 5 and _symspell:
            repeated_noise = 0
            for w in speech_words:
                core = re.sub(r'[^a-zA-Z]', '', w).lower()
                if len(core) >= 2 and len(set(core)) <= 2:
                    # Only count as noise if NOT a dictionary word
                    results = _symspell.lookup(core, max_edit_distance=0, verbosity=0)
                    if not results:
                        repeated_noise += 1
            if repeated_noise >= 2 or (len(speech_words) <= 3 and repeated_noise >= 1 and len(speech_alpha) < 12):
                log.info(f'    DROPPED: repeated-letter noise ({repeated_noise} noise words)')
                filtered_out += 1
                continue

        if len(speech_words) >= 3:
            # Check 1: Too many unknown words (not in dictionary)
            if _symspell:
                gibberish_count = 0
                checked = 0
                for w in speech_words:
                    core = re.sub(r'[^a-zA-Z]', '', w).lower()
                    if len(core) >= 3 and "'" not in w:
                        checked += 1
                        # Try exact match first, then distance=1 for OCR typos
                        results = _symspell.lookup(core, max_edit_distance=1, verbosity=0)
                        if not results:
                            gibberish_count += 1
                if checked >= GIBBERISH_MIN_CHECKED and gibberish_count / max(1, checked) >= GIBBERISH_THRESHOLD:
                    log.info(f'    DROPPED: gibberish ({gibberish_count}/{checked} unknown words)')
                    filtered_out += 1
                    continue

            # Check 2: Symbol garbage — underscores, stray punctuation clusters, math symbols
            symbol_junk = len(re.findall(JUNK_SYMBOL_PATTERN, after_speech))
            if symbol_junk >= SYMBOL_JUNK_LIMIT:
                log.info(f'    DROPPED: symbol garbage ({symbol_junk} junk symbols)')
                filtered_out += 1
                continue

            # Check 3: Too many very short words (0-1 alpha chars) = fragmented OCR noise
            tiny_count = sum(1 for w in after_speech.split() if len(re.sub(r'[^a-zA-Z]', '', w)) <= 1)
            total_words = len(after_speech.split())
            if total_words >= 4 and tiny_count / total_words > 0.35:
                log.info(f'    DROPPED: fragmented noise ({tiny_count}/{total_words} tiny words)')
                filtered_out += 1
                continue

            # Check 4: Too many digits — page numbers, coordinates, not dialogue
            digit_count = sum(1 for c in after_speech if c.isdigit())
            if digit_count >= 3 and digit_count / max(1, len(after_speech.replace(' ',''))) > 0.12:
                log.info(f'    DROPPED: digit-heavy ({digit_count} digits)')
                filtered_out += 1
                continue

            # Check 5: Incoherent word soup — many random words with no sentence structure
            # OCR noise from textures: random dictionary words strung together
            # Must be very strict to avoid dropping real dialogue like "By the guy who's attacking us."
            if len(speech_words) >= 7:
                has_strong_punct = bool(re.search(r'[!?…]', after_speech))
                # Count very short words (1-3 chars)
                short_count = sum(1 for w in speech_words if len(re.sub(r'[^a-zA-Z]', '', w)) <= 3)
                short_ratio = short_count / len(speech_words)
                # Only filter if >70% short words AND no expressive punctuation AND 7+ words
                if short_ratio > 0.7 and not has_strong_punct:
                    log.info(f'    DROPPED: incoherent word soup ({short_count}/{len(speech_words)} short words, no punctuation)')
                    filtered_out += 1
                    continue

        log.info(f'    KEPT')
        _pe(8, f'Text Cleaned: Bubble {len(bubbles)+1}', f'"{after_speech[:80]}"', '', 0,
            {'raw': raw_text.replace('\n',' ').strip()[:120], 'cleanup': after_cleanup[:120],
             'reconstruct': after_reconstruct[:120], 'corrections': after_corrections[:120],
             'final': after_speech[:120], 'status': 'kept'})
        bubbles.append({
            'text': after_speech,
            'conf': round(b.get('score', 1.0), 3),
            'left': round((actual_crop_left + b['x']) / dpr, 1),
            'top': round((actual_crop_top + b['y']) / dpr, 1),
            'width': round(b['w'] / dpr, 1),
            'height': round(b['h'] / dpr, 1),
        })

    log.info(f'  Result: {len(boxes)} boxes -> {ocr_with_text} had OCR text -> {filtered_out} cleaned to empty -> {len(bubbles)} bubbles')

    # 9. Sort reading order (direction validated at top of _do_process)
    bubbles = sort_reading_order(bubbles, direction=reading_dir)

    # Save debug images in background thread (don't block response)
    def _save_debug_bg(cropped_cp, raw_snap, merged_snap, rej_geo, kept_boxes, rej_sfx, ocr_map, boxes_cp, ocr_cp):
        try:
            if DEBUG_VIEW:
                save_debug_frame(cropped_cp, raw_snap, merged_snap,
                                 rej_geo, kept_boxes, rej_sfx, kept_boxes, ocr_map)
            debug_img = cropped_cp.copy()
            for di, db in enumerate(boxes_cp):
                color = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (200, 0, 255), (255, 0, 200)][di % 5]
                cv2.rectangle(debug_img, (db['x'], db['y']), (db['x'] + db['w'], db['y'] + db['h']), color, 3)
                olabel = (ocr_cp[di] if di < len(ocr_cp) else '')[:30].replace('\n', ' ')
                cv2.putText(debug_img, f'{di+1}: {olabel}', (db['x'], db['y'] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), 'debug_last.jpg'), debug_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception:
            pass
    threading.Thread(target=_save_debug_bg, args=(
        cropped.copy(), raw_boxes_snapshot, merged_boxes_snapshot,
        rejected_geo, [dict(b) for b in boxes], rejected_sfx, ocr_texts_map,
        [dict(b) for b in boxes], list(ocr_texts)), daemon=True).start()

    total_ms = int((time.time() - t_start) * 1000)
    ocr_returned = sum(1 for t in ocr_texts if t.strip())
    log.info('STAGE 9: Final result')
    sent_to_ocr = len(merged_boxes_snapshot) - len(rejected_geo)
    log.info(f'  Total detected: {len(raw_boxes_snapshot)}')
    log.info(f'  After merge: {len(merged_boxes_snapshot)}')
    log.info(f'  Rejected by geo filter: {len(rejected_geo)}')
    log.info(f'  Sent to OCR: {sent_to_ocr}')
    log.info(f'  Rejected as SFX: {len(rejected_sfx)}')
    log.info(f'  Returned as dialogue: {len(bubbles)}')
    log.info(f'  Pipeline: {len(raw_boxes_snapshot)} detected -> {ocr_returned} OCR hits -> {len(bubbles)} bubbles ({total_ms}ms)')
    with _pipeline_lock:
        _pipeline_status = 'done'
    _pe(9, 'Done', f'{len(bubbles)} bubbles ready for TTS ({total_ms}ms)', 'latest.png', total_ms,
        {'total_detected': len(raw_boxes_snapshot), 'after_merge': len(merged_boxes_snapshot),
         'geo_rejected': len(rejected_geo), 'sfx_rejected': len(rejected_sfx),
         'final_bubbles': len(bubbles), 'total_ms': total_ms,
         'bubbles': [{'i': i+1, 'text': b['text'], 'pos': f"({b['left']:.0f},{b['top']:.0f})", 'size': f"{b['width']:.0f}x{b['height']:.0f}"} for i, b in enumerate(bubbles)]})
    for i, b in enumerate(bubbles):
        log.info(f'  [{i+1}] {b["text"][:60]}')

    # All I/O (debug saves, learning, auto-learning) runs in background — don't block response
    def _background_saves(cropped_bg, boxes_bg, ocr_bg, rejected_bg, bubbles_bg, total_ms_bg, crop_w_bg, crop_h_bg):
        try:
            base_dir = os.path.dirname(__file__)
            ts_bg = time.strftime('%Y%m%d_%H%M%S')

            # Debug text
            debug_text_path = os.path.join(base_dir, 'debug_frames', 'debug_text_output.txt')
            with open(debug_text_path, 'w') as f:
                for ib, bb in enumerate(bubbles_bg):
                    f.write(f'[{ib+1}] {bb["text"]}\n')

            # Test result
            test_dir = os.path.join(base_dir, 'test_results')
            os.makedirs(test_dir, exist_ok=True)
            with open(os.path.join(test_dir, f'result_{ts_bg}.json'), 'w') as f:
                json.dump({'bubbles': [{'i': i+1, 'text': b['text']} for i, b in enumerate(bubbles_bg)], 'ms': total_ms_bg}, f)

            # Self-learning saves
            learn_dir = os.path.join(base_dir, 'learning')
            os.makedirs(os.path.join(learn_dir, 'kept'), exist_ok=True)
            os.makedirs(os.path.join(learn_dir, 'rejected'), exist_ok=True)

            for ib, bb in enumerate(boxes_bg):
                try:
                    bx1 = max(0, bb['x'] - 5); by1 = max(0, bb['y'] - 5)
                    bx2 = min(crop_w_bg, bb['x'] + bb['w'] + 5); by2 = min(crop_h_bg, bb['y'] + bb['h'] + 5)
                    crop_s = cropped_bg[by1:by2, bx1:bx2]
                    txt = (ocr_bg[ib] if ib < len(ocr_bg) else '')[:30]
                    txt = re.sub(r'[^a-zA-Z0-9_]', '', txt.replace(' ', '_'))[:20]
                    cv2.imwrite(os.path.join(learn_dir, 'kept', f'{ts_bg}_k{ib}_{txt}.png'), crop_s)
                except Exception:
                    pass

            for ib, rej in enumerate(rejected_bg):
                try:
                    bb = rej.get('box', rej) if isinstance(rej, dict) else rej
                    bx1 = max(0, bb['x'] - 5); by1 = max(0, bb['y'] - 5)
                    bx2 = min(crop_w_bg, bb['x'] + bb['w'] + 5); by2 = min(crop_h_bg, bb['y'] + bb['h'] + 5)
                    cv2.imwrite(os.path.join(learn_dir, 'rejected', f'{ts_bg}_r{ib}.png'), cropped_bg[by1:by2, bx1:bx2])
                except Exception:
                    pass

            # Auto-learning: garbage text that got through = save as junk
            train_j = os.path.join(base_dir, 'dataset', 'crops', 'junk')
            os.makedirs(train_j, exist_ok=True)
            for ib, bb in enumerate(boxes_bg):
                txt = ocr_bg[ib] if ib < len(ocr_bg) else ''
                if not txt.strip():
                    continue
                alpha = re.sub(r'[^a-zA-Z]', '', txt)
                if len(alpha) < 2:
                    try:
                        bx1 = max(0, bb['x'] - 5); by1 = max(0, bb['y'] - 5)
                        bx2 = min(crop_w_bg, bb['x'] + bb['w'] + 5); by2 = min(crop_h_bg, bb['y'] + bb['h'] + 5)
                        cv2.imwrite(os.path.join(train_j, f'autofix_{ts_bg}_j{ib}.png'), cropped_bg[by1:by2, bx1:bx2])
                    except Exception:
                        pass
        except Exception:
            pass

    threading.Thread(target=_background_saves, args=(
        cropped.copy(), [dict(b) for b in boxes], list(ocr_texts),
        list(rejected_sfx), list(bubbles), total_ms, crop_w, crop_h), daemon=True).start()

    return jsonify({
        'bubbles': bubbles,
        'actualCropLeft': round(actual_crop_left / dpr, 1),
        'actualCropTop': round(actual_crop_top / dpr, 1),
        'timing': {
            'total_ms': total_ms,
            'decode_ms': decode_ms,
            'detect_ms': det_ms,
            'merge_ms': group_ms,
            'ocr_ms': ocr_ms,
            'boxes_detected': len(boxes),
            'ocr_returned': ocr_returned,
            'bubbles_returned': len(bubbles),
            'used_fallbacks': used_inversion,
        }
    })


@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Profile CPU usage per pipeline stage. POST with a test image (same format as /process).
    Only available when MVR_ENABLE_BENCHMARK=true and from localhost."""
    # Safety: only localhost + explicit env flag
    if os.environ.get('MVR_ENABLE_BENCHMARK', '').lower() not in ('true', '1', 'yes'):
        return jsonify({'error': 'Benchmark disabled. Set MVR_ENABLE_BENCHMARK=true'}), 403
    if request.remote_addr not in ('127.0.0.1', '::1'):
        return jsonify({'error': 'Localhost only'}), 403
    import psutil
    proc = psutil.Process()

    def _cpu_snap():
        return proc.cpu_percent(interval=None)

    # Prime the CPU measurement
    _cpu_snap()

    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image'}), 400

    results = []
    image_data = data['image']

    # Stage 1: Decode
    _cpu_snap()
    t0 = time.time()
    img = decode_base64_image(image_data)
    ms = int((time.time() - t0) * 1000)
    cpu = _cpu_snap()
    results.append({'stage': 'decode', 'ms': ms, 'cpu_pct': cpu})

    if img is None:
        return jsonify({'error': 'Decode failed'}), 400

    crop_h, crop_w = img.shape[:2]
    crop = data.get('cropRect')
    if crop and crop.get('width') and crop.get('height'):
        dpr = data.get('dpr', 2)
        cx = int(crop['left'] * dpr)
        cy = int(crop['top'] * dpr)
        cw = int(crop['width'] * dpr)
        ch = int(crop['height'] * dpr)
        cx, cy = max(0, cx), max(0, cy)
        cw = min(cw, crop_w - cx)
        ch = min(ch, crop_h - cy)
        img = img[cy:cy+ch, cx:cx+cw]
        crop_h, crop_w = img.shape[:2]

    det_img = img
    if crop_w > MAX_DETECT_WIDTH:
        det_scale = MAX_DETECT_WIDTH / crop_w
        det_img = cv2.resize(img, (MAX_DETECT_WIDTH, int(crop_h * det_scale)))

    # Stage 2: Bubble detector ONLY (sequential)
    _cpu_snap()
    t0 = time.time()
    if bubble_detector.available:
        bubble_detector.detect(img, conf_threshold=0.3)
    ms = int((time.time() - t0) * 1000)
    cpu = _cpu_snap()
    results.append({'stage': 'bubble_detector_only', 'ms': ms, 'cpu_pct': cpu})

    # Stage 3: PaddleOCR detection ONLY (sequential)
    _cpu_snap()
    t0 = time.time()
    paddle_boxes = run_detection(det_img, ocr_engines.ocr_engine)
    ms = int((time.time() - t0) * 1000)
    cpu = _cpu_snap()
    results.append({'stage': 'paddle_detect_only', 'ms': ms, 'cpu_pct': cpu})

    # Stage 4: Both in PARALLEL (current pipeline behavior)
    _cpu_snap()
    t0 = time.time()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(lambda: bubble_detector.detect(img, conf_threshold=0.3) if bubble_detector.available else None)
        f2 = ex.submit(lambda: run_detection(det_img, ocr_engines.ocr_engine))
        concurrent.futures.wait([f1, f2])
    ms = int((time.time() - t0) * 1000)
    cpu = _cpu_snap()
    results.append({'stage': 'bubble+paddle_parallel', 'ms': ms, 'cpu_pct': cpu})

    # Stage 5: Florence OCR on first bubble crop
    _cpu_snap()
    t0 = time.time()
    if ocr_engines.florence_available and len(paddle_boxes) > 0:
        b0 = paddle_boxes[0]
        x1 = max(0, b0['x'] - 30)
        y1 = max(0, b0['y'] - 30)
        x2 = min(crop_w, b0['x'] + b0['w'] + 30)
        y2 = min(crop_h, b0['y'] + b0['h'] + 30)
        crop_img = img[y1:y2, x1:x2]
        pil_crop = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        florence_ocr(pil_crop)
    ms = int((time.time() - t0) * 1000)
    cpu = _cpu_snap()
    results.append({'stage': 'florence_ocr_1crop', 'ms': ms, 'cpu_pct': cpu})

    # Stage 6: OpenCV operations (resize, morphology, debug write)
    _cpu_snap()
    t0 = time.time()
    for _ in range(3):
        cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    ms = int((time.time() - t0) * 1000)
    cpu = _cpu_snap()
    results.append({'stage': 'opencv_ops_3x', 'ms': ms, 'cpu_pct': cpu})

    # Summary
    import multiprocessing
    summary = {
        'stages': results,
        'cpu_count': multiprocessing.cpu_count(),
        'thread_caps': {
            'OMP': os.environ.get('OMP_NUM_THREADS', 'unset'),
            'MKL': os.environ.get('MKL_NUM_THREADS', 'unset'),
            'PADDLE': os.environ.get('FLAGS_num_threads', 'unset'),
        },
        'paddle_boxes': len(paddle_boxes),
    }
    return jsonify(summary)


@app.route('/feedback', methods=['POST'])
def feedback():
    """Mark a crop as wrong. Moves it to the correct training folder.
    POST json: {file: 'kept/xxx.png', correct_label: 'junk'} or {file: 'rejected/xxx.png', correct_label: 'dialogue'}"""
    data = request.json
    if not data or 'file' not in data or 'correct_label' not in data:
        return jsonify({'error': 'Need file and correct_label'}), 400

    base = os.path.dirname(__file__)
    learn_dir = os.path.join(base, 'learning')
    src = os.path.join(learn_dir, data['file'])
    label = data['correct_label']

    if not os.path.exists(src):
        return jsonify({'error': 'File not found'}), 404

    # Move to training dataset
    if label == 'junk':
        dest_dir = os.path.join(base, 'dataset', 'crops', 'junk')
    else:
        dest_dir = os.path.join(base, 'dataset', 'crops', 'dialogue')

    os.makedirs(dest_dir, exist_ok=True)
    fname = f'feedback_{os.path.basename(src)}'
    import shutil
    shutil.move(src, os.path.join(dest_dir, fname))
    log.info(f'[FEEDBACK] Moved {data["file"]} -> {label}/{fname}')

    # Count accumulated feedback
    fb_count = len([f for f in os.listdir(os.path.join(base, 'dataset', 'crops', 'junk')) if f.startswith('feedback_')]) + \
               len([f for f in os.listdir(os.path.join(base, 'dataset', 'crops', 'dialogue')) if f.startswith('feedback_')])

    return jsonify({'ok': True, 'moved_to': label, 'total_feedback': fb_count,
                    'message': f'Saved! {fb_count} feedback crops collected. Retrain when ready.'})


@app.route('/feedback/list', methods=['GET'])
def feedback_list():
    """List recent crops for review."""
    base = os.path.dirname(__file__)
    learn_dir = os.path.join(base, 'learning')
    result = {'kept': [], 'rejected': []}
    for folder in ['kept', 'rejected']:
        fdir = os.path.join(learn_dir, folder)
        if os.path.isdir(fdir):
            files = sorted(os.listdir(fdir), reverse=True)[:50]
            result[folder] = [f'{folder}/{f}' for f in files if f.endswith('.png')]
    return jsonify(result)


@app.route('/feedback/image/<path:filename>', methods=['GET'])
def feedback_image(filename):
    """Serve a learning crop image."""
    learn_dir = os.path.join(os.path.dirname(__file__), 'learning')
    return send_from_directory(learn_dir, filename)


@app.route('/', methods=['GET'])
def index():
    return jsonify({'name': 'Manga Voice Reader', 'status': 'ok', 'endpoints': ['/health', '/process', '/debug', '/debug/frame', '/tts', '/tts/status', '/showcase']})

@app.route('/health', methods=['GET'])
def health():
    ocr_name = 'Florence-2 + PaddleOCR' if ocr_engines.florence_available else 'PaddleOCR-v5'
    return jsonify({'status': 'ok', 'detector': ocr_engines.ocr_engine is not None, 'ocr': ocr_name, 'florence': ocr_engines.florence_available})


# ─── Extension Debug Log ──────────────────────────────────────────────────────
_ext_log_path = os.path.join(os.path.dirname(__file__), 'extension_debug.log')

@app.route('/ext-log', methods=['POST'])
def ext_log():
    data = request.get_json()
    msg = data.get('msg', '') if data else ''
    if msg:
        _rotate_ext_log(_ext_log_path)
        with open(_ext_log_path, 'a') as f:
            f.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')
    return jsonify({'ok': True})

@app.route('/showcase/status', methods=['GET'])
def showcase_status():
    """Live system status for the showcase dashboard."""
    import psutil, platform
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()

    # Model file sizes
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    def fsize(p):
        fp = os.path.join(models_dir, p)
        return f'{os.path.getsize(fp)/1e6:.0f}MB' if os.path.isfile(fp) else 'N/A'

    # Voices
    voices = sorted(tts_engine._kokoro_model.get_voices()) if tts_engine._kokoro_model else []

    # Debug frame info
    frame_path = os.path.join(os.path.dirname(__file__), 'debug_frames', 'latest.png')
    has_frame = os.path.isfile(frame_path)
    frame_age = int(time.time() - os.path.getmtime(frame_path)) if has_frame else -1

    # Last debug text
    text_path = os.path.join(os.path.dirname(__file__), 'debug_frames', 'debug_text_output.txt')
    last_text = ''
    if os.path.isfile(text_path):
        with open(text_path, 'r') as f:
            last_text = f.read()

    # GPU info
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'vram_used_mb': torch.cuda.memory_allocated() // 1024 // 1024,
                'vram_total_mb': torch.cuda.get_device_properties(0).total_memory // 1024 // 1024,
            }
    except Exception:
        pass

    return jsonify({
        'system': {
            'platform': platform.system(),
            'arch': platform.machine(),
            'python': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'ram_used_mb': round(mem.rss / 1e6, 1),
            'gpu': gpu_info,
        },
        'models': {
            'florence2': {'status': ocr_engines.florence_available, 'model': 'Florence-2-large-ft', 'params': '0.77B', 'vram_mb': 1469, 'role': 'Primary OCR'},
            'paddle_ocr': {'status': ocr_engines.ocr_engine is not None, 'version': 'PP-OCRv5', 'det': 'mobile', 'rec': 'server', 'role': 'Text Detection + OCR Fallback'},
            'bubble_detector': {'status': bubble_detector.available, 'file': 'detector.onnx', 'size': fsize('detector.onnx'), 'arch': 'RT-DETR-v2', 'training': '11k comic images'},
            'text_segmenter': {'status': text_segmenter.available, 'file': 'comictextdetector.pt.onnx', 'size': fsize('comic_text_detector/comictextdetector.pt.onnx')},
            'kokoro_tts': {'status': tts_engine.kokoro_available, 'file': 'kokoro-v1.0.onnx', 'size': fsize('kokoro/kokoro-v1.0.onnx'), 'params': '82M', 'sample_rate': 24000, 'voices': len(voices), 'role': 'Primary TTS'},
            'piper_tts': {'status': tts_engine.piper_available, 'file': os.path.basename(PIPER_MODEL), 'size': fsize(f'piper/{os.path.basename(PIPER_MODEL)}'), 'role': 'Fallback TTS'},
            'tesseract': {'status': ocr_engines.tesseract_available, 'role': 'Legacy fallback'},
        },
        'tts_voices': voices,
        'debug': {
            'has_frame': has_frame,
            'frame_age_s': frame_age,
            'last_text': last_text,
        },
        'cache_size': len(_page_cache),
    })

@app.route('/showcase/pipeline', methods=['GET'])
def showcase_pipeline():
    """Live pipeline events for the showcase dashboard."""
    since = request.args.get('since', 0, type=int)
    with _pipeline_lock:
        events = [e for e in _pipeline_events if e['id'] >= since]
        return jsonify({
            'status': _pipeline_status,
            'run_id': _pipeline_run_id,
            'events': events,
            'total': len(_pipeline_events),
        })

@app.route('/showcase', methods=['GET'])
def showcase():
    """Full system showcase dashboard."""
    from flask import Response
    html = open(os.path.join(os.path.dirname(__file__), 'showcase.html'), 'r').read()
    return Response(html, mimetype='text/html')

@app.route('/debug', methods=['GET'])
def debug():
    """View the last processed screenshot with detected boxes."""
    from flask import send_file
    debug_path = os.path.join(os.path.dirname(__file__), 'debug_last.jpg')
    if not os.path.isfile(debug_path):
        return 'No debug image yet. Click "Read Page" first.', 404
    return send_file(debug_path, mimetype='image/jpeg')

@app.route('/debug/frame', methods=['GET'])
def debug_frame():
    """Live debug viewer — auto-refreshes every 3 seconds."""
    from flask import Response
    frame_path = os.path.join(os.path.dirname(__file__), 'debug_frames', 'latest.png')
    text_path = os.path.join(os.path.dirname(__file__), 'debug_frames', 'debug_text_output.txt')

    has_frame = os.path.isfile(frame_path)
    frame_mtime = os.path.getmtime(frame_path) if has_frame else 0
    text_content = ''
    if os.path.isfile(text_path):
        with open(text_path, 'r') as f:
            text_content = f.read()

    age = int(time.time() - frame_mtime) if has_frame else -1
    age_str = f'{age}s ago' if age >= 0 else 'no frame yet'

    html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>MVR Debug</title>
<meta http-equiv="refresh" content="3">
<style>
  body {{ background: #111; color: #eee; font-family: -apple-system, sans-serif; margin: 0; padding: 16px; }}
  h1 {{ font-size: 18px; color: #e94560; margin: 0 0 8px; }}
  .meta {{ font-size: 12px; color: #777; margin-bottom: 12px; }}
  img {{ max-width: 100%; border: 1px solid #333; border-radius: 6px; }}
  .text-box {{ background: #1a1a2e; border: 1px solid #333; border-radius: 6px; padding: 12px; margin-top: 12px; font-size: 13px; line-height: 1.6; white-space: pre-wrap; }}
  .no-frame {{ color: #777; font-style: italic; padding: 40px; text-align: center; }}
</style>
</head><body>
<h1>Manga Voice Reader — Debug</h1>
<div class="meta">Auto-refreshes every 3s &middot; Last update: {age_str}</div>
{'<img src="/debug/frame.png?t=' + str(int(frame_mtime)) + '">' if has_frame else '<div class="no-frame">No debug frame yet. Click "Read Page" in the extension.</div>'}
<div class="text-box">{text_content if text_content else 'No text output yet.'}</div>
</body></html>'''
    return Response(html, mimetype='text/html')

@app.route('/debug/frame.png', methods=['GET'])
def debug_frame_png():
    """Raw debug frame image."""
    from flask import send_file
    frame_path = os.path.join(os.path.dirname(__file__), 'debug_frames', 'latest.png')
    if not os.path.isfile(frame_path):
        return 'No debug frame yet.', 404
    return send_file(frame_path, mimetype='image/png')

@app.route('/debug/frames/<path:filename>', methods=['GET'])
def debug_frames_file(filename):
    """Serve any file from the debug_frames directory."""
    from flask import send_file
    import re as _re
    # Security: only allow simple filenames (no path traversal)
    if not _re.match(r'^[a-zA-Z0-9_\-\.]+$', filename):
        return 'Invalid filename', 400
    fp = os.path.join(os.path.dirname(__file__), 'debug_frames', filename)
    if not os.path.isfile(fp):
        return 'Not found', 404
    mime = 'image/png' if fp.endswith('.png') else 'image/jpeg' if fp.endswith('.jpg') else 'application/octet-stream'
    return send_file(fp, mimetype=mime)



@app.route('/tts', methods=['POST'])
def tts_endpoint():
    """Generate speech audio from text using Kokoro (primary) or Piper (fallback)."""
    if _shutting_down:
        return jsonify({'error': 'Server is shutting down'}), 503
    if not _tts_semaphore.acquire(blocking=True, timeout=5):
        return jsonify({'error': 'TTS busy, try again'}), 429
    try:
        return _do_tts()
    finally:
        _tts_semaphore.release()

def _do_tts():
    data = request.json
    text = (data or {}).get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    voice = (data or {}).get('voice', tts_engine._kokoro_voice)
    speed = float((data or {}).get('speed', 1.0))
    log.info(f'[TTS] "{text[:80]}{"..." if len(text) > 80 else ""}" voice={voice} speed={speed}')

    try:
        audio_bytes, engine, synth_ms = tts_engine.synthesize_speech(text, voice=voice, speed=speed)
        log.info(f'[TTS] {engine} ({voice}): {len(audio_bytes)} bytes, {synth_ms}ms')
        _pe('tts', 'TTS Generated', f'"{text[:60]}" — {voice} ({synth_ms}ms, {len(audio_bytes)//1024}KB)', '', synth_ms,
            {'voice': voice, 'text': text[:120], 'bytes': len(audio_bytes), 'engine': engine})
        from flask import Response
        return Response(audio_bytes, mimetype='audio/wav')
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tts/status', methods=['GET'])
def tts_status():
    """Check if local AI TTS is available."""
    return jsonify(tts_engine.get_status())

@app.route('/shutdown', methods=['POST'])
def shutdown_server():
    """Shut down the server gracefully (called by extension on close)."""
    global _last_shutdown_time
    now = time.time()
    if now - _last_shutdown_time < 10:
        return jsonify({'error': 'Shutdown already in progress'}), 429
    _last_shutdown_time = now
    log.info('Shutdown requested by extension')
    global _shutting_down
    _shutting_down = True
    def _do_shutdown():
        # Wait for in-flight requests to finish (up to 30 seconds)
        for _ in range(30):
            process_idle = _process_semaphore.acquire(blocking=False)
            tts_idle = _tts_semaphore.acquire(blocking=False)
            if process_idle:
                _process_semaphore.release()
            if tts_idle:
                _tts_semaphore.release()
            if process_idle and tts_idle:
                break
            time.sleep(1)
        # Free GPU memory before exit
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log.info('GPU cache cleared')
        except Exception:
            pass
        log.info('Exiting cleanly...')
        os._exit(0)
    threading.Thread(target=_do_shutdown, daemon=True).start()
    return jsonify({'ok': True})

# ─── Startup ────────────────────────────────────────────────────────────────



# ─── Comic text segmenter (text mask for clean OCR) ───────────────────────

text_segmenter = ComicTextSegmenter()

if __name__ == '__main__':
    t_startup = time.time()
    log.info('Loading all models in parallel...')

    # Load everything in parallel threads for fast startup
    def _load_ocr_thread():
        load_detector()
        load_tesseract()
        load_manga_cnn()
        load_local_recognizer()

    def _load_visual_thread():
        load_super_res()
        bubble_detector.load()
        text_segmenter.load()

    # Phase 1: Load OCR + visual models in parallel
    threads = [
        threading.Thread(target=_load_ocr_thread, name='load-ocr'),
        threading.Thread(target=_load_visual_thread, name='load-visual'),
    ]
    for t in threads:
        t.start()
    load_manga_corrections()  # fast, run on main thread
    for t in threads:
        t.join()

    # Phase 2: Load Florence-2 OCR + TTS after detection models are done
    ocr_engines._load_florence()
    tts_engine.init()
    if tts_engine.kokoro_available:
        tts_engine._kokoro_model.create('Warm up.', voice=tts_engine._kokoro_voice, speed=1.0)
        log.info('[TTS] Kokoro warmed up')

    log.info(f'All models loaded in {time.time()-t_startup:.1f}s')
    if COLLECT_TRAINING_DATA:
        log.info('Training data collection ENABLED')
    if DEBUG_VIEW:
        log.info('DEBUG_VIEW ENABLED — saving combined debug frames to server/debug_frames/')
        log.info('  View latest: http://127.0.0.1:{}/debug/frame'.format(PORT))
    # Auto-shutdown after idle period (saves GPU resources when gaming)
    # IDLE_SHUTDOWN_MINUTES imported from config.py (default 30 min)
    _idle_tracker = {'last': time.time()}

    @app.before_request
    def _track_activity():
        _idle_tracker['last'] = time.time()

    def _idle_watchdog():
        while True:
            time.sleep(60)
            idle_mins = (time.time() - _idle_tracker['last']) / 60
            if idle_mins >= IDLE_SHUTDOWN_MINUTES:
                log.info(f'No requests for {IDLE_SHUTDOWN_MINUTES} min — auto-shutting down to save resources.')
                os._exit(0)

    watchdog = threading.Thread(target=_idle_watchdog, daemon=True)
    watchdog.start()
    log.info(f'Auto-shutdown enabled: server will stop after {IDLE_SHUTDOWN_MINUTES} min idle')

    log.info(f'Server ready on http://127.0.0.1:{PORT}')
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
