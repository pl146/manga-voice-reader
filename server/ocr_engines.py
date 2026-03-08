"""OCR engine loading, execution, and image decoding utilities.

This module centralises every OCR-related component used by the manga voice
reader server:

  * PaddleOCR detector (speech-bubble detection)
  * Florence-2 (primary GPU-based OCR recognition)
  * Tesseract (CPU fallback OCR)
  * manga-ocr (Japanese manga OCR, optional)
  * MangaCNN ONNX classifier (dialogue vs. junk filtering)
  * Local PaddleOCR font recognizer (hybrid mode)
  * ESPCN x2 super-resolution for OCR crop enhancement
  * Base64 image decoding helper

All public function names are kept identical to the original server module so
that call-sites can simply ``from ocr_engines import …`` with no changes.
"""

import base64
import logging
import os
import threading
import time

import cv2
import numpy as np
import pytesseract
from PIL import Image

from text_processing import _symspell, _have_wordninja
from config import (MAX_DETECT_WIDTH, TESSERACT_CONFIG, USE_LOCAL_RECOGNIZER,
                    FLORENCE_MODEL_NAME, FLORENCE_MAX_TOKENS, FLORENCE_NUM_BEAMS)

# ─── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ─── Configuration constants ─────────────────────────────────────────────────

# Configuration imported from config.py

# ─── Module-level globals ────────────────────────────────────────────────────

ocr_engine = None
_ocr_engine_lock = threading.Lock()

_florence_model = None
_florence_processor = None
_florence_lock = threading.Lock()
florence_available = False

tesseract_available = False

manga_cnn_session = None

manga_ocr_engine = None

local_recognizer = None

_sr_model = None

# ─── PaddleOCR detector (detection only) ─────────────────────────────────────

def load_detector():
    global ocr_engine
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    from paddleocr import PaddleOCR

    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'production_detector')
    has_model = os.path.isdir(model_dir) and any(
        f.endswith(('.pdparams', '.pdmodel', '.pdiparams'))
        for f in os.listdir(model_dir)
    )

    kwargs = {
        'use_textline_orientation': False,
        'use_doc_orientation_classify': False,
        'use_doc_unwarping': False,
        'text_det_thresh': 0.3,
        'text_det_box_thresh': 0.4,
        'text_det_unclip_ratio': 2.5,
        'text_det_limit_side_len': MAX_DETECT_WIDTH,
    }
    if has_model:
        kwargs['text_detection_model_dir'] = model_dir
        log.info(f'Loading custom detector from {model_dir}')
    else:
        # PP-OCRv5 mobile detector + server recognizer
        # v5 mobile det is lightweight, v5 server rec has best accuracy
        kwargs['text_detection_model_name'] = 'PP-OCRv5_mobile_det'
        kwargs['text_recognition_model_name'] = 'PP-OCRv5_server_rec'
        log.info('Using PP-OCRv5 mobile det + server rec')

    ocr_engine = PaddleOCR(**kwargs)
    log.info('Detector loaded.')

    # Warm up with a dummy image so first real request is fast
    import numpy as _np
    dummy = _np.zeros((100, 200, 3), dtype=_np.uint8)
    dummy[20:80, 20:180] = 255
    try:
        ocr_engine.predict(dummy)
        log.info('Detector warmed up.')
    except Exception as e:
        log.warning(f'Warm-up failed (first real request may be slow): {e}')

# ─── Florence-2 (primary OCR recognition) ────────────────────────────────────

def _load_florence():
    """Load Florence-2-large-ft for OCR recognition on bubble crops."""
    global _florence_model, _florence_processor, florence_available
    try:
        import torch
        from config import TORCH_THREAD_CAP
        if TORCH_THREAD_CAP > 0:
            torch.set_num_threads(TORCH_THREAD_CAP)
        if not torch.cuda.is_available():
            log.warning('[OCR] Florence-2 requires CUDA GPU, skipping')
            return
        from transformers import AutoModelForCausalLM, AutoProcessor
        t0 = time.time()
        _florence_model = AutoModelForCausalLM.from_pretrained(
            'microsoft/Florence-2-large-ft',
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to('cuda:0')
        _florence_processor = AutoProcessor.from_pretrained(
            'microsoft/Florence-2-large-ft',
            trust_remote_code=True,
        )
        ms = int((time.time() - t0) * 1000)
        vram = torch.cuda.memory_allocated() // 1024 // 1024
        florence_available = True
        log.info(f'[OCR] Florence-2-large-ft loaded ({ms}ms, VRAM: {vram}MB)')
    except Exception as e:
        import traceback
        log.warning(f'[OCR] Florence-2 not available: {e}')
        log.warning(f'[OCR] {traceback.format_exc()}')


def florence_ocr(pil_image):
    """Run Florence-2 OCR on a PIL image crop. Returns cleaned text string."""
    import torch
    if not florence_available or _florence_model is None:
        return ''
    try:
        with _florence_lock:
            inputs = _florence_processor(
                text='<OCR>', images=pil_image, return_tensors='pt'
            ).to('cuda:0', torch.float16)
            with torch.no_grad():
                gen = _florence_model.generate(
                    input_ids=inputs['input_ids'],
                    pixel_values=inputs['pixel_values'],
                    max_new_tokens=FLORENCE_MAX_TOKENS,
                    num_beams=FLORENCE_NUM_BEAMS,
                )
            text = _florence_processor.batch_decode(gen, skip_special_tokens=False)[0]
            parsed = _florence_processor.post_process_generation(
                text, task='<OCR>',
                image_size=(pil_image.width, pil_image.height),
            )
            raw = parsed.get('<OCR>', '')
        # Join lines and fix cross-line word splits
        return _florence_join_lines(raw)
    except Exception as e:
        log.warning(f'[OCR] Florence-2 error: {e}')
        return ''


def _florence_join_lines(raw_text):
    """Join Florence-2 OCR lines, fix cross-line word splits, and split merged words.
    Florence-2 often merges words within lines: "CAMETO" "HISNUMBERS" "BYASSISTING".
    We split those using wordninja + SymSpell validation."""
    if not raw_text or not raw_text.strip():
        return ''
    lines = [l.strip() for l in raw_text.strip().split('\n') if l.strip()]
    if not lines:
        return ''

    # Step 1: Join cross-line word splits (e.g., last word of line + first word of next)
    if _symspell:
        i = 0
        while i < len(lines) - 1:
            words = lines[i].split()
            next_words = lines[i + 1].split() if lines[i + 1].strip() else []
            if words and next_words:
                last_w = words[-1]
                first_w = next_words[0]
                last_real = _symspell.lookup(last_w.lower(), max_edit_distance=0, verbosity=0)
                first_real = _symspell.lookup(first_w.lower(), max_edit_distance=0, verbosity=0)
                if not (last_real and first_real):
                    joined = last_w + first_w
                    lookup = _symspell.lookup(joined.lower(), max_edit_distance=1, verbosity=0)
                    if lookup and lookup[0].distance <= 1:
                        words[-1] = lookup[0].term.upper()
                        next_words.pop(0)
                        lines[i] = ' '.join(words)
                        lines[i + 1] = ' '.join(next_words) if next_words else ''
            i += 1

    joined_text = ' '.join(l for l in lines if l.strip())

    # Step 2: Split merged words within tokens using wordninja
    # Florence-2 returns "CAMETO" "HISNUMBERS" "BYASSISTING" — all on one line
    if not _have_wordninja:
        return joined_text

    import wordninja
    tokens = joined_text.split()
    result = []
    for token in tokens:
        # Separate punctuation
        lead = ''
        trail = ''
        core = token
        while core and not core[0].isalnum():
            lead += core[0]
            core = core[1:]
        while core and not core[-1].isalnum():
            trail = core[-1] + trail
            core = core[:-1]

        if not core or len(core) <= 3 or not core.isalpha():
            result.append(token)
            continue

        # Skip if it's already a known word
        if _symspell:
            exact = _symspell.lookup(core.lower(), max_edit_distance=0, verbosity=0)
            if exact:
                result.append(token)
                continue

        # Try wordninja split
        parts = wordninja.split(core.lower())
        if len(parts) <= 1:
            result.append(token)
            continue

        # Validate: all parts should be real words (>= 2 chars each)
        # and the split should actually make sense
        all_valid = True
        for p in parts:
            if len(p) < 2:
                all_valid = False
                break
            if _symspell:
                check = _symspell.lookup(p, max_edit_distance=0, verbosity=0)
                if not check:
                    # Allow distance-1 for longer words
                    if len(p) >= 4:
                        check = _symspell.lookup(p, max_edit_distance=1, verbosity=0)
                        if not check or check[0].distance > 1:
                            all_valid = False
                            break
                    else:
                        all_valid = False
                        break

        if all_valid and len(parts) >= 2:
            # Use the split — keep uppercase to match manga style
            split_text = ' '.join(p.upper() for p in parts)
            result.append(lead + split_text + trail)
        else:
            result.append(token)

    return ' '.join(result)

# ─── Tesseract OCR (fallback) ────────────────────────────────────────────────

def load_tesseract():
    global tesseract_available
    try:
        ver = pytesseract.get_tesseract_version()
        log.info(f'Tesseract OCR loaded as fallback (version {ver})')
        tesseract_available = True
    except Exception as e:
        log.warning(f'Tesseract not found: {e}')
        tesseract_available = False

# ─── MangaCNN text classifier (dialogue vs junk) ────────────────────────────

def load_manga_cnn():
    global manga_cnn_session
    import onnxruntime as ort
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'manga_cnn.onnx')
    if os.path.exists(model_path):
        from config import ONNX_THREAD_CAP
        opts = ort.SessionOptions()
        if ONNX_THREAD_CAP > 0:
            opts.inter_op_num_threads = ONNX_THREAD_CAP
            opts.intra_op_num_threads = ONNX_THREAD_CAP
        manga_cnn_session = ort.InferenceSession(model_path, sess_options=opts)
        log.info(f'MangaCNN classifier loaded: {model_path}')
    else:
        log.info('MangaCNN classifier not found (manga_cnn.onnx) — skipping')


def classify_crop_manga_cnn(crop_bgr):
    """Classify a crop as dialogue (True) or junk (False) using MangaCNN.
    Returns (is_dialogue, confidence)."""
    if manga_cnn_session is None:
        return True, 0.5  # no model = pass everything through
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    resized = cv2.resize(gray, (128, 128)).astype(np.float32) / 255.0
    blob = ((resized - 0.5) / 0.5)[np.newaxis, np.newaxis, ...].astype(np.float32)
    out = manga_cnn_session.run(None, {'input': blob})[0]
    score = float(1.0 / (1.0 + np.exp(-out[0][0])))
    is_dialogue = score > 0.5
    conf = score if is_dialogue else 1.0 - score
    return is_dialogue, conf

# ─── Local recognizer (optional, for hybrid mode) ───────────────────────────

def load_local_recognizer():
    global local_recognizer
    if not USE_LOCAL_RECOGNIZER:
        return
    from paddleocr import PaddleOCR

    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'trained_fonts')
    has_model = os.path.isdir(model_dir) and any(
        f.endswith(('.pdparams', '.pdmodel', '.pdiparams'))
        for f in os.listdir(model_dir)
    )
    if not has_model:
        log.warning(f'No trained font model found in {model_dir}. Local recognizer disabled.')
        return

    local_recognizer = PaddleOCR(
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        text_recognition_model_dir=model_dir,
    )
    log.info(f'Local font recognizer loaded from {model_dir}')


def local_recognize_crops(crops_imgs):
    """Run local PaddleOCR recognizer on crop images. Returns list of texts."""
    if not local_recognizer:
        return [''] * len(crops_imgs)
    texts = []
    for img in crops_imgs:
        if img is None or img.size == 0:
            texts.append('')
            continue
        results = local_recognizer.predict(img)
        if results:
            page_text = []
            for r in results:
                res = r.json.get('res', {})
                for t in res.get('rec_texts', []):
                    if t.strip():
                        page_text.append(t.strip())
            texts.append(' '.join(page_text))
        else:
            texts.append('')
    return texts

# ─── Image helpers ───────────────────────────────────────────────────────────

def decode_base64_image(data_url):
    """Decode a base64 data URL to a cv2 BGR image."""
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ─── OCR crop runners ───────────────────────────────────────────────────────

def manga_ocr_crops(crops_imgs):
    """Run manga-ocr on a list of OpenCV images. Returns list of texts."""
    if manga_ocr_engine is None:
        return None  # Signal to use fallback

    texts = []
    for img in crops_imgs:
        if img is None or img.size == 0:
            texts.append('')
            continue
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            text = manga_ocr_engine(pil_img)
            texts.append(text.strip())
        except Exception as e:
            log.warning(f'manga-ocr error: {e}')
            texts.append('')
    return texts


def tesseract_ocr_crops(crops_imgs):
    """Run Tesseract OCR on a list of OpenCV images. Returns list of texts."""
    if not tesseract_available:
        log.warning('Tesseract not available. Returning empty texts.')
        return [''] * len(crops_imgs)

    texts = []
    for img in crops_imgs:
        if img is None or img.size == 0:
            texts.append('')
            continue
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_img, config=TESSERACT_CONFIG)
            texts.append(text.strip())
        except Exception as e:
            log.warning(f'Tesseract error: {e}')
            texts.append('')
    return texts


def ocr_crops(crops_imgs):
    """Run OCR on crops. Uses manga-ocr if available, else Tesseract."""
    result = manga_ocr_crops(crops_imgs)
    if result is not None:
        return result
    return tesseract_ocr_crops(crops_imgs)

# ─── AI Super-Resolution for OCR crops ───────────────────────────────────────

def load_super_res():
    """Load ESPCN x2 super-resolution model for OCR crop enhancement."""
    global _sr_model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'ESPCN_x2.pb')
    if not os.path.isfile(model_path):
        log.info('Super-resolution model not found, using bicubic fallback')
        return
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel('espcn', 2)
        # Warm up
        test = np.ones((50, 100, 3), dtype=np.uint8) * 255
        sr.upsample(test)
        _sr_model = sr
        log.info(f'AI Super-Resolution loaded: ESPCN x2 ({os.path.getsize(model_path)/1024:.0f}KB)')
    except Exception as e:
        log.warning(f'Failed to load super-resolution: {e}')


def ai_upscale_2x(img):
    """Upscale image 2x using AI super-resolution. Falls back to bicubic."""
    if _sr_model is None or img is None or img.size == 0:
        return cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    try:
        return _sr_model.upsample(img)
    except Exception:
        return cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
