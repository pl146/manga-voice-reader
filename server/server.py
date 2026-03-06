"""
Manga Voice Reader — Local Detection + Google Vision OCR Server
Runs on http://127.0.0.1:5055
Endpoint: POST /process
"""

import base64
import io
import json
import logging
import os
import re
import time
from math import ceil

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.cloud import vision
from PIL import Image

# ─── Configuration ───────────────────────────────────────────────────────────

PORT = 5055
MAX_DETECT_WIDTH = 1280
MAX_BOXES = 60
IOU_MERGE_THRESHOLD = 0.45
PROXIMITY_MERGE_RATIO = 2.0  # merge boxes within 2x avg height horizontally
MIN_BOX_AREA = 400
MIN_BOX_SIDE = 12
MAX_ASPECT_RATIO = 15.0
MIN_DET_SCORE = 0.3
LOW_BOX_THRESHOLD = 5  # if fewer boxes, run inverted pass
CROP_PAD = 8  # padding around each box crop for Vision OCR
VISION_BATCH_SIZE = 16  # Google Vision max per batch call
ROW_Y_THRESHOLD_RATIO = 0.6  # fraction of avg height for row grouping

# Hybrid recognition: optional local recognizer trained on manga fonts
USE_LOCAL_RECOGNIZER = os.environ.get('USE_LOCAL_RECOGNIZER', '').lower() in ('true', '1', 'yes')
COLLECT_TRAINING_DATA = os.environ.get('COLLECT_TRAINING_DATA', '').lower() in ('true', '1', 'yes')

logging.basicConfig(level=logging.INFO, format='[MVR] %(message)s')
log = logging.getLogger('mvr')

# ─── App ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ─── Load PaddleOCR (detection only) ────────────────────────────────────────

ocr_engine = None

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
        'text_det_thresh': 0.25,
        'text_det_box_thresh': 0.4,
        'text_det_unclip_ratio': 1.8,
        'text_det_limit_side_len': MAX_DETECT_WIDTH,
    }
    if has_model:
        kwargs['text_detection_model_dir'] = model_dir
        log.info(f'Loading custom detector from {model_dir}')
    else:
        kwargs['text_detection_model_name'] = 'PP-OCRv4_mobile_det'
        log.info('Using PP-OCRv4 mobile detector (detection only, Vision for OCR)')

    ocr_engine = PaddleOCR(**kwargs)
    log.info('Detector loaded.')

# ─── Load Google Vision client ──────────────────────────────────────────────

vision_client = None

def load_vision():
    global vision_client
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    if not creds_path or not os.path.isfile(creds_path):
        log.error(
            'GOOGLE_APPLICATION_CREDENTIALS not set or file not found. '
            'Google Vision OCR is REQUIRED. Set it to your service account JSON path.'
        )
        raise RuntimeError('Google Vision credentials not configured')
    vision_client = vision.ImageAnnotatorClient()
    log.info('Google Vision client loaded.')

# ─── Local recognizer (optional, for hybrid mode) ─────────────────────────

local_recognizer = None

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

# ─── Manga corrections dictionary ─────────────────────────────────────────

manga_corrections = None

def load_manga_corrections():
    global manga_corrections
    corrections_path = os.path.join(os.path.dirname(__file__), 'manga_corrections.json')
    if os.path.isfile(corrections_path):
        with open(corrections_path) as f:
            manga_corrections = json.load(f)
        word_count = len(manga_corrections.get('word_corrections', {}))
        pattern_count = len(manga_corrections.get('pattern_corrections', []))
        log.info(f'Manga corrections loaded: {word_count} words, {pattern_count} patterns')
    else:
        log.info('No manga_corrections.json found, skipping corrections.')
        manga_corrections = {'word_corrections': {}, 'pattern_corrections': [], 'common_sfx': {}}


def apply_manga_corrections(text):
    """Apply manga-specific OCR corrections from the dictionary."""
    if not manga_corrections or not text:
        return text

    # Word-level corrections (case-insensitive)
    word_map = manga_corrections.get('word_corrections', {})
    if word_map:
        words = text.split()
        corrected = []
        for word in words:
            # Strip punctuation for lookup, preserve it
            stripped = word.strip('.,!?;:\'"()[]{}')
            prefix = word[:len(word) - len(word.lstrip('.,!?;:\'"()[]{}'))]
            suffix = word[len(stripped) + len(prefix):]
            lookup = stripped.lower()
            if lookup in word_map:
                replacement = word_map[lookup]
                # Match original case style
                if stripped.isupper():
                    replacement = replacement.upper()
                elif stripped[0].isupper() if stripped else False:
                    replacement = replacement.capitalize()
                corrected.append(prefix + replacement + suffix)
            else:
                corrected.append(word)
        text = ' '.join(corrected)

    # Pattern corrections (regex)
    for rule in manga_corrections.get('pattern_corrections', []):
        pattern = rule.get('pattern', '')
        replacement = rule.get('replacement', '')
        if pattern:
            text = re.sub(pattern, replacement, text)

    # Sound effects normalization for TTS
    sfx_map = manga_corrections.get('common_sfx', {})
    if sfx_map:
        words = text.split()
        corrected = []
        for word in words:
            stripped = word.strip('.,!?;:')
            suffix = word[len(stripped):]
            if stripped.upper() in sfx_map:
                corrected.append(sfx_map[stripped.upper()] + suffix)
            else:
                corrected.append(word)
        text = ' '.join(corrected)

    return text

# ─── Image helpers ──────────────────────────────────────────────────────────

def decode_base64_image(data_url):
    """Decode a base64 data URL to a cv2 BGR image."""
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def cv2_to_png_bytes(img):
    """Encode cv2 image to PNG bytes."""
    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()

# ─── Detection ──────────────────────────────────────────────────────────────

def run_detection(img):
    """Run PaddleOCR detection only. Returns list of {x,y,w,h,score}."""
    results = ocr_engine.predict(img)
    boxes = []
    if not results:
        return boxes

    for r in results:
        res = r.json.get('res', {})
        dt_polys = res.get('dt_polys', [])
        det_scores = res.get('rec_scores', [])

        for i, poly in enumerate(dt_polys):
            points = np.array(poly)
            if points.shape[0] < 4:
                continue
            score = det_scores[i] if i < len(det_scores) else 1.0

            x_min = int(np.min(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            x_max = int(np.max(points[:, 0]))
            y_max = int(np.max(points[:, 1]))
            w = x_max - x_min
            h = y_max - y_min
            if w > 0 and h > 0:
                boxes.append({'x': x_min, 'y': y_min, 'w': w, 'h': h,
                              'score': score})
    return boxes

def detect_with_inversion_fallback(img):
    """Detect text boxes. If too few found, also detect on inverted image and merge."""
    t0 = time.time()
    boxes = run_detection(img)
    normal_count = len(boxes)
    inv_count = 0

    if len(boxes) < LOW_BOX_THRESHOLD:
        inverted = cv2.bitwise_not(img)
        inv_boxes = run_detection(inverted)
        inv_count = len(inv_boxes)
        boxes = merge_box_lists(boxes, inv_boxes)

    det_ms = int((time.time() - t0) * 1000)
    log.info(f'Detection: {normal_count} normal + {inv_count} inverted = {len(boxes)} merged ({det_ms}ms)')
    return boxes, det_ms, inv_count > 0

# ─── Box merging and filtering ──────────────────────────────────────────────

def box_iou(a, b):
    """Compute IoU between two boxes {x,y,w,h}."""
    ax1, ay1, ax2, ay2 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
    bx1, by1, bx2, by2 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / union if union > 0 else 0.0

def merge_two_boxes(a, b):
    """Merge two boxes into their bounding union."""
    x1 = min(a['x'], b['x'])
    y1 = min(a['y'], b['y'])
    x2 = max(a['x'] + a['w'], b['x'] + b['w'])
    y2 = max(a['y'] + a['h'], b['y'] + b['h'])
    return {
        'x': x1, 'y': y1,
        'w': x2 - x1, 'h': y2 - y1,
        'score': max(a.get('score', 1), b.get('score', 1)),
    }

def merge_box_lists(boxes1, boxes2):
    """Combine two box lists, merging duplicates by IoU."""
    merged = list(boxes1)
    for b2 in boxes2:
        is_dup = False
        for i, b1 in enumerate(merged):
            if box_iou(b1, b2) > 0.3:
                merged[i] = merge_two_boxes(b1, b2)
                is_dup = True
                break
        if not is_dup:
            merged.append(b2)
    return merged

def merge_overlapping(boxes, threshold=IOU_MERGE_THRESHOLD):
    """Iteratively merge overlapping boxes until stable."""
    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                if box_iou(current, boxes[j]) >= threshold:
                    current = merge_two_boxes(current, boxes[j])
                    used.add(j)
                    changed = True
            new_boxes.append(current)
            used.add(i)
        boxes = new_boxes
    return boxes

def merge_same_line(boxes):
    """Merge boxes that are on the same text line (close vertically, near horizontally)."""
    if len(boxes) < 2:
        return boxes
    avg_h = np.mean([b['h'] for b in boxes])
    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                b = boxes[j]
                # Vertical centers within 0.5 * avg_h
                cy_a = current['y'] + current['h'] / 2
                cy_b = b['y'] + b['h'] / 2
                if abs(cy_a - cy_b) > avg_h * 0.5:
                    continue
                # Horizontal gap within PROXIMITY_MERGE_RATIO * avg_h
                gap_x = max(0,
                    max(b['x'] - (current['x'] + current['w']),
                        current['x'] - (b['x'] + b['w'])))
                if gap_x < avg_h * PROXIMITY_MERGE_RATIO:
                    current = merge_two_boxes(current, b)
                    used.add(j)
                    changed = True
            new_boxes.append(current)
            used.add(i)
        boxes = new_boxes
    return boxes

def merge_vertical_stack(boxes):
    """Merge boxes stacked vertically (typical manga speech bubble layout).
    Only merges if vertical gap is small and boxes are similar width."""
    if len(boxes) < 2:
        return boxes
    avg_h = np.mean([b['h'] for b in boxes])
    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                b = boxes[j]
                # Check horizontal overlap: boxes must overlap in X range
                cx1, cx2 = current['x'], current['x'] + current['w']
                bx1, bx2 = b['x'], b['x'] + b['w']
                overlap_x = min(cx2, bx2) - max(cx1, bx1)
                min_w = min(current['w'], b['w'])
                if min_w <= 0 or overlap_x < min_w * 0.3:
                    continue
                # Vertical gap must be small: within 0.8x the SMALLER box's height
                small_h = min(current['h'], b['h'])
                gap_y = max(0,
                    max(b['y'] - (current['y'] + current['h']),
                        current['y'] - (b['y'] + b['h'])))
                if gap_y > small_h * 0.8:
                    continue
                # Width ratio check: don't merge a tiny box into a huge one
                max_w = max(current['w'], b['w'])
                if min_w < max_w * 0.2:
                    continue
                current = merge_two_boxes(current, b)
                used.add(j)
                changed = True
            new_boxes.append(current)
            used.add(i)
        boxes = new_boxes
    return boxes

def filter_boxes(boxes):
    """Remove noise boxes by size, aspect ratio, and score."""
    filtered = []
    for b in boxes:
        area = b['w'] * b['h']
        if area < MIN_BOX_AREA:
            continue
        if b['w'] < MIN_BOX_SIDE or b['h'] < MIN_BOX_SIDE:
            continue
        aspect = max(b['w'], b['h']) / max(1, min(b['w'], b['h']))
        if aspect > MAX_ASPECT_RATIO:
            continue
        if b.get('score', 1) < MIN_DET_SCORE:
            continue
        filtered.append(b)
    return filtered

def scale_boxes(boxes, factor):
    """Scale all box coordinates by factor."""
    for b in boxes:
        b['x'] = int(b['x'] * factor)
        b['y'] = int(b['y'] * factor)
        b['w'] = int(b['w'] * factor)
        b['h'] = int(b['h'] * factor)
    return boxes

def cap_boxes(boxes, max_count=MAX_BOXES):
    """Keep top boxes by score * area."""
    if len(boxes) <= max_count:
        return boxes
    boxes.sort(key=lambda b: b.get('score', 1) * b['w'] * b['h'], reverse=True)
    return boxes[:max_count]

# ─── Google Vision batch OCR ────────────────────────────────────────────────

def batch_vision_ocr(crops_bytes):
    """Send all crop images to Google Vision in batch. Returns list of texts."""
    if not vision_client:
        log.warning('Google Vision client not available. Returning empty texts.')
        return [''] * len(crops_bytes)

    all_texts = []
    for i in range(0, len(crops_bytes), VISION_BATCH_SIZE):
        batch = crops_bytes[i:i + VISION_BATCH_SIZE]
        requests = []
        for img_bytes in batch:
            requests.append(
                vision.AnnotateImageRequest(
                    image=vision.Image(content=img_bytes),
                    features=[vision.Feature(
                        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
                    )],
                )
            )
        try:
            response = vision_client.batch_annotate_images(requests=requests)
            for resp in response.responses:
                if resp.error.message:
                    log.warning(f'Vision error: {resp.error.message}')
                    all_texts.append('')
                elif resp.full_text_annotation and resp.full_text_annotation.text:
                    all_texts.append(resp.full_text_annotation.text)
                else:
                    all_texts.append('')
        except Exception as e:
            log.error(f'Vision batch call failed: {e}')
            all_texts.extend([''] * len(batch))

    return all_texts

# ─── Text cleanup ───────────────────────────────────────────────────────────

def join_spaced_letters(text):
    """Join single spaced letters: 'H E L L O' -> 'HELLO'."""
    # Match sequences of single letters separated by spaces
    def replace_spaced(m):
        return m.group(0).replace(' ', '')
    # Pattern: single letter, space, single letter, (repeat)
    text = re.sub(r'\b([A-Za-z]) (?=[A-Za-z](?:\b| ))', replace_spaced, text)
    # Catch remaining pairs at end of string
    text = re.sub(r'\b([A-Za-z]) ([A-Za-z])\b', r'\1\2', text)
    return text

def join_dotted_letters(text):
    """Join dotted letters: 'H.E.L.L.O' -> 'HELLO'."""
    return re.sub(r'\b([A-Za-z])\.([A-Za-z])\.', lambda m: m.group(1) + m.group(2), text)

def fix_slash_as_l(text):
    """Fix OCR slash errors: 'ca/ed' -> 'called', 'contro/' -> 'control'.
    Never touch numeric fractions like 1/2."""
    # Slash between letters -> 'l'
    text = re.sub(r'([a-zA-Z])/([a-zA-Z])', r'\1l\2', text)
    # Slash at end of word after letters -> 'l'
    text = re.sub(r'([a-zA-Z])/$', r'\1l', text)
    # Double-check: "cal" should be "call" in some contexts
    # "caled" -> "called", "controled" -> "controlled"
    text = re.sub(r'([a-zA-Z])led\b', r'\1lled', text)
    return text

def fix_mixed_alphanumeric(text):
    """Fix digits inside words: G0T->GOT, 0KAY->OKAY. Preserve standalone numbers."""
    digit_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}

    def fix_word(m):
        word = m.group(0)
        # All digits = real number, keep as-is
        if word.isdigit():
            return word
        # Ordinals like 1ST, 2ND, 3RD
        if re.match(r'^\d+(ST|ND|RD|TH)$', word, re.IGNORECASE):
            return word
        # Mixed: replace digits surrounded by letters
        result = []
        for i, ch in enumerate(word):
            if ch.isdigit():
                before = word[i - 1].isalpha() if i > 0 else False
                after = word[i + 1].isalpha() if i < len(word) - 1 else False
                if before or after:
                    result.append(digit_map.get(ch, ch))
                else:
                    result.append(ch)
            else:
                result.append(ch)
        return ''.join(result)

    return re.sub(r'\b\S+\b', fix_word, text)

def remove_duplicate_words(text):
    """Remove immediately duplicated words: 'the the cat' -> 'the cat'."""
    return re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)

def remove_duplicate_sentences(text):
    """Remove duplicate sentences/lines."""
    lines = text.split('\n')
    seen = set()
    unique = []
    for line in lines:
        normalized = line.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(line.strip())
        elif not normalized:
            unique.append('')
    return '\n'.join(unique)

def cleanup_text(text):
    """Full text cleanup pipeline."""
    if not text:
        return ''
    # Normalize whitespace
    text = text.strip()
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Join spaced/dotted letters
    text = join_spaced_letters(text)
    text = join_dotted_letters(text)

    # Fix OCR errors
    text = fix_slash_as_l(text)
    text = fix_mixed_alphanumeric(text)

    # Remove duplicates
    text = remove_duplicate_words(text)
    text = remove_duplicate_sentences(text)

    # Final trim
    text = text.strip()
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace remaining newlines with spaces for TTS (single bubble = one utterance)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ─── Speech formatting ─────────────────────────────────────────────────────

# Words that should stay uppercase (proper nouns, acronyms, etc.)
_KEEP_UPPER = {
    'I', 'OK', 'TV', 'US', 'UK', 'DNA', 'FBI', 'CIA', 'USA', 'CEO',
    'AI', 'ID', 'HP', 'MP', 'XP', 'KO', 'OP', 'NPC', 'RPG',
}

# Words that should stay lowercase even at sentence start in dialogue
_LOWERCASE_WORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
    'in', 'on', 'at', 'to', 'of', 'by', 'up', 'as', 'if', 'is', 'it',
}


def _to_sentence_case(text):
    """Convert ALL CAPS text to natural sentence case.
    Preserves acronyms, proper 'I', and numbers."""
    if not text:
        return text

    # Only convert if the text is mostly uppercase
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return text
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    if upper_ratio < 0.6:
        return text  # Already mixed case, don't touch

    words = text.split()
    result = []
    sentence_start = True

    for word in words:
        # Strip trailing punctuation for processing
        trail = ''
        core = word
        while core and core[-1] in '.,!?;:…':
            trail = core[-1] + trail
            core = core[:-1]

        # Preserve numbers and number-letter combos (1ST, 2ND, 300)
        if core and (core.isdigit() or re.match(r'^\d+\w*$', core)):
            result.append(word)
            if trail and trail[-1] in '.!?':
                sentence_start = True
            else:
                sentence_start = False
            continue

        # Preserve known acronyms/abbreviations
        if core.upper() in _KEEP_UPPER:
            result.append(core.upper() + trail)
            sentence_start = trail and trail[-1] in '.!?'
            continue

        # Convert to lowercase
        lower = core.lower()

        # Capitalize first word of sentence
        if sentence_start and lower:
            lower = lower[0].upper() + lower[1:]

        # Always capitalize standalone "I"
        if lower == 'i':
            lower = 'I'
        # Fix "i'm", "i'll", "i've", "i'd"
        if lower.startswith("i'") or lower.startswith("i'"):
            lower = 'I' + lower[1:]

        result.append(lower + trail)

        if trail and trail[-1] in '.!?':
            sentence_start = True
        else:
            sentence_start = False

    return ' '.join(result)


def format_text_for_speech(text):
    """Convert OCR output into natural spoken dialogue.
    Runs after cleanup_text() and apply_manga_corrections().
    Does not modify bubble coordinates."""
    if not text:
        return ''

    # 1. Join broken lines into continuous sentences
    text = re.sub(r'\n+', ' ', text)

    # 2. Join spaced single letters: "H E L L O" -> "HELLO"
    #    Only merge sequences of 3+ single letters to avoid merging real words like "I I" or "I A"
    def merge_spaced(m):
        return m.group(0).replace(' ', '')
    text = re.sub(r'(?<!\w)([A-Za-z]) ([A-Za-z])(?: ([A-Za-z]))+(?!\w)', merge_spaced, text)

    # 3. Fix slash-as-L errors (skip numeric fractions like 1/2)
    text = re.sub(r'([a-zA-Z])/([a-zA-Z])', r'\1l\2', text)
    text = re.sub(r'([a-zA-Z])/$', r'\1l', text)
    text = re.sub(r'([a-zA-Z])led\b', r'\1lled', text)

    # 4. Remove noise tokens (pure symbols or repeated punctuation)
    text = re.sub(r'(?<!\w)[=|_~#*]{2,}(?!\w)', '', text)
    text = re.sub(r'(?:^|\s)[^\w\s]{3,}(?:\s|$)', ' ', text)

    # 5. Normalize repeated punctuation: "!!!" -> "!", "???" -> "?", "..." stays
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'[?!]{3,}', '!', text)

    # 6. Convert ALL CAPS to natural sentence case
    text = _to_sentence_case(text)

    # 7. Remove duplicated adjacent words (after case conversion so "I I" works)
    text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)

    # 8. Numbers are preserved (regex above skips digits)

    # 9. Ensure sentence-ending punctuation for natural TTS pause
    text = text.strip()
    if text and text[-1] not in '.!?…':
        text += '.'

    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ─── Reading order ──────────────────────────────────────────────────────────

def sort_reading_order(bubbles):
    """Sort bubbles top-to-bottom, left-to-right within rows."""
    if not bubbles:
        return bubbles

    # Group into rows by vertical center proximity
    avg_h = np.mean([b['height'] for b in bubbles]) if bubbles else 30
    threshold = avg_h * ROW_Y_THRESHOLD_RATIO

    sorted_by_y = sorted(bubbles, key=lambda b: b['top'])
    rows = []
    current_row = [sorted_by_y[0]]

    for b in sorted_by_y[1:]:
        prev_center = current_row[-1]['top'] + current_row[-1]['height'] / 2
        curr_center = b['top'] + b['height'] / 2
        if abs(curr_center - prev_center) <= threshold:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
    rows.append(current_row)

    # Sort within each row left-to-right
    result = []
    for row in rows:
        row.sort(key=lambda b: b['left'])
        result.extend(row)

    return result

# ─── Main endpoint ──────────────────────────────────────────────────────────

@app.route('/process', methods=['POST'])
def process():
    t_start = time.time()
    data = request.json

    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image field'}), 400

    # 1. Decode screenshot
    t0 = time.time()
    img = decode_base64_image(data['image'])
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400
    decode_ms = int((time.time() - t0) * 1000)

    dpr = data.get('dpr', 2)
    img_h, img_w = img.shape[:2]
    log.info(f'Screenshot: {img_w}x{img_h}, dpr={dpr}')

    # 2. Crop to manga area
    crop = data.get('cropRect')
    if crop and crop.get('width') and crop.get('height'):
        cx = int(crop['left'] * dpr)
        cy = int(crop['top'] * dpr)
        cw = int(crop['width'] * dpr)
        ch = int(crop['height'] * dpr)
        # Clamp to image bounds
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        cw = min(cw, img_w - cx)
        ch = min(ch, img_h - cy)
        cropped = img[cy:cy + ch, cx:cx + cw]
        actual_crop_left = cx
        actual_crop_top = cy
    else:
        cropped = img
        actual_crop_left = 0
        actual_crop_top = 0
        cx, cy = 0, 0

    crop_h, crop_w = cropped.shape[:2]
    log.info(f'Crop: {crop_w}x{crop_h} at ({cx},{cy})')

    # 3. Downscale for detection if needed
    det_scale = 1.0
    if crop_w > MAX_DETECT_WIDTH:
        det_scale = MAX_DETECT_WIDTH / crop_w
        det_img = cv2.resize(cropped, (MAX_DETECT_WIDTH, int(crop_h * det_scale)))
    else:
        det_img = cropped

    # 4. Run detection with inversion fallback
    boxes, det_ms, used_inversion = detect_with_inversion_fallback(det_img)

    # 5. Scale boxes back to original crop coordinates
    if det_scale != 1.0:
        boxes = scale_boxes(boxes, 1.0 / det_scale)

    # 6. Merge, filter, cap
    t0 = time.time()
    log.info(f'Raw boxes ({len(boxes)}):')
    for i, b in enumerate(boxes):
        log.info(f'  raw[{i}] {b["x"]},{b["y"]} {b["w"]}x{b["h"]} score={b.get("score",0):.2f}')
    boxes = merge_overlapping(boxes, IOU_MERGE_THRESHOLD)
    log.info(f'After overlap merge: {len(boxes)}')
    boxes = merge_same_line(boxes)
    log.info(f'After same-line merge: {len(boxes)}')
    boxes = merge_vertical_stack(boxes)
    log.info(f'After vertical merge: {len(boxes)}')
    for i, b in enumerate(boxes):
        log.info(f'  merged[{i}] {b["x"]},{b["y"]} {b["w"]}x{b["h"]}')
    boxes = filter_boxes(boxes)
    log.info(f'After filter: {len(boxes)}')
    boxes = cap_boxes(boxes, MAX_BOXES)
    merge_ms = int((time.time() - t0) * 1000)
    log.info(f'Final: {len(boxes)} boxes ({merge_ms}ms)')

    if len(boxes) == 0:
        # Save debug image even with no boxes
        debug_path = os.path.join(os.path.dirname(__file__), 'debug_last.jpg')
        cv2.imwrite(debug_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jsonify({
            'bubbles': [],
            'actualCropLeft': actual_crop_left / dpr,
            'actualCropTop': actual_crop_top / dpr,
            'timing': {
                'total_ms': int((time.time() - t_start) * 1000),
                'decode_ms': decode_ms,
                'detect_ms': det_ms,
                'merge_ms': merge_ms,
                'ocr_ms': 0,
                'boxes_detected': 0,
                'bubbles_returned': 0,
                'used_inversion': used_inversion,
            }
        })

    # 7. Crop each detected box
    t0 = time.time()
    crops_bytes = []
    crops_imgs = []
    for b in boxes:
        x1 = max(0, b['x'] - CROP_PAD)
        y1 = max(0, b['y'] - CROP_PAD)
        x2 = min(crop_w, b['x'] + b['w'] + CROP_PAD)
        y2 = min(crop_h, b['y'] + b['h'] + CROP_PAD)
        crop_region = cropped[y1:y2, x1:x2]
        if crop_region.size == 0:
            crops_bytes.append(b'')
            crops_imgs.append(None)
            continue
        crops_bytes.append(cv2_to_png_bytes(crop_region))
        crops_imgs.append(crop_region)

    # 7a. Google Vision OCR (primary)
    log.info(f'Cropped {len(crops_bytes)} regions for Vision OCR')
    ocr_texts = batch_vision_ocr(crops_bytes)
    ocr_ms = int((time.time() - t0) * 1000)
    log.info(f'Vision OCR: {ocr_ms}ms')

    # 7b. Local recognizer fallback (if enabled and Vision returned empty)
    if local_recognizer:
        empty_indices = [i for i, t in enumerate(ocr_texts) if not t.strip()]
        if empty_indices:
            empty_imgs = [crops_imgs[i] for i in empty_indices]
            local_texts = local_recognize_crops(empty_imgs)
            for idx, local_text in zip(empty_indices, local_texts):
                if local_text.strip():
                    ocr_texts[idx] = local_text
                    log.info(f'  Box {idx}: Vision empty, local fallback: "{local_text[:50]}"')

    # 7c. Collect training data if enabled
    if COLLECT_TRAINING_DATA:
        from training.collect_dataset import save_crop_pair
        dataset_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'cropped_text')
        for i, crop_img in enumerate(crops_imgs):
            text = ocr_texts[i] if i < len(ocr_texts) else ''
            if crop_img is not None and text.strip():
                save_crop_pair(dataset_dir, crop_img, text.strip(), 'vision')

    # 8. Build bubbles: cleanup -> manga corrections -> speech normalization
    bubbles = []
    for i, b in enumerate(boxes):
        raw_text = ocr_texts[i] if i < len(ocr_texts) else ''
        text = cleanup_text(raw_text)
        text = apply_manga_corrections(text)
        text = format_text_for_speech(text)
        if not text or len(text) < 2:
            continue
        bubbles.append({
            'text': text,
            'conf': round(b.get('score', 1.0), 3),
            'left': round((actual_crop_left + b['x']) / dpr, 1),
            'top': round((actual_crop_top + b['y']) / dpr, 1),
            'width': round(b['w'] / dpr, 1),
            'height': round(b['h'] / dpr, 1),
        })

    # 9. Sort reading order
    bubbles = sort_reading_order(bubbles)

    # Save debug image with boxes and Vision text
    debug_img = cropped.copy()
    for i, b in enumerate(boxes):
        color = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)][i % 5]
        cv2.rectangle(debug_img, (b['x'], b['y']), (b['x'] + b['w'], b['y'] + b['h']), color, 3)
        ocr_label = (ocr_texts[i] if i < len(ocr_texts) else '')[:30].replace('\n', ' ')
        cv2.putText(debug_img, f'{i+1}: {ocr_label}', (b['x'], b['y'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    debug_path = os.path.join(os.path.dirname(__file__), 'debug_last.jpg')
    cv2.imwrite(debug_path, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info(f'Debug image saved: {debug_path}')

    total_ms = int((time.time() - t_start) * 1000)
    log.info(f'Result: {len(bubbles)} bubbles in {total_ms}ms')
    for i, b in enumerate(bubbles):
        log.info(f'  [{i+1}] {b["text"][:60]}')

    return jsonify({
        'bubbles': bubbles,
        'actualCropLeft': round(actual_crop_left / dpr, 1),
        'actualCropTop': round(actual_crop_top / dpr, 1),
        'timing': {
            'total_ms': total_ms,
            'decode_ms': decode_ms,
            'detect_ms': det_ms,
            'merge_ms': merge_ms,
            'ocr_ms': ocr_ms,
            'boxes_detected': len(boxes),
            'bubbles_returned': len(bubbles),
            'used_inversion': used_inversion,
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'detector': ocr_engine is not None, 'vision': vision_client is not None})

@app.route('/debug', methods=['GET'])
def debug():
    """View the last processed screenshot with detected boxes."""
    from flask import send_file
    debug_path = os.path.join(os.path.dirname(__file__), 'debug_last.jpg')
    if not os.path.isfile(debug_path):
        return 'No debug image yet. Click "Read Page" first.', 404
    return send_file(debug_path, mimetype='image/jpeg')

# ─── Startup ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log.info('Loading detector model...')
    load_detector()
    load_vision()
    load_local_recognizer()
    load_manga_corrections()
    if COLLECT_TRAINING_DATA:
        log.info('Training data collection ENABLED')
    log.info(f'Server ready on http://127.0.0.1:{PORT}')
    app.run(host='127.0.0.1', port=PORT, debug=False, threaded=True)
