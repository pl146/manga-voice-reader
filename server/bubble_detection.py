"""Bubble detection, box manipulation, SFX filtering, grouping, reading order,
and debug visualization for the manga voice reader pipeline."""

import cv2
import numpy as np
import math
import re
import time
import os
import logging

log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

from config import (MAX_BOXES, IOU_MERGE_THRESHOLD, PROXIMITY_MERGE_RATIO,
                    MIN_BOX_AREA, MIN_BOX_SIDE, MAX_ASPECT_RATIO, MIN_DET_SCORE,
                    LOW_BOX_THRESHOLD, VERTICAL_MERGE_HEIGHT_FACTOR,
                    VERTICAL_MERGE_WIDTH_FACTOR, DIALOGUE_SCORE_GATE,
                    PRE_SFX_MIN_SIDE, PRE_SFX_MAX_ASPECT, PRE_SFX_MAX_ROTATION,
                    PRE_SFX_MAX_AREA_RATIO, PRE_SFX_MIN_AREA_RATIO)


# ─── Preprocessing for detection ────────────────────────────────────────

def preprocess_for_detection(img):
    """Enhance image for better text detection on manga pages.
    Converts to grayscale, boosts contrast, sharpens edges,
    then converts back to BGR for PaddleOCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen edges to make text boundaries crisp
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # Convert back to BGR (PaddleOCR expects 3-channel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# ─── Detection ──────────────────────────────────────────────────────────────

def run_detection(img, ocr_engine):
    """Run PaddleOCR detection + recognition. Returns list of {x,y,w,h,score,text}."""
    from ocr_engines import _ocr_engine_lock
    with _ocr_engine_lock:
        results = ocr_engine.predict(img)
    boxes = []
    if not results:
        return boxes

    for r in results:
        res = r.json.get('res', {})
        dt_polys = res.get('dt_polys', [])
        rec_scores = res.get('rec_scores', [])
        rec_texts = res.get('rec_texts', [])

        for i, poly in enumerate(dt_polys):
            points = np.array(poly, dtype=np.float32)
            if points.shape[0] < 4:
                continue
            score = rec_scores[i] if i < len(rec_scores) else 1.0
            text = rec_texts[i] if i < len(rec_texts) else ''

            x_min = int(np.min(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            x_max = int(np.max(points[:, 0]))
            y_max = int(np.max(points[:, 1]))
            w = x_max - x_min
            h = y_max - y_min

            # Compute rotation angle from polygon
            angle = 0.0
            if points.shape[0] >= 4:
                # Top edge angle: from top-left to top-right
                dx = float(points[1][0] - points[0][0])
                dy = float(points[1][1] - points[0][1])
                if dx != 0:
                    angle = abs(math.degrees(math.atan2(dy, dx)))
                    if angle > 45:
                        angle = 90 - angle  # normalize to 0-45 range

            if w > 0 and h > 0:
                # Skip very low confidence recognitions (likely Japanese/SFX/artwork)
                if score < 0.3 and text:
                    log.debug(f'  det: skip low-conf box ({score:.2f}) "{text[:30]}"')
                    continue
                boxes.append({'x': x_min, 'y': y_min, 'w': w, 'h': h,
                              'score': score, 'angle': round(angle, 1),
                              'text': text})
    return boxes


def detect_with_fallbacks(img, ocr_engine):
    """Detect text boxes. Runs pass 1 always; extra passes only if pass 1
    finds very few boxes (< LOW_BOX_THRESHOLD), to keep latency low.
    Pass 1: original image
    Pass 2: preprocessed (CLAHE + sharpen) -- fallback
    Pass 3: inverted (white text on dark) -- fallback
    """
    t0 = time.time()

    # Pass 1: original -- always run
    boxes = run_detection(img, ocr_engine)
    normal_count = len(boxes)
    pass1_ms = int((time.time() - t0) * 1000)
    log.info(f'  Pass 1 (original): {normal_count} boxes ({pass1_ms}ms)')

    used_fallback = False
    preproc_count = 0
    inv_count = 0

    if normal_count < LOW_BOX_THRESHOLD:
        log.info(f'  Low box count ({normal_count} < {LOW_BOX_THRESHOLD}), running fallback passes...')
        used_fallback = True

        # Pass 2: preprocessed (CLAHE contrast + sharpen)
        enhanced = preprocess_for_detection(img)
        enh_boxes = run_detection(enhanced, ocr_engine)
        preproc_count = len(enh_boxes)
        boxes = merge_box_lists(boxes, enh_boxes)
        log.info(f'  Pass 2 (enhanced): {preproc_count} boxes -> {len(boxes)} merged')

        # Pass 3: inverted (white text on dark backgrounds)
        inverted = cv2.bitwise_not(img)
        inv_boxes = run_detection(inverted, ocr_engine)
        inv_count = len(inv_boxes)
        boxes = merge_box_lists(boxes, inv_boxes)
        log.info(f'  Pass 3 (inverted): {inv_count} boxes -> {len(boxes)} merged')
    else:
        log.info(f'  Pass 1 sufficient ({normal_count} >= {LOW_BOX_THRESHOLD}), skipping fallback passes')

    det_ms = int((time.time() - t0) * 1000)
    log.info(f'  Detection: {normal_count}+{preproc_count}+{inv_count} = {len(boxes)} merged ({det_ms}ms)')
    return boxes, det_ms, used_fallback


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
        'angle': min(a.get('angle', 0), b.get('angle', 0)),
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
    """Merge boxes that are on the same text line (close vertically, near horizontally).
    Max width cap prevents merging across panel boundaries."""
    if len(boxes) < 2:
        return boxes
    avg_h = np.mean([b['h'] for b in boxes])
    avg_w = np.mean([b['w'] for b in boxes])
    max_merged_w = avg_w * 8  # speech bubbles shouldn't be wider than ~8x avg text width
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
                    # Width cap -- don't merge if result would be too wide
                    merged_x1 = min(current['x'], b['x'])
                    merged_x2 = max(current['x'] + current['w'], b['x'] + b['w'])
                    if (merged_x2 - merged_x1) > max_merged_w:
                        continue
                    current = merge_two_boxes(current, b)
                    used.add(j)
                    changed = True
            new_boxes.append(current)
            used.add(i)
        boxes = new_boxes
    return boxes


def merge_vertical_stack(boxes):
    """Merge text lines that are inside the SAME speech bubble.
    Key idea: lines within a bubble overlap horizontally and are close vertically.
    Lines in DIFFERENT bubbles are separated horizontally or by large vertical gaps.
    A max-height cap prevents merging across panel boundaries."""
    if len(boxes) < 2:
        return boxes

    # Compute average single-line height from original boxes to set caps.
    avg_line_h = np.mean([b['h'] for b in boxes])
    avg_line_w = np.mean([b['w'] for b in boxes])
    max_merged_h = avg_line_h * VERTICAL_MERGE_HEIGHT_FACTOR   # configurable height cap
    max_merged_w = avg_line_w * VERTICAL_MERGE_WIDTH_FACTOR   # configurable width cap

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

                # Vertical gap -- use the SMALLER box height, not the merged one
                small_h = min(current['h'], b['h'])
                c_bot = current['y'] + current['h']
                b_bot = b['y'] + b['h']
                gap = max(0, max(b['y'] - c_bot, current['y'] - b_bot))
                if gap > small_h * 0.8:  # tighter: lines in same bubble are very close
                    continue

                # Horizontal overlap -- at least 50% of smaller box width
                x1_start, x1_end = current['x'], current['x'] + current['w']
                x2_start, x2_end = b['x'], b['x'] + b['w']
                overlap_x = max(0, min(x1_end, x2_end) - max(x1_start, x2_start))
                min_w = min(current['w'], b['w'])
                if min_w <= 0:
                    continue
                if overlap_x < min_w * 0.50:
                    continue

                # Max height cap
                merged_y1 = min(current['y'], b['y'])
                merged_y2 = max(current['y'] + current['h'], b['y'] + b['h'])
                if (merged_y2 - merged_y1) > max_merged_h:
                    continue

                # Max width cap
                merged_x1 = min(current['x'], b['x'])
                merged_x2 = max(current['x'] + current['w'], b['x'] + b['w'])
                if (merged_x2 - merged_x1) > max_merged_w:
                    continue

                current = merge_two_boxes(current, b)
                used.add(j)
                changed = True
            new_boxes.append(current)
            used.add(i)
        boxes = new_boxes
    return boxes


def filter_boxes(boxes):
    """Remove noise boxes by size, aspect ratio, and score.
    Returns (kept, rejected) tuple."""
    filtered = []
    rejected = []
    for i, b in enumerate(boxes):
        area = b['w'] * b['h']
        if area < MIN_BOX_AREA:
            log.info(f'  filter: box {i} dropped (area {area} < {MIN_BOX_AREA})')
            rejected.append(b)
            continue
        if b['w'] < MIN_BOX_SIDE or b['h'] < MIN_BOX_SIDE:
            log.info(f'  filter: box {i} dropped (side {b["w"]}x{b["h"]} < {MIN_BOX_SIDE})')
            rejected.append(b)
            continue
        aspect = max(b['w'], b['h']) / max(1, min(b['w'], b['h']))
        if aspect > MAX_ASPECT_RATIO:
            log.info(f'  filter: box {i} dropped (aspect {aspect:.1f} > {MAX_ASPECT_RATIO})')
            rejected.append(b)
            continue
        if b.get('score', 1) < MIN_DET_SCORE:
            log.info(f'  filter: box {i} dropped (score {b.get("score",0):.2f} < {MIN_DET_SCORE})')
            rejected.append(b)
            continue
        filtered.append(b)
    return filtered, rejected


# ─── Box scaling and capping ────────────────────────────────────────────────

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


# ─── Dialogue vs SFX filtering ──────────────────────────────────────────────

# Common manga sound effect words (lowercase). Single-word OCR matching these = likely SFX.
SFX_WORDS = {
    # Impact / action
    'bam', 'bang', 'bash', 'boom', 'bonk', 'bump', 'burst',
    'clang', 'clank', 'clash', 'click', 'crack', 'crash', 'creak', 'crunch', 'crush',
    'dash', 'dong', 'doom', 'drip', 'drop', 'dun',
    'flash', 'fwip', 'fwish', 'fwoom', 'fwosh',
    'gatan', 'grab', 'grip', 'grind',
    'ka', 'klak',
    'pow', 'punch',
    'rattle', 'rip', 'roar', 'rumble', 'rustle',
    'shatter', 'slam', 'slap', 'slash', 'slice', 'slide', 'smack', 'smash', 'snap',
    'snatch', 'splat', 'splash', 'squish', 'stomp', 'strike', 'swipe', 'swoosh',
    'tap', 'thud', 'thump', 'thwack',
    'vroom',
    'wham', 'whack', 'whip', 'whoosh', 'whump',
    'zap', 'zoom',
    # Visual / energy SFX
    'blaze', 'blaaze', 'blazze', 'bwoo', 'bwooo', 'gleam', 'gleeam',
    'glow', 'shine', 'spark', 'flare', 'beam',
    'fwoosh', 'whomp', 'wham',
    # Vocal / ambient SFX (not dialogue)
    'aaa', 'aaah', 'aaagh', 'ahh', 'ahhh',
    'bwa', 'bwah',
    'cha',
    'fss', 'fssh',
    'gah', 'geh', 'grr', 'grrr',
    'gasp', 'guhk', 'gulp',
    'haa', 'haah', 'hah', 'hmph', 'hss', 'huff',
    'kch', 'kh',
    'ngh',
    'oh', 'ohh', 'ohhh', 'oo', 'ooo',
    'pant',
    'shh', 'shhh', 'sigh',
    'tch', 'tsk',
    'ugh', 'urgh', 'uwaa', 'uwaah', 'uwaaah',
    'whew', 'woah', 'woo', 'wooo',
    'yah', 'yaa', 'yaah', 'yong',
    'zzz',
    'gak', 'gakk', 'gro', 'groo', 'grooo', 'groah', 'groaah',
    'rgh', 'rrgh', 'argh', 'aargh',
    'hyaa', 'hyaah', 'kyaa', 'kyaah',
    'noo', 'nooo', 'noooo',
    'waa', 'waah', 'waaah', 'bwaa', 'bwaah',
    'heh', 'hehe', 'hehehe', 'meh', 'bah', 'pah',
    'hmm', 'hmmm', 'mmm', 'mmmm',
    # Additional manga SFX
    'clack', 'clink', 'clop', 'stoa', 'step', 'stomp',
    'fwoo', 'fwah', 'swooo', 'swoo',
    'don', 'dooon', 'dokkan',
    'sfx', 'plop', 'plip', 'splish',
    # Japanese romanized SFX that OCR sometimes picks up
    'doki', 'gara', 'gata', 'gooo', 'gogo', 'gyaa',
    'paku', 'pata', 'pika',
    'shaa', 'shin',
    'zawa',
}

# Patterns that look like repeated letters / screaming SFX
_SFX_REPEAT_RE = re.compile(r'^(.)\1{2,}$', re.IGNORECASE)  # e.g. AAAA, OOOO
_SFX_REPEAT2_RE = re.compile(r'^(.{1,4})\1{1,}$', re.IGNORECASE)  # e.g. HAHAHA, DOKIDOKI
_ALL_CONSONANTS_RE = re.compile(r'^[^aeiouAEIOU\d\s]{2,}$')  # e.g. GRPH, SHHT


def score_dialogue_box(box, text, image_area):
    """Score a detected box+text as dialogue (positive) or SFX (negative).
    Returns (score, reasons) where reasons is a list of strings."""
    score = 0
    reasons = []
    clean = text.strip()
    clean_lower = clean.lower()

    # --- Geometric heuristics (pre-OCR compatible) ---

    # 1. Aspect ratio: normal dialogue boxes are roughly 0.3-5.0
    aspect = max(box['w'], box['h']) / max(1, min(box['w'], box['h']))
    if aspect <= 5.0:
        score += 2
        reasons.append(f'aspect {aspect:.1f} normal (+2)')
    elif aspect > 8.0:
        score -= 2
        reasons.append(f'aspect {aspect:.1f} extreme (-2)')
    else:
        reasons.append(f'aspect {aspect:.1f} borderline (0)')

    # 2. Size relative to page: very small boxes = likely SFX
    box_area = box['w'] * box['h']
    area_ratio = box_area / max(1, image_area)
    if area_ratio < 0.001:
        score -= 1
        reasons.append(f'tiny area {area_ratio:.4f} (-1)')
    elif area_ratio > 0.005:
        score += 1
        reasons.append(f'decent area {area_ratio:.4f} (+1)')

    # 3. Rotation: SFX are often slanted
    angle = box.get('angle', 0)
    if angle > 15:
        score -= 2
        reasons.append(f'rotated {angle} deg (-2)')
    elif angle > 8:
        score -= 1
        reasons.append(f'slightly rotated {angle} deg (-1)')

    # --- Text-based heuristics (post-OCR) ---

    if not clean:
        score -= 3
        reasons.append('empty text (-3)')
        return score, reasons

    # 3b. Mostly non-letter text = garbage/SFX (e.g. "0 0W", "2 ??? 1", "sqrt")
    alpha_chars = sum(1 for c in clean if c.isalpha())
    if len(clean) > 0 and alpha_chars / len(clean) < 0.4:
        score -= 3
        reasons.append(f'mostly non-letter ({alpha_chars}/{len(clean)} alpha) (-3)')

    # 3c. Very short text with few real words = likely SFX or noise
    # But don't penalize if it has punctuation (real dialogue like "ME.", "NO WAY!")
    real_words = [w for w in clean.split() if len(w) >= 2 and any(c.isalpha() for c in w)]
    has_sentence_punct = bool(re.search(r'[.!?,;:…]', clean))
    if len(real_words) < 2 and len(clean) < 10 and not has_sentence_punct:
        score -= 2
        reasons.append(f'too few real words ({len(real_words)}) (-2)')

    # 4. Text length
    if len(clean) > 10:
        score += 2
        reasons.append(f'long text {len(clean)} chars (+2)')
    elif len(clean) >= 4:
        score += 1
        reasons.append(f'medium text {len(clean)} chars (+1)')
    elif len(clean) <= 2:
        score -= 2
        reasons.append(f'very short {len(clean)} chars (-2)')

    # 5. Punctuation = likely dialogue
    if re.search(r'[.!?,;:\'"…]', clean):
        score += 1
        reasons.append('has punctuation (+1)')

    # 6. Multiple words = likely dialogue or narration
    word_count = len(clean.split())
    if word_count >= 3:
        score += 2
        reasons.append(f'{word_count} words (+2)')
    elif word_count == 2:
        score += 1
        reasons.append(f'2 words (+1)')

    # 7. SFX pattern matching
    # Strip non-alphanumeric for pattern matching
    alpha_only = re.sub(r'[^a-zA-Z]', '', clean_lower)

    # Text starting with "Sfx" or "SFX" is OCR reading sound effect labels
    # Also handle "SFX:", "SFX=", "SFX -" variants
    if re.match(r'^sfx\b', clean_lower):
        score -= 10
        reasons.append(f'starts with SFX label (-10)')
        return score, reasons  # Definitely not dialogue, skip other checks

    if alpha_only in SFX_WORDS:
        # Single word SFX = very likely not dialogue, heavy penalty
        if word_count <= 1:
            score -= 8
            reasons.append(f'single-word SFX "{alpha_only}" (-8)')
        else:
            score -= 4
            reasons.append(f'SFX word "{alpha_only}" (-4)')
    elif _SFX_REPEAT_RE.match(alpha_only):
        score -= 5
        reasons.append(f'repeated letter "{alpha_only}" (-5)')
    elif _SFX_REPEAT2_RE.match(alpha_only) and len(alpha_only) <= 12:
        score -= 5
        reasons.append(f'repeated pattern "{alpha_only}" (-5)')
    elif _ALL_CONSONANTS_RE.match(alpha_only):
        score -= 3
        reasons.append(f'all consonants "{alpha_only}" (-3)')

    # 9. Check if ALL words are SFX-like (catches multi-word SFX like "OH OHH OHHH")
    if word_count >= 2:
        words_lower = [re.sub(r'[^a-z]', '', w.lower()) for w in clean.split()]
        words_lower = [w for w in words_lower if w]  # remove empty
        sfx_word_count = sum(1 for w in words_lower if w in SFX_WORDS or _SFX_REPEAT_RE.match(w) or _SFX_REPEAT2_RE.match(w) or len(w) <= 1)
        if sfx_word_count == len(words_lower):
            score -= 4
            reasons.append(f'all words are SFX ({sfx_word_count}/{len(words_lower)}) (-4)')

        # Same word repeated = SFX (e.g. "GAK GAK GAK GAK")
        unique_words = set(words_lower)
        if len(unique_words) == 1 and len(words_lower) >= 2:
            the_word = list(unique_words)[0]
            score -= 8
            reasons.append(f'same word repeated "{the_word}" x{len(words_lower)} (-8)')
        elif len(unique_words) <= 2 and len(words_lower) >= 3 and all(len(w) <= 5 for w in unique_words):
            score -= 5
            reasons.append(f'only {len(unique_words)} unique short words repeated (-5)')

    # 10. Very short text with no real words = likely junk (numbers, symbols, 1-2 letter garbage)
    if word_count <= 2:
        real_words = [w for w in clean.split() if len(re.sub(r'[^a-zA-Z]', '', w)) >= 3]
        if len(real_words) == 0:
            score -= 4
            reasons.append(f'no real words in short text (-4)')

    # 10. Gibberish detector -- text with mostly non-dictionary short fragments
    #     Catches "2 bla a ze", "Gasp hll a", "Glee e am", "Cree ech 7"
    if word_count >= 2 and alpha_only:
        # Count short/nonsense fragments (<=3 chars alpha, or single digits)
        short_frags = [w for w in clean.split() if (len(w) <= 3 and w.isalpha()) or (len(w) <= 2 and w.isdigit())]
        if len(short_frags) >= word_count * 0.5 and len(alpha_only) < 15:
            score -= 3
            reasons.append(f'mostly short fragments ({len(short_frags)}/{word_count}) (-3)')

    # 11. Symbol/punctuation garbage detector
    #     Catches OCR noise like "Vi ll}", "* xe I", "FR aN 2.", ". ~)"
    #     Real dialogue rarely has brackets, asterisks, or tildes
    symbol_chars = sum(1 for c in clean if c in '{}[]<>*~\\|@#^`')
    if symbol_chars > 0:
        score -= 3
        reasons.append(f'has {symbol_chars} symbol chars (-3)')

    # 12. Very low alpha ratio with short text = likely art/noise
    #     Catches "24)", "1 X 8", ". ~)"
    if len(clean) < 10 and len(clean) > 0:
        alpha_ratio = sum(1 for c in clean if c.isalpha()) / len(clean)
        if alpha_ratio < 0.5:
            score -= 3
            reasons.append(f'short text low alpha ratio {alpha_ratio:.2f} (-3)')

    # 13. Non-English text detector -- skip Japanese, Chinese, Korean, etc.
    #     Manga pages often have Japanese title text, author credits, SFX in kanji/kana
    #     If text has CJK/Hangul/Arabic/Hebrew chars, it's not English dialogue
    cjk_chars = sum(1 for c in clean if (
        '\u3000' <= c <= '\u9FFF' or   # CJK Unified + Japanese kana + punctuation
        '\uAC00' <= c <= '\uD7AF' or   # Korean Hangul
        '\uF900' <= c <= '\uFAFF' or   # CJK Compatibility
        '\u0600' <= c <= '\u06FF' or   # Arabic
        '\u0590' <= c <= '\u05FF'      # Hebrew
    ))
    if cjk_chars > 0:
        cjk_ratio = cjk_chars / max(1, len(clean))
        if cjk_ratio > 0.3:
            score -= 20
            reasons.append(f'non-English text ({cjk_chars} CJK chars, {cjk_ratio:.0%}) (-20)')
        elif cjk_ratio > 0.1:
            score -= 15
            reasons.append(f'mixed non-English ({cjk_chars} CJK chars, {cjk_ratio:.0%}) (-15)')
        elif cjk_chars >= 2:
            score -= 8
            reasons.append(f'minor CJK contamination ({cjk_chars} chars, {cjk_ratio:.0%}) (-8)')
        else:
            score -= 4
            reasons.append(f'has CJK char (-4)')

    # 14. Fragmented nonsense -- many single/double char words with no real sentences
    #     Catches OCR garbage like "Fc a den bet me", "E hi ca", "Vi ll a ge"
    if word_count >= 3:
        tiny_words = sum(1 for w in clean.split() if len(re.sub(r'[^a-zA-Z]', '', w)) <= 2)
        if tiny_words >= word_count * 0.5:
            # Check if any 3+ word subsequence forms a real English phrase
            has_real_phrase = any(len(w) >= 4 and w.isalpha() for w in clean.split())
            if not has_real_phrase:
                score -= 4
                reasons.append(f'fragmented nonsense ({tiny_words}/{word_count} tiny) (-4)')

    return score, reasons


def filter_dialogue_boxes(boxes, ocr_texts, image_shape):
    """Filter out likely SFX boxes using multi-heuristic scoring.
    Returns (kept_boxes, kept_texts, rejected_boxes) tuple."""
    image_area = image_shape[0] * image_shape[1]
    kept_boxes = []
    kept_texts = []
    rejected_boxes = []

    log.info(f'  Scoring {len(boxes)} boxes for dialogue vs SFX...')

    for i, box in enumerate(boxes):
        text = ocr_texts[i] if i < len(ocr_texts) else ''
        score, reasons = score_dialogue_box(box, text, image_area)

        text_preview = text.replace('\n', ' ').strip()[:50]
        reason_str = '; '.join(reasons)

        if score >= DIALOGUE_SCORE_GATE:
            log.info(f'  box[{i}] KEEP (score={score:+d}, gate={DIALOGUE_SCORE_GATE}) "{text_preview}" [{reason_str}]')
            kept_boxes.append(box)
            kept_texts.append(text)
        else:
            log.info(f'  box[{i}] DROP (score={score:+d}, gate={DIALOGUE_SCORE_GATE}) "{text_preview}" [{reason_str}]')
            rejected_boxes.append(box)

    log.info(f'  Dialogue filter: {len(boxes)} -> {len(kept_boxes)} kept, {len(rejected_boxes)} SFX dropped')
    return kept_boxes, kept_texts, rejected_boxes


def pre_filter_sfx_geometric(boxes, image_shape):
    """Pre-filter likely SFX/noise BEFORE bubble grouping, using geometry only.
    This prevents SFX regions from influencing dialogue grouping.
    Returns (dialogue_boxes, sfx_boxes)."""
    if not boxes:
        return boxes, []

    img_h, img_w = image_shape[:2]
    image_area = img_h * img_w
    kept = []
    rejected = []

    for b in boxes:
        reasons = []
        reject = False

        # 1. Too small -- noise fragments
        if b['w'] < PRE_SFX_MIN_SIDE or b['h'] < PRE_SFX_MIN_SIDE:
            reasons.append(f'tiny {b["w"]}x{b["h"]}')
            reject = True
        # 2. Extreme aspect ratio -- likely decorative SFX
        aspect = max(b['w'], b['h']) / max(1, min(b['w'], b['h']))
        if aspect > PRE_SFX_MAX_ASPECT:
            reasons.append(f'extreme aspect {aspect:.1f}')
            reject = True
        # 3. Large rotation -- SFX are often slanted
        angle = b.get('angle', 0)
        if angle > PRE_SFX_MAX_ROTATION:
            reasons.append(f'rotated {angle:.0f}deg')
            reject = True
        # 4. Very large boxes -- likely full art text, not dialogue
        box_area = b['w'] * b['h']
        if box_area > image_area * PRE_SFX_MAX_AREA_RATIO:
            reasons.append(f'huge area {box_area/image_area:.2%}')
            reject = True
        # 5. Very tiny area relative to page
        if box_area < image_area * PRE_SFX_MIN_AREA_RATIO:
            reasons.append(f'micro area {box_area}')
            reject = True

        if reject:
            log.info(f'  pre-SFX: DROP {b["w"]}x{b["h"]} [{"; ".join(reasons)}]')
            rejected.append(b)
        else:
            kept.append(b)

    log.info(f'  Pre-SFX filter: {len(boxes)} -> {len(kept)} kept, {len(rejected)} removed')
    return kept, rejected


# ─── Bubble grouping ────────────────────────────────────────────────────────

def _estimate_line_height(raw_boxes):
    """Estimate typical text line height from raw detection boxes."""
    if not raw_boxes:
        return 20
    heights = sorted([b['h'] for b in raw_boxes])
    # Use median to avoid outliers (SFX, merged boxes)
    median_h = heights[len(heights) // 2]
    return max(10, median_h)


def mask_based_bubble_grouping(text_mask, raw_boxes, crop_shape, pad_pct=0.15, min_pad=12):
    """Group text into speech bubbles using the text segmentation mask.

    Two-phase approach:
    Phase 1: Find connected components on RAW (undilated) mask.
             Each component is a hard boundary -- boxes in different
             components NEVER merge.
    Phase 2: Within each component, cluster raw detection boxes by
             vertical proximity to form bubble groups.

    The key insight: the raw mask already separates different bubbles
    because there's whitespace (bubble boundary) between them. We only
    need geometric clustering to handle fragmented text within one bubble.

    Returns list of boxes with 'shape' and 'raw_indices' metadata."""
    if text_mask is None:
        return None

    h, w = text_mask.shape[:2]

    # 1. Threshold the mask to binary
    _, binary = cv2.threshold(text_mask, 100, 255, cv2.THRESH_BINARY)

    # 2. NO dilation for component finding -- the raw mask already separates
    #    bubbles via white borders/gutters between them. Dilation bridges those gaps.
    #    We only use a tiny 3x1 horizontal kernel to connect broken character strokes.
    kernel_char = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    mild_dilated = cv2.dilate(binary, kernel_char, iterations=1)

    # 3. Find connected components on the (barely) dilated mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mild_dilated)

    log.info(f'  Mask components (raw+3x1): {num_labels - 1} regions')

    # 4. Assign each raw detection box to its mask component
    #    A box belongs to the component that contains its center pixel.
    box_to_component = {}
    for ri, rb in enumerate(raw_boxes):
        rcx = int(min(w - 1, max(0, rb['x'] + rb['w'] / 2)))
        rcy = int(min(h - 1, max(0, rb['y'] + rb['h'] / 2)))
        comp_id = int(labels[rcy, rcx])
        if comp_id == 0:
            # Center is on background -- check if any part overlaps a component
            for dx, dy in [(0, 0), (rb['w']//2, 0), (0, rb['h']//2),
                           (rb['w']//4, rb['h']//4)]:
                sx = int(min(w - 1, max(0, rb['x'] + dx)))
                sy = int(min(h - 1, max(0, rb['y'] + dy)))
                if labels[sy, sx] > 0:
                    comp_id = int(labels[sy, sx])
                    break
        box_to_component[ri] = comp_id

    # 5. Group boxes by their mask component
    component_boxes = {}
    for ri, comp_id in box_to_component.items():
        if comp_id == 0:
            continue  # background -- skip
        if comp_id not in component_boxes:
            component_boxes[comp_id] = []
        component_boxes[comp_id].append(ri)

    log.info(f'  Boxes assigned to {len(component_boxes)} mask regions')

    # 6. Since we use minimal dilation, we get many small components (per-word or per-line).
    #    We need to merge nearby components that belong to the same speech bubble.
    #    Strategy: merge components whose bounding boxes are close AND aligned.
    #    Two components merge if:
    #    - Their bboxes overlap or gap < line_h vertically
    #    - They have significant horizontal overlap (>30%)
    #    - The merged bbox doesn't grow excessively (area limit)
    line_h = _estimate_line_height(raw_boxes) if raw_boxes else 20

    # Build component info list (only components with boxes)
    comp_infos = []
    for comp_id, box_indices in component_boxes.items():
        comp_area = stats[comp_id, cv2.CC_STAT_AREA]
        cbw = stats[comp_id, cv2.CC_STAT_WIDTH]
        cbh = stats[comp_id, cv2.CC_STAT_HEIGHT]
        if comp_area < 100 or cbw < 8 or cbh < 8:
            continue
        # Use the bounding box of the raw detection boxes (more accurate than mask stats)
        cboxes = [raw_boxes[ri] for ri in box_indices]
        bx1 = min(b['x'] for b in cboxes)
        by1 = min(b['y'] for b in cboxes)
        bx2 = max(b['x'] + b['w'] for b in cboxes)
        by2 = max(b['y'] + b['h'] for b in cboxes)
        comp_infos.append({
            'comp_id': comp_id,
            'indices': box_indices,
            'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
            'text_area': sum(b['w'] * b['h'] for b in cboxes),
        })

    # Max merged bubble size: prevent cross-panel merges.
    # A single speech bubble should never span more than ~35% of page height
    # or ~55% of page width. This stops title page text from merging with
    # bottom panel dialogue.
    max_merged_h = int(h * 0.35)
    max_merged_w = int(w * 0.55)
    log.info(f'  Merge caps: max_h={max_merged_h}px ({h}*0.35), max_w={max_merged_w}px ({w}*0.55)')

    # Greedy merge: repeatedly merge closest compatible pair
    changed = True
    while changed:
        changed = False
        for i in range(len(comp_infos)):
            if comp_infos[i] is None:
                continue
            a = comp_infos[i]
            best_j = -1
            best_dist = float('inf')
            for j in range(i + 1, len(comp_infos)):
                if comp_infos[j] is None:
                    continue
                b = comp_infos[j]

                # Vertical gap between bounding boxes
                v_gap = max(0, max(a['y1'], b['y1']) - min(a['y2'], b['y2']))
                # Horizontal gap
                h_gap = max(0, max(a['x1'], b['x1']) - min(a['x2'], b['x2']))

                # Must be close vertically (within 1.5 * line height)
                if v_gap > line_h * 1.5:
                    continue
                # Must not be too far horizontally (separate columns)
                if h_gap > line_h * 0.5:
                    continue

                # Must have horizontal overlap (for vertically stacked lines)
                h_overlap = min(a['x2'], b['x2']) - max(a['x1'], b['x1'])
                min_w = min(a['x2'] - a['x1'], b['x2'] - b['x1'])
                if h_overlap < min_w * 0.25 and h_gap > 0:
                    continue

                # Merged bbox dimensions
                mx1 = min(a['x1'], b['x1'])
                my1 = min(a['y1'], b['y1'])
                mx2 = max(a['x2'], b['x2'])
                my2 = max(a['y2'], b['y2'])

                # SIZE CAP: reject merge if result would be too tall or wide
                # This prevents cross-panel merges (e.g. title page + bottom dialogue)
                if (my2 - my1) > max_merged_h:
                    continue
                if (mx2 - mx1) > max_merged_w:
                    continue

                # Area growth check: merged bbox must not be too much bigger
                merged_bbox_area = (mx2 - mx1) * (my2 - my1)
                combined_text_area = a['text_area'] + b['text_area']
                if merged_bbox_area > combined_text_area * 6.0:
                    continue

                # Distance metric: prefer closest pairs
                dist = v_gap + h_gap * 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                b = comp_infos[best_j]
                a['indices'] = a['indices'] + b['indices']
                a['x1'] = min(a['x1'], b['x1'])
                a['y1'] = min(a['y1'], b['y1'])
                a['x2'] = max(a['x2'], b['x2'])
                a['y2'] = max(a['y2'], b['y2'])
                a['text_area'] = a['text_area'] + b['text_area']
                comp_infos[best_j] = None
                changed = True

    # Remove None entries
    comp_infos = [c for c in comp_infos if c is not None]

    # 7. Create bubbles from merged component groups
    bubbles = []
    for info in comp_infos:
        bw = info['x2'] - info['x1']
        bh = info['y2'] - info['y1']
        _add_bubble_v2(bubbles, info['x1'], info['y1'], bw, bh,
                      info['text_area'], info['indices'], w, h, pad_pct, min_pad)

    log.info(f'  Mask-based grouping: {num_labels - 1} components -> {len(comp_infos)} groups -> {len(bubbles)} bubbles')
    for i, b in enumerate(bubbles):
        log.info(f'    bubble[{i}] {b["shape"]} x={b["x"]} y={b["y"]} {b["w"]}x{b["h"]} '
                 f'(raw boxes: {b["raw_indices"]})')

    return bubbles, mild_dilated


def _add_bubble_v2(bubbles, x, y, bw, bh, area, raw_indices, img_w, img_h, pad_pct, min_pad):
    """Create a bubble entry with proportional padding."""
    aspect = max(bw, bh) / max(1, min(bw, bh))
    if aspect < 2.0:
        shape = 'oval'
    elif aspect < 4.0:
        shape = 'rect'
    else:
        shape = 'wide'

    # Proportional padding: max(min_pad, smaller_dim * pad_pct)
    pad = max(min_pad, int(min(bw, bh) * pad_pct))
    # Extra padding for oval bubbles (text often near edges)
    if shape == 'oval':
        pad = int(pad * 1.3)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + bw + pad)
    y2 = min(img_h, y + bh + pad)

    bubbles.append({
        'x': x1, 'y': y1,
        'w': x2 - x1, 'h': y2 - y1,
        'score': 1.0,
        'angle': 0,
        'shape': shape,
        'raw_indices': raw_indices,
        'mask_area': area,
    })


# ─── Reading order ──────────────────────────────────────────────────────────

def sort_reading_order(bubbles, direction='rtl'):
    """Sort bubbles in reading order using panel-row detection.

    direction:
      'rtl' -- manga (right-to-left within rows, top-to-bottom rows)
      'ltr' -- Western comics (left-to-right within rows, top-to-bottom rows)
      'vertical' -- webtoons (pure top-to-bottom, ignore columns)

    Algorithm:
    1. Group bubbles into panel rows by detecting large vertical gaps
    2. Within each row, sort bubbles by direction (RTL or LTR)
    3. Within the same panel column, sort top-to-bottom first"""
    if not bubbles:
        return bubbles
    if len(bubbles) == 1:
        return bubbles

    # Webtoon/vertical scroll: pure top-to-bottom, no column logic
    if direction == 'vertical':
        return sorted(bubbles, key=lambda b: b['top'])

    # Step 1: Detect panel rows by grouping bubbles with similar Y ranges
    sorted_by_y = sorted(bubbles, key=lambda b: b['top'])

    # Calculate vertical gaps between consecutive bubbles
    avg_h = np.mean([b['height'] for b in bubbles])

    # Group into rows: large Y gap = new row
    # Row threshold: bubbles within 1.2x average height are in the same row
    row_gap_threshold = avg_h * 1.2

    rows = [[sorted_by_y[0]]]
    for b in sorted_by_y[1:]:
        # Compare against the bottom edge of the last bubble in current row
        prev_bottom = max(rb['top'] + rb['height'] for rb in rows[-1])
        curr_top = b['top']

        if curr_top - prev_bottom > row_gap_threshold:
            # Large gap -- new panel row
            rows.append([b])
        else:
            # Check if this bubble overlaps vertically with current row
            row_top = min(rb['top'] for rb in rows[-1])
            row_bottom = max(rb['top'] + rb['height'] for rb in rows[-1])
            b_center_y = b['top'] + b['height'] / 2

            if row_top <= b_center_y <= row_bottom + row_gap_threshold:
                rows[-1].append(b)
            else:
                rows.append([b])

    log.info(f'  Reading order: {len(bubbles)} bubbles -> {len(rows)} panel rows')

    # Step 2: Within each row, sort right-to-left (manga reading direction)
    # But group sub-columns first: bubbles stacked vertically in the same
    # horizontal zone should be read top-to-bottom before moving left
    result = []
    for row_idx, row in enumerate(rows):
        if len(row) == 1:
            result.extend(row)
            log.info(f'    row[{row_idx}]: 1 bubble')
            continue

        # Sub-group by horizontal proximity (same panel column)
        row.sort(key=lambda b: b['left'])
        avg_w = np.mean([b['width'] for b in row])
        col_gap_threshold = avg_w * 0.8  # gap between columns

        columns = [[row[0]]]
        for b in row[1:]:
            prev_right = max(cb['left'] + cb['width'] for cb in columns[-1])
            if b['left'] - prev_right > col_gap_threshold:
                columns.append([b])
            else:
                columns[-1].append(b)

        # Sort columns by reading direction
        if direction == 'rtl':
            columns.sort(key=lambda col: -np.mean([b['left'] for b in col]))
        else:  # ltr
            columns.sort(key=lambda col: np.mean([b['left'] for b in col]))

        # Within each column, sort top-to-bottom
        for col in columns:
            col.sort(key=lambda b: b['top'])
            result.extend(col)

        col_sizes = [len(c) for c in columns]
        log.info(f'    row[{row_idx}]: {len(row)} bubbles, {len(columns)} columns {col_sizes}')

    return result


# ─── BubbleDetector (RT-DETR-v2 ONNX model) ────────────────────────────────

class BubbleDetector:
    """Detects speech bubbles and text regions using RT-DETR-v2 ONNX model.
    Classes: 0=bubble, 1=text_bubble (dialogue), 2=text_free (SFX/narration)"""

    CLASS_NAMES = {0: 'bubble', 1: 'text_bubble', 2: 'text_free'}
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        self.session = None
        self.available = False

    def load(self):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'detector.onnx')
        if not os.path.exists(model_path):
            log.warning(f'Bubble detector model not found at {model_path}')
            return
        try:
            import onnxruntime as ort
            from config import ONNX_THREAD_CAP
            opts = ort.SessionOptions()
            if ONNX_THREAD_CAP > 0:
                opts.inter_op_num_threads = ONNX_THREAD_CAP
                opts.intra_op_num_threads = ONNX_THREAD_CAP
            self.session = ort.InferenceSession(model_path, sess_options=opts)
            self.available = True
            log.info(f'Bubble detector loaded: {model_path} ({os.path.getsize(model_path)/1e6:.0f}MB)')
        except Exception as e:
            log.warning(f'Failed to load bubble detector: {e}')

    def detect(self, img_bgr, conf_threshold=0.4):
        """Run bubble detection on a BGR image. Returns dict with bubble/text_bubble/text_free lists."""
        if not self.available:
            return None

        orig_h, orig_w = img_bgr.shape[:2]

        # Preprocess: resize to 640x640, normalize with ImageNet stats
        resized = cv2.resize(img_bgr, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - self.MEAN) / self.STD).transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0).astype(np.float32)
        orig_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)

        labels, boxes, scores = self.session.run(None, {
            'images': blob,
            'orig_target_sizes': orig_sizes
        })

        # Collect all detections by class
        raw = {'bubble': [], 'text_bubble': [], 'text_free': []}
        for i in range(len(labels[0])):
            score = float(scores[0][i])
            if score < conf_threshold:
                continue
            label = int(labels[0][i])
            box = boxes[0][i]
            cls = self.CLASS_NAMES.get(label, 'unknown')
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(orig_w, int(box[2]))
            y2 = min(orig_h, int(box[3]))
            if x2 > x1 and y2 > y1:
                raw[cls].append({
                    'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                    'score': score, 'class': cls
                })

        # NMS per class: remove overlapping detections (IoU > 0.3)
        results = {}
        for cls, dets in raw.items():
            results[cls] = self._nms(dets, iou_threshold=0.3)

        # Cross-class NMS: suppress overlapping text_bubble vs text_free only.
        # Do NOT suppress bubble vs text_bubble -- bubble outlines are SUPPOSED to
        # contain text_bubble regions inside them. Removing one kills the filtering.
        text_dets = results.get('text_bubble', []) + results.get('text_free', [])
        text_dets.sort(key=lambda d: d['score'], reverse=True)
        keep_text = []
        for d in text_dets:
            dominated = False
            for k in keep_text:
                iou, containment = BubbleDetector._box_overlap(d, k)
                if iou > 0.3 or containment > 0.6:
                    dominated = True
                    break
            if not dominated:
                keep_text.append(d)
        # Rebuild: keep all bubble outlines, only NMS'd text regions
        results = {'bubble': results.get('bubble', []), 'text_bubble': [], 'text_free': []}
        for d in keep_text:
            results[d['class']].append(d)

        return results

    @staticmethod
    def _nms(detections, iou_threshold=0.3):
        """Non-Maximum Suppression: remove overlapping/contained boxes, keep highest score."""
        if len(detections) <= 1:
            return detections
        # Sort by score descending
        dets = sorted(detections, key=lambda d: d['score'], reverse=True)
        keep = []
        for d in dets:
            dominated = False
            for k in keep:
                iou, containment = BubbleDetector._box_overlap(d, k)
                # Drop if IoU is high OR if this box is mostly inside a kept box
                if iou > iou_threshold or containment > 0.6:
                    dominated = True
                    break
            if not dominated:
                keep.append(d)
        return keep

    @staticmethod
    def _box_overlap(a, b):
        """Compute IoU and containment (fraction of smaller box inside larger) between two boxes."""
        ax1, ay1, ax2, ay2 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
        bx1, by1, bx2, by2 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = a['w'] * a['h']
        area_b = b['w'] * b['h']
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0.0
        # Containment: how much of the smaller box is inside the larger one
        min_area = min(area_a, area_b)
        containment = inter / min_area if min_area > 0 else 0.0
        return iou, containment


# ─── Debug visualization ───────────────────────────────────────────────────

def save_debug_frame(image, raw_boxes, merged_boxes, rejected_geo_boxes,
                     final_ocr_boxes, rejected_sfx_boxes, kept_boxes,
                     ocr_texts_map):
    """Build and save a combined debug overlay showing every pipeline stage.
    Colors:
      Red    = raw detections (before merge)
      Yellow = merged boxes (before filter)
      Gray   = rejected by geometric filter
      Cyan   = rejected by SFX filter (after OCR)
      Green  = final kept dialogue boxes
    Each box is labeled with its index and OCR text if available.
    Saved to server/debug_frames/debug_page_TIMESTAMP.png"""

    frames_dir = os.path.join(os.path.dirname(__file__), 'debug_frames')
    os.makedirs(frames_dir, exist_ok=True)

    overlay = image.copy()
    h_img, w_img = overlay.shape[:2]

    # Scale font based on image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.7, w_img / 2000))
    thickness = max(1, int(font_scale * 2.5))

    def draw_boxes(boxes, color, label_prefix, line_w, texts=None):
        for i, b in enumerate(boxes):
            x, y, w, bh = b['x'], b['y'], b['w'], b['h']
            cv2.rectangle(overlay, (x, y), (x + w, y + bh), color, line_w)
            # Label
            lbl = f'{label_prefix}{i}'
            if texts and i in texts and texts[i].strip():
                lbl += ': ' + texts[i].replace('\n', ' ').strip()[:35]
            elif 'score' in b:
                lbl += f' ({b["score"]:.2f})'
            # Background for readability
            (tw, th), _ = cv2.getTextSize(lbl, font, font_scale, thickness)
            ty = max(y - 6, th + 4)
            cv2.rectangle(overlay, (x, ty - th - 4), (x + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(overlay, lbl, (x + 2, ty), font, font_scale, color, thickness)

    # Layer 1: Raw detections (red, thin)
    draw_boxes(raw_boxes, (0, 0, 255), 'R', 1)

    # Layer 2: Merged boxes (yellow, thin dashed via dotted rectangles)
    draw_boxes(merged_boxes, (0, 255, 255), 'M', 1)

    # Layer 3: Rejected by geometric filter (gray, thin)
    draw_boxes(rejected_geo_boxes, (128, 128, 128), 'X', 1)

    # Layer 4: Rejected by SFX filter (cyan, medium)
    sfx_texts = {i: ocr_texts_map.get(id(b), '') for i, b in enumerate(rejected_sfx_boxes)}
    draw_boxes(rejected_sfx_boxes, (255, 200, 0), 'SFX', 2, sfx_texts)

    # Layer 5: Final kept boxes (green, thick) -- drawn last so they're on top
    kept_texts = {i: ocr_texts_map.get(id(b), '') for i, b in enumerate(kept_boxes)}
    draw_boxes(kept_boxes, (0, 255, 0), '#', 3, kept_texts)

    # Legend in top-left corner
    legend_y = 25
    legend_items = [
        ((0, 0, 255), f'Red = raw detections ({len(raw_boxes)})'),
        ((0, 255, 255), f'Yellow = merged ({len(merged_boxes)})'),
        ((128, 128, 128), f'Gray = rejected geo ({len(rejected_geo_boxes)})'),
        ((255, 200, 0), f'Cyan = rejected SFX ({len(rejected_sfx_boxes)})'),
        ((0, 255, 0), f'Green = final kept ({len(kept_boxes)})'),
    ]
    for color, text in legend_items:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(overlay, (8, legend_y - th - 2), (16 + tw, legend_y + 4), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, legend_y), font, font_scale, color, thickness)
        legend_y += th + 12

    # Save with timestamp
    ts = time.strftime('%Y%m%d_%H%M%S')
    filename = f'debug_page_{ts}.png'
    filepath = os.path.join(frames_dir, filename)
    cv2.imwrite(filepath, overlay)
    log.info(f'  Debug frame saved: {filepath}')

    # Also save as latest for quick access
    latest = os.path.join(frames_dir, 'latest.png')
    cv2.imwrite(latest, overlay)

    return filepath
