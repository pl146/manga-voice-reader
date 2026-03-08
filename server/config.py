"""
Manga Voice Reader — Centralized Configuration

All tunable parameters in one place. Every value can be overridden
via environment variable with MVR_ prefix (e.g. MVR_PORT=8080).
"""

import os

_dir = os.path.dirname(os.path.abspath(__file__))

import logging as _logging
_log = _logging.getLogger(__name__)

def _env(name, default, cast=str):
    v = os.environ.get(f'MVR_{name}', '')
    if not v:
        return default
    try:
        return cast(v)
    except (ValueError, TypeError):
        _log.warning(f'Invalid MVR_{name}={v!r}, using default {default!r}')
        return default

# ─── Server ──────────────────────────────────────────────────────────────────
PORT = _env('PORT', 5055, int)
IDLE_SHUTDOWN_MINUTES = _env('IDLE_TIMEOUT', 30, int)  # was 10, raised per improvement plan

# ─── Model paths ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_dir, 'models')
KOKORO_MODEL_PATH = os.path.join(MODELS_DIR, 'kokoro', 'kokoro-v1.0.onnx')
KOKORO_VOICES_PATH = os.path.join(MODELS_DIR, 'kokoro', 'voices-v1.0.bin')
PIPER_BIN = _env('PIPER_BIN', 'piper')
PIPER_MODEL = _env('PIPER_MODEL', os.path.join(MODELS_DIR, 'piper', 'en_US-lessac-medium.onnx'))
DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, 'detector.onnx')
TEXT_SEGMENTER_PATH = os.path.join(MODELS_DIR, 'comictextdetector.pt.onnx')

# ─── Detection ───────────────────────────────────────────────────────────────
MAX_DETECT_WIDTH = _env('MAX_DETECT_WIDTH', 1280, int)
MAX_BOXES = _env('MAX_BOXES', 80, int)
IOU_MERGE_THRESHOLD = _env('IOU_MERGE_THRESHOLD', 0.45, float)
PROXIMITY_MERGE_RATIO = _env('PROXIMITY_MERGE_RATIO', 0.5, float)
MIN_BOX_AREA = _env('MIN_BOX_AREA', 60, int)        # was 120, lowered to catch small bubbles
MIN_BOX_SIDE = _env('MIN_BOX_SIDE', 5, int)         # was 8, lowered to catch small bubbles
MAX_ASPECT_RATIO = _env('MAX_ASPECT_RATIO', 20.0, float)
MIN_DET_SCORE = _env('MIN_DET_SCORE', 0.10, float)
LOW_BOX_THRESHOLD = _env('LOW_BOX_THRESHOLD', 5, int)

# ─── Merging ─────────────────────────────────────────────────────────────────
VERTICAL_MERGE_HEIGHT_FACTOR = _env('VERTICAL_MERGE_HEIGHT_FACTOR', 6, int)   # was 4
VERTICAL_MERGE_WIDTH_FACTOR = _env('VERTICAL_MERGE_WIDTH_FACTOR', 4, int)     # was 3

# ─── Crop padding ────────────────────────────────────────────────────────────
CROP_PAD_MIN = _env('CROP_PAD_MIN', 20, int)
CROP_PAD_PCT = _env('CROP_PAD_PCT', 0.12, float)
OCR_CROP_PAD_PCT = 0.20
OCR_CROP_PAD_MIN = 30
TESSERACT_CONFIG = '--oem 3 --psm 6'
ROW_Y_THRESHOLD_RATIO = _env('ROW_Y_THRESHOLD_RATIO', 0.6, float)

# ─── Filtering ───────────────────────────────────────────────────────────────
CNN_REJECTION_CONFIDENCE = _env('CNN_REJECTION_CONFIDENCE', 0.93, float)  # was 0.88
CNN_RESCUE_CONFIDENCE = 0.6
DIALOGUE_SCORE_GATE = _env('DIALOGUE_SCORE_GATE', -4, int)               # was -2
GIBBERISH_THRESHOLD = _env('GIBBERISH_THRESHOLD', 0.85, float)           # was 0.80
GIBBERISH_MIN_CHECKED = _env('GIBBERISH_MIN_CHECKED', 5, int)            # was 3, short fragments exempt
SYMBOL_JUNK_LIMIT = _env('SYMBOL_JUNK_LIMIT', 6, int)                   # was 4
# Removed %, ;, : from junk symbols — they appear in valid manga dialogue
JUNK_SYMBOL_PATTERN = r'[_¢§©®™°¶•€£¥|\\<>{}@#^`=\[\]+~/]'

# ─── Pre-SFX geometric filter ─────────────────────────────────────────────────
PRE_SFX_MIN_SIDE = 6                # reject boxes with width or height < this
PRE_SFX_MAX_ASPECT = 12.0           # reject boxes with aspect ratio > this
PRE_SFX_MAX_ROTATION = 20           # reject boxes rotated more than this (degrees)
PRE_SFX_MAX_AREA_RATIO = 0.15       # reject boxes larger than this fraction of image
PRE_SFX_MIN_AREA_RATIO = 0.0003     # reject boxes smaller than this fraction of image

# ─── OCR ─────────────────────────────────────────────────────────────────────
FLORENCE_MODEL_NAME = 'microsoft/Florence-2-large-ft'
FLORENCE_MAX_TOKENS = 512
FLORENCE_NUM_BEAMS = 3
PADDLE_USE_GPU = True
PADDLE_LANG = 'en'

# ─── TTS ─────────────────────────────────────────────────────────────────────
DEFAULT_VOICE = 'af_heart'
TTS_SPEED_MIN = 0.5
TTS_SPEED_MAX = 2.0

# ─── Debug ───────────────────────────────────────────────────────────────────
SAVE_DEBUG_CROPS = _env('SAVE_DEBUG_CROPS', True, lambda v: v.lower() in ('true', '1', 'yes'))
DEBUG_VIEW = _env('DEBUG_VIEW', True, lambda v: v.lower() in ('true', '1', 'yes'))
SAVE_DEBUG_IMAGES = _env('SAVE_DEBUG_IMAGES', True, lambda v: v.lower() in ('true', '1', 'yes'))
DEBUG_FRAMES_DIR = os.path.join(_dir, 'debug_frames')
MAX_DEBUG_FRAMES = 20           # rotation: keep only this many
DEBUG_MAX_AGE_SECONDS = 3600    # rotation: delete frames older than 1 hour

# ─── Feature flags ───────────────────────────────────────────────────────────
USE_LOCAL_RECOGNIZER = _env('USE_LOCAL_RECOGNIZER', False, lambda v: v.lower() in ('true', '1', 'yes'))
COLLECT_TRAINING_DATA = _env('COLLECT_TRAINING_DATA', False, lambda v: v.lower() in ('true', '1', 'yes'))

# ─── Thread / CPU caps ──────────────────────────────────────────────────────
# Controls peak CPU usage by limiting per-library thread counts.
# Lower = less CPU spike but slightly more latency. 0 = use library defaults.
# Validated: default 2 reduces system CPU peak from ~50% to ~25% with no
# quality regression (same bubble count/text on benchmark pages).
# Rollback: set all to 0 or remove MVR_ env vars to restore defaults.
#   MVR_CPU_THREAD_CAP=0 MVR_ONNX_THREAD_CAP=0 MVR_TORCH_THREAD_CAP=0
# Recommended range: 1-4. Values above CPU core count have no effect.
CPU_THREAD_CAP = _env('CPU_THREAD_CAP', 2, int)       # OMP/MKL/BLAS/Paddle threads
ONNX_THREAD_CAP = _env('ONNX_THREAD_CAP', 2, int)     # ONNX Runtime inter/intra threads
TORCH_THREAD_CAP = _env('TORCH_THREAD_CAP', 2, int)   # PyTorch CPU threads

# ─── Rate limiting ───────────────────────────────────────────────────────────
MAX_CONCURRENT_PROCESS = 2
MAX_QUEUED_PROCESS = 3
MAX_CONCURRENT_TTS = 5

# ─── Input validation ────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 50 * 1024 * 1024   # 50MB decoded
MAX_IMAGE_DIMENSION = 8000            # pixels
MAX_DPR = 4
VALID_READING_DIRECTIONS = ('rtl', 'ltr', 'vertical')

# ─── Storage ─────────────────────────────────────────────────────────────────
PAGE_CACHE_MAX = _env('PAGE_CACHE_MAX', 5, int)
EXT_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB, then rotate
EXT_LOG_MAX_ROTATED = 2

# ─── Protected vocabulary (never spell-corrected) ────────────────────────────
PROTECTED_VOCAB_PATH = os.path.join(_dir, 'protected_vocab.txt')
