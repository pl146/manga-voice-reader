"""TTS engine module for Manga Voice Reader.

Provides text-to-speech synthesis using two backends:
  - Kokoro TTS (primary) — 82M param ONNX model, fast (~850ms/bubble), 54 voices
  - Piper TTS (fallback) — lightweight ONNX model loaded via Python API

The main entry point is synthesize_speech(), which tries Kokoro first and
falls back to Piper automatically. Call init() at startup to load whichever
engines are available.
"""

import io
import logging
import os
import subprocess
import threading
import time
import wave

import numpy as np

from config import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, PIPER_BIN, PIPER_MODEL, DEFAULT_VOICE

log = logging.getLogger(__name__)

# ─── Model path constants ───────────────────────────────────────────────────

# Model paths imported from config.py

# ─── Kokoro TTS globals ─────────────────────────────────────────────────────

_kokoro_model = None
_kokoro_lock = threading.Lock()
_kokoro_voice = DEFAULT_VOICE
kokoro_available = False

# ─── Piper TTS globals ──────────────────────────────────────────────────────

_piper_voice = None
_piper_lock = threading.Lock()
piper_available = False


# ─── Kokoro TTS ─────────────────────────────────────────────────────────────

def _load_kokoro():
    """Load the Kokoro ONNX model into memory. Called once at startup."""
    global _kokoro_model, kokoro_available
    if not os.path.isfile(KOKORO_MODEL_PATH) or not os.path.isfile(KOKORO_VOICES_PATH):
        log.warning(f'[TTS] Kokoro model files not found')
        return
    try:
        from kokoro_onnx import Kokoro
        t0 = time.time()
        _kokoro_model = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
        ms = int((time.time() - t0) * 1000)
        voices = _kokoro_model.get_voices()
        kokoro_available = True
        log.info(f'[TTS] Kokoro loaded ({ms}ms, {len(voices)} voices, sr=24000)')
    except Exception as e:
        log.error(f'[TTS] Failed to load Kokoro: {e}')


# ─── Piper TTS ──────────────────────────────────────────────────────────────

def check_piper():
    """Check if Piper TTS is installed and a model is configured."""
    global piper_available
    log.info('─── Piper TTS check ───')

    # Check model file
    if not PIPER_MODEL:
        log.warning('Piper TTS: PIPER_MODEL not set and no default model found')
        log.warning('  Fix: download a model to server/models/piper/ or set PIPER_MODEL env var')
        return
    if not os.path.isfile(PIPER_MODEL):
        log.warning(f'Piper TTS: model file not found: {PIPER_MODEL}')
        log.warning('  Fix: download from https://huggingface.co/rhasspy/piper-voices')
        return
    log.info(f'  Model found: {PIPER_MODEL} ({os.path.getsize(PIPER_MODEL) / 1024 / 1024:.1f} MB)')

    # Check .json config alongside model
    json_config = PIPER_MODEL + '.json'
    if not os.path.isfile(json_config):
        log.warning(f'  Model config missing: {json_config}')
        log.warning('  Fix: download the .onnx.json file alongside the .onnx model')
        return
    log.info(f'  Model config found: {json_config}')

    # Check piper binary
    try:
        r = subprocess.run([PIPER_BIN, '--help'], capture_output=True, text=True, timeout=5)
        log.info(f'  Piper binary: {PIPER_BIN} (ok)')
    except FileNotFoundError:
        log.warning(f'  Piper binary not found: {PIPER_BIN}')
        log.warning('  Fix: pip3 install piper-tts')
        return
    except subprocess.TimeoutExpired:
        log.warning(f'  Piper binary timed out: {PIPER_BIN}')
        return

    # Quick synthesis test (use binary mode — Piper outputs raw audio bytes)
    try:
        test_result = subprocess.run(
            [PIPER_BIN, '--model', PIPER_MODEL, '--output-raw'],
            input=b'test',
            capture_output=True,
            timeout=15,
        )
        if test_result.returncode != 0:
            log.warning(f'  Piper test synthesis failed: {test_result.stderr[:200]}')
            return
        log.info('  Piper test synthesis: ok')
    except Exception as e:
        log.warning(f'  Piper test synthesis error: {e}')
        return

    piper_available = True
    log.info(f'  Piper TTS: READY (model: {os.path.basename(PIPER_MODEL)})')


def _load_piper_voice():
    """Load Piper voice model into memory. Called once at startup."""
    global _piper_voice
    try:
        from piper.voice import PiperVoice
        t0 = time.time()
        _piper_voice = PiperVoice.load(PIPER_MODEL)
        ms = int((time.time() - t0) * 1000)
        log.info(f'[TTS] Piper model loaded in-process ({ms}ms, sr={_piper_voice.config.sample_rate})')
    except Exception as e:
        log.error(f'[TTS] Failed to load Piper model: {e}')
        _piper_voice = None


# ─── Unified synthesis ──────────────────────────────────────────────────────

def synthesize_speech(text, voice=None, speed=1.0):
    """Synthesize speech from text, returning WAV bytes.

    Tries Kokoro first (fast, many voices), falls back to Piper.

    Args:
        text: The text to synthesize.
        voice: Voice name (Kokoro only). Defaults to _kokoro_voice.
        speed: Playback speed multiplier (0.5–2.0). Defaults to 1.0.

    Returns:
        A tuple of (audio_bytes, engine_name, synth_ms) on success,
        or raises RuntimeError if all engines fail.
    """
    if voice is None:
        voice = _kokoro_voice
    speed = max(0.5, min(2.0, float(speed)))

    t0 = time.time()

    # Kokoro TTS — fast (850ms/bubble), 54 voices, parallel-friendly
    if kokoro_available and _kokoro_model is not None:
        try:
            with _kokoro_lock:
                samples, sr = _kokoro_model.create(text, voice=voice, speed=speed)
            # Convert float32 samples to int16 WAV
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm16.tobytes())
            audio_bytes = buf.getvalue()
            synth_ms = int((time.time() - t0) * 1000)
            log.info(f'[TTS] Kokoro ({voice}): {len(audio_bytes)} bytes, {synth_ms}ms')
            return audio_bytes, 'kokoro', synth_ms
        except Exception as e:
            log.error(f'[TTS] Kokoro failed: {e}, falling back to Piper')

    # Fallback to Piper
    if piper_available and _piper_voice is not None:
        try:
            with _piper_lock:
                chunks = list(_piper_voice.synthesize(text))
            raw_pcm = b''.join(c.audio_int16_bytes for c in chunks)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_piper_voice.config.sample_rate)
                wf.writeframes(raw_pcm)
            audio_bytes = buf.getvalue()
            synth_ms = int((time.time() - t0) * 1000)
            log.info(f'[TTS] Piper fallback: {len(audio_bytes)} bytes, {synth_ms}ms')
            return audio_bytes, 'piper', synth_ms
        except Exception as e:
            synth_ms = int((time.time() - t0) * 1000)
            log.error(f'[TTS] Piper failed ({synth_ms}ms): {e}')
            raise RuntimeError(f'TTS failed: {e}')

    raise RuntimeError('No TTS engine available')


def get_status():
    """Return a dict describing the current TTS engine status."""
    if kokoro_available:
        return {
            'available': True,
            'engine': 'kokoro',
            'model': 'kokoro-v1.0',
            'voice': _kokoro_voice,
            'voices': sorted(_kokoro_model.get_voices()) if _kokoro_model else [],
        }
    return {
        'available': piper_available,
        'engine': 'piper' if piper_available else None,
        'model': os.path.basename(PIPER_MODEL) if piper_available else None,
    }


def init():
    """Initialize all available TTS engines. Call once at startup."""
    _load_kokoro()
    check_piper()
    if piper_available:
        _load_piper_voice()
