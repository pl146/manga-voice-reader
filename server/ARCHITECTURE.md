# Server Architecture

The server is structured as a Flask app (`server.py`) that imports all processing
logic from extracted modules. Each module is self-contained and imports configuration
from `config.py`.

## Module Structure

```
server/
  config.py              # Centralized configuration — all thresholds, paths, feature flags
  server.py              # Flask app — routes, pipeline orchestration, rate limiting, lifecycle
  text_processing.py     # Text cleanup, reconstruction, word splitting, formatting
  bubble_detection.py    # Bubble detector, box merging, SFX filter, reading order
  ocr_engines.py         # Florence-2, PaddleOCR, Tesseract, super-resolution
  tts_engine.py          # Kokoro TTS (primary), Piper TTS (fallback)
  comic_text_segmenter.py # Text mask generation for bubble grouping
  manga_corrections.json # 46 manga-specific OCR corrections
  protected_vocab.txt    # Words that should never be spell-corrected (names, honorifics)
  showcase.html          # Live dashboard
  launcher.py            # Lightweight auto-start service
```

## Current State

`server.py` imports all processing functions from the extracted modules and orchestrates
the pipeline. It contains only Flask routes, rate limiting, lifecycle management, and the
main `/process` pipeline logic.

**server.py imports from:**
- `config` — all thresholds and settings
- `ocr_engines` — detector loading, Florence-2 OCR, CNN classifier, super-resolution
- `text_processing` — text cleanup, reconstruction, formatting, manga corrections
- `bubble_detection` — detection, merging, filtering, grouping, sorting, debug frames
- `tts_engine` — speech synthesis (Kokoro + Piper), status, initialization
- `comic_text_segmenter` — text mask generation

**Module dependencies:**
```
config.py (all tunable parameters, env-var overridable)
  ^
  ├── text_processing.py (cleanup, reconstruction, formatting)
  │     └── config.py (protected vocab path)
  ├── ocr_engines.py (Florence-2, PaddleOCR, Tesseract)
  │     ├── text_processing.py (SymSpell, wordninja)
  │     └── config.py
  ├── bubble_detection.py (RT-DETR-v2, box merging, SFX filter, grouping)
  │     └── config.py
  └── tts_engine.py (Kokoro, Piper)
        └── config.py
```

## Processing Pipeline

The main `/process` endpoint orchestrates all stages:

1. **Validate** — Input validation (image size, DPR, reading direction), rate limiting
2. **Decode** — Base64 screenshot from Chrome extension
3. **Crop** — Extract manga area using extension-provided coordinates
4. **Detect** — Run RT-DETR-v2 bubble detection + PaddleOCR text detection in parallel
5. **Filter** — Pre-SFX geometric filter removes obvious sound effects
6. **Group** — Merge text regions into speech bubbles using text mask
7. **OCR** — Florence-2 (primary) + PaddleOCR (secondary) per bubble
8. **Classify** — MangaCNN junk rejection (93% confidence gate)
9. **Reconstruct** — Fix OCR errors, split merged words, protected vocab, spell-check
10. **Format** — Sentence case, speech formatting, manga corrections
11. **Quality gates** — Gibberish (85%), symbol junk (6), dialogue score (≥-4)
12. **Sort** — Reading order (RTL for manga, LTR for comics)
13. **Return** — Bubble coordinates + cleaned text to extension

## Safety Features

- **Rate limiting** — Semaphore-based: 2 concurrent /process, 5 concurrent /tts
- **Input validation** — Image size, DPR range, reading direction whitelist
- **Graceful shutdown** — Waits for in-flight /process and /tts requests before exit
- **Heartbeat** — Extension sends keep-alive to prevent premature idle shutdown
- **Debug frame rotation** — Auto-cleanup: keeps latest 20 frames, deletes >1hr old
- **Log rotation** — Extension log rotated at 5MB
