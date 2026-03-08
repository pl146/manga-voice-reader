# Manga Voice Reader

A Chrome extension + local Python server that detects speech bubbles in manga pages, reads them aloud with AI voice, and highlights each bubble as it's spoken. Fully local — no cloud APIs, no paid services, no data leaves your machine.

## How It Works

1. Open any manga page in Chrome
2. Click **Read Page** — the extension screenshots the visible page
3. The screenshot is sent to a local Python server running on your machine
4. **RT-DETR-v2** detects speech bubbles, **PaddleOCR + Florence-2** extract the text
5. **Kokoro TTS** generates natural speech for each bubble
6. The extension plays audio and highlights each bubble in reading order

The entire pipeline runs in ~3-5 seconds per page depending on hardware.

## Features

- **6 AI models** working together — bubble detection, text segmentation, OCR (x2), text classification, TTS
- **54 voices** across 8 languages (English, British, Spanish, French, Hindi, Italian, Japanese, Chinese, Portuguese)
- **RTL/LTR/Vertical** reading direction support for manga, comics, and webtoons
- **Auto-read** — automatically advances and reads the next page when done
- **Smart text cleanup** — handles OCR noise, merged words, SFX filtering, gibberish detection
- **Bubble overlay** — highlights the bubble currently being read
- **Keyboard shortcuts** — Space (play/pause), arrow keys (skip bubble)
- **Speed control** — 0.5x to 2.0x playback speed
- **Page caching** — won't re-process identical screenshots
- **Debug dashboard** — live pipeline visualization at `localhost:5055/debug/frame`
- **Auto-shutdown** — server shuts down after 30 minutes of inactivity
- **CPU throttling** — configurable thread caps to prevent fan noise / CPU spikes

## AI Models Used

| Model | Purpose | Size | Speed |
|-------|---------|------|-------|
| RT-DETR-v2 (ONNX) | Speech bubble detection | 168MB | ~600-900ms |
| Comic Text Segmenter (ONNX) | Text region segmentation | 90MB | ~200ms |
| PaddleOCR PP-OCRv5 | Text recognition (primary) | ~150MB | ~1-2s |
| Florence-2 Large FT | VLM OCR (GPU, secondary) | ~1.5GB VRAM | ~1-2s |
| MangaCNN (ONNX) | Dialogue vs SFX classifier | ~5MB | ~10ms |
| Kokoro v1.0 (ONNX) | Text-to-speech (54 voices) | 337MB | ~850ms/bubble |
| Piper (ONNX) | TTS fallback | 60MB | ~94ms/bubble |

## Processing Pipeline

```
Screenshot → Crop → Bubble Detection (RT-DETR-v2)
                  → PaddleOCR Detection (parallel)
                        ↓
              Bubble Grouping → Pre-SFX Filter
                        ↓
              Multi-OCR (PaddleOCR + Tesseract + Florence-2)
                        ↓
              Dialogue vs SFX Classification (MangaCNN)
                        ↓
              Text Cleanup → Case Normalization → Word Splitting
              → Spell Correction → Manga Corrections → Sentence Case
                        ↓
              Reading Order Sort (RTL/LTR/Vertical)
                        ↓
              TTS Generation (Kokoro) → Audio Playback
```

## Requirements

- **Python 3.11+**
- **GPU recommended** (NVIDIA with CUDA for Florence-2), CPU-only works but slower
- **Tesseract OCR** installed on system (`brew install tesseract` / `apt install tesseract-ocr`)
- **Chrome browser** with Developer Mode enabled

### Hardware Tested On

- **Windows PC**: RTX 3080 10GB, CUDA 12.6 — full pipeline ~3s/page
- **MacBook Air M4**: 16GB RAM, CPU-only — works but slower (~6-8s/page)

## Setup

### 1. Install the Server

```bash
cd server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Additional packages needed (not all in requirements.txt yet):

```bash
pip install kokoro-onnx soundfile wordninja symspellpy piper-tts torch torchvision transformers
```

### 2. Download Models

Create `server/models/` directory and download:

- **Kokoro TTS**: `kokoro-v1.0.onnx` + `voices-v1.0.bin` → `server/models/kokoro/`
- **Piper TTS**: `en_US-lessac-medium.onnx` + `.json` → `server/models/piper/`
- **Bubble Detector**: `detector.onnx` → `server/models/`
- **Text Segmenter**: `comictextdetector.pt.onnx` → `server/models/`

### 3. Start the Server

```bash
cd server
python server.py
```

Server runs on `http://127.0.0.1:5055`.

### 4. Install the Chrome Extension

1. Open `chrome://extensions`
2. Enable **Developer Mode**
3. Click **Load unpacked** → select the `extension/` folder
4. Pin the extension icon in your toolbar

## Usage

1. Go to any manga website (MangaFire, MangaDex, Webtoons, etc.)
2. Click the extension icon → **Open Reader on This Page**
3. A floating panel appears in the bottom-right
4. Click **Read Page** — bubbles are detected, text is read aloud
5. Each bubble highlights as it's spoken

### Controls

| Control | Action |
|---------|--------|
| Read Page | Screenshot + detect + read all bubbles |
| Space | Play / Pause |
| Left/Right Arrow | Skip to previous / next bubble |
| Stop | Stop reading |
| Settings | Voice, speed, overlay mode, reading direction |

### Reading Directions

- **RTL** — Right-to-left (standard manga)
- **LTR** — Left-to-right (Western comics)
- **Vertical** — Top-to-bottom (webtoons/manhwa)

## Configuration

All settings can be overridden via environment variables with `MVR_` prefix:

```bash
# Reduce CPU usage (default: 2 threads per library)
MVR_CPU_THREAD_CAP=2
MVR_ONNX_THREAD_CAP=2
MVR_TORCH_THREAD_CAP=2

# Server port
MVR_PORT=5055

# Idle shutdown timer (minutes)
MVR_IDLE_TIMEOUT=30

# Detection tuning
MVR_MAX_DETECT_WIDTH=1280
MVR_MIN_DET_SCORE=0.10
```

See `server/config.py` for all available options.

## Auto-Start (Windows)

A lightweight launcher service (`launcher.py`) can run on boot via Windows Task Scheduler:

1. Listens on port 5056 with minimal resources
2. When the Chrome extension connects, it starts the full server on port 5055
3. Server auto-shuts down after 30 minutes of inactivity

## Project Structure

```
manga-voice-reader/
├── extension/                  # Chrome extension
│   ├── manifest.json           # Extension manifest (MV3)
│   ├── content.js              # UI panel, overlay, TTS playback, auto-read
│   ├── background.js           # Service worker, HTTP proxy, server detection
│   ├── popup.html / popup.js   # Extension popup
│   ├── styles.css              # Floating panel styles
│   └── icons/                  # Extension icons
├── server/                     # Local Python server
│   ├── server.py               # Flask server, main pipeline
│   ├── config.py               # All tunable parameters
│   ├── bubble_detection.py     # RT-DETR-v2 + merging + sorting
│   ├── ocr_engines.py          # PaddleOCR, Florence-2, Tesseract, MangaCNN
│   ├── text_processing.py      # Cleanup, spell correction, formatting
│   ├── tts_engine.py           # Kokoro + Piper TTS
│   ├── comic_text_segmenter.py # Text mask segmentation
│   ├── launcher.py             # Lightweight boot launcher (Windows)
│   ├── requirements.txt        # Python dependencies
│   └── models/                 # AI model files (not in repo)
└── README.md
```

## Supported Sites

Works on any manga website. Pre-configured content script injection for:

- MangaFire
- MangaDex
- MangaKakalot / MangaNato
- MangaReader
- ReadM
- ManhuaPlus
- Webtoons
- MangaBuddy
- MangaPill
- MangaRead
- AsuraScans / ReaperScans / FlameScans

Can be used on any other site via the extension popup → "Open Reader on This Page".

## Known Issues & Solutions

### CPU Spikes / Fan Noise
The AI models can cause CPU spikes (~50%) during processing. Thread caps were added to reduce peak CPU to ~25%. Adjust via `MVR_CPU_THREAD_CAP`, `MVR_ONNX_THREAD_CAP`, `MVR_TORCH_THREAD_CAP` environment variables.

### RTL Page Navigation
Manga sites using Swiper with RTL layout have inverted page numbering (page 14 is first, page 1 is last). The auto-next system accounts for this — advancing forward means decreasing page numbers in RTL mode.

### Chrome Manifest Apex Domain
`*://*.domain.com/*` in Chrome manifest doesn't match the apex domain (`domain.com`). Both patterns are needed for full coverage.

### Synthetic Keyboard Events
`dispatchEvent(new KeyboardEvent(...))` creates events with `isTrusted=false` that sites ignore. Page navigation uses direct DOM element clicks (`a[data-page]`) instead of simulated keypresses.

### PaddleOCR Windows Crashes
PaddlePaddle's oneDNN and PIR mode can crash on Windows. Disabled via environment variables (`FLAGS_use_mkldnn=0`, `FLAGS_enable_pir_in_executor=0`) before import.

### Mixed Content Errors
HTTPS manga sites block HTTP requests to localhost. All server communication is proxied through the background service worker to avoid mixed content restrictions.

## Cost

Everything is free. No cloud APIs, no subscriptions, no usage limits.

| Component | Cost |
|-----------|------|
| All AI models | Free (open source) |
| Server | Free (runs locally) |
| Chrome extension | Free |
| TTS voices | Free (54 included) |

## License

MIT
