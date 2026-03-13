# MangaVoice

<p align="center">
  <img src="extension/icons/icon128.png" alt="MangaVoice" width="80">
</p>

<p align="center">
  <strong>Hear your manga come alive.</strong><br>
  AI-powered speech bubble detection + text-to-speech. 100% local, no cloud.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-macOS-000?logo=apple&logoColor=white" alt="macOS">
  <img src="https://img.shields.io/badge/chrome-extension-4285F4?logo=googlechrome&logoColor=white" alt="Chrome Extension">
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="GPL-3.0">
  <img src="https://img.shields.io/badge/cost-free-brightgreen" alt="Free">
</p>

---

MangaVoice is a Chrome extension + local Python server that detects speech bubbles in manga pages, reads the text aloud, and highlights each bubble as it's spoken. Everything runs on your Mac. No cloud APIs, no subscriptions, no data leaves your machine.

## Demo

### Original Manga Page
![Original Page](docs/original-page.png)

### AI Bubble Detection
![Bubble Detection](docs/detection-demo.png)
*Green boxes = detected speech bubbles. The system finds every bubble, reads the text, and speaks it in reading order.*

### Individual Bubble Crops
| | | |
|:---:|:---:|:---:|
| ![Crop 1](docs/crop-bubble-1.png) | ![Crop 2](docs/crop-bubble-2.png) | ![Crop 3](docs/crop-bubble-3.png) |
| "We need to get rid of these arrows before the kids are carried off!!" | "I doubt Chopper will be back anytime soon..." | "Master Dorri! Master Brog'!!" |

## How It Works

1. Open any manga page in Chrome
2. Click **Play** in the floating panel
3. The extension screenshots the page and sends it to the local server
4. **RT-DETR** detects speech bubbles, **Apple Vision OCR** reads the text
5. **Piper TTS** generates natural speech for each bubble
6. Audio plays while each bubble highlights in reading order
7. Auto-skip advances to the next page when done

The entire pipeline runs in ~1-3 seconds per page on Apple Silicon.

## Features

- **Bubble detection** with RT-DETR neural network
- **Apple Vision OCR** using the macOS Neural Engine for fast, accurate text recognition
- **Piper TTS** for natural-sounding voice output
- **Auto-read** mode that advances through pages automatically
- **Free text detection** for text outside speech bubbles (narration, floating dialogue)
- **Dark bubble support** with automatic inversion for white-on-black text
- **RTL / LTR / Vertical** reading directions for manga, comics, and webtoons
- **Keyboard shortcuts**: Space (play/pause), arrow keys (skip bubble)
- **Speed control** from 0.5x to 2.0x
- **Smart text cleanup** with OCR noise filtering, SFX removal, garbage detection
- **Japanese/CJK filter** to reject misread characters from artwork
- **Starfield UI** with animated background in the extension popup and reader panel
- **Auto-start** via macOS launchd (server starts when you open the extension)
- **Auto-shutdown** after 10 minutes of inactivity

## AI Models

| Model | Purpose | Size |
|-------|---------|------|
| RT-DETR-v2 (ONNX) | Speech bubble detection | 168 MB |
| Apple Vision OCR | Text recognition (macOS native) | Built-in |
| Florence-2 (ONNX, optional) | VLM quality pass | ~320 MB |
| MangaCNN (ONNX) | Dialogue vs SFX classifier | ~5 MB |
| Piper TTS (ONNX) | Text-to-speech | 60 MB |

## Pipeline

```
Screenshot
    |
    v
Bubble Detection (RT-DETR-v2)
    |
    v
Apple Vision OCR (per bubble crop)
    |
    +-- Dark bubble? --> Invert before OCR
    |
    v
Text Cleanup
    |-- SFX filtering (word list + MangaCNN)
    |-- Garbage/noise rejection
    |-- Japanese character filter
    |-- Smart word splitting
    |-- Punctuation + case normalization
    |
    v
Reading Order Sort (RTL/LTR/Vertical)
    |
    v
Piper TTS --> Audio Playback + Bubble Highlight
```

## Requirements

- **macOS** (Apple Silicon recommended, Intel works too)
- **Python 3.11+**
- **Tesseract OCR** (`brew install tesseract`)
- **Chrome** with Developer Mode enabled

## Setup

### 1. Install Dependencies

```bash
cd server
pip install flask flask-cors pillow opencv-python-headless numpy \
            onnxruntime piper-tts wordninja pyobjc-framework-Vision
```

### 2. Download Models

Create `server/models/` and download:

- **RT-DETR bubble detector** --> `server/models/detector.onnx`
  - [comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) (168 MB)
- **Piper TTS** --> `server/models/piper/`
  - [`en_US-lessac-medium.onnx`](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) + `.json` config
- **MangaCNN** (optional) --> `server/models/manga_cnn.onnx`

### 3. Start the Server

```bash
cd server
python server_lite.py
```

Server runs on `http://127.0.0.1:5055`. A launcher service on port 5056 can auto-start it via macOS launchd.

### 4. Install the Chrome Extension

1. Open `chrome://extensions`
2. Enable **Developer Mode**
3. Click **Load unpacked** and select the `extension/` folder
4. Pin the MangaVoice icon in your toolbar

## Usage

1. Go to any manga site (MangaFire, MangaDex, Webtoons, etc.)
2. Click the extension icon, then **Open Reader**
3. A floating panel appears in the bottom-right corner
4. Hit the play button to start reading
5. Each bubble highlights as it's spoken

### Controls

| Control | Action |
|---------|--------|
| Play button | Read all bubbles on the page |
| Space | Play / Pause |
| Left/Right arrow | Skip to previous / next bubble |
| Stop | Stop reading |
| Auto-read | Automatically advance and read next page |
| Free text | Detect text outside speech bubbles |
| Show marks | Toggle bubble highlight overlays |

### Settings

- **Voice**: Piper (local) or System Voice (browser Web Speech API)
- **Speed**: 0.5x to 2.0x
- **Direction**: RTL (manga), LTR (comics), Vertical (webtoons)
- **Overlay**: Reader, Border, Debug, or Hidden

## Supported Sites

Works on any website that displays manga images. Tested on:

MangaFire, MangaDex, MangaKakalot, MangaNato, MangaReader, ReadM, ManhuaPlus, Webtoons, MangaBuddy, MangaPill, AsuraScans, ReaperScans, FlameScans

## Project Structure

```
MangaVoice/
├── extension/              # Chrome extension (Manifest V3)
│   ├── manifest.json
│   ├── content.js          # Reader panel, overlay, TTS playback, auto-read
│   ├── background.js       # Service worker, server communication
│   ├── popup.html/js       # Extension popup with starfield
│   ├── stars.js            # Star animation
│   ├── styles.css          # Panel styles
│   └── icons/
├── server/
│   ├── server_lite.py      # Flask server with full pipeline
│   ├── launcher.py         # Auto-start launcher (port 5056)
│   ├── protected_vocab.txt # Words protected from spell correction
│   ├── requirements_lite.txt
│   └── models/             # AI model files (not in repo)
└── README.md
```

## Privacy

- Zero cloud calls. Everything runs on your Mac.
- No analytics, no telemetry, no tracking.
- No API keys or accounts needed.
- Open source. Every line of code is here.

## Cost

Free. No cloud APIs, no subscriptions, no usage limits.

## License

**GPL-3.0** with additional terms. See [LICENSE](LICENSE).

You're free to use, modify, and share this project, but:
- You must credit the original author (pl146)
- Any modified version must stay open source under GPL-3.0
- Commercial use requires written permission

Copyright 2026 pl146
