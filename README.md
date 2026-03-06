# Manga Voice Reader

A Chrome extension that automatically reads manga speech bubbles aloud on any manga website. No uploading required — it reads directly from images on the page.

## How It Works

1. You open a manga page in Chrome
2. The extension detects all manga panels on the page (by reading image URLs from the DOM)
3. Each panel image is sent to the backend API
4. Google Cloud Vision extracts all text and returns it in reading order
5. The browser reads the text aloud using Web Speech API (free, built-in)

---

## Setup (Two Parts)

### Part 1 — Deploy the Backend (Vercel)

You need to deploy the backend **once**. It's free.

**Step 1: Get a free Google Vision API key**

1. Go to https://console.cloud.google.com
2. Create a new project
3. Go to APIs & Services → Enable APIs → search "Cloud Vision API" → Enable it
4. Go to APIs & Services → Credentials → Create Credentials → API Key
5. Copy your API key

**Step 2: Deploy to Vercel**

1. Install Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Go into the backend folder:
   ```
   cd backend
   ```

3. Deploy:
   ```
   vercel
   ```
   Follow the prompts. Choose default settings.

4. Set your Google Vision API key as an environment variable:
   ```
   vercel env add GOOGLE_VISION_API_KEY
   ```
   Paste your key when prompted.

5. Redeploy to apply the env variable:
   ```
   vercel --prod
   ```

6. Copy your deployment URL (looks like `https://manga-voice-reader-abc123.vercel.app`)

**Step 3: Update the extension with your backend URL**

Open `extension/content.js` and change line 4:
```js
const API_URL = 'https://manga-voice-reader.vercel.app/api/ocr';
```
Replace it with your actual Vercel URL + `/api/ocr`.

---

### Part 2 — Install the Chrome Extension

1. Open Chrome and go to: `chrome://extensions`
2. Enable **Developer Mode** (toggle in top-right corner)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. The extension is now installed — you'll see the icon in your toolbar

---

## Using the Extension

1. Go to any manga website (MangaDex, Webtoon, MangaPlus, etc.)
2. Click the Manga Reader icon in your toolbar → **Open Reader on This Page**
3. A floating player will appear in the bottom-right corner
4. Click **Scan Page** — it finds all manga panels
5. Click **▶ Play** — it reads every panel aloud, one by one
6. You can also **click any panel** directly to jump to it and read from there

### Player Controls

| Button | Action |
|--------|--------|
| Scan Page | Detects all manga image panels |
| ▶ Play | Start reading from current panel |
| ⏸ Pause | Pause speech |
| ⏮ Prev | Go to previous panel |
| ⏭ Next | Skip to next panel |
| ■ Stop | Stop reading |
| Speed slider | 0.5x to 2x reading speed |
| Voice dropdown | Choose browser voice |

---

## Cost

| Service | Cost |
|---------|------|
| Google Cloud Vision | **Free** up to 1,000 images/month |
| Vercel hosting | **Free** (hobby tier) |
| Web Speech API (TTS) | **Free** (built into Chrome) |
| Chrome extension | **Free** |

For casual reading (a few manga chapters per day), you'll stay well within the free tier.

---

## Supported Sites

Works on any website with manga images, including:
- MangaDex
- Webtoon
- MangaPlus
- MangaKakalot
- MangaFox
- Any site where manga is served as `<img>` tags

---

## Project Structure

```
manga-voice-reader/
├── extension/          # Chrome extension
│   ├── manifest.json
│   ├── content.js      # Core logic: panel detection, OCR, TTS
│   ├── styles.css      # Floating player UI styles
│   ├── popup.html      # Extension popup
│   ├── popup.js
│   ├── background.js
│   └── icons/
├── backend/            # Vercel API
│   ├── api/
│   │   └── ocr.js      # Google Vision OCR endpoint
│   ├── package.json
│   └── vercel.json
└── README.md
```
