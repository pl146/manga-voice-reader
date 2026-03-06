// Manga Voice Reader - OCR API (Vercel Serverless Function)
// Uses Google Cloud Vision to extract speech bubble text from manga images

const https = require('https');
const http = require('http');
const { URL } = require('url');

const GOOGLE_VISION_KEY = (process.env.GOOGLE_VISION_API_KEY || '').trim();

// ─── Fetch image as base64 ────────────────────────────────────────────────────

function fetchImageAsBase64(imageUrl) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(imageUrl);
    const protocol = parsedUrl.protocol === 'https:' ? https : http;

    const options = {
      hostname: parsedUrl.hostname,
      path: parsedUrl.pathname + parsedUrl.search,
      method: 'GET',
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; MangaReader/1.0)',
        'Referer': parsedUrl.origin,
      },
      timeout: 15000,
    };

    const req = protocol.request(options, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        return fetchImageAsBase64(res.headers.location).then(resolve).catch(reject);
      }

      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const buffer = Buffer.concat(chunks);
        resolve(buffer.toString('base64'));
      });
    });

    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Image fetch timeout')); });
    req.end();
  });
}

// ─── Google Vision OCR ────────────────────────────────────────────────────────

async function runGoogleVision(base64Image) {
  const body = JSON.stringify({
    requests: [{
      image: { content: base64Image },
      features: [{ type: 'DOCUMENT_TEXT_DETECTION', maxResults: 1 }],
      imageContext: { languageHints: ['en'] },
    }]
  });

  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'vision.googleapis.com',
      path: `/v1/images:annotate?key=${GOOGLE_VISION_KEY}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error('Failed to parse Vision API response'));
        }
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ─── Text Ordering Logic ──────────────────────────────────────────────────────

/**
 * Groups word-level annotations into logical speech bubbles
 * and returns them sorted in manga reading order (left→right, top→bottom)
 *
 * Google Vision gives us bounding polygons for each block of text.
 * We use those to sort and group into coherent speech bubbles.
 */
function extractOrderedTexts(visionResponse) {
  const response = visionResponse?.responses?.[0];

  // Fallback: use simple text annotation if fullTextAnnotation fails
  const annotation = response?.fullTextAnnotation;
  if (!annotation) {
    const simple = response?.textAnnotations?.[0];
    if (simple) {
      const verts = simple.boundingPoly?.vertices || [];
      const xs = verts.map(v => v.x || 0);
      const ys = verts.map(v => v.y || 0);
      return [{ text: simple.description.trim(), x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) }];
    }
    return [];
  }

  const blocks = annotation.pages?.[0]?.blocks || [];
  if (blocks.length === 0) return [];

  // Extract each text block with its bounding box
  const textBlocks = blocks.map(block => {
    const vertices = block.boundingBox?.vertices || [];
    const xs = vertices.map(v => v.x || 0);
    const ys = vertices.map(v => v.y || 0);
    const x = Math.min(...xs);
    const y = Math.min(...ys);
    const width = Math.max(...xs) - x;
    const height = Math.max(...ys) - y;

    // Reconstruct text from paragraphs → words → symbols
    const text = (block.paragraphs || []).map(para =>
      (para.words || []).map(word =>
        (word.symbols || []).map(s => s.text || '').join('')
      ).join(' ')
    ).join('\n').trim();

    return { text, x, y, width, height, centerX: x + width / 2, centerY: y + height / 2 };
  }).filter(b => b.text.length > 1); // skip single characters

  // Sort by reading order: row by row (group blocks within ~100px vertically)
  textBlocks.sort((a, b) => {
    const rowThreshold = 80;
    if (Math.abs(a.y - b.y) > rowThreshold) return a.y - b.y;
    return a.x - b.x;
  });

  // Merge nearby blocks that are part of the same speech bubble
  const merged = mergeNearbyBlocks(textBlocks);

  return merged.filter(b => b.text.length > 0);
}

/**
 * Merges text blocks that are spatially close (likely same speech bubble)
 */
function mergeNearbyBlocks(blocks) {
  if (blocks.length === 0) return [];

  const MERGE_DISTANCE = 60; // pixels
  const groups = [];
  const used = new Set();

  for (let i = 0; i < blocks.length; i++) {
    if (used.has(i)) continue;

    const group = [blocks[i]];
    used.add(i);

    for (let j = i + 1; j < blocks.length; j++) {
      if (used.has(j)) continue;

      const a = blocks[i];
      const b = blocks[j];
      const dist = Math.sqrt(Math.pow(a.centerX - b.centerX, 2) + Math.pow(a.centerY - b.centerY, 2));

      if (dist < MERGE_DISTANCE) {
        group.push(blocks[j]);
        used.add(j);
      }
    }

    // Sort group members top-to-bottom
    group.sort((a, b) => a.y - b.y);
    groups.push({
      text: group.map(g => g.text).join(' '),
      x: Math.min(...group.map(g => g.x)),
      y: Math.min(...group.map(g => g.y)),
      centerX: group.reduce((s, g) => s + g.centerX, 0) / group.length,
      centerY: group.reduce((s, g) => s + g.centerY, 0) / group.length,
    });
  }

  // Final sort by position
  groups.sort((a, b) => {
    const rowThreshold = 80;
    if (Math.abs(a.y - b.y) > rowThreshold) return a.y - b.y;
    return a.x - b.x;
  });

  return groups;
}

// ─── CORS Headers ─────────────────────────────────────────────────────────────

function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

// ─── Main Handler ─────────────────────────────────────────────────────────────

module.exports = async function handler(req, res) {
  setCorsHeaders(res);

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { imageUrl, imageBase64 } = req.body || {};

  if (!imageUrl && !imageBase64) {
    return res.status(400).json({ error: 'imageUrl or imageBase64 is required' });
  }

  if (!GOOGLE_VISION_KEY) {
    return res.status(500).json({ error: 'Google Vision API key not configured' });
  }

  try {
    // 1. Get image as base64 (prefer direct base64 from extension)
    const base64 = imageBase64 || await fetchImageAsBase64(imageUrl);

    // 2. Run OCR
    const visionResult = await runGoogleVision(base64);

    // Check for Vision API errors
    const visionError = visionResult?.responses?.[0]?.error;
    if (visionError) {
      throw new Error(`Vision API: ${visionError.message}`);
    }

    // 3. Extract and order texts with positions
    const blocks = extractOrderedTexts(visionResult);
    const texts = blocks.map(b => b.text);

    return res.status(200).json({ texts, blocks, count: texts.length });

  } catch (err) {
    console.error('[OCR Error]', err.message);
    return res.status(500).json({ error: err.message });
  }
};
