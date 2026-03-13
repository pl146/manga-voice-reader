// Background service worker — handles screenshot relay + TTS

const MAC_SERVER = 'http://127.0.0.1:5055';
const MAC_LAUNCHER = 'http://127.0.0.1:5056';
let ACTIVE_SERVER = MAC_SERVER;

// Connection state machine: DISCONNECTED → CONNECTING → CONNECTED → RECONNECTING
let _connState = 'DISCONNECTED';
let _reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;

// Heartbeat: keep server alive while extension is open
let _heartbeatInterval = null;
function startHeartbeat() {
  if (_heartbeatInterval) return;
  _heartbeatInterval = setInterval(() => {
    if (_connState === 'CONNECTED') {
      fetch(`${ACTIVE_SERVER}/heartbeat`, { method: 'POST', signal: AbortSignal.timeout(3000) })
        .catch(() => {
          _connState = 'RECONNECTING';
          _reconnectAttempts++;
          if (_reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            _connState = 'DISCONNECTED';
            stopHeartbeat();
            console.log('[MVR] Lost connection after', _reconnectAttempts, 'failed heartbeats');
          } else {
            ensureServerRunning().catch(() => {});
          }
        });
    }
  }, 2 * 60 * 1000); // every 2 minutes
}
function stopHeartbeat() {
  if (_heartbeatInterval) { clearInterval(_heartbeatInterval); _heartbeatInterval = null; }
}

chrome.runtime.onInstalled.addListener(() => {
  console.log('[MVR] Installed.');
});

// Auto-detect: Mac server first, Mac launcher second
async function ensureServerRunning() {
  _connState = 'CONNECTING';

  // 1. Try Mac server (instant if already running)
  try {
    const r = await fetch(`${MAC_SERVER}/health`, { signal: AbortSignal.timeout(1500) });
    if (r.ok) { ACTIVE_SERVER = MAC_SERVER; _connState = 'CONNECTED'; _reconnectAttempts = 0; startHeartbeat(); console.log('[MVR] Using Mac server'); return true; }
  } catch (_) {}

  // 2. Try Mac launcher to auto-start server
  try {
    console.log('[MVR] Trying Mac launcher...');
    const r = await fetch(`${MAC_LAUNCHER}/start`, { signal: AbortSignal.timeout(20000) });
    const d = await r.json();
    if (d.ok) { ACTIVE_SERVER = MAC_SERVER; _connState = 'CONNECTED'; _reconnectAttempts = 0; startHeartbeat(); console.log('[MVR] Mac started via launcher'); return true; }
  } catch (_) {}

  _connState = 'DISCONNECTED';
  stopHeartbeat();
  console.log('[MVR] No server available');
  return false;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'READ_SCREEN') {
    handleReadScreen(request.cropRect, request.dpr || 2, request.pageTitle || '', request.pageUrl || '', request.readingDirection || 'rtl', request.voice || '_piper', request.speed || 0.9, request.freeText || false)
      .then(r => sendResponse(r))
      .catch(e => sendResponse({ ok: false, error: e.message }));
    return true;
  }
  if (request.type === 'GET_SERVER') {
    sendResponse({ server: ACTIVE_SERVER });
    return;
  }
  if (request.type === 'MVR_GET_STATE') {
    sendResponse({ state: _connState, server: ACTIVE_SERVER });
    return;
  }
  if (request.type === 'MVR_FETCH_IMAGE') {
    (async () => {
      try {
        const serverUp = await ensureServerRunning();
        if (!serverUp) { sendResponse({ ok: false, error: 'SERVER_NOT_RUNNING' }); return; }
        // Fetch the image from the manga site
        const imgRes = await fetch(request.url);
        if (!imgRes.ok) { sendResponse({ ok: false, error: 'Image fetch failed: ' + imgRes.status }); return; }
        const blob = await imgRes.blob();
        // Convert to base64 data URL
        const reader = new FileReader();
        const dataUrl = await new Promise((resolve) => {
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(blob);
        });
        // Send to server for processing (retry if models still loading)
        const res = await fetchWithRetry(`${ACTIVE_SERVER}/process`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: dataUrl,
            cropRect: null,
            dpr: 1,
            pageTitle: request.pageTitle || '',
            pageUrl: request.pageUrl || '',
            readingDirection: request.readingDirection || 'rtl',
          }),
        });
        if (!res.ok) { sendResponse({ ok: false, error: 'Server error ' + res.status }); return; }
        const data = await res.json();
        sendResponse({
          ok: true,
          bubbles: data.bubbles || [],
          timing: data.timing || {},
        });
      } catch (e) { sendResponse({ ok: false, error: e.message }); }
    })();
    return true;
  }
  if (request.type === 'MVR_PROCESS_IMAGE') {
    // Direct data URL image — send straight to server for processing
    (async () => {
      try {
        const serverUp = await ensureServerRunning();
        if (!serverUp) { sendResponse({ ok: false, error: 'SERVER_NOT_RUNNING' }); return; }
        const res = await fetchWithRetry(`${ACTIVE_SERVER}/process`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: request.image,
            cropRect: null,
            dpr: 1,
            pageTitle: request.pageTitle || '',
            pageUrl: request.pageUrl || '',
            readingDirection: request.readingDirection || 'rtl',
          }),
        });
        if (!res.ok) { sendResponse({ ok: false, error: 'Server error ' + res.status }); return; }
        const data = await res.json();
        sendResponse({ ok: true, bubbles: data.bubbles || [], timing: data.timing || {} });
      } catch (e) { sendResponse({ ok: false, error: e.message }); }
    })();
    return true;
  }
  if (request.type === 'MVR_ENSURE_SERVER') {
    ensureServerRunning()
      .then(ok => sendResponse({ ok, server: ACTIVE_SERVER }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
  if (request.type === 'MVR_TTS') {
    (async () => {
      try {
        const res = await fetch(`${ACTIVE_SERVER}/tts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: request.text, voice: request.voice, speed: request.speed }),
        });
        if (!res.ok) { sendResponse({ ok: false }); return; }
        const buf = await res.arrayBuffer();
        const contentType = res.headers.get('content-type') || 'audio/wav';
        // MV3 sendResponse uses JSON — must base64 encode binary data
        const bytes = new Uint8Array(buf);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        const b64 = btoa(binary);
        sendResponse({ ok: true, audioB64: b64, type: contentType });
      } catch (e) { sendResponse({ ok: false }); }
    })();
    return true;
  }
  if (request.type === 'MVR_TTS_STATUS') {
    (async () => {
      try {
        const res = await fetch(`${ACTIVE_SERVER}/tts/status`, { signal: AbortSignal.timeout(2000) });
        if (!res.ok) { sendResponse({ ok: false }); return; }
        const data = await res.json();
        sendResponse({ ok: true, ...data });
      } catch (e) { sendResponse({ ok: false }); }
    })();
    return true;
  }
  if (request.type === 'MVR_LOG') {
    fetch(`${ACTIVE_SERVER}/ext-log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ msg: request.msg }),
    }).catch(() => {});
    return; // fire-and-forget
  }
  if (request.type === 'MVR_QUALITY_POLL') {
    (async () => {
      try {
        const res = await fetch(`${ACTIVE_SERVER}/process/quality?id=${encodeURIComponent(request.requestId)}`, {
          signal: AbortSignal.timeout(3000),
        });
        if (!res.ok) { sendResponse({ ready: false }); return; }
        const data = await res.json();
        sendResponse(data);
      } catch (e) { sendResponse({ ready: false }); }
    })();
    return true;
  }
  if (request.type === 'MVR_SHUTDOWN') {
    stopHeartbeat();
    _connState = 'DISCONNECTED';
    return;
  }
  if (request.type === 'MVR_HEARTBEAT') {
    fetch(`${ACTIVE_SERVER}/heartbeat`, { method: 'POST', signal: AbortSignal.timeout(3000) }).catch(() => {});
    return;
  }
  if (request.type === 'MVR_WARM_VOICE') {
    fetch(`${ACTIVE_SERVER}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: 'Warm up.', voice: request.voice, speed: 1.0 }),
    }).catch(() => {});
    return;
  }
});


// ─── Retry helper for 503 (models still loading) ─────────────────────────

async function fetchWithRetry(url, options, maxRetries = 20, delayMs = 2000) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const res = await fetch(url, options);
    if (res.status === 503 && attempt < maxRetries) {
      console.log(`[MVR] Server loading models, retry ${attempt + 1}/${maxRetries} in ${delayMs/1000}s...`);
      await new Promise(r => setTimeout(r, delayMs));
      continue;
    }
    return res;
  }
}

// ─── Screenshot + server relay ────────────────────────────────────────────

async function handleReadScreen(cropRect, dpr, pageTitle, pageUrl, readingDirection, voice, speed, freeText) {
  let dataUrl;
  try {
    dataUrl = await chrome.tabs.captureVisibleTab(null, { format: 'png' });
  } catch (err) {
    throw new Error('Screenshot failed: ' + err.message);
  }

  const serverUp = await ensureServerRunning();
  if (!serverUp) throw new Error('SERVER_NOT_RUNNING');

  let res;
  try {
    res = await fetchWithRetry(`${ACTIVE_SERVER}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: dataUrl,
        cropRect: cropRect || null,
        dpr: dpr,
        pageTitle: pageTitle || '',
        pageUrl: pageUrl || '',
        readingDirection: readingDirection || 'rtl',
        voice: voice || '_piper',
        speed: speed || 0.9,
      }),
    });
  } catch (err) {
    throw new Error('SERVER_NOT_RUNNING');
  }

  if (!res.ok) {
    const errText = await res.text();
    throw new Error('Server error ' + res.status + ': ' + errText);
  }

  const data = await res.json();
  const bubbles = data.bubbles || [];

  // Free text pass: detect text outside bubbles (separate from bubble pipeline)
  // Check storage directly in case message flag didn't come through
  let freeTexts = [];
  let ftEnabled = freeText;
  try {
    const stored = await chrome.storage.local.get('mvrFreeText');
    if (stored.mvrFreeText) ftEnabled = true;
  } catch (_) {}
  console.log('[MVR-bg] freeText flag:', freeText, 'storage:', ftEnabled, 'bubbles:', bubbles.length);
  if (ftEnabled && bubbles.length === 0) {
    try {
      const ftRes = await fetch(`${ACTIVE_SERVER}/process/freetext`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataUrl,
          cropRect: cropRect || null,
          dpr: dpr,
          readingDirection: readingDirection || 'rtl',
          bubbleBoxes: bubbles.map(b => ({
            left: b.left, top: b.top, width: b.width, height: b.height,
          })),
        }),
      });
      if (ftRes.ok) {
        const ftData = await ftRes.json();
        freeTexts = ftData.freeTexts || [];
        console.log('[MVR] Free text pass:', freeTexts.length, 'regions found');
      }
    } catch (e) {
      console.log('[MVR] Free text pass failed (non-fatal):', e.message);
    }
  }

  return {
    ok: true,
    bubbles: bubbles,
    freeTexts: freeTexts,
    actualCropTop: data.actualCropTop || 0,
    actualCropLeft: data.actualCropLeft || 0,
    timing: data.timing || {},
    requestId: data.requestId || null,
    qualityPending: data.qualityPending || false,
    audioId: data.audioId || null,
    actionPage: bubbles.length === 0 && freeTexts.length === 0,
  };
}
