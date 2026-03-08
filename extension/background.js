// Background service worker — handles screenshot relay + TTS

const PC_SERVER = 'http://192.168.2.183:5055';
const PC_LAUNCHER = 'http://192.168.2.183:5056';
const MAC_SERVER = 'http://127.0.0.1:5055';
let ACTIVE_SERVER = PC_SERVER;

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
  }, 5 * 60 * 1000); // every 5 minutes
}
function stopHeartbeat() {
  if (_heartbeatInterval) { clearInterval(_heartbeatInterval); _heartbeatInterval = null; }
}

chrome.runtime.onInstalled.addListener(() => {
  console.log('[MVR] Installed.');
});

// Auto-detect which server to use: PC first, then MacBook fallback
async function ensureServerRunning() {
  _connState = 'CONNECTING';

  // 1. Try PC server
  try {
    const r = await fetch(`${PC_SERVER}/health`, { signal: AbortSignal.timeout(1500) });
    if (r.ok) { ACTIVE_SERVER = PC_SERVER; _connState = 'CONNECTED'; _reconnectAttempts = 0; startHeartbeat(); return true; }
  } catch (_) {}

  // 2. Try PC launcher
  try {
    console.log('[MVR] PC server down, trying launcher...');
    const r = await fetch(`${PC_LAUNCHER}/start`, { signal: AbortSignal.timeout(60000) });
    const d = await r.json();
    if (d.ok) { ACTIVE_SERVER = PC_SERVER; _connState = 'CONNECTED'; _reconnectAttempts = 0; startHeartbeat(); return true; }
  } catch (_) {}

  // 3. Fall back to MacBook localhost
  try {
    const r = await fetch(`${MAC_SERVER}/health`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) { ACTIVE_SERVER = MAC_SERVER; _connState = 'CONNECTED'; _reconnectAttempts = 0; startHeartbeat(); console.log('[MVR] PC offline, using MacBook'); return true; }
  } catch (_) {}

  _connState = 'DISCONNECTED';
  stopHeartbeat();
  console.log('[MVR] No server available');
  return false;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'READ_SCREEN') {
    handleReadScreen(request.cropRect, request.dpr || 2, request.pageTitle || '', request.pageUrl || '', request.readingDirection || 'rtl')
      .then(r => sendResponse(r))
      .catch(e => sendResponse({ ok: false, error: e.message }));
    return true;
  }
  if (request.type === 'GET_SERVER') {
    sendResponse({ server: ACTIVE_SERVER });
    return;
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
        // Send as base64 since sendResponse can't transfer blobs
        const bytes = new Uint8Array(buf);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        sendResponse({ ok: true, audio: btoa(binary), type: res.headers.get('content-type') || 'audio/wav' });
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
  if (request.type === 'MVR_SHUTDOWN') {
    stopHeartbeat();
    _connState = 'DISCONNECTED';
    fetch(`${ACTIVE_SERVER}/shutdown`, { method: 'POST' }).catch(() => {});
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
  if (request.type === 'MVR_DUMP_MAIN') {
    chrome.scripting.executeScript({
      target: { tabId: sender.tab.id },
      world: 'MAIN',
      func: () => {
        const info = [];
        info.push('=== MAIN WORLD DUMP ===');
        info.push('jQuery: ' + (typeof window.jQuery !== 'undefined'));
        info.push('$: ' + (typeof window.$ !== 'undefined'));
        if (window.jQuery) {
          const events = window.jQuery._data(document, 'events');
          if (events) info.push('Document events: ' + Object.keys(events).join(', '));
          const body = document.body;
          const bodyEvents = window.jQuery._data(body, 'events');
          if (bodyEvents) {
            info.push('Body events: ' + Object.keys(bodyEvents).join(', '));
            for (const [type, handlers] of Object.entries(bodyEvents)) {
              handlers.forEach((h, i) => {
                info.push('  ' + type + '[' + i + ']: selector="' + (h.selector || '') + '" handler=' + (h.handler?.toString()?.substring(0, 200) || 'n/a'));
              });
            }
          }
        }
        info.push('');
        info.push('=== KEYBOARD LISTENER TEST ===');
        let keyHandled = false;
        const testHandler = (e) => { keyHandled = true; info.push('KeyDown caught: key=' + e.key); };
        document.addEventListener('keydown', testHandler);
        document.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', code: 'ArrowRight', keyCode: 39, bubbles: true}));
        document.removeEventListener('keydown', testHandler);
        info.push('Keyboard event reached handler: ' + keyHandled);
        console.log('[MVR MAIN DUMP]\n' + info.join('\n'));
      },
    });
  }
  if (request.type === 'MVR_SAVE_DUMP') {
    const blob = new Blob([request.text], { type: 'text/plain' });
    const reader = new FileReader();
    reader.onloadend = () => {
      chrome.downloads.download({
        url: reader.result,
        filename: 'mvr-dump.txt',
        saveAs: false,
      });
    };
    reader.readAsDataURL(blob);
  }
  if (request.type === 'MVR_CLICK_PAGE') {
    chrome.scripting.executeScript({
      target: { tabId: sender.tab.id },
      world: 'MAIN',
      func: (readDir) => {
        const getPage = () => {
          const s = document.querySelector('span.current-page, b.current-page');
          return s ? parseInt(s.textContent.trim()) : 0;
        };
        const before = getPage();
        const isRTL = readDir === 'rtl';
        let method = 'none';

        // 1. Try data-page link (mangafire.to — most reliable)
        const activePage = document.querySelector('a[data-page].active');
        if (activePage) {
          const cur = parseInt(activePage.dataset.page);
          // RTL: forward = lower page number. LTR: forward = higher.
          const target = isRTL ? cur - 1 : cur + 1;
          const link = document.querySelector(`a[data-page="${target}"]`);
          if (link) {
            link.click();
            method = `data-page ${cur}->${target}`;
          }
        }

        // 2. If no data-page, try Swiper API
        if (method === 'none') {
          const swiperEls = document.querySelectorAll('.swiper-container, .swiper, [class*="swiper"]');
          for (const el of swiperEls) {
            if (el.swiper) {
              // Try slideNext first, then check if it went the right direction
              el.swiper.slideNext();
              method = 'swiper-slideNext';
              // Validate after a tick
              setTimeout(() => {
                const after = getPage();
                const wentForward = isRTL ? (after < before) : (after > before);
                if (!wentForward && before > 0) {
                  // Wrong direction — try slidePrev instead
                  el.swiper.slidePrev();
                  el.swiper.slidePrev(); // undo slideNext + go forward
                  console.log('[MVR] slideNext went wrong direction, used slidePrev');
                }
              }, 400);
              break;
            }
          }
        }

        console.log(`[MVR] MVR_CLICK_PAGE: method=${method}, before=${before}, dir=${readDir}`);
      },
      args: [request.direction || 'rtl'],
    });
  }
});


// ─── Screenshot + server relay ────────────────────────────────────────────

async function handleReadScreen(cropRect, dpr, pageTitle, pageUrl, readingDirection) {
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
    res = await fetch(`${ACTIVE_SERVER}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: dataUrl,
        cropRect: cropRect || null,
        dpr: dpr,
        pageTitle: pageTitle || '',
        pageUrl: pageUrl || '',
        readingDirection: readingDirection || 'rtl',
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
  return {
    ok: true,
    bubbles: data.bubbles || [],
    actualCropTop: data.actualCropTop || 0,
    actualCropLeft: data.actualCropLeft || 0,
    timing: data.timing || {},
    actionPage: (data.bubbles || []).length === 0,
  };
}
