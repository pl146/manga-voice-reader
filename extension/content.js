// MangaVoice — Content Script

function mvrLog(msg) {
  console.log('[MVR] ' + msg);
  chrome.runtime.sendMessage({ type: 'MVR_LOG', msg }).catch(() => {});
}

let speaking = false;
let paused = false;
let panelCreated = false;
let currentAudio = null;
let autoReadEnabled = false;
let lastPageHash = null;
let lastReadPageNum = 0;  // Track page number to detect Swiper wrap-around
let currentBubbleIndex = -1;
let totalBubbles = 0;
let skipDirection = 0; // +1 skip forward, -1 skip back, 0 none
let availableVoices = [];
let selectedVoice = '_piper'; // default: Piper TTS
let _prefetchInvalid = false;
let freeTextEnabled = false;
const LOCAL_SERVER = 'http://127.0.0.1:5055';
let SERVER = LOCAL_SERVER;

// Pre-load browser voices (they load async in Chrome)
let _browserVoices = [];
function _loadBrowserVoices() {
  _browserVoices = window.speechSynthesis.getVoices();
  mvrLog('Browser voices loaded: ' + _browserVoices.length + ' — ' +
    _browserVoices.filter(v => v.lang.startsWith('en')).map(v => v.name).join(', '));
}
_loadBrowserVoices();
window.speechSynthesis.onvoiceschanged = _loadBrowserVoices;

async function _ensureServer() {
  // Route through background.js service worker (single source of truth)
  // Retry once if first attempt fails (server may still be starting)
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await chrome.runtime.sendMessage({ type: 'MVR_ENSURE_SERVER' });
      if (res && res.ok) {
        SERVER = res.server;
        mvrLog('Connected to ' + SERVER);
        const statusEl = document.getElementById('mvr-status');
        if (statusEl) statusEl.textContent = 'Connected';
        return true;
      }
    } catch (_) {}
    if (attempt === 0) {
      mvrLog('Server not found, retrying in 2s...');
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  mvrLog('No server available');
  return false;
}

function _shutdownServer() {
  stopSpeaking();
  // Actually shut down the server to free CPU/GPU
  fetch(SERVER + '/shutdown', { method: 'POST' }).catch(() => {});
  mvrLog('Shutdown request sent to server');
}


// Overlay modes: 'reader' (default), 'border', 'debug', 'hidden'
let overlayMode = 'reader';

// ─── Activation gate ─────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'MVR_ACTIVATE') {
    // Auto-start server if needed
    _ensureServer().then((connected) => {
      if (!panelCreated) {
        createUI();
        panelCreated = true;
      } else {
        const panel = document.getElementById('mvr');
        if (panel) panel.style.display = '';
      }
      if (connected) {
        // Warm up Kokoro voice (skip if using browser TTS)
        if (selectedVoice !== '_browser') {
          chrome.runtime.sendMessage({ type: 'MVR_WARM_VOICE', voice: selectedVoice }).catch(() => {});
        }
      } else {
        // Show the panel anyway so user sees the status, but warn them
        const statusEl = document.getElementById('mvr-status');
        if (statusEl) statusEl.textContent = 'No server. Run: python3 ~/MangaVoice/server/server_lite.py';
      }
    });
    sendResponse({ ok: true });
  }
  if (msg.type === 'MVR_DUMP_BUTTONS') {
    try {
      var lines = [];
      lines.push('URL: ' + location.href);
      lines.push('Viewport: ' + window.innerWidth + 'x' + window.innerHeight);
      lines.push('');

      lines.push('=== IMAGES ===');
      document.querySelectorAll('img').forEach(function(img, i) {
        var r = img.getBoundingClientRect();
        if (r.width < 30 && r.height < 30) return;
        lines.push('IMG ' + i + ': ' + (img.src||'').substring(0,150));
        lines.push('  class=' + img.className + ' id=' + img.id + ' ' + Math.round(r.width) + 'x' + Math.round(r.height) + ' at (' + Math.round(r.left) + ',' + Math.round(r.top) + ')');
        lines.push('  parent=<' + (img.parentElement?.tagName||'') + ' class=' + (img.parentElement?.className||'') + '>');
      });
      lines.push('');

      lines.push('=== PAGE CONTROLS ===');
      ['span.current-page', '.page-toggler', 'a[data-page]', 'a[data-page].active'].forEach(function(sel) {
        var els = document.querySelectorAll(sel);
        if (els.length === 0) return;
        lines.push(sel + ': ' + els.length + ' found');
        els.forEach(function(el, i) {
          if (i > 3) return;
          var r = el.getBoundingClientRect();
          lines.push('  <' + el.tagName + ' class=' + el.className + '> text=' + el.textContent.trim().substring(0,40) + ' pos=(' + Math.round(r.left) + ',' + Math.round(r.top) + ') ' + Math.round(r.width) + 'x' + Math.round(r.height));
          lines.push('  html=' + el.outerHTML.substring(0,300));
        });
      });
      lines.push('');

      lines.push('=== SCRIPTS ===');
      document.querySelectorAll('script:not([src])').forEach(function(s, i) {
        var c = s.textContent.trim();
        if (c.length > 0 && (c.indexOf('page') !== -1 || c.indexOf('chapter') !== -1)) {
          lines.push('SCRIPT ' + i + ': ' + c.substring(0, 800));
          lines.push('---');
        }
      });

      // Also dump via MAIN world execution to find event handlers
      var text = lines.join('\n');

      // Get extra info from MAIN world
      chrome.runtime.sendMessage({ type: 'MVR_DUMP_MAIN' });

      chrome.runtime.sendMessage({ type: 'MVR_SAVE_DUMP', text: text });
      sendResponse({ ok: true });
    } catch (err) {
      sendResponse({ ok: false, error: err.message });
    }
    return true;
  }
});

// ─── UI ──────────────────────────────────────────────────────────────────────

function createUI() {
  if (document.getElementById('mvr')) return;

  const d = document.createElement('div');
  d.id = 'mvr';
  d.innerHTML = `
    <canvas id="mvr-stars"></canvas>
    <div id="mvr-top">
      <div id="mvr-title">
        <span class="mvr-dot"></span>
        <div>
          <div class="mvr-title-main">MangaVoice</div>
          <div id="mvr-status" class="mvr-status-label">Ready</div>
        </div>
      </div>
      <div id="mvr-top-actions">
        <button id="mvr-controls-toggle" aria-label="Toggle settings" aria-expanded="false">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6">
            <path d="M12 15.5a3.5 3.5 0 100-7 3.5 3.5 0 000 7z"/>
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09a1.65 1.65 0 00-1-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09a1.65 1.65 0 001.51-1 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9c.01.69.35 1.32.9 1.71z"/>
          </svg>
          <span class="mvr-sr-only">Settings</span>
        </button>
        <button id="mvr-x" aria-label="Close MangaVoice">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 6L18 18"/>
            <path d="M18 6L6 18"/>
          </svg>
        </button>
      </div>
    </div>
    <div id="mvr-body">
      <div id="mvr-progress-wrap" style="display:none;">
        <div id="mvr-progress-bar"></div>
      </div>
      <div class="mvr-text-row">
        <div id="mvr-text" role="log" aria-live="polite"></div>
      </div>
      <div id="mvr-url-bar"></div>
      <div class="mvr-actions">
        <div id="mvr-row-idle">
          <button id="mvr-go" class="mvr-icon-pill" aria-label="Read this page">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z"/>
            </svg>
          </button>
        </div>
        <div id="mvr-row-reading" style="display:none;">
          <button id="mvr-pause" class="mvr-icon-pill" aria-label="Pause reading">
            <svg class="mvr-icon-pause" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 5h3v14H7zM14 5h3v14h-3z"/>
            </svg>
            <svg class="mvr-icon-play" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z"/>
            </svg>
          </button>
          <button id="mvr-stop" class="mvr-icon-pill" aria-label="Stop reading">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <rect x="7" y="7" width="10" height="10" rx="2"/>
            </svg>
          </button>
        </div>
      </div>
      <div id="mvr-controls-wrapper">
        <div id="mvr-controls" class="collapsed">
          <label>
            <span class="mvr-label-title">Voice</span>
            <select id="mvr-voice"><option value="_piper">Piper</option><option value="_browser">System Voice</option></select>
          </label>
          <label>
            <span class="mvr-label-title">Speed</span>
            <div class="mvr-speed-row">
              <input type="range" id="mvr-speed" min="0.5" max="2" step="0.1" value="0.85">
              <span id="mvr-sv">0.85x</span>
            </div>
          </label>
          <label>
            <span class="mvr-label-title">Overlay</span>
            <select id="mvr-overlay-mode">
              <option value="reader">Reader</option>
              <option value="border">Border</option>
              <option value="debug">Debug</option>
              <option value="hidden">Hidden</option>
            </select>
          </label>
          <label>
            <span class="mvr-label-title">Read →</span>
            <select id="mvr-direction">
              <option value="rtl">RTL</option>
              <option value="ltr">LTR</option>
              <option value="vertical">Vertical</option>
            </select>
          </label>
          <label id="mvr-auto-label" class="mvr-checkbox">
            <input type="checkbox" id="mvr-auto">
            <span>Auto-read</span>
          </label>
          <label class="mvr-checkbox">
            <input type="checkbox" id="mvr-show-marks" checked>
            <span>Show marks</span>
          </label>
          <label class="mvr-checkbox">
            <input type="checkbox" id="mvr-free-text">
            <span>Free text</span>
          </label>
          <div id="mvr-keys">Space: play/pause · ←→: skip</div>
        </div>
      </div>
    </div>`;
  document.body.appendChild(d);

  // Starfield animation
  (() => {
    const c = document.getElementById('mvr-stars');
    if (!c) return;
    const ctx = c.getContext('2d');
    let stars = [], raf;
    function resize() {
      const rect = d.getBoundingClientRect();
      c.width = rect.width * 2;
      c.height = rect.height * 2;
      c.style.width = rect.width + 'px';
      c.style.height = rect.height + 'px';
    }
    function init() {
      resize();
      stars = [];
      for (let i = 0; i < 40; i++) {
        stars.push({
          x: Math.random() * c.width,
          y: Math.random() * c.height,
          r: Math.random() * 2 + 0.6,
          dx: (Math.random() - 0.5) * 0.5,
          dy: Math.random() * 0.3 + 0.1,
          o: Math.random() * 0.7 + 0.3,
        });
      }
    }
    function draw() {
      ctx.clearRect(0, 0, c.width, c.height);
      for (const s of stars) {
        s.x += s.dx;
        s.y += s.dy;
        s.o += (Math.random() - 0.5) * 0.015;
        s.o = Math.max(0.08, Math.min(0.6, s.o));
        if (s.y > c.height) { s.y = 0; s.x = Math.random() * c.width; }
        if (s.x < 0) s.x = c.width;
        if (s.x > c.width) s.x = 0;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${s.o})`;
        ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    }
    init();
    draw();
    // Resize when panel expands/collapses
    new ResizeObserver(() => { resize(); }).observe(d);
  })();

  document.getElementById('mvr-x').addEventListener('click', () => {
    stopSpeaking(); removeHighlights(); d.remove(); panelCreated = false; autoReadEnabled = false;
  }, true);
  // Settings toggle — start collapsed
  const ctrlPanel = document.getElementById('mvr-controls');
  const ctrlToggle = document.getElementById('mvr-controls-toggle');
  ctrlPanel.classList.add('collapsed');
  ctrlToggle.setAttribute('aria-expanded', 'false');
  ctrlToggle.addEventListener('click', () => {
    const collapsed = ctrlPanel.classList.toggle('collapsed');
    ctrlToggle.setAttribute('aria-expanded', (!collapsed).toString());
    ctrlToggle.classList.toggle('open', !collapsed);
  });
  document.getElementById('mvr-go').addEventListener('click', () => {
    if (speaking) {
      stopSpeaking();
      setTimeout(readPage, 100);
    } else {
      readPage();
    }
  }, true);
  document.getElementById('mvr-stop').addEventListener('click', stopSpeaking, true);
  document.getElementById('mvr-pause').addEventListener('click', togglePause, true);
  document.getElementById('mvr-speed').addEventListener('input', (e) => {
    document.getElementById('mvr-sv').textContent = parseFloat(e.target.value).toFixed(2) + 'x';
    chrome.storage.local.set({ mvrSpeed: parseFloat(e.target.value) });
  }, true);
  document.getElementById('mvr-overlay-mode').addEventListener('change', (e) => {
    overlayMode = e.target.value; refreshOverlayStyle();
    chrome.storage.local.set({ mvrOverlay: overlayMode });
  }, true);
  const autoCheckbox = document.getElementById('mvr-auto');
  autoCheckbox.checked = false;  // default OFF — user prefers manual
  autoCheckbox.addEventListener('change', (e) => {
    autoReadEnabled = e.target.checked;
    chrome.storage.local.set({ mvrAutoRead: autoReadEnabled });
    if (autoReadEnabled) {
      setStatus('Auto-read ON. Reading...');
      if (!speaking) readPage();
    } else {
      setStatus('Auto-read OFF.');
    }
  }, true);
  document.getElementById('mvr-show-marks').addEventListener('change', (e) => {
    if (!e.target.checked) removeHighlights();
    chrome.storage.local.set({ mvrShowMarks: e.target.checked });
  }, true);
  document.getElementById('mvr-free-text').addEventListener('change', (e) => {
    freeTextEnabled = e.target.checked;
    chrome.storage.local.set({ mvrFreeText: freeTextEnabled });
  }, true);

  // Voice picker — warm up new voice on server when changed
  const voiceSelect = document.getElementById('mvr-voice');
  voiceSelect.addEventListener('change', (e) => {
    selectedVoice = e.target.value;
    chrome.storage.local.set({ mvrVoice: selectedVoice });
    // Pre-warm the new voice on the server so there's no delay
    chrome.runtime.sendMessage({ type: 'MVR_WARM_VOICE', voice: selectedVoice }).catch(() => {});
    // Invalidate pre-fetched audio so next bubble uses new voice
    _prefetchInvalid = true;
  }, true);
  loadVoices();

  // Restore persisted settings
  chrome.storage.local.get(['mvrVoice', 'mvrSpeed', 'mvrOverlay', 'mvrAutoRead', 'mvrShowMarks', 'mvrFreeText'], (data) => {
    if (data.mvrVoice) {
      // Reset old Kokoro voices to Piper
      if (data.mvrVoice !== '_piper' && data.mvrVoice !== '_browser') {
        data.mvrVoice = '_piper';
        chrome.storage.local.set({ mvrVoice: '_piper' });
      }
      selectedVoice = data.mvrVoice;
      const vs = document.getElementById('mvr-voice');
      if (vs) vs.value = selectedVoice;
    }
    if (data.mvrSpeed !== undefined) {
      const sp = document.getElementById('mvr-speed');
      const sv = document.getElementById('mvr-sv');
      if (sp) { sp.value = data.mvrSpeed; }
      if (sv) { sv.textContent = parseFloat(data.mvrSpeed).toFixed(2) + 'x'; }
    }
    if (data.mvrOverlay) {
      overlayMode = data.mvrOverlay;
      const om = document.getElementById('mvr-overlay-mode');
      if (om) om.value = overlayMode;
    }
    if (data.mvrAutoRead !== undefined) {
      autoReadEnabled = data.mvrAutoRead;
      const ac = document.getElementById('mvr-auto');
      if (ac) ac.checked = autoReadEnabled;
    }
    const sm = document.getElementById('mvr-show-marks');
    if (sm) sm.checked = data.mvrShowMarks !== false; // default ON
    if (data.mvrFreeText !== undefined) {
      freeTextEnabled = data.mvrFreeText;
      const ft = document.getElementById('mvr-free-text');
      if (ft) ft.checked = freeTextEnabled;
    }
  });
  updateUrlBar();
}

async function loadVoices() {
  const select = document.getElementById('mvr-voice');
  if (!select) return;
  // Simple: just Piper and System Voice
  select.innerHTML = '';
  const piperOpt = document.createElement('option');
  piperOpt.value = '_piper';
  piperOpt.textContent = 'Piper';
  if (selectedVoice !== '_browser') piperOpt.selected = true;
  select.appendChild(piperOpt);
  const browserOpt = document.createElement('option');
  browserOpt.value = '_browser';
  browserOpt.textContent = 'System Voice';
  if (selectedVoice === '_browser') browserOpt.selected = true;
  select.appendChild(browserOpt);
}

// ─── Keyboard shortcuts ───────────────────────────────────────────────────────

let _audioResolve = null; // resolve function for current audio playback

function skipBubble(dir) {
  if (!speaking) return;
  skipDirection = dir;
  // Kill current audio playback
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  // Kill browser speech
  window.speechSynthesis.cancel();
  // Resolve the pending audio/speech promise so the loop continues
  if (_audioResolve) { _audioResolve(); _audioResolve = null; }
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
  if (!panelCreated) return;

  if (e.code === 'Space') {
    e.preventDefault();
    if (speaking) { togglePause(); } else { readPage(); }
  } else if (e.code === 'ArrowRight' && speaking) {
    e.preventDefault();
    skipBubble(1);
  } else if (e.code === 'ArrowLeft' && speaking) {
    e.preventDefault();
    skipBubble(-1);
  }
}, true);

function updateUrlBar() {
  const el = document.getElementById('mvr-url-bar');
  if (el) el.textContent = location.pathname;
}
function setStatus(msg) {
  const el = document.getElementById('mvr-status');
  if (el) el.textContent = msg;
  updateUrlBar();
}
function showText(msg) {
  const el = document.getElementById('mvr-text');
  if (el) el.textContent = msg;
}
function setButtons(reading) {
  const idleRow = document.getElementById('mvr-row-idle');
  const readRow = document.getElementById('mvr-row-reading');
  if (idleRow) idleRow.style.display = reading ? 'none' : '';
  if (readRow) readRow.style.display = reading ? '' : 'none';
  if (!reading) {
    paused = false;
    const pauseBtn = document.getElementById('mvr-pause');
    if (pauseBtn) { pauseBtn.classList.remove('active'); pauseBtn.removeAttribute('aria-pressed'); }
  }
}
function setProgress(current, total) {
  const wrap = document.getElementById('mvr-progress-wrap');
  const bar = document.getElementById('mvr-progress-bar');
  if (!wrap || !bar) return;
  if (total <= 0) { wrap.style.display = 'none'; return; }
  wrap.style.display = '';
  bar.style.width = Math.round((current / total) * 100) + '%';
}

// ─── Page hash for auto-read ─────────────────────────────────────────────────

function getPageHash() {
  const imgs = document.querySelectorAll('img, canvas');
  let hash = window.scrollY + '|';
  for (const el of imgs) {
    const r = el.getBoundingClientRect();
    if (r.width > 150 && r.height > 150) {
      const src = el.src || el.getAttribute('data-src') || '';
      hash += `${src.slice(-30)},${Math.round(r.top)}|`;
    }
  }
  hash += location.href + '|' + document.title;
  return hash;
}

// ─── Find manga crop rect ──────────────────────────────────────────────────

function getObjectFitOffset(img) {
  // For images with object-fit: contain, the rendered content may be smaller
  // than the element box. Calculate the offset of actual content within the element.
  if (!(img instanceof HTMLImageElement) || !img.naturalWidth || !img.naturalHeight) {
    return { dx: 0, dy: 0, scale: 1 };
  }
  const style = getComputedStyle(img);
  const fit = style.objectFit;
  if (fit !== 'contain' && fit !== 'scale-down') {
    return { dx: 0, dy: 0, scale: 1 };
  }
  const r = img.getBoundingClientRect();
  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  const elemW = r.width;
  const elemH = r.height;
  const scaleX = elemW / natW;
  const scaleY = elemH / natH;
  const scale = Math.min(scaleX, scaleY);
  const renderedW = natW * scale;
  const renderedH = natH * scale;
  // object-position defaults to 50% 50% (centered)
  const dx = (elemW - renderedW) / 2;
  const dy = (elemH - renderedH) / 2;
  if (dx > 1 || dy > 1) {
    console.log(`[MVR] object-fit offset: dx=${dx.toFixed(1)}, dy=${dy.toFixed(1)}, scale=${scale.toFixed(3)}`);
  }
  return { dx, dy, scale };
}

function getMangaCropRect() {
  const elements = document.querySelectorAll('img, canvas');
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  // Collect all visible images with their areas
  const candidates = [];
  for (const el of elements) {
    const r = el.getBoundingClientRect();
    if (r.width < 100 || r.height < 100) continue;
    // Must be at least partially visible in viewport
    if (r.right < 0 || r.bottom < 0 || r.left > vw || r.top > vh) continue;
    candidates.push({ rect: r, area: r.width * r.height, el });
  }

  if (candidates.length === 0) {
    console.log('[MVR] No manga images found, returning null cropRect (server will use full screenshot)');
    return null;
  }

  // Find the largest image — that's almost always the manga panel
  candidates.sort((a, b) => b.area - a.area);
  const largest = candidates[0];

  // Account for object-fit: contain offset on the main manga image
  const fitOffset = getObjectFitOffset(largest.el);

  // Only include images that horizontally overlap with the largest one
  // This excludes sidebar images while keeping stacked manga panels
  const mangaRects = [];
  const mainLeft = largest.rect.left + fitOffset.dx;
  const mainRight = largest.rect.right - fitOffset.dx;
  const mainWidth = mainRight - mainLeft;

  for (const c of candidates) {
    const r = c.rect;
    const cFit = getObjectFitOffset(c.el);
    const adjLeft = r.left + cFit.dx;
    const adjRight = r.right - cFit.dx;
    const adjTop = r.top + cFit.dy;
    const adjBottom = r.bottom - cFit.dy;
    // Check horizontal overlap with the main manga image
    const overlapLeft = Math.max(adjLeft, mainLeft);
    const overlapRight = Math.min(adjRight, mainRight);
    const overlap = Math.max(0, overlapRight - overlapLeft);
    // Must overlap at least 50% of the smaller width
    const adjWidth = adjRight - adjLeft;
    const minW = Math.min(adjWidth, mainWidth);
    if (overlap > minW * 0.5) {
      mangaRects.push({ left: adjLeft, top: adjTop, right: adjRight, bottom: adjBottom });
    }
  }

  if (mangaRects.length === 0) {
    mangaRects.push({
      left: largest.rect.left + fitOffset.dx,
      top: largest.rect.top + fitOffset.dy,
      right: largest.rect.right - fitOffset.dx,
      bottom: largest.rect.bottom - fitOffset.dy,
    });
  }

  let minLeft = Infinity, minTop = Infinity, maxRight = -Infinity, maxBottom = -Infinity;
  for (const r of mangaRects) {
    minLeft = Math.min(minLeft, r.left);
    minTop = Math.min(minTop, r.top);
    maxRight = Math.max(maxRight, r.right);
    maxBottom = Math.max(maxBottom, r.bottom);
  }

  const pad = 5;
  const cropW = Math.ceil(maxRight - minLeft) + pad * 2;
  const cropH = Math.ceil(maxBottom - minTop) + pad * 2;

  // Sanity check: crop must be at least 20% of viewport in both dimensions
  // If not, the detection failed — return null so server uses full screenshot
  if (cropW < vw * 0.2 || cropH < vh * 0.2) {
    console.log(`[MVR] cropRect too small (${cropW}x${cropH} vs viewport ${vw}x${vh}), returning null`);
    return null;
  }

  console.log(`[MVR] cropRect: left=${Math.floor(minLeft)}, top=${Math.floor(minTop)}, ${cropW}x${cropH} from ${mangaRects.length} images`);

  return {
    left: Math.max(0, Math.floor(minLeft) - pad),
    top: Math.max(0, Math.floor(minTop) - pad),
    width: cropW,
    height: cropH,
  };
}

// ─── Auto-read: click next page and keep reading ─────────────────────────────
//
// After readPage finishes all bubbles, if auto-read is ON:
// 1. Click the right side of the manga image (= next page on most readers)
// 2. Wait for the page to change
// 3. Call readPage again
//
// This creates a loop: read → click next → read → click next → ...

function findNextChapterUrl() {
  // Try inline script JSON first
  const scripts = document.querySelectorAll('script:not([src])');
  for (const s of scripts) {
    const match = s.textContent.match(/"next_chapter_url"\s*:\s*"([^"]+)"/);
    if (match && match[1]) return match[1];
  }
  return null;
}

function _countdownNavigate(url) {
  // 3-second countdown before navigating to next chapter, cancellable via Stop
  let remaining = 3;
  setStatus('Next chapter in ' + remaining + '...');
  return new Promise((resolve) => {
    function tick() {
      remaining--;
      if (remaining <= 0) {
        _countdownTimer = null;
        window.location.href = url;
        resolve();
        return;
      }
      setStatus('Next chapter in ' + remaining + '...');
      _countdownTimer = setTimeout(tick, 1000);
    }
    _countdownTimer = setTimeout(tick, 1000);
  });
}

function _goNextPage(isRTL) {
  // Click the page-go-right button (mangafire.to — always advances one page)
  const btn = document.querySelector('#page-go-right');
  if (btn) {
    btn.click();
    mvrLog('Next page: clicked #page-go-right');
    return;
  }
  // Fallback for other sites
  const fallbacks = ['a.next', '.next-page', '.btn-next', '[rel="next"]'];
  for (const sel of fallbacks) {
    const el = document.querySelector(sel);
    if (el) {
      el.click();
      mvrLog('Next page: clicked ' + sel);
      return;
    }
  }
  mvrLog('Next page: no button found');
}

function _waitForImageLoad(callback) {
  // Poll until the largest visible image is fully loaded (complete + nonzero size).
  // This prevents reading a stale or half-loaded page after navigation.
  const IMG_WAIT_MAX = 4000;
  const IMG_POLL_MS = 200;
  const imgStart = Date.now();

  function checkImage() {
    const imgs = document.querySelectorAll('img');
    let best = null;
    let bestArea = 0;
    for (const img of imgs) {
      const r = img.getBoundingClientRect();
      const area = r.width * r.height;
      if (area > bestArea && r.width > 100 && r.height > 100) {
        bestArea = area;
        best = img;
      }
    }

    const elapsed = Date.now() - imgStart;
    if (best && best.complete && best.naturalWidth > 0 && best.naturalHeight > 0) {
      // Image loaded. Give a small extra buffer for rendering.
      mvrLog(`Image ready after ${elapsed}ms (${best.naturalWidth}x${best.naturalHeight})`);
      setTimeout(callback, 200);
      return;
    }

    if (elapsed >= IMG_WAIT_MAX) {
      mvrLog(`Image wait timeout (${elapsed}ms), proceeding anyway`);
      callback();
      return;
    }

    setTimeout(checkImage, IMG_POLL_MS);
  }

  // Small initial delay to let the DOM update
  setTimeout(checkImage, 150);
}

let _autoNavPending = false;
let _consecutiveEmpty = 0;
const MAX_EMPTY_SKIPS = 2;

// Build a "chapter fingerprint" using multiple signals so we detect chapter changes
// even if some signals update before others during SPA navigation
function _getChapterFingerprint() {
  const url = location.href;
  const title = document.title || '';
  const totalSpan = document.querySelector('b.total-page');
  const totalPages = totalSpan ? totalSpan.textContent.trim() : '?';

  // Extract chapter from URL if available
  const m = url.match(/chapter[_-]?(\d+(?:\.\d+)?)/i);
  const chapterNum = m ? m[1] : '';

  // Try to get chapter from page DOM (chapter selector, breadcrumbs, etc.)
  let domChapter = '';
  // mangafire: look for active chapter selector or visible chapter text
  const selectors = [
    'select.chapter-selector option[selected]',
    'select[name="chapter"] option[selected]',
    '.chapter-selector .active',
    '.active-chapter',
    '[data-chapter].active',
    'a.active[href*="chapter"]',
    '.chapters .active',
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el) {
      const txt = el.textContent.trim();
      const cm = txt.match(/(\d+(?:\.\d+)?)/);
      if (cm) { domChapter = cm[1]; break; }
    }
  }

  // Also check inline scripts for chapter data
  let scriptChapter = '';
  if (!chapterNum && !domChapter) {
    const scripts = document.querySelectorAll('script:not([src])');
    for (const s of scripts) {
      const sm = s.textContent.match(/"current_chapter"\s*:\s*(\d+)/);
      if (sm) { scriptChapter = sm[1]; break; }
    }
  }

  return { url, title, totalPages, chapterNum, domChapter, scriptChapter };
}

function _chapterChanged(before, after) {
  // URL chapter number changed
  if (before.chapterNum && after.chapterNum && before.chapterNum !== after.chapterNum) return true;
  // DOM chapter indicator changed
  if (before.domChapter && after.domChapter && before.domChapter !== after.domChapter) return true;
  // Script chapter changed
  if (before.scriptChapter && after.scriptChapter && before.scriptChapter !== after.scriptChapter) return true;
  // Full URL changed (covers non-chapter URL patterns)
  if (before.url !== after.url) return true;
  // Title changed (many sites put chapter in title)
  if (before.title !== after.title) return true;
  // Total pages changed (different chapters have different page counts)
  if (before.totalPages !== '?' && after.totalPages !== '?' && before.totalPages !== after.totalPages) return true;
  return false;
}

function autoGoNext() {
  mvrLog(`autoGoNext called: autoReadEnabled=${autoReadEnabled}, speaking=${speaking}, pending=${_autoNavPending}`);
  if (!autoReadEnabled || speaking) return;
  if (_autoNavPending) return;

  const dir = document.getElementById('mvr-direction')?.value || 'rtl';
  const isRTL = dir === 'rtl';

  // Get current page number before clicking
  const currentSpan = document.querySelector('span.current-page');
  const currentNum = currentSpan ? parseInt(currentSpan.textContent.trim()) : 0;
  const urlBefore = location.href;

  _autoNavPending = true;
  mvrLog(`Advancing from page ${currentNum} (dir=${dir})`);
  _goNextPage(isRTL);

  // Poll until something changes (page number or URL)
  const pollStart = Date.now();
  const MAX_WAIT = 5000;
  const POLL_MS = 150;
  const pollForChange = () => {
    if (!autoReadEnabled || speaking) { _autoNavPending = false; return; }

    const newSpan = document.querySelector('span.current-page');
    const newNum = newSpan ? parseInt(newSpan.textContent.trim()) : 0;
    const urlNow = location.href;
    const elapsed = Date.now() - pollStart;

    // Something changed (page number or URL) or timeout
    if (newNum !== currentNum || urlNow !== urlBefore || elapsed >= MAX_WAIT) {
      _autoNavPending = false;

      if (urlNow !== urlBefore) {
        mvrLog(`URL changed: ${urlBefore} -> ${urlNow}. Continuing.`);
        updateUrlBar();
      }
      mvrLog(`Post-click: page ${currentNum}->${newNum} (${elapsed}ms)`);

      // Wait for image to load, then read
      _waitForImageLoad(() => {
        if (autoReadEnabled && !speaking) {
          readPage();
        }
      });
      return;
    }

    setTimeout(pollForChange, POLL_MS);
  };
  setTimeout(pollForChange, POLL_MS);
}

// ─── Highlighting ─────────────────────────────────────────────────────────────

function removeHighlights() {
  const el = document.getElementById('mvr-overlay');
  if (el) el.remove();
}

const _debugColors = [
  { bg: 'rgba(255,69,96,0.25)', border: 'rgba(255,69,96,0.7)' },
  { bg: 'rgba(69,160,255,0.25)', border: 'rgba(69,160,255,0.7)' },
  { bg: 'rgba(69,255,130,0.25)', border: 'rgba(69,255,130,0.7)' },
  { bg: 'rgba(255,200,69,0.25)', border: 'rgba(255,200,69,0.7)' },
  { bg: 'rgba(200,69,255,0.25)', border: 'rgba(200,69,255,0.7)' },
];

function drawBubbles(bubbles) {
  removeHighlights();
  if (!bubbles || bubbles.length === 0) return;
  if (overlayMode === 'hidden') return;
  if (!document.getElementById('mvr-show-marks')?.checked) return;

  const overlay = document.createElement('div');
  overlay.id = 'mvr-overlay';
  overlay.style.cssText = `position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483646;`;

  bubbles.forEach((bubble, i) => {
    const pad = 2;
    const outline = document.createElement('div');
    outline.className = 'mvr-bubble mvr-b' + i;
    outline.dataset.idx = i;
    const basePos = `position:absolute;left:${bubble.left-pad}px;top:${bubble.top-pad}px;width:${bubble.width+pad*2}px;height:${bubble.height+pad*2}px;transition:all 0.25s ease;`;

    if (overlayMode === 'debug') {
      const c = _debugColors[i % _debugColors.length];
      outline.style.cssText = basePos + `border:2px solid ${c.border};border-radius:4px;background:${c.bg};`;
      const label = document.createElement('div');
      label.style.cssText = `position:absolute;top:-12px;left:-2px;background:${c.border};color:#fff;font-size:10px;font-weight:bold;padding:1px 4px;border-radius:3px;font-family:sans-serif;`;
      label.textContent = i + 1;
      outline.appendChild(label);
    } else if (overlayMode === 'border') {
      outline.style.cssText = basePos + `border:1px solid rgba(255,255,255,0.15);border-radius:4px;background:none;`;
    } else {
      // Reader mode: very subtle borders that don't distract from reading
      outline.style.cssText = basePos + `border:1px solid rgba(255,255,255,0.04);border-radius:6px;background:none;`;
    }
    overlay.appendChild(outline);
  });

  document.body.appendChild(overlay);
}

function refreshOverlayStyle() {
  const overlay = document.getElementById('mvr-overlay');
  if (!overlay) return;
  const bubbleEls = overlay.querySelectorAll('.mvr-bubble');
  if (overlayMode === 'hidden') { overlay.style.display = 'none'; return; }
  overlay.style.display = '';
  bubbleEls.forEach(el => {
    const idx = parseInt(el.dataset.idx);
    el.querySelectorAll('div').forEach(d => d.remove());
    if (overlayMode === 'debug') {
      const c = _debugColors[idx % _debugColors.length];
      el.style.border = `2px solid ${c.border}`; el.style.background = c.bg;
      const label = document.createElement('div');
      label.style.cssText = `position:absolute;top:-12px;left:-2px;background:${c.border};color:#fff;font-size:10px;font-weight:bold;padding:1px 4px;border-radius:3px;font-family:sans-serif;`;
      label.textContent = idx + 1; el.appendChild(label);
    } else if (overlayMode === 'border') {
      el.style.border = '1px solid rgba(255,255,255,0.15)'; el.style.background = 'none'; el.style.boxShadow = 'none';
    } else {
      el.style.border = '1px solid rgba(255,255,255,0.08)'; el.style.background = 'none'; el.style.boxShadow = 'none';
    }
  });
}

function setActiveBubble(activeIdx) {
  const bubbleEls = document.querySelectorAll('.mvr-bubble');
  if (overlayMode === 'hidden') return;
  bubbleEls.forEach(el => {
    const idx = parseInt(el.dataset.idx);
    const isActive = idx === activeIdx;
    if (overlayMode === 'debug') {
      const c = _debugColors[idx % _debugColors.length];
      el.style.borderWidth = isActive ? '3px' : '2px';
      el.style.boxShadow = isActive ? `0 0 12px ${c.border}` : 'none';
    } else if (overlayMode === 'border') {
      el.style.border = isActive ? '2px solid rgba(233,69,96,0.7)' : '1px solid rgba(255,255,255,0.12)';
      el.style.boxShadow = isActive ? '0 0 10px rgba(233,69,96,0.3)' : 'none';
      el.style.background = isActive ? 'rgba(233,69,96,0.06)' : 'none';
    } else {
      // Reader mode: active bubble is clearly highlighted, others are nearly invisible
      el.style.border = isActive ? '2px solid rgba(233,69,96,0.5)' : '1px solid rgba(255,255,255,0.04)';
      el.style.boxShadow = isActive ? '0 0 8px rgba(233,69,96,0.2), inset 0 0 12px rgba(233,69,96,0.05)' : 'none';
      el.style.background = isActive ? 'rgba(233,69,96,0.05)' : 'none';
      el.style.borderRadius = isActive ? '8px' : '4px';
    }
  });
}

// ─── Background Quality Pass (Florence-2) ────────────────────────────────────

let _activeQualityPollId = null;

function _pollQualityPass(requestId, bubbles) {
  _activeQualityPollId = requestId; // cancel any previous poll
  let attempts = 0;
  const maxAttempts = 30; // 30 * 2s = 60s max wait
  const pollInterval = 2000;

  const poll = async () => {
    if (_activeQualityPollId !== requestId) {
      mvrLog('Quality pass: cancelled (new page)');
      return;
    }
    if (attempts >= maxAttempts) {
      mvrLog('Quality pass: timed out');
      return;
    }
    attempts++;
    try {
      const res = await new Promise((resolve) => {
        chrome.runtime.sendMessage(
          { type: 'MVR_QUALITY_POLL', requestId },
          (r) => resolve(r || { ready: false })
        );
      });

      if (!res.ready) {
        setTimeout(poll, pollInterval);
        return;
      }

      // Quality results arrived — update unread bubbles
      const qBubbles = res.bubbles || [];
      if (qBubbles.length === 0) return;

      let updated = 0;
      for (let qi = 0; qi < qBubbles.length && qi < bubbles.length; qi++) {
        // Only update bubbles we haven't read yet
        if (qi > currentBubbleIndex && qBubbles[qi].text !== bubbles[qi].text) {
          const oldText = bubbles[qi].text;
          bubbles[qi].text = qBubbles[qi].text;
          updated++;
          mvrLog(`Quality upgrade [${qi}]: "${oldText.substring(0,30)}" → "${qBubbles[qi].text.substring(0,30)}"`);
        }
      }

      if (updated > 0) {
        mvrLog(`Quality pass: ${updated} bubbles upgraded (${res.timing_ms}ms)`);
        // Refresh displayed text
        const allText = bubbles.map((b, i) => `[${i + 1}] ${b.text}`).join('\n');
        showText(allText);
      } else {
        mvrLog(`Quality pass: no improvements (${res.timing_ms}ms)`);
      }
    } catch (e) {
      mvrLog('Quality poll error: ' + e.message);
    }
  };

  // Start polling after a short delay (give Florence-2 time to process)
  setTimeout(poll, 3000);
}


// ─── Main Flow ────────────────────────────────────────────────────────────────

async function readPage() {
  if (speaking) return; // prevent double-run
  speaking = true;
  setButtons(true);
  setStatus('Capturing page...');
  setProgress(0, 0);
  showText('');
  removeHighlights();

  // Track current page number for wrap detection
  const curSpan = document.querySelector('span.current-page');
  const curNum = curSpan ? parseInt(curSpan.textContent.trim()) : 0;
  if (curNum > 0) lastReadPageNum = curNum;
  mvrLog(`readPage start: page ${curNum}, lastReadPageNum=${lastReadPageNum}`);

  const cropRect = getMangaCropRect();

  try {
    const panel = document.getElementById('mvr');
    if (panel) panel.style.display = 'none';
    // Use requestAnimationFrame for fastest possible panel hide
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

    setStatus('Reading...');

    const result = await Promise.race([
      new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(
          { type: 'READ_SCREEN', cropRect, dpr: window.devicePixelRatio || 2, pageTitle: document.title, pageUrl: location.href, readingDirection: document.getElementById('mvr-direction')?.value || 'rtl', voice: document.getElementById('mvr-voice')?.value || '_piper', speed: parseFloat(document.getElementById('mvr-speed')?.value || '0.9'), freeText: freeTextEnabled },
          (res) => {
            if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
            resolve(res);
          }
        );
      }),
      new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout: server took too long')), 60000))
    ]);

    if (panel) panel.style.display = '';
    if (!speaking) return;

    if (!result?.ok) {
      const err = result?.error || 'Unknown error';
      if (err === 'SERVER_NOT_RUNNING') {
        setStatus('Server off. Run: python3 ~/MangaVoice/server/server_lite.py');
      } else if (err.includes('503')) {
        setStatus('Models still loading... try again in a few seconds.');
      } else {
        setStatus('Error: ' + err);
      }
      setButtons(false); speaking = false;
      return;
    }

    const bubbles = result.bubbles || [];
    const freeTexts = result.freeTexts || [];
    const qualityRequestId = result.requestId || null;

    // Tag original bubbles with their audio index before merging
    bubbles.forEach((b, i) => { b._audioIdx = i; });

    // Append free text results to bubbles (they come pre-sorted from server)
    if (freeTexts.length > 0) {
      mvrLog(`Free text: ${freeTexts.length} regions found outside bubbles`);
      for (const ft of freeTexts) {
        bubbles.push(ft);
      }
      // Re-sort everything together by reading order (top-to-bottom, then by direction)
      const dir = document.getElementById('mvr-direction')?.value || 'rtl';
      const rowH = 150; // approximate row height in CSS pixels
      if (dir === 'rtl') {
        bubbles.sort((a, b) => {
          const rowA = Math.floor(a.top / rowH), rowB = Math.floor(b.top / rowH);
          if (rowA !== rowB) return rowA - rowB;
          return b.left - a.left;
        });
      } else {
        bubbles.sort((a, b) => {
          const rowA = Math.floor(a.top / rowH), rowB = Math.floor(b.top / rowH);
          if (rowA !== rowB) return rowA - rowB;
          return a.left - b.left;
        });
      }
    }

    // Start background quality polling if server has a quality pass running
    if (qualityRequestId) {
      _pollQualityPass(qualityRequestId, bubbles);
    }

    if (bubbles.length === 0) {
      _consecutiveEmpty++;
      mvrLog(`readPage: 0 bubbles found (${_consecutiveEmpty} consecutive empty)`);
      if (autoReadEnabled && _consecutiveEmpty <= MAX_EMPTY_SKIPS) {
        setStatus(`No dialogue, skipping... (${_consecutiveEmpty}/${MAX_EMPTY_SKIPS})`);
        // Still "speaking" so autoGoNext fires at the end
      } else {
        setStatus('No dialogue found.');
        setButtons(false); speaking = false;
        _consecutiveEmpty = 0;
        return;
      }
    } else {
      // Found bubbles, reset empty counter
      _consecutiveEmpty = 0;
    }

    if (result.timing) {
      console.log('[MVR] Timing:', result.timing);
    }

    const ftCount = freeTexts.length;
    const bCount = bubbles.length - ftCount;
    const statusMsg = ftCount > 0
      ? `Found ${bCount} bubble${bCount !== 1 ? 's' : ''} + ${ftCount} free text`
      : `Found ${bubbles.length} bubble${bubbles.length !== 1 ? 's' : ''}`;
    setStatus(statusMsg);
    drawBubbles(bubbles);

    const allText = bubbles.map((b, i) => `[${i + 1}] ${b.text}`).join('\n');
    showText(allText);

    // TTS mode: "kokoro" (server AI voice) or "browser" (instant system voice)
    const voiceVal = document.getElementById('mvr-voice')?.value || '_piper';
    const useKokoro = voiceVal !== '_browser';
    totalBubbles = bubbles.length;
    const audioId = result.audioId;

    if (useKokoro && audioId) {
      // ─── STREAMING PAGE AUDIO ────────────────────────────────────
      // Server generates bubble audio in background. We buffer ahead
      // so there's no gap between bubbles during playback.

      skipDirection = 0;

      // Wait for first 2 bubbles (or all if fewer) to buffer before playing
      const bufferTarget = Math.min(2, bubbles.length);
      setStatus(`Buffering voice...`);
      for (let wait = 0; wait < 80 && speaking; wait++) {
        try {
          const resp = await fetch(`${LOCAL_SERVER}/process/audio?id=${audioId}&bubble=${bufferTarget - 1}`);
          if (resp.status === 200) break;
        } catch (e) { /* retry */ }
        await new Promise(r => setTimeout(r, 150));
      }

      // Pre-fetch helper: fetch bubble audio, return blob URL or null
      async function fetchBubbleAudio(idx) {
        const bubble = bubbles[idx];
        if (!bubble) return null;

        // Free text items don't have pre-cached audio, use /tts directly
        if (bubble.freeText) {
          try {
            const resp = await fetch(`${LOCAL_SERVER}/tts`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text: bubble.text, voice: voiceVal, speed: parseFloat(document.getElementById('mvr-speed')?.value || '0.9') }),
            });
            if (resp.ok) {
              const blob = await resp.blob();
              return URL.createObjectURL(blob);
            }
          } catch (e) { /* fallback to null */ }
          return null;
        }

        // Regular bubble: use pre-cached audio by original index
        const audioIdx = bubble._audioIdx !== undefined ? bubble._audioIdx : idx;
        for (let attempt = 0; attempt < 60; attempt++) {
          if (!speaking) return null;
          try {
            const resp = await fetch(`${LOCAL_SERVER}/process/audio?id=${audioId}&bubble=${audioIdx}`);
            if (resp.status === 200) {
              const contentType = resp.headers.get('content-type') || '';
              if (contentType.includes('audio')) {
                const blob = await resp.blob();
                return URL.createObjectURL(blob);
              } else {
                const data = await resp.json();
                if (data.empty) return null;
              }
            } else if (resp.status === 202) {
              const progress = await resp.json();
              if (idx === 0) setStatus(`Generating voice... (${progress.generated}/${progress.total})`);
            }
          } catch (e) { /* retry */ }
          await new Promise(r => setTimeout(r, 100));
        }
        return null;
      }

      // Pre-fetch first bubble
      let nextAudioPromise = fetchBubbleAudio(0);

      for (let i = 0; i < bubbles.length; i++) {
        if (!speaking) break;
        currentBubbleIndex = i;
        setActiveBubble(i);
        setProgress(i + 1, bubbles.length);
        setStatus(`Reading ${i + 1}/${bubbles.length}`);

        // Await this bubble's audio (already being fetched)
        let audioBlobUrl = await nextAudioPromise;

        // Start pre-fetching NEXT bubble immediately (overlaps with playback)
        nextAudioPromise = (i + 1 < bubbles.length) ? fetchBubbleAudio(i + 1) : Promise.resolve(null);

        if (!speaking) break;

        // Handle skip
        if (skipDirection !== 0) {
          const newIdx = i + skipDirection;
          skipDirection = 0;
          if (audioBlobUrl) URL.revokeObjectURL(audioBlobUrl);
          // Reset pre-fetch for the new target
          nextAudioPromise = (newIdx >= 0 && newIdx < bubbles.length) ? fetchBubbleAudio(newIdx) : Promise.resolve(null);
          if (newIdx >= 0 && newIdx < bubbles.length) { i = newIdx - 1; }
          continue;
        }

        if (audioBlobUrl) {
          mvrLog(`Bubble ${i}: audio ready, playing`);
          await playAudioData(audioBlobUrl);
        }

        // Handle skip after playback
        if (skipDirection !== 0) {
          const newIdx = i + skipDirection;
          skipDirection = 0;
          if (newIdx >= 0 && newIdx < bubbles.length) { i = newIdx - 1; }
          continue;
        }

        // Respect pause
        while (paused && speaking && skipDirection === 0) await new Promise(r => setTimeout(r, 100));
        if (skipDirection !== 0) {
          const newIdx = i + skipDirection;
          skipDirection = 0;
          if (newIdx >= 0 && newIdx < bubbles.length) { i = newIdx - 1; }
          continue;
        }
      }

    } else {
      // ─── BROWSER TTS FALLBACK ────────────────────────────────────
      skipDirection = 0;
      for (let i = 0; i < bubbles.length; i++) {
        if (!speaking) break;
        currentBubbleIndex = i;
        setActiveBubble(i);
        setProgress(i + 1, bubbles.length);
        setStatus(`Reading ${i + 1}/${bubbles.length}`);
        await speak(bubbles[i].text);

        if (skipDirection !== 0) {
          const newIdx = i + skipDirection;
          skipDirection = 0;
          if (newIdx >= 0 && newIdx < bubbles.length) { i = newIdx - 1; }
          continue;
        }
        while (paused && speaking && skipDirection === 0) await new Promise(r => setTimeout(r, 100));
      }
    }

    if (speaking) {
      setProgress(0, 0);
      if (autoReadEnabled) {
        setStatus('Done. Going to next page...');
        mvrLog(`readPage: all ${bubbles.length} bubbles done, will call autoGoNext`);
      } else {
        setStatus('Done. Press Read Page for next.');
      }
    }

  } catch (err) {
    setStatus('Error: ' + err.message);
    mvrLog(`readPage ERROR: ${err.message}`);
    console.error('[MVR]', err);
    const panel = document.getElementById('mvr');
    if (panel) panel.style.display = '';
  } finally {
    // ALWAYS restore panel visibility no matter what
    const p = document.getElementById('mvr');
    if (p) p.style.display = '';
  }

  const shouldAutoNext = speaking && autoReadEnabled;
  mvrLog(`readPage end: speaking=${speaking}, autoRead=${autoReadEnabled}, shouldAutoNext=${shouldAutoNext}`);
  setButtons(false);
  speaking = false;

  if (shouldAutoNext) {
    // Small delay before auto-advancing to prevent rapid-fire on empty pages
    setTimeout(() => autoGoNext(), 500);
  }
}

// ─── TTS with emotion detection ─────────────────────────────────────────────

function detectEmotion(text) {
  if (/!{1,}/.test(text) && text === text.toUpperCase()) return 'shout';
  if (/\?!|!\?/.test(text)) return 'surprise';
  if (/\.{3}|…/.test(text)) return 'trailing';
  if (/\?$/.test(text)) return 'question';
  if (/!$/.test(text)) return 'exclaim';
  return 'normal';
}

async function speak(text) {
  if (!speaking) return;
  const baseRate = parseFloat(document.getElementById('mvr-speed')?.value || '0.85');
  const emotion = detectEmotion(text);

  return new Promise((resolve) => {
    if (!speaking || skipDirection !== 0) { resolve(); return; }
    window.speechSynthesis.cancel();
    _audioResolve = resolve;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-US';
    u.volume = 1;
    // Pick best voice: Google US English > Google UK English > any English Google voice
    if (_browserVoices.length === 0) _browserVoices = window.speechSynthesis.getVoices();
    const gv = _browserVoices.find(v => v.name === 'Google US English')
            || _browserVoices.find(v => v.name.includes('Google') && v.lang.startsWith('en'))
            || _browserVoices.find(v => v.lang === 'en-US' && !v.localService);
    if (gv) u.voice = gv;

    switch (emotion) {
      case 'shout':
        u.rate = baseRate + 0.1; u.pitch = 1.3; u.volume = 1; break;
      case 'surprise':
        u.rate = baseRate + 0.05; u.pitch = 1.2; break;
      case 'question':
        u.rate = baseRate - 0.05; u.pitch = 1.1; break;
      case 'trailing':
        u.rate = baseRate - 0.1; u.pitch = 0.9; break;
      case 'exclaim':
        u.rate = baseRate + 0.05; u.pitch = 1.15; break;
      default:
        u.rate = baseRate; u.pitch = 1.0;
    }

    const done = () => { _audioResolve = null; resolve(); };
    u.onend = done;
    u.onerror = done;
    window.speechSynthesis.speak(u);
  });
}

function playAudioData(blobUrl) {
  /**Play audio blob URL. Respects pause state. Returns promise that resolves when done.*/
  return new Promise(async (resolve) => {
    if (!speaking) { resolve(); return; }
    while (paused && speaking && skipDirection === 0) await new Promise(r => setTimeout(r, 100));
    if (!speaking || skipDirection !== 0) { resolve(); return; }
    _audioResolve = resolve;
    try {
      const audio = new Audio(blobUrl);
      currentAudio = audio;
      const done = () => { currentAudio = null; _audioResolve = null; URL.revokeObjectURL(blobUrl); resolve(); };
      audio.onended = done;
      audio.onerror = done;
      audio.play().catch(done);
    } catch (e) {
      _audioResolve = null;
      resolve();
    }
  });
}

// ─── Pause / Resume ──────────────────────────────────────────────────────────

function togglePause() {
  if (!speaking) return;
  paused = !paused;
  const pauseBtn = document.getElementById('mvr-pause');
  if (paused) {
    if (pauseBtn) { pauseBtn.classList.add('active'); pauseBtn.setAttribute('aria-pressed', 'false'); }
    if (currentAudio) currentAudio.pause();
    setStatus('Paused');
  } else {
    if (pauseBtn) { pauseBtn.classList.remove('active'); pauseBtn.setAttribute('aria-pressed', 'true'); }
    if (currentAudio) currentAudio.play();
    setStatus('Reading...');
  }
}

// ─── Stop ────────────────────────────────────────────────────────────────────

let _countdownTimer = null; // for next-chapter countdown cancellation

function stopSpeaking() {
  speaking = false;
  paused = false;
  // Cancel any next-chapter countdown in progress
  if (_countdownTimer) { clearTimeout(_countdownTimer); _countdownTimer = null; }
  window.speechSynthesis.cancel();
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
  removeHighlights();
  setProgress(0, 0);
  setStatus('Ready');
  setButtons(false);
}
