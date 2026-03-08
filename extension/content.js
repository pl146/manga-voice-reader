// Manga Voice Reader — Content Script

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
let selectedVoice = 'af_heart';
let _prefetchInvalid = false;
const PC_SERVER = 'http://192.168.2.183:5055';
const PC_LAUNCHER = 'http://192.168.2.183:5056';
const LOCAL_SERVER = 'http://127.0.0.1:5055';
let SERVER = PC_SERVER;  // active server, switches automatically

async function _ensureServer() {
  // Route through background.js service worker to avoid Mixed Content blocking
  try {
    const res = await chrome.runtime.sendMessage({ type: 'MVR_ENSURE_SERVER' });
    if (res && res.ok) {
      SERVER = res.server;
      mvrLog('Connected to ' + SERVER);
      return true;
    }
  } catch (_) {}
  mvrLog('No server available');
  return false;
}

function _shutdownServer() {
  stopSpeaking();
  if (SERVER === PC_SERVER) {
    chrome.runtime.sendMessage({ type: 'MVR_SHUTDOWN' }).catch(() => {});
    mvrLog('PC shutdown signal sent');
  }
}

// Overlay modes: 'reader' (default), 'border', 'debug', 'hidden'
let overlayMode = 'reader';

// ─── Activation gate ─────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'MVR_ACTIVATE') {
    // Auto-start server if needed
    _ensureServer().then(() => {
      if (!panelCreated) {
        createUI();
        panelCreated = true;
      } else {
        const panel = document.getElementById('mvr');
        if (panel) panel.style.display = '';
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
    <div id="mvr-head">
      <div id="mvr-head-title">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <path d="M4 4h16a2 2 0 012 2v10a2 2 0 01-2 2H8l-4 4V6a2 2 0 012-2z" fill="rgba(255,255,255,0.85)"/>
          <rect x="8" y="9" width="2" height="6" rx="1" fill="#c22d4e"/>
          <rect x="11" y="7" width="2" height="10" rx="1" fill="#c22d4e"/>
          <rect x="14" y="8" width="2" height="8" rx="1" fill="#c22d4e"/>
        </svg>
        <span>Manga Voice Reader</span>
      </div>
      <button id="mvr-x">\u2715</button>
    </div>
    <div id="mvr-body">
      <div id="mvr-status">Ready</div>
      <div id="mvr-progress-wrap" style="display:none;">
        <div id="mvr-progress-bar"></div>
      </div>
      <div id="mvr-row-idle">
        <button id="mvr-go">\u25B6 Read Page</button>
      </div>
      <div id="mvr-row-reading" style="display:none;">
        <button id="mvr-pause">\u275A\u275A</button>
        <button id="mvr-stop">\u25A0 Stop</button>
      </div>
      <div id="mvr-text"></div>
      <button id="mvr-controls-toggle">\u25BC Settings</button>
      <div id="mvr-controls">
        <label>Voice <select id="mvr-voice"><option value="af_heart">Heart</option></select></label>
        <label>Speed <input type="range" id="mvr-speed" min="0.5" max="2" step="0.1" value="0.85"> <span id="mvr-sv">0.85x</span></label>
        <label>Overlay <select id="mvr-overlay-mode">
          <option value="reader">Reader</option>
          <option value="border">Border</option>
          <option value="debug">Debug</option>
          <option value="hidden">Hidden</option>
        </select></label>
        <label>Read \u2192 <select id="mvr-direction">
          <option value="rtl">RTL</option>
          <option value="ltr">LTR</option>
          <option value="vertical">Vertical</option>
        </select></label>
        <label id="mvr-auto-label"><input type="checkbox" id="mvr-auto"> Auto-read</label>
        <div id="mvr-keys">Space: play/pause \u00B7 \u2190\u2192: skip</div>
      </div>
    </div>`;
  document.body.appendChild(d);

  // Drag
  const head = d.querySelector('#mvr-head');
  let dragging = false, dx = 0, dy = 0;
  head.addEventListener('mousedown', (e) => {
    dragging = true; dx = e.clientX - d.offsetLeft; dy = e.clientY - d.offsetTop; e.preventDefault();
  }, true);
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    d.style.left = (e.clientX - dx) + 'px'; d.style.top = (e.clientY - dy) + 'px';
    d.style.right = 'auto'; d.style.bottom = 'auto';
  }, true);
  document.addEventListener('mouseup', () => { dragging = false; }, true);

  document.getElementById('mvr-x').addEventListener('click', () => {
    stopSpeaking(); removeHighlights(); d.remove(); panelCreated = false; autoReadEnabled = false;
    _shutdownServer();
  }, true);
  // Settings toggle — start collapsed
  const ctrlPanel = document.getElementById('mvr-controls');
  const ctrlToggle = document.getElementById('mvr-controls-toggle');
  ctrlPanel.classList.add('collapsed');
  ctrlToggle.addEventListener('click', () => {
    const collapsed = ctrlPanel.classList.toggle('collapsed');
    ctrlToggle.textContent = collapsed ? '\u25BC Settings' : '\u25B2 Settings';
  });
  document.getElementById('mvr-go').addEventListener('click', () => {
    // Force stop any current reading first, then start fresh
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
  chrome.storage.local.get(['mvrVoice', 'mvrSpeed', 'mvrOverlay', 'mvrAutoRead'], (data) => {
    if (data.mvrVoice) {
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
  });
}

async function loadVoices() {
  const select = document.getElementById('mvr-voice');
  if (!select) return;
  try {
    const data = await chrome.runtime.sendMessage({ type: 'MVR_TTS_STATUS' });
    if (!data || !data.ok) return;
    if (!data.voices || data.voices.length === 0) return;
    availableVoices = data.voices;
    select.innerHTML = '';
    // Group voices by language prefix
    const groups = {};
    for (const v of data.voices) {
      const prefix = v.substring(0, 2);
      if (!groups[prefix]) groups[prefix] = [];
      groups[prefix].push(v);
    }
    const voiceNames = {
      af_alloy: 'Alloy', af_aoede: 'Aoede', af_bella: 'Bella', af_heart: 'Heart', af_jessica: 'Jessica',
      af_kore: 'Kore', af_nicole: 'Nicole', af_nova: 'Nova', af_river: 'River', af_sarah: 'Sarah', af_sky: 'Sky',
      am_adam: 'Adam', am_echo: 'Echo', am_eric: 'Eric', am_fenrir: 'Fenrir', am_liam: 'Liam',
      am_michael: 'Michael', am_onyx: 'Onyx', am_puck: 'Puck', am_santa: 'Santa',
      bf_alice: 'Alice', bf_emma: 'Emma', bf_isabella: 'Isabella', bf_lily: 'Lily',
      bm_daniel: 'Daniel', bm_fable: 'Fable', bm_george: 'George', bm_lewis: 'Lewis',
      ef_dora: 'Dora', em_alex: 'Alex', em_santa: 'Santa',
      ff_siwis: 'Siwis',
      hf_alpha: 'Alpha', hf_beta: 'Beta', hm_omega: 'Omega', hm_psi: 'Psi',
      if_sara: 'Sara', im_nicola: 'Nicola',
      jf_alpha: 'Alpha', jf_gongitsune: 'Gongitsune', jf_nezumi: 'Nezumi', jf_tebukuro: 'Tebukuro', jm_kumo: 'Kumo',
      pf_dora: 'Dora', pm_alex: 'Alex', pm_santa: 'Santa',
      zf_xiaobei: 'Xiaobei', zf_xiaoni: 'Xiaoni', zf_xiaoxiao: 'Xiaoxiao', zf_xiaoyi: 'Xiaoyi',
      zm_yunjian: 'Yunjian', zm_yunxi: 'Yunxi', zm_yunxia: 'Yunxia', zm_yunyang: 'Yunyang',
    };
    const groupNames = { af: 'English Female', am: 'English Male', bf: 'British Female', bm: 'British Male', ef: 'Spanish Female', em: 'Spanish Male', ff: 'French', hf: 'Hindi Female', hm: 'Hindi Male', 'if': 'Italian Female', im: 'Italian Male', jf: 'Japanese Female', jm: 'Japanese Male', pf: 'Portuguese Female', pm: 'Portuguese Male', zf: 'Chinese Female', zm: 'Chinese Male' };
    for (const [prefix, voices] of Object.entries(groups)) {
      const group = document.createElement('optgroup');
      group.label = groupNames[prefix] || prefix;
      for (const v of voices) {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = voiceNames[v] || v.split('_').pop();
        if (v === selectedVoice) opt.selected = true;
        group.appendChild(opt);
      }
      select.appendChild(group);
    }
  } catch (_) {}
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

function setStatus(msg) {
  const el = document.getElementById('mvr-status');
  if (el) el.textContent = msg;
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
    if (pauseBtn) { pauseBtn.textContent = '\u275A\u275A'; pauseBtn.classList.remove('active'); }
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

function autoGoNext() {
  mvrLog(`autoGoNext called: autoReadEnabled=${autoReadEnabled}, speaking=${speaking}`);
  if (!autoReadEnabled || speaking) return;

  // Get current and total page from mangafire.to's UI
  // Use the specific span inside the button (most reliable)
  const currentSpan = document.querySelector('span.current-page');
  const totalSpan = document.querySelector('b.total-page');
  const currentNum = currentSpan ? parseInt(currentSpan.textContent.trim()) : 0;
  const totalNum = totalSpan ? parseInt(totalSpan.textContent.trim()) : 0;

  const dir = document.getElementById('mvr-direction')?.value || 'rtl';
  const isRTL = dir === 'rtl';
  mvrLog(`autoGoNext: page ${currentNum}/${totalNum}, lastReadPageNum=${lastReadPageNum}, dir=${dir}`);

  // LAST PAGE CHECK
  // RTL: page 1 is last (pages count down). LTR: page N is last (pages count up).
  const isLastPage = isRTL
    ? (currentNum > 0 && currentNum <= 1)
    : (currentNum > 0 && totalNum > 0 && currentNum >= totalNum);
  if (isLastPage) {
    const nextChapterUrl = findNextChapterUrl();
    mvrLog(`Last page detected (page=${currentNum}, dir=${dir})! next_chapter_url: ${nextChapterUrl}`);
    if (nextChapterUrl) {
      _countdownNavigate(nextChapterUrl);
    } else {
      setStatus('Done — last page of last chapter.');
    }
    return;
  }

  // WRAP-AROUND CHECK
  // RTL: wrap = page went UP (e.g. 2→14). LTR: wrap = page went DOWN (e.g. 13→1).
  const wrapped = isRTL
    ? (lastReadPageNum > 0 && currentNum > 0 && currentNum > lastReadPageNum)
    : (lastReadPageNum > 0 && currentNum > 0 && currentNum < lastReadPageNum);
  if (wrapped) {
    const nextChapterUrl = findNextChapterUrl();
    mvrLog(`Wrap detected: was page ${lastReadPageNum}, now ${currentNum}. next_chapter_url: ${nextChapterUrl}`);
    if (nextChapterUrl) {
      _countdownNavigate(nextChapterUrl);
    } else {
      setStatus('Done — no next chapter found.');
    }
    return;
  }

  lastReadPageNum = currentNum;

  mvrLog(`Sending MVR_CLICK_PAGE to advance from page ${currentNum} (dir=${dir})`);
  chrome.runtime.sendMessage({ type: 'MVR_CLICK_PAGE', direction: dir });

  // After click, wait for page transition to fully complete before reading
  setTimeout(() => {
    if (!autoReadEnabled || speaking) return;

    // Re-check page number after the click
    const newSpan = document.querySelector('span.current-page');
    const newNum = newSpan ? parseInt(newSpan.textContent.trim()) : 0;
    mvrLog(`Post-click: page is now ${newNum} (was ${currentNum})`);

    // If page didn't change or wrapped around — go to next chapter
    // RTL: forward = page number decreases, wrap = page went up
    // LTR: forward = page number increases, wrap = page went down
    const postClickWrap = isRTL
      ? (newNum > 0 && newNum >= currentNum && currentNum <= 2)
      : (newNum > 0 && currentNum > 0 && newNum <= currentNum && currentNum >= totalNum - 1);
    if (postClickWrap) {
      const nextChapterUrl = findNextChapterUrl();
      mvrLog(`Post-click wrap detected (${currentNum}->${newNum}). next_chapter_url: ${nextChapterUrl}`);
      if (nextChapterUrl) {
        _countdownNavigate(nextChapterUrl);
        return;
      }
    }

    if (autoReadEnabled && !speaking) {
      readPage();
    }
  }, 1500);
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
  lastScrollY = window.scrollY;
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

    const result = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { type: 'READ_SCREEN', cropRect, dpr: window.devicePixelRatio || 2, pageTitle: document.title, pageUrl: location.href, readingDirection: document.getElementById('mvr-direction')?.value || 'rtl' },
        (res) => {
          if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
          resolve(res);
        }
      );
    });

    if (panel) panel.style.display = '';
    if (!speaking) return;

    if (!result?.ok) {
      const err = result?.error || 'Unknown error';
      setStatus(err === 'SERVER_NOT_RUNNING' ? 'Server not running. Start: python server/server.py' : 'Error: ' + err);
      setButtons(false); speaking = false;
      return;
    }

    const bubbles = result.bubbles || [];

    if (bubbles.length === 0) {
      // No dialogue — auto-skip to next page immediately if auto-read is on
      mvrLog(`readPage: 0 bubbles found, autoReadEnabled=${autoReadEnabled}`);
      if (autoReadEnabled) {
        setStatus('No dialogue — skipping...');
        setButtons(false); speaking = false;
        autoGoNext();
        return;
      }
      setStatus('No dialogue found.');
      setButtons(false); speaking = false;
      return;
    }

    if (result.timing) {
      console.log('[MVR] Timing:', result.timing);
    }

    setStatus('Found ' + bubbles.length + ' bubble' + (bubbles.length !== 1 ? 's' : ''));
    drawBubbles(bubbles);

    const allText = bubbles.map((b, i) => `[${i + 1}] ${b.text}`).join('\n');
    showText(allText);

    // Pre-fetch ALL bubble audio in parallel for zero-gap playback
    totalBubbles = bubbles.length;
    const audioCache = new Array(bubbles.length).fill(null);
    const audioPromises = bubbles.map((b, idx) => {
      const p = fetchTTSAudio(b.text).then(url => { audioCache[idx] = url; return url; });
      return p;
    });

    skipDirection = 0;
    for (let i = 0; i < bubbles.length; i++) {
      if (!speaking) break;
      currentBubbleIndex = i;
      setActiveBubble(i);
      setProgress(i + 1, bubbles.length);
      setStatus(`Reading ${i + 1}/${bubbles.length}`);

      // Wait for this bubble's audio (likely already cached from batch pre-fetch)
      let audioData = audioCache[i] || await audioPromises[i];

      // Re-fetch if voice was changed while pre-fetching
      if (_prefetchInvalid) {
        _prefetchInvalid = false;
        audioData = await fetchTTSAudio(bubbles[i].text);
      }

      // Check if skip happened during fetch
      if (skipDirection !== 0) {
        const newIdx = i + skipDirection;
        skipDirection = 0;
        if (newIdx >= 0 && newIdx < bubbles.length) {
          i = newIdx - 1;
        }
        continue;
      }

      if (!speaking) break;

      // Pre-buffer NEXT bubble's audio while this one plays (zero-gap transition)
      if (i + 1 < bubbles.length) {
        const nextAudio = audioCache[i + 1] || await audioPromises[i + 1];
        if (nextAudio) preloadNextAudio(nextAudio);
      }

      // Play the audio
      if (audioData) {
        await playAudioData(audioData);
      } else {
        await speak(bubbles[i].text);
      }

      // Handle skip after playback interrupted
      if (skipDirection !== 0) {
        const newIdx = i + skipDirection;
        skipDirection = 0;
        if (newIdx >= 0 && newIdx < bubbles.length) {
          i = newIdx - 1;
        }
        continue;
      }

      // Wait while paused, then brief natural pause between bubbles
      while (paused && speaking && skipDirection === 0) await new Promise(r => setTimeout(r, 100));
      if (skipDirection !== 0) {
        const newIdx = i + skipDirection;
        skipDirection = 0;
        if (newIdx >= 0 && newIdx < bubbles.length) {
          i = newIdx - 1;
        }
        continue;
      }
      if (speaking && i < bubbles.length - 1) {
        await new Promise(r => setTimeout(r, 30));
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
  }

  const shouldAutoNext = speaking && autoReadEnabled;
  mvrLog(`readPage end: speaking=${speaking}, autoRead=${autoReadEnabled}, shouldAutoNext=${shouldAutoNext}`);
  setButtons(false);
  speaking = false;

  if (shouldAutoNext) {
    autoGoNext();
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

// ─── AI TTS (local Piper via server) ─────────────────────────────────────────

async function fetchTTSAudio(text, _retryCount = 0) {
  /**Fetch audio from server via background.js proxy. Returns blob URL or null.
   * Retries once on failure before giving up (no SpeechSynthesis fallback).*/
  try {
    const speed = parseFloat(document.getElementById('mvr-speed')?.value || '0.85');
    const res = await chrome.runtime.sendMessage({
      type: 'MVR_TTS', text, voice: selectedVoice, speed,
    });
    if (!res || !res.ok || !res.audio) {
      if (_retryCount === 0) {
        await new Promise(r => setTimeout(r, 500));
        return fetchTTSAudio(text, 1);
      }
      return null;
    }
    // Decode base64 audio back to blob
    const binary = atob(res.audio);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: res.type || 'audio/wav' });
    return URL.createObjectURL(blob);
  } catch (err) {
    if (_retryCount === 0) {
      await new Promise(r => setTimeout(r, 500));
      return fetchTTSAudio(text, 1);
    }
    return null;
  }
}

let _preloadedAudio = null; // Pre-buffered Audio element for next bubble

function preloadNextAudio(blobUrl) {
  /**Pre-buffer the next bubble's audio so it plays instantly when needed.*/
  if (!blobUrl) return;
  try {
    const audio = new Audio(blobUrl);
    audio.preload = 'auto';
    audio.load(); // Start buffering immediately
    _preloadedAudio = { url: blobUrl, audio };
  } catch(e) { _preloadedAudio = null; }
}

function playAudioData(blobUrl) {
  /**Play pre-fetched audio blob URL. Respects pause state. Returns promise that resolves when done.*/
  return new Promise(async (resolve) => {
    if (!speaking) { resolve(); return; }
    while (paused && speaking && skipDirection === 0) await new Promise(r => setTimeout(r, 100));
    if (!speaking || skipDirection !== 0) { resolve(); return; }
    _audioResolve = resolve;
    try {
      let audio;
      // Use pre-buffered audio if it matches this URL
      if (_preloadedAudio && _preloadedAudio.url === blobUrl) {
        audio = _preloadedAudio.audio;
        _preloadedAudio = null;
      } else {
        audio = new Audio(blobUrl);
      }
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
    if (pauseBtn) { pauseBtn.textContent = '\u25B6'; pauseBtn.classList.add('active'); }
    if (currentAudio) currentAudio.pause();
    setStatus('Paused');
  } else {
    if (pauseBtn) { pauseBtn.textContent = '\u275A\u275A'; pauseBtn.classList.remove('active'); }
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
