// Manga Voice Reader — Content Script

// Panel only appears when user clicks "Open Reader" in the popup.
let speaking = false;
let currentUtterance = null;
let panelCreated = false;

// ─── Activation gate ─────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'MVR_ACTIVATE') {
    if (!panelCreated) {
      createUI();
      panelCreated = true;
    } else {
      const panel = document.getElementById('mvr');
      if (panel) panel.style.display = '';
    }
    sendResponse({ ok: true });
  }
});

// ─── UI ──────────────────────────────────────────────────────────────────────

function createUI() {
  if (document.getElementById('mvr')) return;

  const d = document.createElement('div');
  d.id = 'mvr';
  d.innerHTML = `
    <div id="mvr-head">
      <span>Manga Reader</span>
      <button id="mvr-x">\u2715</button>
    </div>
    <div id="mvr-body">
      <div id="mvr-status">Press "Read Page" to start</div>
      <button id="mvr-go">Read Page</button>
      <button id="mvr-stop" style="display:none">Stop</button>
      <div id="mvr-text"></div>
      <label>Speed: <input type="range" id="mvr-speed" min="0.5" max="2" step="0.1" value="0.85"> <span id="mvr-sv">0.85x</span></label>
    </div>`;
  document.body.appendChild(d);

  // Drag to move
  const head = d.querySelector('#mvr-head');
  let dragging = false, dx = 0, dy = 0;
  head.addEventListener('mousedown', (e) => {
    dragging = true;
    dx = e.clientX - d.offsetLeft;
    dy = e.clientY - d.offsetTop;
    e.preventDefault();
  }, true);
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    d.style.left = (e.clientX - dx) + 'px';
    d.style.top = (e.clientY - dy) + 'px';
    d.style.right = 'auto';
    d.style.bottom = 'auto';
  }, true);
  document.addEventListener('mouseup', () => { dragging = false; }, true);

  document.getElementById('mvr-x').addEventListener('click', () => {
    stopSpeaking(); removeHighlights(); d.remove(); panelCreated = false;
  }, true);
  document.getElementById('mvr-go').addEventListener('click', readPage, true);
  document.getElementById('mvr-stop').addEventListener('click', stopSpeaking, true);
  document.getElementById('mvr-speed').addEventListener('input', (e) => {
    document.getElementById('mvr-sv').textContent = parseFloat(e.target.value).toFixed(2) + 'x';
  }, true);
}

function setStatus(msg) {
  const el = document.getElementById('mvr-status');
  if (el) el.textContent = msg;
}
function showText(msg) {
  const el = document.getElementById('mvr-text');
  if (el) el.textContent = msg;
}
function setButtons(reading) {
  const go = document.getElementById('mvr-go');
  const st = document.getElementById('mvr-stop');
  if (go) go.style.display = reading ? 'none' : '';
  if (st) st.style.display = reading ? '' : 'none';
}

// ─── Find manga crop rect ──────────────────────────────────────────────────

function getMangaCropRect() {
  const elements = document.querySelectorAll('img, canvas');
  let best = null, bestArea = 0;
  for (const el of elements) {
    const r = el.getBoundingClientRect();
    if (r.width * r.height > bestArea && r.width > 200 && r.height > 200) {
      bestArea = r.width * r.height;
      best = r;
    }
  }
  if (!best) return null;
  const pad = 5;
  return {
    left: Math.max(0, Math.floor(best.left) - pad),
    top: Math.max(0, Math.floor(best.top) - pad),
    width: Math.ceil(best.width) + pad * 2,
    height: Math.ceil(best.height) + pad * 2,
  };
}

// ─── Stale highlight cleanup on scroll ───────────────────────────────────────

let lastScrollY = window.scrollY;
let scrollDebounce = null;

window.addEventListener('scroll', () => {
  clearTimeout(scrollDebounce);
  scrollDebounce = setTimeout(() => {
    const delta = Math.abs(window.scrollY - lastScrollY);
    if (delta > 100) {
      removeHighlights();
      lastScrollY = window.scrollY;
      if (!speaking) {
        setStatus('Scrolled. Press "Read Page" to scan this view.');
      }
    }
  }, 300);
}, { passive: true });

// ─── Highlighting ─────────────────────────────────────────────────────────────

function removeHighlights() {
  const el = document.getElementById('mvr-overlay');
  if (el) el.remove();
}

function drawBubbles(bubbles) {
  removeHighlights();
  if (!bubbles || bubbles.length === 0) return;

  const overlay = document.createElement('div');
  overlay.id = 'mvr-overlay';
  overlay.style.cssText = `
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none; z-index: 2147483646;
  `;

  const colors = [
    { bg: 'rgba(255,69,96,0.25)', border: 'rgba(255,69,96,0.7)' },
    { bg: 'rgba(69,160,255,0.25)', border: 'rgba(69,160,255,0.7)' },
    { bg: 'rgba(69,255,130,0.25)', border: 'rgba(69,255,130,0.7)' },
    { bg: 'rgba(255,200,69,0.25)', border: 'rgba(255,200,69,0.7)' },
    { bg: 'rgba(200,69,255,0.25)', border: 'rgba(200,69,255,0.7)' },
  ];

  bubbles.forEach((bubble, i) => {
    const c = colors[i % colors.length];
    const pad = 4;

    // Server returns coords in CSS pixels relative to viewport — use directly
    const outline = document.createElement('div');
    outline.className = 'mvr-bubble mvr-b' + i;
    outline.dataset.idx = i;
    outline.style.cssText = `
      position: absolute;
      left: ${bubble.left - pad}px; top: ${bubble.top - pad}px;
      width: ${bubble.width + pad * 2}px; height: ${bubble.height + pad * 2}px;
      border: 2px solid ${c.border}; border-radius: 6px;
      background: ${c.bg};
      transition: all 0.3s ease;
    `;

    const label = document.createElement('div');
    label.style.cssText = `
      position: absolute; top: -12px; left: -2px;
      background: ${c.border}; color: #fff;
      font-size: 11px; font-weight: bold; padding: 1px 5px;
      border-radius: 4px; font-family: sans-serif;
    `;
    label.textContent = i + 1;
    outline.appendChild(label);
    overlay.appendChild(outline);
  });

  document.body.appendChild(overlay);
  lastScrollY = window.scrollY;
}

function setActiveBubble(activeIdx) {
  const bubbleEls = document.querySelectorAll('.mvr-bubble');
  bubbleEls.forEach(el => {
    const idx = parseInt(el.dataset.idx);
    if (idx === activeIdx) {
      el.style.borderColor = 'rgba(255,69,96,1)';
      el.style.borderWidth = '3px';
      el.style.boxShadow = '0 0 15px rgba(255,69,96,0.6)';
    } else {
      el.style.borderColor = 'rgba(150,150,150,0.3)';
      el.style.borderWidth = '1px';
      el.style.boxShadow = 'none';
    }
  });
}

// ─── Main Flow ────────────────────────────────────────────────────────────────

async function readPage() {
  speaking = true;
  setButtons(true);
  setStatus('Taking screenshot...');
  showText('');
  removeHighlights();

  const cropRect = getMangaCropRect();

  try {
    const panel = document.getElementById('mvr');
    if (panel) panel.style.display = 'none';
    await new Promise(r => setTimeout(r, 50));

    setStatus('Detecting text...');

    const result = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { type: 'READ_SCREEN', cropRect, dpr: window.devicePixelRatio || 2 },
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
      if (err === 'SERVER_NOT_RUNNING') {
        setStatus('Local server not running. Start: python server/server.py');
      } else {
        setStatus('Error: ' + err);
      }
      setButtons(false);
      return;
    }

    const bubbles = result.bubbles || [];

    if (bubbles.length === 0) {
      setStatus('No dialogue found. This may be an action page.');
      setButtons(false);
      speaking = false;
      return;
    }

    // Log timing info
    if (result.timing) {
      console.log('[MVR] Timing:', result.timing);
    }

    drawBubbles(bubbles);

    const allText = bubbles.map((b, i) => `[${i + 1}] ${b.text}`).join('\n');
    showText(allText);

    for (let i = 0; i < bubbles.length; i++) {
      if (!speaking) break;
      setActiveBubble(i);
      setStatus(`Reading ${i + 1} of ${bubbles.length}...`);
      await speak(bubbles[i].text);
      if (speaking && i < bubbles.length - 1) {
        await new Promise(r => setTimeout(r, 500));
      }
    }

    if (speaking) {
      setStatus('Done! Scroll and press "Read Page" again.');
    }

  } catch (err) {
    setStatus('Error: ' + err.message);
    console.error('[MVR]', err);
    const panel = document.getElementById('mvr');
    if (panel) panel.style.display = '';
  }

  setButtons(false);
  speaking = false;
}

// ─── TTS (browser voice — instant, no network lag) ───────────────────────────

async function speak(text) {
  if (!speaking) return;
  const rate = parseFloat(document.getElementById('mvr-speed')?.value || '0.85');

  return new Promise((resolve) => {
    if (!speaking) { resolve(); return; }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = rate;
    u.lang = 'en-US';
    u.volume = 1;
    u.onend = resolve;
    u.onerror = resolve;
    window.speechSynthesis.speak(u);
  });
}

function stopSpeaking() {
  speaking = false;
  window.speechSynthesis.cancel();
  currentUtterance = null;
  removeHighlights();
  setStatus('Stopped.');
  setButtons(false);
}
