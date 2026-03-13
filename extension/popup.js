let activeServer = null;

document.getElementById('btn-open').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  try {
    const response = await chrome.tabs.sendMessage(tab.id, { type: 'MVR_ACTIVATE' });
    if (response?.ok) {
      window.close();
      return;
    }
  } catch (_) {
    // Content script not loaded yet — inject it
  }

  try {
    await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ['styles.css'] });
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
    await new Promise(r => setTimeout(r, 200));
    await chrome.tabs.sendMessage(tab.id, { type: 'MVR_ACTIVATE' });
    window.close();
  } catch (err) {
    document.getElementById('status').textContent =
      'Cannot run on this page. Open a manga website first.';
  }
});

// Check server via background.js (single source of truth)
async function checkServer() {
  const dot = document.getElementById('server-dot');
  const label = document.getElementById('server-label');
  const detail = document.getElementById('server-detail');

  // First check current state without triggering reconnect
  try {
    const state = await chrome.runtime.sendMessage({ type: 'MVR_GET_STATE' });
    if (state && state.state === 'CONNECTED') {
      activeServer = state.server;
      dot.classList.add('online');
      label.textContent = 'Server online';
      label.classList.add('connected');
      detail.textContent = '';
      loadModelStatus();
      return;
    }
  } catch (_) {}

  // Not connected — trigger connection attempt
  dot.classList.add('loading');
  label.textContent = 'Connecting...';
  detail.textContent = '';

  try {
    const res = await chrome.runtime.sendMessage({ type: 'MVR_ENSURE_SERVER' });
    if (res && res.ok) {
      activeServer = res.server;
      dot.classList.remove('loading');
      dot.classList.add('online');
      label.textContent = 'Server online';
      label.classList.add('connected');
      detail.textContent = '';
      loadModelStatus();
      return;
    }
  } catch (_) {}

  dot.classList.remove('loading');
  dot.classList.add('offline');
  label.textContent = 'No server';
  detail.textContent = 'Run: python3 ~/MangaVoice/server/server_lite.py';
}

// Load model status pills
async function loadModelStatus() {
  if (!activeServer) return;
  try {
    const healthRes = await fetch(activeServer + '/health', { signal: AbortSignal.timeout(2000) });
    if (!healthRes.ok) return;
    const h = healthRes.ok ? await healthRes.json() : null;

    const pills = document.getElementById('model-pills');
    if (pills) pills.style.display = 'flex';

    const set = (id, on) => {
      const el = document.getElementById(id);
      if (el) el.className = 'pill ' + (on ? 'on' : '');
    };

    if (h) {
      set('pill-florence', h.quality_pass);
      set('pill-paddle', !!h.ocr);
      set('pill-bubble', h.detector);
      set('pill-tts', h.tts !== undefined ? h.tts : true);

      const detail = document.getElementById('server-detail');
      if (detail) detail.textContent = h.ocr || '';
    }
  } catch (_) {}
}

checkServer();
