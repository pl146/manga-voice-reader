const PC_SERVER = 'http://192.168.2.183:5055';
const PC_LAUNCHER = 'http://192.168.2.183:5056';
const MAC_SERVER = 'http://127.0.0.1:5055';
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

// Check server: PC first, then MacBook fallback
async function checkServer() {
  const dot = document.getElementById('server-dot');
  const label = document.getElementById('server-label');
  const detail = document.getElementById('server-detail');

  // 1. Try PC server
  try {
    const res = await fetch(PC_SERVER + '/health', { signal: AbortSignal.timeout(1500) });
    if (res.ok) {
      const data = await res.json();
      dot.classList.add('online');
      label.textContent = 'PC server online';
      detail.textContent = data.ocr || '';
      activeServer = PC_SERVER;
      return;
    }
  } catch (_) {}

  // 2. Try PC launcher
  try {
    dot.classList.add('offline');
    label.textContent = 'Starting PC server...';
    detail.textContent = 'Waking up...';
    const r = await fetch(PC_LAUNCHER + '/start', { signal: AbortSignal.timeout(60000) });
    const d = await r.json();
    if (d.ok) {
      dot.classList.remove('offline');
      dot.classList.add('online');
      label.textContent = 'PC server online';
      detail.textContent = 'Auto-started!';
      activeServer = PC_SERVER;
      return;
    }
  } catch (_) {}

  // 3. Fall back to MacBook
  try {
    const res = await fetch(MAC_SERVER + '/health', { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      const data = await res.json();
      dot.classList.remove('offline');
      dot.classList.add('online');
      label.textContent = 'MacBook server online';
      detail.textContent = (data.ocr || '') + ' (local)';
      activeServer = MAC_SERVER;
      return;
    }
  } catch (_) {}

  dot.classList.add('offline');
  label.textContent = 'No server available';
  detail.textContent = 'PC off & MacBook server not running';
}

checkServer();

// AI TTS toggle
const aiToggle = document.getElementById('ai-tts-toggle');
const aiStatus = document.getElementById('ai-tts-status');

chrome.storage.local.get('aiTtsEnabled', (data) => {
  if (aiToggle) aiToggle.checked = !!data.aiTtsEnabled;
});

if (aiToggle) {
  aiToggle.addEventListener('change', () => {
    chrome.storage.local.set({ aiTtsEnabled: aiToggle.checked });
  });
}

async function checkAiTts() {
  if (!aiStatus || !activeServer) return;
  try {
    const res = await fetch(activeServer + '/tts/status', { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      const data = await res.json();
      if (data.available) {
        aiStatus.textContent = data.model || 'Ready';
        aiStatus.style.color = '#3fb950';
        if (aiToggle) aiToggle.checked = true;
      } else {
        aiStatus.textContent = 'Not configured';
      }
    }
  } catch (_) {}
}

// Check TTS after server check completes
setTimeout(checkAiTts, 2000);

// Forensic page dump
document.getElementById('btn-dump').addEventListener('click', async () => {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Dumping...';
  statusEl.style.color = '#f0c040';
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try {
    await chrome.tabs.sendMessage(tab.id, { type: 'MVR_DUMP_BUTTONS' });
    statusEl.textContent = 'Done! Check Downloads for mvr-dump.txt';
    statusEl.style.color = '#3fb950';
  } catch (err) {
    statusEl.textContent = 'Open Reader on the page first, then dump.';
    statusEl.style.color = '#f85149';
  }
});
