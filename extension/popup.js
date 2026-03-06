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
      'Cannot run on this page. Go to a manga website first.';
    document.getElementById('status').style.color = '#e94560';
  }
});

// Check if local server is running
async function checkServer() {
  const el = document.getElementById('status');
  if (!el) return;
  try {
    const res = await fetch('http://127.0.0.1:5055/health', { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      const data = await res.json();
      el.textContent = 'Server: running';
      el.style.color = '#69ff82';
    } else {
      el.textContent = 'Server: error';
      el.style.color = '#e94560';
    }
  } catch (_) {
    el.textContent = 'Server: not running';
    el.style.color = '#e94560';
  }
}

checkServer();

// ─── AI TTS toggle ──────────────────────────────────────────────────────────

const aiToggle = document.getElementById('ai-tts-toggle');
const aiStatus = document.getElementById('ai-tts-status');

// Load saved state
chrome.storage.local.get('aiTtsEnabled', (data) => {
  if (aiToggle) aiToggle.checked = !!data.aiTtsEnabled;
});

// Save on toggle
if (aiToggle) {
  aiToggle.addEventListener('change', () => {
    chrome.storage.local.set({ aiTtsEnabled: aiToggle.checked });
  });
}

// Check if Piper is available on server
async function checkAiTts() {
  if (!aiStatus) return;
  try {
    const res = await fetch('http://127.0.0.1:5055/tts/status', { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      const data = await res.json();
      if (data.available) {
        aiStatus.textContent = 'AI voice: ' + (data.model || 'ready');
        aiStatus.style.color = '#69ff82';
      } else {
        aiStatus.textContent = 'AI voice: not configured';
        aiStatus.style.color = '#777';
      }
    }
  } catch (_) {
    aiStatus.textContent = '';
  }
}

checkAiTts();
