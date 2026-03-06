// Background service worker — handles screenshot relay to localhost + TTS

const LOCAL_SERVER = 'http://127.0.0.1:5055';
const GOOGLE_TTS_API = 'https://texttospeech.googleapis.com/v1/text:synthesize';
const GOOGLE_API_KEY = 'AIzaSyCn4R9nNogsMSJoOUeUmbfOBvjPAcNqhwE';

chrome.runtime.onInstalled.addListener(() => {
  console.log('[MVR] Installed.');
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'READ_SCREEN') {
    handleReadScreen(request.cropRect, request.dpr || 2)
      .then(r => sendResponse(r))
      .catch(e => sendResponse({ ok: false, error: e.message }));
    return true;
  }
  if (request.type === 'SPEAK') {
    handleSpeak(request.text, request.rate || 1.0)
      .then(r => sendResponse(r))
      .catch(e => sendResponse({ ok: false, error: e.message }));
    return true;
  }
});

// ─── Screenshot + Localhost relay ────────────────────────────────────────────

async function handleReadScreen(cropRect, dpr) {
  let dataUrl;
  try {
    dataUrl = await chrome.tabs.captureVisibleTab(null, { format: 'png' });
  } catch (err) {
    throw new Error('Screenshot failed: ' + err.message);
  }

  let res;
  try {
    res = await fetch(`${LOCAL_SERVER}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: dataUrl,
        cropRect: cropRect || null,
        dpr: dpr,
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

// ─── TTS with Google Cloud Neural2 ──────────────────────────────────────────

function detectEmotion(text) {
  const t = text.trim();
  if (/!{2,}/.test(t)) return 'shout';
  if (/\?!|!\?/.test(t)) return 'surprise';
  if (/\?$/.test(t)) return 'question';
  if (/\.{3}/.test(t)) return 'trailing';
  if (/!$/.test(t)) return 'exclaim';
  return 'normal';
}

function buildSSML(text, rate) {
  const emotion = detectEmotion(text);
  const basePercent = Math.round(rate * 100);
  let pitch, speakRate;

  switch (emotion) {
    case 'shout':
      pitch = '+2st'; speakRate = `${basePercent + 5}%`;
      break;
    case 'surprise':
      pitch = '+2st'; speakRate = `${basePercent}%`;
      break;
    case 'question':
      pitch = '+1st'; speakRate = `${basePercent - 5}%`;
      break;
    case 'trailing':
      pitch = '-1st'; speakRate = `${basePercent - 10}%`;
      break;
    case 'exclaim':
      pitch = '+1st'; speakRate = `${basePercent}%`;
      break;
    default:
      pitch = '+0st'; speakRate = `${basePercent}%`;
  }

  const rateNum = parseInt(speakRate);
  const clampedRate = Math.max(50, Math.min(200, rateNum));
  speakRate = `${clampedRate}%`;

  let clean = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  if (clean === clean.toUpperCase() && clean.length > 2) {
    clean = clean.charAt(0) + clean.slice(1).toLowerCase();
  }

  return `<speak>
    <prosody rate="${speakRate}" pitch="${pitch}">
      ${clean}
    </prosody>
  </speak>`;
}

async function getVoiceName() {
  const data = await chrome.storage.local.get('voiceName');
  return data.voiceName || 'en-US-Neural2-D';
}

async function handleSpeak(text, rate) {
  const ssml = buildSSML(text, rate);
  const voiceName = await getVoiceName();
  console.log('[MVR] TTS emotion:', detectEmotion(text), 'voice:', voiceName, 'text:', text.substring(0, 50));

  let res;
  try {
    res = await fetch(`${GOOGLE_TTS_API}?key=${GOOGLE_API_KEY}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input: { ssml },
        voice: { languageCode: 'en-US', name: voiceName },
        audioConfig: {
          audioEncoding: 'MP3',
          speakingRate: 1.0,
          effectsProfileId: ['headphone-class-device'],
        }
      })
    });
  } catch (netErr) {
    throw new Error('TTS network error: ' + netErr.message);
  }

  if (res.status === 429) {
    throw new Error('QUOTA_EXCEEDED');
  }
  if (res.status === 403) {
    const body = await res.text();
    if (body.includes('quota') || body.includes('QUOTA')) {
      throw new Error('QUOTA_EXCEEDED');
    }
    throw new Error('TTS auth failed (403). Check API key.');
  }
  if (!res.ok) {
    const err = await res.text();
    console.error('[MVR] TTS error:', err);
    throw new Error('TTS failed: ' + res.status);
  }

  const data = await res.json();
  return { ok: true, audioBase64: data.audioContent };
}
