<?php
session_start();
include("db.php");

if (!isset($_SESSION['ID'])) {
    header("Location: index.php");
    exit();
}

// Hardcoded Render backend URL — no need for the user to type it in anymore.
$SALINGO_API_BASE = "https://salingo-api.onrender.com";
?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>SALINGO — Translate</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
  :root {
    --bg: #0E1422;
    --panel: #131b2e;
    --panel-border: #1f2b45;
    --gold: #ffce1b;
    --text: #e9ecf5;
    --text-dim: #8b93a8;
    --ok: #3ddc84;
    --err: #ff5d5d;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
  body { background: var(--bg); color: var(--text); min-height: 100vh; padding: 40px 20px; }
  .wrap { max-width: 960px; margin: 0 auto; }
  header { margin-bottom: 32px; display: flex; align-items: center; justify-content: space-between; }
  header .eyebrow { color: var(--gold); letter-spacing: 3px; font-size: 12px; text-transform: uppercase; }
  header h1 { font-size: 28px; margin-top: 6px; }
  header p { color: var(--text-dim); margin-top: 6px; font-size: 14px; }
  .back-link { color: var(--text-dim); text-decoration: none; font-size: 14px; display: inline-flex; align-items: center; gap: 6px; }
  .back-link:hover { color: var(--gold); }

  .status-bar {
    display: flex; gap: 10px; align-items: center; margin-bottom: 28px;
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 10px;
    padding: 12px 16px; font-size: 13px; color: var(--text-dim);
  }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 12px;
    padding: 20px;
  }
  .card h2 { font-size: 16px; color: var(--gold); margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
  .card h2 .num { background: rgba(255,206,27,0.12); color: var(--gold); width: 22px; height: 22px;
    border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; }

  label.field-label { display: block; font-size: 12px; color: var(--text-dim); margin: 12px 0 4px; }
  input[type=text], textarea, select {
    width: 100%; background: #0b1120; border: 1px solid var(--panel-border); color: var(--text);
    border-radius: 8px; padding: 10px 12px; font-size: 14px; resize: vertical;
  }
  textarea { min-height: 90px; }
  select { appearance: none; }

  .direction-toggle { display: flex; gap: 8px; margin-top: 12px; }
  .direction-toggle button {
    flex: 1; background: #0b1120; border: 1px solid var(--panel-border); color: var(--text-dim);
    padding: 8px; border-radius: 8px; cursor: pointer; font-size: 12px;
  }
  .direction-toggle button.active { border-color: var(--gold); color: var(--gold); background: rgba(255,206,27,0.08); }

  button.primary {
    margin-top: 16px; width: 100%; background: var(--gold); color: #14100a; border: none;
    padding: 12px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;
  }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }

  .result { margin-top: 16px; padding: 14px; border-radius: 8px; background: #0b1120; border: 1px solid var(--panel-border);
    font-size: 14px; line-height: 1.5; min-height: 20px; white-space: pre-wrap; }
  .result.ok { border-color: var(--ok); }
  .result.err { border-color: var(--err); color: var(--err); }
  .meta { font-size: 11px; color: var(--text-dim); margin-top: 8px; }

  .file-drop { margin-top: 12px; border: 1px dashed var(--panel-border); border-radius: 8px; padding: 16px;
    text-align: center; color: var(--text-dim); font-size: 13px; }
  .file-drop input { display: none; }
  .file-drop label { cursor: pointer; color: var(--gold); }

  .trained-list { margin-top: 24px; }
  .trained-list h2 { font-size: 14px; color: var(--text-dim); margin-bottom: 8px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { background: rgba(255,206,27,0.1); color: var(--gold); border: 1px solid rgba(255,206,27,0.3);
    padding: 4px 10px; border-radius: 999px; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="eyebrow">SALINGO</div>
      <h1>Translate</h1>
      <p>Powered by the SALINGO translation memory service.</p>
    </div>
    <a class="back-link" href="languageManagement.php"><i class="fa-solid fa-arrow-left"></i> Back to Language Management</a>
  </header>

  <div class="status-bar">
    <span>Service status:</span>
    <span id="healthStatus">checking...</span>
  </div>

  <div class="grid">
    <div class="card">
      <h2><span class="num">1</span> Translate</h2>

      <div style="display:flex; gap:8px; align-items:flex-end;">
        <div style="flex:1;">
          <label class="field-label">From</label>
          <select id="fromLang" onchange="onLangSelectChange('fromLang','fromLangCustom')">
            <option value="Kapampangan">Kapampangan</option>
            <option value="Tagalog">Tagalog</option>
            <option value="English">English</option>
            <option value="__custom__">Other...</option>
          </select>
          <input type="text" id="fromLangCustom" placeholder="Type language" style="display:none; margin-top:6px;" />
        </div>
        <button type="button" onclick="swapLangs()" title="Swap" style="height:38px; width:38px; flex-shrink:0; background:#0b1120; border:1px solid var(--panel-border); color:var(--gold); border-radius:8px; cursor:pointer;">⇄</button>
        <div style="flex:1;">
          <label class="field-label">To</label>
          <select id="toLang" onchange="onLangSelectChange('toLang','toLangCustom')">
            <option value="Tagalog">Tagalog</option>
            <option value="English" selected>English</option>
            <option value="Kapampangan">Kapampangan</option>
            <option value="__custom__">Other...</option>
          </select>
          <input type="text" id="toLangCustom" placeholder="Type language" style="display:none; margin-top:6px;" />
        </div>
      </div>

      <label class="field-label">Text to translate</label>
      <textarea id="translateText" placeholder="Type or paste text here..."></textarea>

      <button class="primary" id="translateBtn" onclick="doTranslate()">Translate</button>

      <div id="translateResult" class="result" style="display:none;"></div>
    </div>

    <div class="card">
      <h2><span class="num">2</span> Speech to Text</h2>
      <p class="meta" style="margin-bottom:12px;">Speech-to-text uses Gemini's audio understanding — works for any language, including Kapampangan. Text-to-speech uses your browser's built-in voices, which don't include Kapampangan.</p>

      <label class="field-label" style="margin-top:4px;">Please Select Language you want to Speak</label>
      <select id="speakLang" onchange="validateSpeechLangs()">
        <option value="Kapampangan" selected>Kapampangan</option>
        <option value="Tagalog">Tagalog</option>
        <option value="English">English</option>
      </select>

     <label class="field-label" style="margin-top:4px;">Please Select Language you want to Translate</label>
      <select id="speechToLang" onchange="validateSpeechLangs()">
        <option value="Tagalog" selected>Tagalog</option>
        <option value="English">English</option>
        <option value="Kapampangan">Kapampangan</option>
      </select>
      <div id="speechLangWarning" class="meta" style="display:none; color:var(--err); margin-top:6px;">
        "Speak" and "Translate" languages must be different.
      </div>





      <button class="primary" id="micBtn" onclick="toggleRecording()" style="margin-top:8px;">
        <i class="fa-solid fa-microphone"></i> Start recording
      </button>
      <div id="sttResult" class="result" style="display:none;"></div>

      <label class="field-label" style="margin-top:10px;">Translation Result</label>
      <textarea id="sttTranslationResult" placeholder="Translation result will appear here..." readonly style="margin-top:6px;"></textarea>




      <label class="field-label" style="margin-top:20px;">Text → Speech</label>
      <select id="speechLang">
        <option value="en-US">English</option>
        <option value="fil-PH">Filipino / Tagalog</option>
        <option value="__kapampangan__">Kapampangan (no browser voice available)</option>
      </select>
      <div id="kapampanganNote" class="meta" style="display:none; color:var(--err); margin-top:6px;">
        No browser or OS ships a Kapampangan voice, so this text can't be spoken aloud. The transcription/translation above still works fine for Kapampangan.
      </div>
      <textarea id="ttsText" placeholder="Type text to hear it spoken, or click below to speak the translation result..." style="margin-top:10px;"></textarea>
      <div style="display:flex; gap:8px; margin-top:8px;">
        <button class="primary" id="speakBtn" onclick="speakText()" style="margin-top:0;">
          <i class="fa-solid fa-volume-high"></i> Speak
        </button>
        <button class="primary" id="stopSpeakBtn" onclick="stopSpeaking()" style="margin-top:0; background:#0b1120; color:var(--gold); border:1px solid var(--panel-border);">
          Stop
        </button>
      </div>
      <button class="primary" style="margin-top:8px; background:#0b1120; color:var(--gold); border:1px solid var(--panel-border);" onclick="useTranslationResultForSpeech()">
        Use last translation result ↑
      </button>
    </div>
  </div>
</div>

<script>
// Hardcoded from the PHP side — no user input needed.
const API_BASE = <?php echo json_encode(rtrim($SALINGO_API_BASE, '/')); ?>;
let lastTranslationText = '';

function onLangSelectChange(selectId, customId) {
  const select = document.getElementById(selectId);
  const custom = document.getElementById(customId);
  custom.style.display = select.value === '__custom__' ? 'block' : 'none';
}

function getLangValue(selectId, customId) {
  const select = document.getElementById(selectId);
  if (select.value === '__custom__') {
    return document.getElementById(customId).value.trim();
  }
  return select.value;
}

function swapLangs() {
  const fromSel = document.getElementById('fromLang');
  const toSel = document.getElementById('toLang');
  const tmp = fromSel.value;
  fromSel.value = toSel.value;
  toSel.value = tmp;
  onLangSelectChange('fromLang', 'fromLangCustom');
  onLangSelectChange('toLang', 'toLangCustom');
}

async function checkHealth() {
  const el = document.getElementById('healthStatus');
  el.innerText = 'checking...';
  try {
    const res = await fetch(API_BASE + '/health');
    if (res.ok) {
      el.innerText = '● online';
      el.style.color = '#3ddc84';
    } else {
      el.innerText = '● error ' + res.status;
      el.style.color = '#ff5d5d';
    }
  } catch (e) {
    el.innerText = '● unreachable (service may be waking up, try again in a moment)';
    el.style.color = '#ff5d5d';
  }
}

async function doTranslate() {
  const fromLang = getLangValue('fromLang', 'fromLangCustom');
  const toLang = getLangValue('toLang', 'toLangCustom');
  const text = document.getElementById('translateText').value.trim();
  const resultEl = document.getElementById('translateResult');
  const btn = document.getElementById('translateBtn');

  if (!fromLang || !toLang || !text) {
    resultEl.style.display = 'block';
    resultEl.className = 'result err';
    resultEl.innerText = 'Please select both languages and enter text.';
    return;
  }
  if (fromLang.toLowerCase() === toLang.toLowerCase()) {
    resultEl.style.display = 'block';
    resultEl.className = 'result err';
    resultEl.innerText = '"From" and "To" languages must be different.';
    return;
  }

  btn.disabled = true;
  btn.innerText = 'Translating...';
  resultEl.style.display = 'block';
  resultEl.className = 'result';
  resultEl.innerText = 'Translating (this may take up to a minute if the service was asleep)...';

  try {
    const res = await fetch(API_BASE + '/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language: fromLang, target_language: toLang })
    });
    const data = await res.json();

    if (!res.ok) {
      resultEl.className = 'result err';
      resultEl.innerText = data.detail || 'Translation failed.';
    } else {
      resultEl.className = 'result ok';
      resultEl.innerText = data.translation;
      lastTranslationText = data.translation;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.innerText = `Examples used from translation memory: ${data.examples_used} · Trained: ${data.trained ? 'yes' : 'no (using base model knowledge)'}`;
      resultEl.appendChild(meta);
    }
  } catch (e) {
    resultEl.className = 'result err';
    resultEl.innerText = 'Could not reach the translation service: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.innerText = 'Translate';
  }
}

// ---------------------------------------------------------------------
// Speech: language select behavior (warn if Kapampangan picked — no
// browser/OS ships a speech engine for it)
// ---------------------------------------------------------------------
document.getElementById('speechLang').addEventListener('change', function () {
  document.getElementById('kapampanganNote').style.display =
    this.value === '__kapampangan__' ? 'block' : 'none';
});

// ---------------------------------------------------------------------
// Prevent picking the same language for "Speak" and "Translate" in the
// Speech-to-Text card.
// ---------------------------------------------------------------------
function validateSpeechLangs() {
  const speakLang = document.getElementById('speakLang').value;
  const toLang = document.getElementById('speechToLang').value;
  const warningEl = document.getElementById('speechLangWarning');
  const micBtn = document.getElementById('micBtn');
  const same = speakLang.toLowerCase() === toLang.toLowerCase();

  warningEl.style.display = same ? 'block' : 'none';
  micBtn.disabled = same;
  return !same;
}

// ---------------------------------------------------------------------
// Speech → Text → Translation (records audio, sends to Gemini via our
// backend — works for ANY language, including Kapampangan, unlike the
// browser's built-in SpeechRecognition which has no Kapampangan support)
// ---------------------------------------------------------------------
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function toggleRecording() {
  const micBtn = document.getElementById('micBtn');
  const resultEl = document.getElementById('sttResult');

  if (isRecording) {
    mediaRecorder.stop();
    return;
  }

  if (!validateSpeechLangs()) {
    return;
  }

  if (!navigator.mediaDevices || !window.MediaRecorder) {
    resultEl.style.display = 'block';
    resultEl.className = 'result err';
    resultEl.innerText = 'Microphone recording is not supported in this browser.';
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    resultEl.style.display = 'block';
    resultEl.className = 'result err';
    resultEl.innerText = 'Microphone access was denied or unavailable: ' + e.message;
    return;
  }

  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunks.push(e.data);
  };

  mediaRecorder.onstart = () => {
    isRecording = true;
    micBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop recording';
    resultEl.style.display = 'block';
    resultEl.className = 'result';
    resultEl.innerText = 'Recording... click "Stop recording" when done.';
  };

  mediaRecorder.onstop = async () => {
    isRecording = false;
    micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i> Start recording';
    stream.getTracks().forEach((t) => t.stop());

    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    await sendAudioForTranslation(audioBlob);
  };

  mediaRecorder.start();
}

async function sendAudioForTranslation(audioBlob) {
  const resultEl = document.getElementById('sttResult');
  const sttTranslationResultEl = document.getElementById('sttTranslationResult');
  const fromLang = document.getElementById('speakLang').value;
  const toLang = document.getElementById('speechToLang').value;

  resultEl.className = 'result';
  resultEl.innerText = 'Transcribing and translating audio (this can take a bit)...';
  sttTranslationResultEl.value = '';

  const form = new FormData();
  form.append('audio', audioBlob, 'recording.webm');
  form.append('source_language', fromLang);
  form.append('target_language', toLang);

  try {
    const res = await fetch(API_BASE + '/translate-audio', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      resultEl.className = 'result err';
      resultEl.innerText = data.detail || 'Audio translation failed.';
      return;
    }

    resultEl.className = 'result ok';
    resultEl.innerText = `Heard (${fromLang}): "${data.transcript}"`;

    // Result shows up right away in the Speech-to-Text card's own box.
    sttTranslationResultEl.value = data.translation;
    lastTranslationText = data.translation;
  } catch (e) {
    resultEl.className = 'result err';
    resultEl.innerText = 'Could not reach the translation service: ' + e.message;
  }
}

// ---------------------------------------------------------------------
// Text → Speech (Web Speech API — browser/OS built-in, no server call)
// ---------------------------------------------------------------------
function speakText() {
  const text = document.getElementById('ttsText').value.trim();
  const langSelect = document.getElementById('speechLang');

  if (langSelect.value === '__kapampangan__') {
    alert('No browser or OS ships a Kapampangan voice yet, so this text cannot be spoken aloud.');
    return;
  }
  if (!text) {
    alert('Type some text first, or click "Use last translation result".');
    return;
  }
  if (!('speechSynthesis' in window)) {
    alert('Text-to-speech is not supported in this browser.');
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = langSelect.value;
  window.speechSynthesis.speak(utterance);
}

function stopSpeaking() {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
}

function useTranslationResultForSpeech() {
  if (!lastTranslationText) {
    alert('No translation result yet — translate something first.');
    return;
  }
  document.getElementById('ttsText').value = lastTranslationText;
}

checkHealth();
validateSpeechLangs();
</script>
</body>
</html>
