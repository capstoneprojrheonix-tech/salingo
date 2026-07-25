<?php
session_start();
include("db.php");


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

      <label class="field-label">Language</label>
      <input type="text" id="translateLang" placeholder="e.g. Tagalog" />

      <div class="direction-toggle">
        <button id="dirToEn" class="active" onclick="setDirection('to_english')">Language → English</button>
        <button id="dirFromEn" onclick="setDirection('from_english')">English → Language</button>
      </div>

      <label class="field-label">Text to translate</label>
      <textarea id="translateText" placeholder="Type or paste text here..."></textarea>

      <button class="primary" id="translateBtn" onclick="doTranslate()">Translate</button>

      <div id="translateResult" class="result" style="display:none;"></div>
    </div>

    <div class="card">
      <h2><span class="num">2</span> Train a language</h2>

      <label class="field-label">Language name</label>
      <input type="text" id="trainLang" placeholder="e.g. Tagalog" />

      <label class="field-label">Dataset (CSV: language column + english column)</label>
      <div class="file-drop">
        <input type="file" id="trainFile" accept=".csv" onchange="fileChosen()" />
        <label for="trainFile">Choose CSV file</label>
        <div id="fileName" class="meta"></div>
      </div>

      <button class="primary" id="trainBtn" onclick="doTrain()">Train</button>

      <div id="trainResult" class="result" style="display:none;"></div>

      <div class="trained-list">
        <h2>Currently trained languages</h2>
        <div class="chips" id="trainedChips"><span class="meta">— none loaded yet —</span></div>
      </div>
    </div>
  </div>
</div>

<script>
// Hardcoded from the PHP side — no user input needed.
const API_BASE = <?php echo json_encode(rtrim($SALINGO_API_BASE, '/')); ?>;

let direction = 'to_english';

function setDirection(d) {
  direction = d;
  document.getElementById('dirToEn').classList.toggle('active', d === 'to_english');
  document.getElementById('dirFromEn').classList.toggle('active', d === 'from_english');
}

function fileChosen() {
  const f = document.getElementById('trainFile').files[0];
  document.getElementById('fileName').innerText = f ? f.name : '';
}

async function checkHealth() {
  const el = document.getElementById('healthStatus');
  el.innerText = 'checking...';
  try {
    const res = await fetch(API_BASE + '/health');
    if (res.ok) {
      el.innerText = '● online';
      el.style.color = '#3ddc84';
      loadTrainedLanguages();
    } else {
      el.innerText = '● error ' + res.status;
      el.style.color = '#ff5d5d';
    }
  } catch (e) {
    el.innerText = '● unreachable (service may be waking up, try again in a moment)';
    el.style.color = '#ff5d5d';
  }
}

async function loadTrainedLanguages() {
  const container = document.getElementById('trainedChips');
  try {
    const res = await fetch(API_BASE + '/languages');
    const data = await res.json();
    const langs = data.trained_languages || [];
    container.innerHTML = langs.length
      ? langs.map(l => `<span class="chip">${l}</span>`).join('')
      : '<span class="meta">— no languages trained yet —</span>';
  } catch (e) {
    container.innerHTML = '<span class="meta">could not load</span>';
  }
}

async function doTranslate() {
  const lang = document.getElementById('translateLang').value.trim();
  const text = document.getElementById('translateText').value.trim();
  const resultEl = document.getElementById('translateResult');
  const btn = document.getElementById('translateBtn');

  if (!lang || !text) {
    resultEl.style.display = 'block';
    resultEl.className = 'result err';
    resultEl.innerText = 'Please enter both a language and text.';
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
      body: JSON.stringify({ text, language: lang, direction })
    });
    const data = await res.json();

    if (!res.ok) {
      resultEl.className = 'result err';
      resultEl.innerText = data.detail || 'Translation failed.';
    } else {
      resultEl.className = 'result ok';
      resultEl.innerText = data.translation;
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

async function doTrain() {
  const lang = document.getElementById('trainLang').value.trim();
  const file = document.getElementById('trainFile').files[0];
  const resultEl = document.getElementById('trainResult');
  const btn = document.getElementById('trainBtn');

  if (!lang || !file) {
    resultEl.style.display = 'block';
    resultEl.className = 'result err';
    resultEl.innerText = 'Please enter a language name and choose a CSV file.';
    return;
  }

  btn.disabled = true;
  btn.innerText = 'Training...';
  resultEl.style.display = 'block';
  resultEl.className = 'result';
  resultEl.innerText = 'Uploading and building translation memory...';

  const form = new FormData();
  form.append('language', lang);
  form.append('file', file);

  try {
    const res = await fetch(API_BASE + '/train', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      resultEl.className = 'result err';
      resultEl.innerText = data.detail || 'Training failed.';
    } else {
      resultEl.className = 'result ok';
      resultEl.innerText = data.message;
      loadTrainedLanguages();
    }
  } catch (e) {
    resultEl.className = 'result err';
    resultEl.innerText = 'Could not reach the training service: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.innerText = 'Train';
  }
}

checkHealth();
</script>
</body>
</html>