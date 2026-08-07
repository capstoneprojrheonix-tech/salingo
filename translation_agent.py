"""
SALINGO Translation Engine (Gemini edition, multi-language-pair)
------------------------------------------------------------------
"Training" here means building a translation memory (RAG), not fine-tuning.

Unlike the original version (which only supported "Language <-> English"),
this version supports ANY language pair, e.g. Kapampangan <-> Tagalog,
Kapampangan <-> English, Tagalog <-> English, etc.

Each trained pair is stored under:
    vectorstores/<lang_a>__<lang_b>/embeddings.npy
    vectorstores/<lang_a>__<lang_b>/metadata.json
    vectorstores/<lang_a>__<lang_b>/info.json   (original casing of both names)

<lang_a> and <lang_b> are alphabetically sorted slugs, so training
"Kapampangan -> Tagalog" and "Tagalog -> Kapampangan" both land in the
same store (bidirectional).

Supported training file formats: .csv, .xlsx, .pdf (glossary-style).
"""

import os
import re
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "vectorstores"
VECTORSTORE_DIR.mkdir(exist_ok=True)
AUDIO_TRAIN_DIR = BASE_DIR / "audio_training"
AUDIO_TRAIN_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
EMBEDDING_BATCH_SIZE = 100
MAX_AUDIO_EXAMPLES_PER_REQUEST = 5  # how many reference clips to feed Gemini per transcription
MAX_AUDIO_SAMPLE_BYTES = 5 * 1024 * 1024  # 5 MB cap per training clip

# TTS (voice cloning) — reuses the same audio_training/<language>/ samples
# collected by train_audio_sample(), but as reference voice for synthesis
# instead of as few-shot calibration for transcription.
MIN_TTS_REFERENCE_SECONDS = 3  # a clip shorter than this is too short to clone a voice from
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE_FALLBACK = {
    # XTTS doesn't know "Kapampangan" as a language code — it has no
    # dedicated phoneme set for it. We pass the closest supported code
    # so the model's text-to-phoneme step doesn't error out; the actual
    # voice timbre/accent still comes from the reference clip itself.
    "kapampangan": "tl",
    "tagalog": "tl",
    "filipino": "tl",
    "english": "en",
}

_tts_model = None


def _get_tts_model():
    """
    Lazily loads the Coqui XTTS v2 model. This is intentionally NOT loaded
    at import time — it's a large model (needs a few GB of RAM/VRAM) and
    most deployments of this service (e.g. a small Render instance) won't
    want to pay that cost unless /synthesize-speech is actually called.

    Requires: pip install TTS
    In production, this model should run on its own worker with a GPU
    (or at least several CPU cores) — training/inference here is far
    heavier than the Gemini API calls used elsewhere in this file.
    """
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS  # local import: optional heavy dependency
        _tts_model = TTS(XTTS_MODEL_NAME)
    return _tts_model

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "_", name.strip().lower())
    return slug.strip("_") or "unknown"


def _pair_slug(lang_a: str, lang_b: str) -> str:
    a, b = sorted([_slug(lang_a), _slug(lang_b)])
    return f"{a}__{b}"


def _store_path(lang_a: str, lang_b: str) -> Path:
    return VECTORSTORE_DIR / _pair_slug(lang_a, lang_b)


def _embeddings_file(store_path: Path) -> Path:
    return store_path / "embeddings.npy"


def _metadata_file(store_path: Path) -> Path:
    return store_path / "metadata.json"


def _info_file(store_path: Path) -> Path:
    return store_path / "info.json"


# ---------------------------------------------------------------------
# Parsing: CSV / XLSX (tabular, two language columns)
# ---------------------------------------------------------------------

def _detect_columns(df: pd.DataFrame, lang_a: str, lang_b: str) -> tuple[str, str]:
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    a_col = lang_a.strip().lower() if lang_a.strip().lower() in cols else None
    b_col = lang_b.strip().lower() if lang_b.strip().lower() in cols else None

    # Backward-compatible fallback for the old "english"/"en" convention.
    if b_col is None and lang_b.strip().lower() == "english":
        b_col = next((c for c in ("english", "en") if c in cols), None)
    if a_col is None and lang_a.strip().lower() == "english":
        a_col = next((c for c in ("english", "en") if c in cols), None)

    if a_col is None or b_col is None:
        if len(cols) < 2:
            raise ValueError(f"File must have at least 2 columns, got: {df.columns.tolist()}")
        a_col = a_col or cols[0]
        b_col = b_col or cols[1]

    return a_col, b_col


def _pairs_from_dataframe(df: pd.DataFrame, lang_a: str, lang_b: str) -> list[tuple[str, str]]:
    if df.empty:
        return []
    a_col, b_col = _detect_columns(df, lang_a, lang_b)

    pairs = []
    for _, row in df.iterrows():
        a_text = str(row.get(a_col, "")).strip()
        b_text = str(row.get(b_col, "")).strip()
        if a_text and b_text and a_text.lower() != "nan" and b_text.lower() != "nan":
            pairs.append((a_text, b_text))
    return pairs


def _pairs_from_csv(csv_path: str, lang_a: str, lang_b: str) -> list[tuple[str, str]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return _pairs_from_dataframe(df, lang_a, lang_b)


def _pairs_from_xlsx(xlsx_path: str, lang_a: str, lang_b: str) -> list[tuple[str, str]]:
    df = pd.read_excel(xlsx_path, dtype=str, engine="openpyxl")
    df = df.fillna("")
    return _pairs_from_dataframe(df, lang_a, lang_b)


# ---------------------------------------------------------------------
# Parsing: PDF glossary (best-effort — glossary layouts vary a lot)
# ---------------------------------------------------------------------

_GLOSSARY_LINE_RE = re.compile(
    r"^\s*(?P<source>.+?)\s*(?:[-–—:]|\t|\s{3,})\s*(?P<target>.+?)\s*$"
)


def _pairs_from_pdf(pdf_path: str) -> list[tuple[str, str]]:
    reader = PdfReader(pdf_path)
    pairs = []

    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _GLOSSARY_LINE_RE.match(line)
            if not match:
                continue
            source = match.group("source").strip()
            target = match.group("target").strip()
            if not source or not target:
                continue
            if source.replace(" ", "").isdigit() or target.replace(" ", "").isdigit():
                continue
            pairs.append((source, target))

    return pairs


# ---------------------------------------------------------------------
# Embedding storage
# ---------------------------------------------------------------------

def _embed_texts(texts: list[str]) -> np.ndarray:
    client = get_client()
    vectors = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        resp = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        vectors.extend([e.values for e in resp.embeddings])
    return np.array(vectors, dtype=np.float32)


def _load_store(store_path: Path) -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(_embeddings_file(store_path))
    with open(_metadata_file(store_path), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings, metadata


def _save_store(store_path: Path, embeddings: np.ndarray, metadata: list[dict]):
    store_path.mkdir(parents=True, exist_ok=True)
    np.save(_embeddings_file(store_path), embeddings)
    with open(_metadata_file(store_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def train_language(lang_a: str, file_path: str, lang_b: str = "English") -> dict:
    """
    Ingest a CSV, XLSX, or PDF dataset and add it to the (lang_a <-> lang_b)
    translation memory. Creates the store if it doesn't exist, or merges
    into it.

    Returns: {"success": bool, "count": int, "message": str}
    """
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".csv":
            pairs = _pairs_from_csv(file_path, lang_a, lang_b)
        elif ext in (".xlsx", ".xlsm"):
            pairs = _pairs_from_xlsx(file_path, lang_a, lang_b)
        elif ext == ".pdf":
            pairs = _pairs_from_pdf(file_path)
        else:
            return {"success": False, "count": 0, "message": f"Unsupported file type: {ext}"}
    except Exception as e:
        return {"success": False, "count": 0, "message": f"Could not read file: {e}"}

    if not pairs:
        return {
            "success": False,
            "count": 0,
            "message": (
                f"No valid '{lang_a}' / '{lang_b}' sentence pairs found in the file. "
                "Make sure it has two columns — one per language — with matching translations "
                "in each row (not just single-language word lists or tagging data)."
            ),
        }

    texts: list[str] = []
    metadata: list[dict] = []
    for a_text, b_text in pairs:
        texts.append(a_text)
        metadata.append(
            {"source_lang": lang_a, "target_lang": lang_b, "source_text": a_text, "target_text": b_text}
        )
        texts.append(b_text)
        metadata.append(
            {"source_lang": lang_b, "target_lang": lang_a, "source_text": b_text, "target_text": a_text}
        )

    try:
        new_embeddings = _embed_texts(texts)
    except Exception as e:
        return {"success": False, "count": 0, "message": f"Embedding request failed: {e}"}

    store_path = _store_path(lang_a, lang_b)
    if _embeddings_file(store_path).exists():
        existing_embeddings, existing_metadata = _load_store(store_path)
        all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        all_metadata = existing_metadata + metadata
    else:
        all_embeddings = new_embeddings
        all_metadata = metadata

    _save_store(store_path, all_embeddings, all_metadata)
    with open(_info_file(store_path), "w", encoding="utf-8") as f:
        json.dump({"lang_a": lang_a, "lang_b": lang_b}, f, ensure_ascii=False)

    return {
        "success": True,
        "count": len(pairs),
        "message": f"Trained on {len(pairs)} pairs for '{lang_a}' <-> '{lang_b}' (from {ext} file).",
    }


def language_pair_is_trained(lang_a: str, lang_b: str) -> bool:
    return _embeddings_file(_store_path(lang_a, lang_b)).exists()


def list_trained_languages() -> list[str]:
    """Returns human-readable pair labels, e.g. ['Kapampangan ↔ Tagalog', 'Tagalog ↔ English']."""
    if not VECTORSTORE_DIR.exists():
        return []
    labels = []
    for p in VECTORSTORE_DIR.iterdir():
        if not (p.is_dir() and _embeddings_file(p).exists()):
            continue
        info_path = _info_file(p)
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            labels.append(f"{info['lang_a']} ↔ {info['lang_b']}")
        else:
            labels.append(p.name.replace("__", " ↔ "))
    return labels


def delete_language_pair(lang_a: str, lang_b: str) -> bool:
    path = _store_path(lang_a, lang_b)
    if path.exists():
        shutil.rmtree(path)
        return True
    return False


# ---------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------

def _cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    if matrix.shape[0] == 0:
        return []
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    scores = matrix_norm @ query_norm
    top_k = min(k, len(scores))
    return list(np.argsort(-scores)[:top_k])


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "(no matching examples found)"
    return "\n".join(
        f"- Source: {e['source_text']}\n  Target: {e['target_text']}" for e in examples
    )


def translate_text(
    text: str,
    source_language: str,
    target_language: str = "English",
    k: int = 4,
) -> dict:
    """
    Translate `text` from `source_language` into `target_language`.
    Works for ANY language pair (e.g. Kapampangan -> Tagalog), not just
    a fixed "-> English" direction.
    """
    store_path = _store_path(source_language, target_language)
    examples: list[dict] = []
    trained = _embeddings_file(store_path).exists()

    if trained:
        embeddings, metadata = _load_store(store_path)
        direction_indices = [
            i for i, m in enumerate(metadata)
            if m["source_lang"].strip().lower() == source_language.strip().lower()
            and m["target_lang"].strip().lower() == target_language.strip().lower()
        ]

        if direction_indices:
            sub_matrix = embeddings[direction_indices]
            query_vec = _embed_texts([text])[0]
            top_local = _cosine_top_k(query_vec, sub_matrix, k)
            examples = [metadata[direction_indices[i]] for i in top_local]

    system_prompt = (
        f"You are a professional translator working from '{source_language}' into "
        f"'{target_language}'. Use the example translations below (retrieved from a "
        "verified translation memory) to match terminology, tone, and phrasing. If no "
        "examples are relevant, translate using your own knowledge of both languages. "
        "Respond with ONLY the translated text — no explanations, no quotes, no extra "
        "commentary.\n\nExamples:\n" + _format_examples(examples)
    )
    user_prompt = f"Translate this text from {source_language} into {target_language}:\n\n{text}"

    client = get_client()
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )

    translation = response.text.strip()

    return {
        "translation": translation,
        "examples_used": len(examples),
        "trained": trained,
    }


# ---------------------------------------------------------------------
# Voice pronunciation training (audio reference samples)
# ---------------------------------------------------------------------
# Unlike train_language() (text glossary -> embeddings), this stores short
# audio clips + their correct transcript per language. There's no audio
# similarity search here — a capped batch of the most recently added
# samples for that language is fed to Gemini as few-shot reference audio
# on every transcription call, to calibrate it to the speaker's accent,
# pronunciation, and spelling conventions for that language.

def _audio_lang_dir(language: str) -> Path:
    return AUDIO_TRAIN_DIR / _slug(language)


def _audio_metadata_file(lang_dir: Path) -> Path:
    return lang_dir / "metadata.json"


def _load_audio_metadata(lang_dir: Path) -> list[dict]:
    meta_file = _audio_metadata_file(lang_dir)
    if not meta_file.exists():
        return []
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_audio_metadata(lang_dir: Path, metadata: list[dict]):
    lang_dir.mkdir(parents=True, exist_ok=True)
    with open(_audio_metadata_file(lang_dir), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def train_audio_sample(language: str, file_path: str, mime_type: str, transcript: str) -> dict:
    """
    Save a short audio clip + its correct transcript as a pronunciation
    reference sample for `language`.

    Returns: {"success": bool, "count": int, "message": str}
    """
    transcript = transcript.strip()
    if not transcript:
        return {"success": False, "count": 0, "message": "Please provide the correct transcript for this recording."}

    src = Path(file_path)
    if not src.exists():
        return {"success": False, "count": 0, "message": "Audio file not found."}

    size = src.stat().st_size
    if size > MAX_AUDIO_SAMPLE_BYTES:
        return {
            "success": False,
            "count": 0,
            "message": (
                f"Audio clip is too large ({size // 1024} KB). Keep clips under "
                f"{MAX_AUDIO_SAMPLE_BYTES // (1024 * 1024)} MB — a few seconds is enough."
            ),
        }

    lang_dir = _audio_lang_dir(language)
    lang_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_audio_metadata(lang_dir)

    sample_id = uuid.uuid4().hex[:12]
    ext = src.suffix or ".webm"
    dest_filename = f"{sample_id}{ext}"
    shutil.copyfile(src, lang_dir / dest_filename)

    metadata.append(
        {
            "id": sample_id,
            "filename": dest_filename,
            "mime_type": mime_type,
            "transcript": transcript,
            "language": language,
        }
    )
    _save_audio_metadata(lang_dir, metadata)

    return {
        "success": True,
        "count": len(metadata),
        "message": f"Saved pronunciation sample for '{language}' ({len(metadata)} total).",
    }


def list_audio_samples(language: str) -> list[dict]:
    """Returns [{"id": ..., "transcript": ...}, ...] for a given language."""
    lang_dir = _audio_lang_dir(language)
    metadata = _load_audio_metadata(lang_dir)
    return [{"id": m["id"], "transcript": m["transcript"]} for m in metadata]


def delete_audio_sample(language: str, sample_id: str) -> bool:
    lang_dir = _audio_lang_dir(language)
    metadata = _load_audio_metadata(lang_dir)
    match = next((m for m in metadata if m["id"] == sample_id), None)
    if not match:
        return False

    file_path = lang_dir / match["filename"]
    if file_path.exists():
        file_path.unlink()

    metadata = [m for m in metadata if m["id"] != sample_id]
    _save_audio_metadata(lang_dir, metadata)
    return True


def _load_audio_examples(language: str, max_examples: int = MAX_AUDIO_EXAMPLES_PER_REQUEST) -> list[dict]:
    lang_dir = _audio_lang_dir(language)
    metadata = _load_audio_metadata(lang_dir)
    if not metadata:
        return []

    chosen = metadata[-max_examples:]  # most recently added samples
    examples = []
    for m in chosen:
        file_path = lang_dir / m["filename"]
        if not file_path.exists():
            continue
        with open(file_path, "rb") as f:
            data = f.read()
        examples.append({"data": data, "mime_type": m["mime_type"], "transcript": m["transcript"]})
    return examples



def _pick_reference_clip(language: str) -> Optional[Path]:
    """
    Picks the best available reference clip for voice cloning: the most
    recently added sample for this language, since newer samples are
    likely to have been recorded with the current mic/setup in mind.
    """
    lang_dir = _audio_lang_dir(language)
    metadata = _load_audio_metadata(lang_dir)
    if not metadata:
        return None

    for m in reversed(metadata):  # most recent first
        candidate = lang_dir / m["filename"]
        if candidate.exists():
            return candidate
    return None


def synthesize_speech(text: str, language: str) -> dict:
    """
    Synthesize `text` spoken in `language`, cloning the voice from the
    trained pronunciation samples in audio_training/<language>/ (the
    same samples collected by train_audio_sample() for STT calibration).

    This is how languages with no OS/browser TTS voice (e.g. Kapampangan)
    can still be "spoken" — the voice comes from real recorded samples of
    a speaker of that language, not from a pre-built system voice.

    Returns: {"success": bool, "audio": bytes | None, "mime_type": str,
              "message": str}
    """
    text = text.strip()
    if not text:
        return {"success": False, "audio": None, "mime_type": "", "message": "No text to speak."}

    reference_clip = _pick_reference_clip(language)
    if reference_clip is None:
        return {
            "success": False,
            "audio": None,
            "mime_type": "",
            "message": (
                f"No voice samples trained for '{language}' yet. Record a few short, clear "
                "clips via the pronunciation trainer first — even 3-5 short samples from one "
                "speaker are enough to clone a voice from."
            ),
        }

    xtts_lang = XTTS_LANGUAGE_FALLBACK.get(language.strip().lower(), "en")

    try:
        model = _get_tts_model()
    except Exception as e:
        return {
            "success": False,
            "audio": None,
            "mime_type": "",
            "message": f"TTS engine unavailable: {e}",
        }

    tmp_dir = Path(tempfile.mkdtemp())
    out_path = tmp_dir / "speech.wav"
    try:
        model.tts_to_file(
            text=text,
            speaker_wav=str(reference_clip),
            language=xtts_lang,
            file_path=str(out_path),
        )
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        return {
            "success": False,
            "audio": None,
            "mime_type": "",
            "message": f"Speech synthesis failed: {e}",
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "success": True,
        "audio": audio_bytes,
        "mime_type": "audio/wav",
        "message": "ok",
    }


def _transcribe_audio(
    file_path: str,
    mime_type: str,
    spoken_language: str,
    reference_examples: Optional[list[dict]] = None,
) -> str:
    """
    Transcribe a short audio clip using Gemini's audio understanding.
    Works for any language the model has been told to expect, including
    languages with no dedicated browser speech-recognition support
    (e.g. Kapampangan).

    If `reference_examples` is provided (each a dict with "data",
    "mime_type", "transcript"), they're fed to Gemini first as few-shot
    reference recordings — trained pronunciation samples for this
    language — to calibrate its understanding of accent, pronunciation,
    and spelling before it transcribes the real clip.
    """
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    contents: list = [
        f"You are an expert speech transcriber for the '{spoken_language}' language, "
        "including regional accents and non-standard spellings."
    ]

    if reference_examples:
        contents.append(
            "Here are reference recordings from a trained speaker in this language, each "
            "followed by its correct transcript. Use these ONLY to calibrate your "
            "understanding of pronunciation and spelling conventions — do NOT transcribe "
            "these reference clips themselves."
        )
        for example in reference_examples:
            contents.append(types.Part.from_bytes(data=example["data"], mime_type=example["mime_type"]))
            contents.append(f"Reference transcript: {example['transcript']}")

    contents.append(
        f"Now transcribe ONLY the following new audio clip. Respond with ONLY the "
        f"verbatim transcript in '{spoken_language}' — no explanations, no quotes, no "
        "extra commentary. If the audio is silent or unintelligible, respond with an "
        "empty string."
    )
    contents.append(types.Part.from_bytes(data=audio_bytes, mime_type=mime_type))

    client = get_client()
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.0),
    )

    return (response.text or "").strip()


def transcribe_and_translate_audio(
    file_path: str,
    mime_type: str,
    source_language: str,
    target_language: str,
) -> dict:
    """
    Transcribe spoken audio in `source_language`, then translate the
    transcript into `target_language` using the same translation-memory
    pipeline as translate_text().

    Returns: {"transcript": str, "translation": str, "examples_used": int, "trained": bool}
    """
    reference_examples = _load_audio_examples(source_language)
    transcript = _transcribe_audio(file_path, mime_type, source_language, reference_examples)

    if not transcript:
        return {
            "transcript": "",
            "translation": "",
            "examples_used": 0,
            "trained": False,
            "message": "Could not make out any speech in the recording. Please try again.",
        }

    result = translate_text(transcript, source_language, target_language)

    return {
        "transcript": transcript,
        "translation": result["translation"],
        "examples_used": result["examples_used"],
        "trained": result["trained"],
        "pronunciation_samples_used": len(reference_examples),
    }
