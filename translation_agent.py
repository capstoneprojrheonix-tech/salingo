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
import uuid
import itertools
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
# Parsing: CSV / XLSX (tabular — ANY number of language columns)
# ---------------------------------------------------------------------
# Instead of only reading two fixed columns, every column that doesn't
# look like an ID/index column is treated as a language column (using its
# header as the language name), and a training pair is built for every
# 2-column combination found — e.g. a sheet with English, Tagalog, and
# Kapampangan columns yields English↔Tagalog, English↔Kapampangan, and
# Tagalog↔Kapampangan pairs, all from a single upload.

_ID_COLUMN_NAME_RE = re.compile(r"^(id|no\.?|num(ber)?|#|index|row)$", re.IGNORECASE)


def _is_id_like_column(col_name, series: pd.Series) -> bool:
    if _ID_COLUMN_NAME_RE.match(str(col_name).strip()):
        return True

    values = [str(v).strip() for v in series if str(v).strip() and str(v).strip().lower() != "nan"]
    if not values:
        return True  # fully empty column — not usable as a language column

    # If every non-empty value is a plain integer, it's almost certainly an
    # auto-numbered index column, not translated text.
    return all(v.lstrip("-").isdigit() for v in values)


def _detect_language_columns(df: pd.DataFrame) -> list:
    return [col for col in df.columns if not _is_id_like_column(col, df[col])]


def _all_pairs_from_dataframe(df: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """
    Returns {(lang_x, lang_y): [(text_x, text_y), ...]} for every pairwise
    combination of detected language columns, using the original column
    headers (trimmed) as the language names.
    """
    if df.empty:
        return {}

    lang_cols = _detect_language_columns(df)
    if len(lang_cols) < 2:
        return {}

    result: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for col_x, col_y in itertools.combinations(lang_cols, 2):
        pairs = []
        for _, row in df.iterrows():
            x_text = str(row.get(col_x, "")).strip()
            y_text = str(row.get(col_y, "")).strip()
            if x_text and y_text and x_text.lower() != "nan" and y_text.lower() != "nan":
                pairs.append((x_text, y_text))
        if pairs:
            result[(str(col_x).strip(), str(col_y).strip())] = pairs
    return result


def _all_pairs_from_csv(csv_path: str) -> dict[tuple[str, str], list[tuple[str, str]]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return _all_pairs_from_dataframe(df)


def _all_pairs_from_xlsx(xlsx_path: str) -> dict[tuple[str, str], list[tuple[str, str]]]:
    # sheet_name=None loads every sheet as {sheet_name: DataFrame}, instead
    # of silently defaulting to just the first sheet — some exports (e.g.
    # FAQ workbooks) split content across multiple sheets like
    # "Questions" / "Answers".
    sheets = pd.read_excel(xlsx_path, dtype=str, engine="openpyxl", sheet_name=None)

    combined: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for sheet_df in sheets.values():
        sheet_df = sheet_df.fillna("")
        for combo, pairs in _all_pairs_from_dataframe(sheet_df).items():
            combined.setdefault(combo, []).extend(pairs)
    return combined


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


def _train_pairs_into_store(lang_a: str, lang_b: str, pairs: list[tuple[str, str]]) -> int:
    """Embed `pairs` and merge them into the (lang_a <-> lang_b) store. Returns count added."""
    if not pairs:
        return 0

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

    new_embeddings = _embed_texts(texts)

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

    return len(pairs)


def train_language(lang_a: str, file_path: str, lang_b: str = "English") -> dict:
    """
    Ingest a CSV, XLSX, or PDF dataset into the translation memory.

    For CSV/XLSX files, EVERY language column found in the file (any
    column that isn't an ID/index column) is used — not just two fixed
    ones. A separate trained (bidirectional) pair is created for every
    2-column combination, e.g. a sheet with English, Tagalog, and
    Kapampangan columns yields English↔Tagalog, English↔Kapampangan, and
    Tagalog↔Kapampangan pairs from a single upload.

    `lang_a`/`lang_b` are used as the pair for PDF glossaries only, since
    those are parsed line-by-line as strictly two-column (source/target).

    Returns: {"success": bool, "count": int, "message": str}
    """
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".csv":
            pairs_by_combo = _all_pairs_from_csv(file_path)
        elif ext in (".xlsx", ".xlsm"):
            pairs_by_combo = _all_pairs_from_xlsx(file_path)
        elif ext == ".pdf":
            pdf_pairs = _pairs_from_pdf(file_path)
            pairs_by_combo = {(lang_a, lang_b): pdf_pairs} if pdf_pairs else {}
        else:
            return {"success": False, "count": 0, "message": f"Unsupported file type: {ext}"}
    except Exception as e:
        return {"success": False, "count": 0, "message": f"Could not read file: {e}"}

    pairs_by_combo = {combo: p for combo, p in pairs_by_combo.items() if p}

    if not pairs_by_combo:
        return {
            "success": False,
            "count": 0,
            "message": (
                "No valid sentence pairs found in the file. Make sure it has at least two "
                "language columns — one per language — with matching translations in each "
                "row (not just single-language word lists or tagging data)."
            ),
        }

    total_count = 0
    trained_summaries = []
    errors = []
    for (col_a, col_b), pairs in pairs_by_combo.items():
        try:
            count = _train_pairs_into_store(col_a, col_b, pairs)
        except Exception as e:
            errors.append(f"{col_a} <-> {col_b}: {e}")
            continue
        total_count += count
        trained_summaries.append(f"{col_a} ↔ {col_b} ({count})")

    if total_count == 0:
        return {
            "success": False,
            "count": 0,
            "message": "Training failed for all detected language pairs: " + "; ".join(errors),
        }

    message = f"Trained on {total_count} pairs from {ext} file: " + ", ".join(trained_summaries)
    if errors:
        message += f". Some pairs failed: {'; '.join(errors)}"

    return {"success": True, "count": total_count, "message": message}


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
