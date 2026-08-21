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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "vectorstores"
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Pronunciation/voice recordings (used to be a local audio_training/<lang>/
# folder — moved to Supabase Postgres's "recordingManagement" table so
# recordings survive redeploys/restarts/spin-downs on hosts with an
# ephemeral filesystem (e.g. Render's free tier).
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


def _get_db_connection():
    """
    Opens a fresh connection to the Supabase Postgres database.
    Requires SUPABASE_DB_URL in the environment, e.g.:
        postgresql://postgres:<password>@<host>:5432/postgres
    Find this under Supabase dashboard -> Project Settings -> Database
    -> Connection string -> URI.
    """
    if not SUPABASE_DB_URL:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. Add it to your .env (local) or "
            "your host's environment variables (Render/Hostinger/etc)."
        )
    return psycopg2.connect(SUPABASE_DB_URL)


# ---------------------------------------------------------------------
# languageManagement table (admin dashboard rows) — Supabase-backed
# ---------------------------------------------------------------------
# This is separate from the RAG translation memory in vectorstores/.
# It's the bookkeeping table languageManagement.php's dashboard reads
# and writes: one row per "language added" action, tracking its display
# name, a running translation-pair count, the uploaded filename, and an
# Active/Inactive status. PHP on InfinityFree can't reach Supabase
# directly (outbound DB ports are blocked there), so PHP calls these
# through HTTP endpoints on this Python service instead — see main.py's
# /language-records routes and ai_bridge.php's matching PHP functions.

def db_insert_language_record(language_name: str, translation: int, file_name: str, status: str = "Active") -> int:
    """Insert a new languageManagement row. Returns the new row's ID."""
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO "languageManagement" ("LanguageName", "Translation", "FileName", "Status")
                VALUES (%s, %s, %s, %s)
                RETURNING "ID"
                """,
                (language_name, translation, file_name, status),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
    finally:
        conn.close()
    return new_id


def db_get_language_record(record_id: int) -> Optional[dict]:
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT "ID", "LanguageName", "Translation", "FileName", "Status" '
                'FROM "languageManagement" WHERE "ID" = %s',
                (record_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"id": row[0], "language_name": row[1], "translation": row[2], "file_name": row[3], "status": row[4]}


def db_update_language_record(
    record_id: int,
    language_name: str,
    status: str,
    file_name: Optional[str] = None,
    translation: Optional[int] = None,
) -> bool:
    """Update a languageManagement row. If file_name/translation are None,
    those columns are left unchanged (matches the old PHP behavior where
    editing without picking a new file kept the existing FileName/Translation)."""
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            if file_name is not None:
                cur.execute(
                    """
                    UPDATE "languageManagement"
                    SET "LanguageName" = %s, "FileName" = %s, "Translation" = %s, "Status" = %s
                    WHERE "ID" = %s
                    """,
                    (language_name, file_name, translation, status, record_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE "languageManagement"
                    SET "LanguageName" = %s, "Status" = %s
                    WHERE "ID" = %s
                    """,
                    (language_name, status, record_id),
                )
            updated = cur.rowcount > 0
            conn.commit()
    finally:
        conn.close()
    return updated


def db_list_language_records() -> list[dict]:
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT "ID", "LanguageName", "Translation", "FileName", "Status" '
                'FROM "languageManagement" ORDER BY "ID" ASC'
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "language_name": r[1], "translation": r[2], "file_name": r[3], "status": r[4]}
        for r in rows
    ]

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
EMBEDDING_BATCH_SIZE = 100
MAX_AUDIO_EXAMPLES_PER_REQUEST = 5  # how many reference clips to feed Gemini per transcription
MAX_AUDIO_SAMPLE_BYTES = 5 * 1024 * 1024  # 5 MB cap per training clip

# TTS (premade voice) — speaks translated text using one of ElevenLabs'
# stock/premade voices (e.g. "Josh"), NOT a cloned voice. This works on
# the ElevenLabs FREE plan, unlike Instant Voice Cloning which requires a
# paid plan.
#
# Why ElevenLabs instead of a self-hosted model (XTTS/Chatterbox/etc): those
# all need several GB of RAM to load, which reliably OOM-crashes small hosts
# (Render free/starter tier). ElevenLabs runs the model on their servers —
# this backend just makes a couple of small HTTPS calls.
#
# NOTE: ElevenLabs has no official Kapampangan language support — the
# premade voice's timbre/accent is whatever that stock voice sounds like,
# and pronunciation accuracy for Kapampangan text is best-effort from
# their multilingual model, not guaranteed.
#
# NOTE: audio generated on an ElevenLabs FREE plan key cannot be used
# commercially (no monetization, requires attribution). For a live/public
# deployment, use a Starter plan ($6/mo+) key instead — nothing in this
# code needs to change, only the API key.
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_TTS_MODEL = "eleven_multilingual_v2"
# Name of the premade ElevenLabs voice to speak with, e.g. "Josh", "Rachel",
# "Bella". Must match a voice name visible under the account's Voice Library
# (My Voices / Default voices) exactly (case-insensitive match is used below).
ELEVENLABS_VOICE_NAME = os.getenv("ELEVENLABS_VOICE_NAME", "Josh")

_premade_voice_cache: dict[str, str] = {}  # voice_name (lowercased) -> voice_id, in-memory only

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

# ---------------------------------------------------------------------
# Parsing: CSV / XLSX (tabular, two language columns)
# ---------------------------------------------------------------------

# Column headers that are clearly not language text (row numbers, IDs,
# etc.) â€” excluded from guessing so they're never mistaken for a
# language column.
_ID_LIKE_COLUMNS = {"id", "no", "no.", "num", "number", "#", "index", "row", "count", "item"}


def _looks_like_id_column(col_name: str) -> bool:
    return col_name.strip().lower() in _ID_LIKE_COLUMNS


def _find_col(name: str, cols: list[str]) -> Optional[str]:
    """Exact match first, then substring match (handles things like a
    'Kapampangan Text' header when the user typed just 'Kapampangan')."""
    name = name.strip().lower()
    if not name:
        return None
    if name in cols:
        return name
    for c in cols:
        if name in c or c in name:
            return c
    return None


def _detect_columns(df: pd.DataFrame, lang_a: str, lang_b: str) -> tuple[str, str, Optional[str]]:
    """
    Figures out which two columns hold `lang_a` and `lang_b` text.

    Returns (a_col, b_col, note). `note` is None when the columns were
    matched with confidence â€” either by name, or because there was only
    one possible column left to guess. When there's genuine ambiguity
    (2+ equally-plausible columns and no name match to break the tie),
    this raises a clear error instead of silently guessing â€” guessing
    wrong here means silently training on the wrong language's data,
    which is worse than asking the person to type a name that matches
    a column header.
    """
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    a_col = _find_col(lang_a, cols)
    b_col = _find_col(lang_b, cols)

    # Backward-compatible fallback for the old "english"/"en" convention.
    if b_col is None and lang_b.strip().lower() == "english":
        b_col = next((c for c in ("english", "en") if c in cols), None)
    if a_col is None and lang_a.strip().lower() == "english":
        a_col = next((c for c in ("english", "en") if c in cols), None)

    if a_col and b_col and a_col != b_col:
        return a_col, b_col, None

    candidate_cols = [c for c in cols if not _looks_like_id_column(c)]

    if len(candidate_cols) < 2:
        raise ValueError(
            f"File must have at least 2 usable language columns (excluding "
            f"ID-style columns), got: {df.columns.tolist()}"
        )

    if a_col and not b_col:
        remaining = [c for c in candidate_cols if c != a_col]
        if len(remaining) == 1:
            b_col = remaining[0]
            return a_col, b_col, None
        raise ValueError(
            f"'{lang_b}' didn't match any column header, and there's more than "
            f"one column it could be ({remaining}) â€” too ambiguous to guess safely. "
            f"Type '{lang_b}' to exactly match one of these column names, or "
            f"rename the file's columns."
        )

    if b_col and not a_col:
        remaining = [c for c in candidate_cols if c != b_col]
        if len(remaining) == 1:
            a_col = remaining[0]
            return a_col, b_col, None
        raise ValueError(
            f"'{lang_a}' didn't match any column header, and there's more than "
            f"one column it could be ({remaining}) â€” too ambiguous to guess safely. "
            f"Type '{lang_a}' to exactly match one of these column names "
            f"(e.g. one of {remaining}), or rename the file's columns."
        )

    # Neither language name matched anything. Only safe to guess positionally
    # when there are EXACTLY 2 usable columns total (nothing else it could
    # mean). With 3+, silently picking the first two risks training the
    # wrong pair, so ask instead.
    if len(candidate_cols) == 2:
        a_col, b_col = candidate_cols[0], candidate_cols[1]
        note = (
            f"Neither '{lang_a}' nor '{lang_b}' matched a column header, so "
            f"the file's only two columns ('{a_col}' and '{b_col}') were used."
        )
        return a_col, b_col, note

    raise ValueError(
        f"Neither '{lang_a}' nor '{lang_b}' matched a column header, and this "
        f"file has {len(candidate_cols)} possible language columns ({candidate_cols}) "
        f"â€” too ambiguous to guess safely. Type language names that exactly match "
        f"two of these column names."
    )


def _pairs_from_dataframe(df: pd.DataFrame, lang_a: str, lang_b: str) -> tuple[list[tuple[str, str]], Optional[str]]:
    if df.empty:
        return [], None
    a_col, b_col, note = _detect_columns(df, lang_a, lang_b)

    pairs = []
    for _, row in df.iterrows():
        a_text = str(row.get(a_col, "")).strip()
        b_text = str(row.get(b_col, "")).strip()
        if a_text and b_text and a_text.lower() != "nan" and b_text.lower() != "nan":
            pairs.append((a_text, b_text))
    return pairs, note


def _pairs_from_csv(csv_path: str, lang_a: str, lang_b: str) -> tuple[list[tuple[str, str]], Optional[str]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return _pairs_from_dataframe(df, lang_a, lang_b)


def _pairs_from_xlsx(xlsx_path: str, lang_a: str, lang_b: str) -> tuple[list[tuple[str, str]], Optional[str]]:
    df = pd.read_excel(xlsx_path, dtype=str, engine="openpyxl")
    df = df.fillna("")
    return _pairs_from_dataframe(df, lang_a, lang_b)




# ---------------------------------------------------------------------
# Parsing: PDF glossary (best-effort â€” glossary layouts vary a lot)
# ---------------------------------------------------------------------

_GLOSSARY_LINE_RE = re.compile(
    r"^\s*(?P<source>.+?)\s*(?:[-â€“â€”:]|\t|\s{3,})\s*(?P<target>.+?)\s*$"
)


def _pairs_via_gemini_extraction(text: str, lang_a: str, lang_b: str) -> list[tuple[str, str]]:
    """
    Best-effort extraction of (source, target) translation pairs from
    unstructured text using Gemini itself. This is the fallback for PDFs
    (or any dataset) that aren't laid out as a clean two-column glossary â€”
    e.g. translations embedded in prose, tables that didn't extract
    cleanly, or mixed formatting â€” so training isn't limited to rigidly
    formatted files.
    """
    text = text[:60000]  # cap input â€” this is for glossary-style docs, not whole books

    lang_hint = f" The two languages involved are '{lang_a}' and '{lang_b}'." if lang_a and lang_b else ""

    prompt = (
        "Extract every translation pair you can find in the text below and return "
        "them as a JSON array of [source, target] pairs â€” respond with ONLY the "
        "JSON array, no explanations, no markdown code fences." + lang_hint +
        " Only include genuine word/phrase/sentence translation pairs â€” skip "
        "headers, page numbers, and unrelated text.\n\nText:\n" + text
    )

    client = get_client()
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )

    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    pairs = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            a_text, b_text = str(item[0]).strip(), str(item[1]).strip()
            if a_text and b_text:
                pairs.append((a_text, b_text))
    return pairs


def _pairs_from_pdf(pdf_path: str, lang_a: str = "", lang_b: str = "") -> tuple[list[tuple[str, str]], Optional[str]]:
    reader = PdfReader(pdf_path)
    pairs = []
    full_text_parts = []

    for page in reader.pages:
        text = page.extract_text() or ""
        full_text_parts.append(text)
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

    if pairs:
        return pairs, None

    # No clean "source - target" glossary lines found â€” the PDF might still
    # have translations in it (a table that didn't extract as neat lines,
    # prose with inline translations, etc.). Fall back to asking Gemini to
    # pull pairs out of the raw extracted text directly.
    full_text = "\n".join(full_text_parts).strip()
    if not full_text:
        return [], None

    try:
        pairs = _pairs_via_gemini_extraction(full_text, lang_a, lang_b)
    except Exception:
        return [], None

    note = None
    if pairs:
        note = (
            "This PDF wasn't a simple 'source - target' line-by-line glossary, so "
            "Gemini was used to extract the translation pairs from the text directly "
            "â€” spot-check a few entries after training to make sure they're correct."
        )
    return pairs, note


def _pairs_from_txt(txt_path: str, lang_a: str = "", lang_b: str = "") -> tuple[list[tuple[str, str]], Optional[str]]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pairs = []
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

    if pairs:
        return pairs, None

    if not text.strip():
        return [], None

    try:
        pairs = _pairs_via_gemini_extraction(text, lang_a, lang_b)
    except Exception:
        return [], None

    note = None
    if pairs:
        note = (
            "This text file wasn't a simple 'source - target' line-by-line glossary, "
            "so Gemini was used to extract the translation pairs directly â€” "
            "spot-check a few entries after training."
        )
    return pairs, note




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

    Column/pair detection is always best-effort: the language name typed
    doesn't need to exactly match anything in the file. When Claude has to
    guess, a short `note` explaining the guess is folded into the returned
    message so it can be spot-checked.

    Returns: {"success": bool, "count": int, "message": str}
    """
    ext = Path(file_path).suffix.lower()
    note: Optional[str] = None

    try:
        if ext == ".csv":
            pairs, note = _pairs_from_csv(file_path, lang_a, lang_b)
        elif ext in (".xlsx", ".xlsm"):
            pairs, note = _pairs_from_xlsx(file_path, lang_a, lang_b)
        elif ext == ".pdf":
            pairs, note = _pairs_from_pdf(file_path, lang_a, lang_b)
        elif ext == ".txt":
            pairs, note = _pairs_from_txt(file_path, lang_a, lang_b)
        else:
            return {"success": False, "count": 0, "message": f"Unsupported file type: {ext}"}
    except Exception as e:
        return {"success": False, "count": 0, "message": f"Could not read file: {e}"}

    if not pairs:
        return {
            "success": False,
            "count": 0,
            "message": (
                f"No translation pairs could be found in this file for '{lang_a}' / "
                f"'{lang_b}'. Make sure it actually contains matching translations â€” "
                "either two columns (one per language) with the same rows lined up, "
                "or, for PDFs, some recognizable source/target text â€” not just a "
                "single-language word list or unrelated data."
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

    message = f"Trained on {len(pairs)} pairs for '{lang_a}' <-> '{lang_b}' (from {ext} file)."
    if note:
        message += f" Note: {note}"

    return {
        "success": True,
        "count": len(pairs),
        "message": message,
    }




def language_pair_is_trained(lang_a: str, lang_b: str) -> bool:
    return _embeddings_file(_store_path(lang_a, lang_b)).exists()


def list_trained_languages() -> list[str]:
    """Returns human-readable pair labels, e.g. ['Kapampangan â†” Tagalog', 'Tagalog â†” English']."""
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
            labels.append(f"{info['lang_a']} â†” {info['lang_b']}")
        else:
            labels.append(p.name.replace("__", " â†” "))
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
        "Respond with ONLY the translated text â€” no explanations, no quotes, no extra "
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
# audio clips + their correct transcript per language, in the Supabase
# "recordingManagement" table (see salingo_supabase_migration.sql).
# There's no audio similarity search here â€” a capped batch of the most
# recently added samples for that language is fed to Gemini as few-shot
# reference audio on every transcription call, to calibrate it to the
# speaker's accent, pronunciation, and spelling conventions for that
# language. The same samples double as the reference voice for TTS
# (see synthesize_speech / _prepare_reference_clip below).

def train_audio_sample(language: str, file_path: str, mime_type: str, transcript: str) -> dict:
    """
    Save a short audio clip + its correct transcript as a pronunciation
    reference sample for `language`, into Supabase.

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
                f"{MAX_AUDIO_SAMPLE_BYTES // (1024 * 1024)} MB â€” a few seconds is enough."
            ),
        }

    language_key = _slug(language)
    with open(src, "rb") as f:
        audio_bytes = f.read()

    try:
        conn = _get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO "recordingManagement" ("Language", "Transcript", "AudioData", "MimeType")
                    VALUES (%s, %s, %s, %s)
                    """,
                    (language_key, transcript, psycopg2.Binary(audio_bytes), mime_type),
                )
                conn.commit()
                cur.execute(
                    'SELECT COUNT(*) FROM "recordingManagement" WHERE "Language" = %s',
                    (language_key,),
                )
                count = cur.fetchone()[0]
        finally:
            conn.close()
    except Exception as e:
        return {"success": False, "count": 0, "message": f"Database error while saving sample: {e}"}

    return {
        "success": True,
        "count": count,
        "message": f"Saved pronunciation sample for '{language}' ({count} total).",
    }


def list_audio_samples(language: str) -> list[dict]:
    """Returns [{"id": ..., "transcript": ...}, ...] for a given language."""
    language_key = _slug(language)
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT "ID", "Transcript" FROM "recordingManagement" '
                'WHERE "Language" = %s ORDER BY "CreatedAt" ASC',
                (language_key,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"id": str(row[0]), "transcript": row[1]} for row in rows]


def delete_audio_sample(language: str, sample_id: str) -> bool:
    language_key = _slug(language)
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'DELETE FROM "recordingManagement" WHERE "ID" = %s AND "Language" = %s',
                (sample_id, language_key),
            )
            deleted = cur.rowcount > 0
            conn.commit()
    finally:
        conn.close()

    return deleted


def _load_audio_examples(language: str, max_examples: int = MAX_AUDIO_EXAMPLES_PER_REQUEST) -> list[dict]:
    language_key = _slug(language)
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT "AudioData", "MimeType", "Transcript" FROM "recordingManagement" '
                'WHERE "Language" = %s ORDER BY "CreatedAt" DESC LIMIT %s',
                (language_key, max_examples),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {"data": bytes(row[0]), "mime_type": row[1], "transcript": row[2]}
        for row in rows
    ]


def _get_premade_voice_id(voice_name: str = ELEVENLABS_VOICE_NAME) -> dict:
    """
    Returns {"voice_id": str} for an existing ElevenLabs premade/stock voice
    matched by name (e.g. "Josh") — no cloning, works on the Free plan.
    Result is cached in-memory per process so we don't call GET /v1/voices
    on every single "Speak result" click.

    Returns {"error": message} on failure (bad key, name not found, etc).
    """
    if not ELEVENLABS_API_KEY:
        return {"error": "ELEVENLABS_API_KEY is not set on the server."}

    cache_key = voice_name.strip().lower()
    if cache_key in _premade_voice_cache:
        return {"voice_id": _premade_voice_cache[cache_key]}

    try:
        resp = requests.get(
            f"{ELEVENLABS_API_BASE}/voices",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=30,
        )
    except requests.RequestException as e:
        return {"error": f"Could not reach ElevenLabs: {e}"}

    if resp.status_code >= 400:
        detail = resp.text[:300]
        return {"error": f"Could not list ElevenLabs voices ({resp.status_code}): {detail}"}

    voices = resp.json().get("voices", [])
    # ElevenLabs premade voice names often have a descriptor suffix, e.g.
    # "Josh - Natural Narrator" rather than plain "Josh". Match exactly
    # first; if that fails, fall back to matching the leading word(s)
    # before " - ", then a plain substring match.
    match = next((v for v in voices if (v.get("name") or "").strip().lower() == cache_key), None)
    if not match:
        match = next(
            (v for v in voices if (v.get("name") or "").split(" - ")[0].strip().lower() == cache_key),
            None,
        )
    if not match:
        match = next((v for v in voices if cache_key in (v.get("name") or "").strip().lower()), None)

    if not match:
        available = ", ".join(v.get("name", "?") for v in voices) or "(none returned)"
        return {
            "error": (
                f"No ElevenLabs voice named '{voice_name}' found on this account. "
                f"Available voices: {available}"
            )
        }

    voice_id = match.get("voice_id")
    if not voice_id:
        return {"error": "Matched voice has no voice_id."}

    _premade_voice_cache[cache_key] = voice_id
    return {"voice_id": voice_id}


def synthesize_speech(text: str, language: str) -> dict:
    """
    Synthesize `text` using a premade ElevenLabs voice (ELEVENLABS_VOICE_NAME,
    e.g. "Josh") — NOT a cloned voice. This is how languages with no OS/
    browser TTS voice (e.g. Kapampangan) can still be "spoken" on the
    frontend. `language` is accepted for interface compatibility with the
    old cloned-voice version and for future per-language voice mapping, but
    currently every language is spoken with the same premade voice.

    Returns: {"success": bool, "audio": bytes | None, "mime_type": str,
              "message": str}
    """
    text = text.strip()
    if not text:
        return {"success": False, "audio": None, "mime_type": "", "message": "No text to speak."}

    voice_result = _get_premade_voice_id()
    if "error" in voice_result:
        return {"success": False, "audio": None, "mime_type": "", "message": voice_result["error"]}

    voice_id = voice_result["voice_id"]

    try:
        resp = requests.post(
            f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": ELEVENLABS_TTS_MODEL,
            },
            timeout=60,
        )
    except requests.RequestException as e:
        return {"success": False, "audio": None, "mime_type": "", "message": f"Could not reach ElevenLabs: {e}"}

    if resp.status_code >= 400:
        detail = resp.text[:300]
        return {
            "success": False,
            "audio": None,
            "mime_type": "",
            "message": f"Speech synthesis failed ({resp.status_code}): {detail}",
        }

    return {
        "success": True,
        "audio": resp.content,
        "mime_type": "audio/mpeg",
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
    reference recordings â€” trained pronunciation samples for this
    language â€” to calibrate its understanding of accent, pronunciation,
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
            "understanding of pronunciation and spelling conventions â€” do NOT transcribe "
            "these reference clips themselves."
        )
        for example in reference_examples:
            contents.append(types.Part.from_bytes(data=example["data"], mime_type=example["mime_type"]))
            contents.append(f"Reference transcript: {example['transcript']}")

    contents.append(
        f"Now transcribe ONLY the following new audio clip. Respond with ONLY the "
        f"verbatim transcript in '{spoken_language}' â€” no explanations, no quotes, no "
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
