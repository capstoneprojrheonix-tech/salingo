"""
SALINGO Translation Engine (Gemini edition)
--------------------------------------------
"Training" here means building a translation memory (RAG), not fine-tuning:
  - Each uploaded dataset (CSV or PDF glossary) becomes sentence/term pairs.
  - Every pair becomes two entries: language -> english, english -> language.
  - Each entry is embedded (Gemini embeddings) and stored per language under
    vectorstores/<language_slug>/ as a numpy array + a metadata JSON file.
  - This does NOT change model weights. It's a searchable example set the
    agent consults before every translation — the same idea as a CAT tool's
    translation memory (Trados, MemoQ).

Translation:
  - Embed the input text, cosine-similarity search the stored vectors for
    that language + direction, take the top-k matches, and feed them to the
    chat model as few-shot examples.
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Literal

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

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
EMBEDDING_BATCH_SIZE = 100

Direction = Literal["to_english", "from_english"]

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def _slug(language_name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", language_name.strip().lower())
    return slug.strip("_") or "unknown"


def _store_path(language_name: str) -> Path:
    return VECTORSTORE_DIR / _slug(language_name)


def _embeddings_file(store_path: Path) -> Path:
    return store_path / "embeddings.npy"


def _metadata_file(store_path: Path) -> Path:
    return store_path / "metadata.json"


# ---------------------------------------------------------------------
# Parsing: CSV
# ---------------------------------------------------------------------

def _detect_columns(df: pd.DataFrame, language_name: str) -> tuple[str, str]:
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    english_col = next((c for c in ("english", "en") if c in cols), None)
    lang_col = language_name.strip().lower() if language_name.strip().lower() in cols else None

    if english_col is None or lang_col is None:
        if len(cols) < 2:
            raise ValueError(f"CSV must have at least 2 columns, got: {df.columns.tolist()}")
        lang_col = lang_col or cols[0]
        english_col = english_col or cols[1]

    return lang_col, english_col


def _pairs_from_csv(csv_path: str, language_name: str) -> list[tuple[str, str]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if df.empty:
        return []

    lang_col, en_col = _detect_columns(df, language_name)

    pairs = []
    for _, row in df.iterrows():
        source_text = str(row.get(lang_col, "")).strip()
        target_text = str(row.get(en_col, "")).strip()
        if source_text and target_text:
            pairs.append((source_text, target_text))
    return pairs


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


def train_language(language_name: str, file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".csv":
            pairs = _pairs_from_csv(file_path, language_name)
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
            "message": "No valid sentence/term pairs found in the file.",
        }

    texts: list[str] = []
    metadata: list[dict] = []
    for source_text, target_text in pairs:
        texts.append(source_text)
        metadata.append(
            {
                "language": language_name,
                "direction": "to_english",
                "source_text": source_text,
                "target_text": target_text,
            }
        )
        texts.append(target_text)
        metadata.append(
            {
                "language": language_name,
                "direction": "from_english",
                "source_text": target_text,
                "target_text": source_text,
            }
        )

    try:
        new_embeddings = _embed_texts(texts)
    except Exception as e:
        return {"success": False, "count": 0, "message": f"Embedding request failed: {e}"}

    store_path = _store_path(language_name)
    if _embeddings_file(store_path).exists():
        existing_embeddings, existing_metadata = _load_store(store_path)
        all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        all_metadata = existing_metadata + metadata
    else:
        all_embeddings = new_embeddings
        all_metadata = metadata

    _save_store(store_path, all_embeddings, all_metadata)

    return {
        "success": True,
        "count": len(pairs),
        "message": f"Trained on {len(pairs)} pairs for '{language_name}' (from {ext} file).",
    }


def language_is_trained(language_name: str) -> bool:
    return _embeddings_file(_store_path(language_name)).exists()


def list_trained_languages() -> list[str]:
    if not VECTORSTORE_DIR.exists():
        return []
    return [
        p.name for p in VECTORSTORE_DIR.iterdir()
        if p.is_dir() and _embeddings_file(p).exists()
    ]


def delete_language(language_name: str) -> bool:
    path = _store_path(language_name)
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
    text: str, language_name: str, direction: Direction = "to_english", k: int = 4
) -> dict:
    store_path = _store_path(language_name)
    examples: list[dict] = []
    trained = _embeddings_file(store_path).exists()

    if trained:
        embeddings, metadata = _load_store(store_path)
        direction_indices = [i for i, m in enumerate(metadata) if m["direction"] == direction]

        if direction_indices:
            sub_matrix = embeddings[direction_indices]
            query_vec = _embed_texts([text])[0]
            top_local = _cosine_top_k(query_vec, sub_matrix, k)
            examples = [metadata[direction_indices[i]] for i in top_local]

    direction_label = (
        f"from {language_name} into English"
        if direction == "to_english"
        else f"from English into {language_name}"
    )

    system_prompt = (
        f"You are a professional translator for the '{language_name}' language. "
        "Use the example translations below (retrieved from a verified translation "
        "memory) to match terminology, tone, and phrasing. If no examples are "
        "relevant, translate using your own knowledge of the language. Respond "
        "with ONLY the translated text — no explanations, no quotes, no extra "
        "commentary.\n\nExamples:\n" + _format_examples(examples)
    )
    user_prompt = f"Translate this text ({direction_label}):\n\n{text}"

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