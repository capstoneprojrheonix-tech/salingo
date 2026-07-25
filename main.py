"""
SALINGO Translation Service (FastAPI)
-------------------------------------
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

This service is meant to run on its own host/server. languageManagement.php
(via ai_bridge.php) and translate.php both call this over HTTP instead of
calling Gemini directly.

Backward compatibility notes:
  - /translate still accepts the old {language, direction} shape used by
    ai_bridge.php's translateText(). If `target_language` is omitted, the
    old direction ("to_english" / "from_english") is used to infer it,
    defaulting to English.
  - /train still accepts the old {language, file} shape used by
    ai_bridge.php's trainSalingoAI(). If `target_language` is omitted, it
    defaults to "English" (same behavior as before).
  - New callers (like translate.php) can pass `target_language` directly
    to translate/train between ANY two languages, e.g. Kapampangan <-> Tagalog.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import translation_agent as agent

app = FastAPI(title="SALINGO Translation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str
    language: str
    target_language: Optional[str] = None
    direction: str = "to_english"  # legacy: "to_english" or "from_english"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/languages")
def languages():
    return {"trained_languages": agent.list_trained_languages()}


@app.post("/translate")
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")

    if req.target_language:
        source_language = req.language
        target_language = req.target_language
    else:
        if req.direction not in ("to_english", "from_english"):
            raise HTTPException(400, "direction must be 'to_english' or 'from_english'")
        if req.direction == "to_english":
            source_language, target_language = req.language, "English"
        else:
            source_language, target_language = "English", req.language

    try:
        result = agent.translate_text(req.text, source_language, target_language)
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {e}")

    return result


@app.post("/train")
def train(
    language: str = Form(...),
    target_language: str = Form("English"),
    file: UploadFile = File(...),
):
    allowed_ext = (".csv", ".xlsx", ".xlsm", ".pdf")
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(400, f"Only {', '.join(allowed_ext)} files are supported")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = agent.train_language(language, str(tmp_path), target_language)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not result["success"]:
        raise HTTPException(400, result["message"])

    return result


@app.delete("/languages")
def delete_language_pair(language: str, target_language: str = "English"):
    deleted = agent.delete_language_pair(language, target_language)
    if not deleted:
        raise HTTPException(404, "Language pair not found / not trained yet")
    return {"success": True, "message": f"Deleted training data for '{language}' <-> '{target_language}'"}