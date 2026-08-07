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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
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


@app.post("/translate-audio")
def translate_audio(
    source_language: str = Form(...),
    target_language: str = Form(...),
    audio: UploadFile = File(...),
):
    mime_type = audio.content_type or "audio/webm"

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / (audio.filename or "recording.webm")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        try:
            result = agent.transcribe_and_translate_audio(
                str(tmp_path), mime_type, source_language, target_language
            )
        except Exception as e:
            raise HTTPException(500, f"Audio translation failed: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


@app.post("/train-audio")
def train_audio(
    language: str = Form(...),
    transcript: str = Form(...),
    audio: UploadFile = File(...),
):
    mime_type = audio.content_type or "audio/webm"

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / (audio.filename or "sample.webm")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        result = agent.train_audio_sample(language, str(tmp_path), mime_type, transcript)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not result["success"]:
        raise HTTPException(400, result["message"])

    return result


@app.post("/synthesize-speech")
def synthesize_speech(
    text: str = Form(...),
    language: str = Form(...),
):
    """
    Voice-clones speech for `language` from the trained pronunciation
    samples in audio_training/<language>/ (same corpus collected via
    /train-audio). Lets languages with no OS/browser TTS voice — e.g.
    Kapampangan — still be spoken aloud on the frontend.
    """
    result = agent.synthesize_speech(text, language)

    if not result["success"]:
        raise HTTPException(400, result["message"])

    return Response(content=result["audio"], media_type=result["mime_type"])


@app.get("/audio-samples")
def audio_samples(language: str):
    return {"language": language, "samples": agent.list_audio_samples(language)}


@app.delete("/audio-samples")
def delete_audio_sample(language: str, sample_id: str):
    deleted = agent.delete_audio_sample(language, sample_id)
    if not deleted:
        raise HTTPException(404, "Pronunciation sample not found")
    return {"success": True, "message": "Pronunciation sample deleted"}


@app.delete("/languages")
def delete_language_pair(language: str, target_language: str = "English"):
    deleted = agent.delete_language_pair(language, target_language)
    if not deleted:
        raise HTTPException(404, "Language pair not found / not trained yet")
    return {"success": True, "message": f"Deleted training data for '{language}' <-> '{target_language}'"}
