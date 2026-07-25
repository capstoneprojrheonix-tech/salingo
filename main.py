"""
SALINGO Translation Service (FastAPI)
-------------------------------------
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

This service is meant to run on its own host/server. languageManagement.php
(via ai_bridge.php) and the test website both call this over HTTP instead
of calling Gemini directly.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import translation_agent as agent

app = FastAPI(title="SALINGO Translation Service")

# Allow the PHP server and the test website (different hosts) to call this.
# Lock this down to your actual domains in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str
    language: str
    direction: str = "to_english"  # "to_english" or "from_english"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/languages")
def languages():
    return {"trained_languages": agent.list_trained_languages()}


@app.post("/translate")
def translate(req: TranslateRequest):
    if req.direction not in ("to_english", "from_english"):
        raise HTTPException(400, "direction must be 'to_english' or 'from_english'")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")

    if not agent.language_is_trained(req.language):
        # Still allow translation (Gemini's own knowledge), just flag it.
        pass

    try:
        result = agent.translate_text(req.text, req.language, req.direction)
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {e}")

    return result


@app.post("/train")
def train(language: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = agent.train_language(language, str(tmp_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not result["success"]:
        raise HTTPException(400, result["message"])

    return result


@app.delete("/languages/{language}")
def delete_language(language: str):
    deleted = agent.delete_language(language)
    if not deleted:
        raise HTTPException(404, "Language not found / not trained yet")
    return {"success": True, "message": f"Deleted training data for '{language}'"}
