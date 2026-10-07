"""
SALINGO Translation Service (FastAPI)
-------------------------------------
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

This service is meant to run on its own host/server.
languageManagement.php (via ai_bridge.php) and translate.php
both call this over HTTP instead of calling Gemini directly.

Backward compatibility notes:
  - /translate still accepts the old {language, direction} shape.
  - /train still accepts the old {language, file} shape.
  - New callers can pass target_language directly.
"""

import os
import hmac
import shutil
import tempfile
import json
import re
import urllib.parse
import urllib.request

from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Response,
    Depends,
    Query,
)

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint

import translation_agent as agent


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="SALINGO Translation Service"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request models
# ============================================================

class TranslateRequest(BaseModel):
    text: str
    language: str
    target_language: Optional[str] = None
    direction: str = "to_english"


class DeviceRegisterRequest(BaseModel):
    device_id: str
    mac_address: str
    firmware_version: str = "unknown"


class DeviceHeartbeatRequest(BaseModel):
    device_id: str
    mac_address: str = ""
    firmware_version: str = "unknown"
    wifi_rssi: Optional[int] = None
    battery_percent: Optional[int] = None


class DeviceReportRequest(BaseModel):
    device_id: str
    category: str
    message: str


class LanguageRecordUpdate(BaseModel):
    language_name: str
    status: str
    file_name: Optional[str] = None
    translation: Optional[int] = None


# ============================================================
# SALINGO physical device authentication
# ============================================================

device_security = HTTPBearer(auto_error=False)

SALINGO_DEVICE_SECRET = os.getenv(
    "SALINGO_DEVICE_SECRET",
    ""
).strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()

def _require_device_auth(
    credentials: Optional[HTTPAuthorizationCredentials]
) -> None:

    if not SALINGO_DEVICE_SECRET:
        raise HTTPException(
            503,
            "SALINGO_DEVICE_SECRET is not configured on the server"
        )

    if credentials is None:
        raise HTTPException(
            401,
            "Missing device authorization"
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            401,
            "Invalid authorization scheme"
        )

    supplied = credentials.credentials.strip()

    if not hmac.compare_digest(
        supplied,
        SALINGO_DEVICE_SECRET
    ):
        raise HTTPException(
            401,
            "Invalid device authorization"
        )

def _supabase_request(method: str, table: str, query: str = "", body=None):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise HTTPException(503, "Supabase service is not configured")

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}"
    if query:
        url += f"?{query}"

    data = None
    headers = {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Accept": "application/json",
    }

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Prefer"] = "return=representation"

    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response_text = response.read().decode("utf-8")
            return json.loads(response_text) if response_text else None
    except Exception as e:
        raise HTTPException(502, f"Supabase request failed: {e}")


def _get_latest_firmware():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise HTTPException(503, "Supabase firmware service is not configured")

    query = urllib.parse.urlencode({
        "select": "ID,VersionCode,Status,UpdateInfo,FirmwareUrl,FirmwareSha256",
        "Status": "eq.Latest",
        "order": "ID.desc",
        "limit": "1",
    })

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/updateManagement?{query}"

    request = urllib.request.Request(
        url,
        headers={
            "apikey": SUPABASE_SECRET_KEY,
            "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            rows = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(
            502,
            f"Could not read firmware information from {url}: {e}"
        )

    if not rows:
        return None

    return rows[0]

# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# SALINGO physical device API
# ============================================================

@app.post("/api/device/register")
def register_device(
    body: DeviceRegisterRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)

    if not body.device_id.strip():
        raise HTTPException(
            400,
            "device_id is required"
        )

    if not body.mac_address.strip():
        raise HTTPException(
            400,
            "mac_address is required"
        )

    try:
        device = agent.db_register_device(
            body.device_id.strip(),
            body.mac_address.strip(),
            body.firmware_version.strip()
            or "unknown",
        )

    except Exception as e:
        raise HTTPException(
            500,
            f"Device registration failed: {e}"
        )

    return {
        "success": True,
        "device": device,
    }


@app.post("/api/device/heartbeat")
def device_heartbeat(
    body: DeviceHeartbeatRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)

    if not body.device_id.strip():
        raise HTTPException(
            400,
            "device_id is required"
        )

    try:
        device = agent.db_heartbeat_device(
            device_id=body.device_id.strip(),
            mac_address=body.mac_address.strip(),
            firmware_version=(
                body.firmware_version.strip()
                or "unknown"
            ),
            wifi_rssi=body.wifi_rssi,
            battery_percent=body.battery_percent,
        )

    except Exception as e:
        raise HTTPException(
            500,
            f"Device heartbeat failed: {e}"
        )

    return {
        "success": True,
        "device": device,
    }


@app.get("/api/devices")
def list_devices(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)

    try:
        devices = agent.db_list_devices()

    except Exception as e:
        raise HTTPException(
            500,
            f"Could not list devices: {e}"
        )

    return {
        "devices": devices
    }

@app.post("/api/device/reports")
def create_device_report(
    body: DeviceReportRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)

    device_id = body.device_id.strip()
    category = body.category.strip()
    message = body.message.strip()

    if not device_id:
        raise HTTPException(400, "device_id is required")

    if not category:
        raise HTTPException(400, "category is required")

    if not message:
        raise HTTPException(400, "message is required")

    rows = _supabase_request(
        "POST",
        "deviceReports",
        body={
            "DeviceID": device_id,
            "Category": category,
            "Message": message,
            "Status": "Pending",
        },
    )

    return {
        "success": True,
        "report": rows[0] if rows else None,
    }

@app.get("/api/device/notifications")
def device_notifications(
    device_id: str,
    current_version: str = "",
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)

    device_id = device_id.strip()
    installed = current_version.strip().lstrip("vV")

    if not device_id:
        raise HTTPException(400, "device_id is required")

    firmware = _get_latest_firmware()
    latest = ""

    if firmware:
        latest = str(firmware.get("VersionCode") or "").strip().lstrip("vV")

    update_available = bool(
        latest
        and _version_tuple(latest) > _version_tuple(installed)
    )

    query = urllib.parse.urlencode({
        "select": "ID",
        "DeviceID": f"eq.{device_id}",
        "Status": "eq.Pending",
    })

    pending_rows = _supabase_request(
        "GET",
        "deviceReports",
        query=query,
    )

    return {
        "success": True,
        "new_update": update_available,
        "current_version": installed,
        "latest_version": latest,
        "update_info": str(firmware.get("UpdateInfo") or "") if firmware else "",
        "pending_reports": len(pending_rows or []),
    }

def _version_tuple(version: str):
    clean = version.strip().lstrip("vV")
    parts = clean.split(".")
    values = []

    for part in parts:
        try:
            values.append(int(part))
        except ValueError:
            values.append(0)

    while len(values) < 4:
        values.append(0)

    return tuple(values[:4])

@app.get("/api/firmware/latest")
def latest_firmware(
    current_version: str = "",
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)

    firmware = _get_latest_firmware()

    if firmware is None:
        raise HTTPException(404, "No firmware release is available")

    installed = current_version.strip().lstrip("vV")
    latest = str(firmware.get("VersionCode") or "").strip().lstrip("vV")

    return {
        "success": True,
        "update_available": bool(latest and _version_tuple(latest) > _version_tuple(installed)),
        "current_version": installed,
        "latest_version": latest,
        "update_info": str(firmware.get("UpdateInfo") or ""),
        "firmware_url": str(firmware.get("FirmwareUrl") or ""),
        "sha256": str(firmware.get("FirmwareSha256") or ""),
    }

# ============================================================
# ESP32 speech-to-text
# ============================================================

@app.post("/api/device/transcribe")
def device_transcribe(
    language: str = Form(...),
    audio: UploadFile = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)

    mime_type = (
        audio.content_type
        or "audio/wav"
    )

    tmp_dir = Path(
        tempfile.mkdtemp()
    )

    tmp_path = tmp_dir / (
        audio.filename
        or "recording.wav"
    )

    try:

        # Save the WAV uploaded by the ESP32
        with open(
            tmp_path,
            "wb"
        ) as f:

            shutil.copyfileobj(
                audio.file,
                f
            )

        try:
            transcript = agent.transcribe_audio(
                str(tmp_path),
                mime_type,
                language,
            )

        except Exception as e:
            raise HTTPException(
                500,
                f"Transcription failed: {e}"
            )

    finally:
        shutil.rmtree(
            tmp_dir,
            ignore_errors=True
        )

    return {
        "success": True,
        "transcript": transcript,
    }


# ============================================================
# Languages
# ============================================================

@app.get("/languages")
def languages():

    return {
        "trained_languages":
            agent.list_trained_languages()
    }


# ============================================================
# Text translation
# ============================================================

@app.post("/translate")
def translate(
    req: TranslateRequest
):

    if not req.text.strip():
        raise HTTPException(
            400,
            "text is empty"
        )

    if req.target_language:

        source_language = req.language
        target_language = req.target_language

    else:

        if req.direction not in (
            "to_english",
            "from_english"
        ):
            raise HTTPException(
                400,
                "direction must be "
                "'to_english' or 'from_english'"
            )

        if req.direction == "to_english":

            source_language = req.language
            target_language = "English"

        else:

            source_language = "English"
            target_language = req.language

    try:

        result = agent.translate_text(
            req.text,
            source_language,
            target_language
        )

    except Exception as e:

        raise HTTPException(
            500,
            f"Translation failed: {e}"
        )

    return result


# ============================================================
# Train translation datasets
# ============================================================

@app.post("/train")
def train(
    language: str = Form(...),
    target_language: str = Form("English"),
    file: UploadFile = File(...),
):

    allowed_ext = (
        ".csv",
        ".xlsx",
        ".xlsm",
        ".pdf",
        ".txt",
    )

    filename = (
        file.filename
        or "upload"
    )

    if not filename.lower().endswith(
        allowed_ext
    ):
        raise HTTPException(
            400,
            "Only "
            + ", ".join(allowed_ext)
            + " files are supported"
        )

    tmp_dir = Path(
        tempfile.mkdtemp()
    )

    tmp_path = (
        tmp_dir
        / filename
    )

    try:

        with open(
            tmp_path,
            "wb"
        ) as f:

            shutil.copyfileobj(
                file.file,
                f
            )

        result = agent.train_language(
            language,
            str(tmp_path),
            target_language
        )

    finally:

        shutil.rmtree(
            tmp_dir,
            ignore_errors=True
        )

    if not result["success"]:

        raise HTTPException(
            400,
            result["message"]
        )

    return result


# ============================================================
# Website/browser audio translation
# ============================================================

@app.post("/translate-audio")
def translate_audio(
    source_language: str = Form(...),
    target_language: str = Form(...),
    audio: UploadFile = File(...),
):

    mime_type = (
        audio.content_type
        or "audio/webm"
    )

    tmp_dir = Path(
        tempfile.mkdtemp()
    )

    tmp_path = tmp_dir / (
        audio.filename
        or "recording.webm"
    )

    try:

        with open(
            tmp_path,
            "wb"
        ) as f:

            shutil.copyfileobj(
                audio.file,
                f
            )

        try:

            result = (
                agent.transcribe_and_translate_audio(
                    str(tmp_path),
                    mime_type,
                    source_language,
                    target_language
                )
            )

        except Exception as e:

            raise HTTPException(
                500,
                f"Audio translation failed: {e}"
            )

    finally:

        shutil.rmtree(
            tmp_dir,
            ignore_errors=True
        )

    return result


# ============================================================
# Pronunciation training
# ============================================================

@app.post("/train-audio")
def train_audio(
    language: str = Form(...),
    transcript: str = Form(...),
    audio: UploadFile = File(...),
):

    mime_type = (
        audio.content_type
        or "audio/webm"
    )

    tmp_dir = Path(
        tempfile.mkdtemp()
    )

    tmp_path = tmp_dir / (
        audio.filename
        or "sample.webm"
    )

    try:

        with open(
            tmp_path,
            "wb"
        ) as f:

            shutil.copyfileobj(
                audio.file,
                f
            )

        result = agent.train_audio_sample(
            language,
            str(tmp_path),
            mime_type,
            transcript
        )

    finally:

        shutil.rmtree(
            tmp_dir,
            ignore_errors=True
        )

    if not result["success"]:

        raise HTTPException(
            400,
            result["message"]
        )

    return result


# ============================================================
# Text-to-speech
# ============================================================

@app.post("/synthesize-speech")
def synthesize_speech(
    text: str = Form(...),
    language: str = Form(...),
):

    result = agent.synthesize_speech(
        text,
        language
    )

    if not result["success"]:

        raise HTTPException(
            400,
            result["message"]
        )

    return Response(
        content=result["audio"],
        media_type=result["mime_type"],
        headers={
            "X-Sample-Rate":
                str(result["sample_rate"]),

            "X-Audio-Format":
                "pcm_s16le",
        },
    )


# ============================================================
# Audio sample management
# ============================================================

@app.get("/audio-samples")
def audio_samples(
    language: str
):

    return {
        "language": language,
        "samples":
            agent.list_audio_samples(
                language
            )
    }


@app.delete("/audio-samples")
def delete_audio_sample(
    language: str,
    sample_id: str
):

    deleted = (
        agent.delete_audio_sample(
            language,
            sample_id
        )
    )

    if not deleted:

        raise HTTPException(
            404,
            "Pronunciation sample not found"
        )

    return {
        "success": True,
        "message":
            "Pronunciation sample deleted"
    }


# ============================================================
# Delete trained language pair
# ============================================================

@app.delete("/languages")
def delete_language_pair(
    language: str,
    target_language: str = "English",
):

    deleted = (
        agent.delete_language_pair(
            language,
            target_language
        )
    )

    if not deleted:

        raise HTTPException(
            404,
            "Language pair not found / not trained yet"
        )

    return {
        "success": True,
        "message":
            f"Deleted training data for "
            f"'{language}' <-> "
            f"'{target_language}'"
    }


# ============================================================
# languageManagement admin dashboard
# ============================================================

@app.post("/language-records")
def create_language_record(
    language_name: str = Form(...),
    translation: int = Form(0),
    file_name: str = Form(""),
    status: str = Form("Active"),
):

    new_id = (
        agent.db_insert_language_record(
            language_name,
            translation,
            file_name,
            status
        )
    )

    return {
        "success": True,
        "id": new_id
    }


@app.get("/language-records")
def list_language_records():

    return {
        "records":
            agent.db_list_language_records()
    }


@app.get(
    "/language-records/{record_id}"
)
def get_language_record(
    record_id: int
):

    record = (
        agent.db_get_language_record(
            record_id
        )
    )

    if record is None:

        raise HTTPException(
            404,
            "Language record not found"
        )

    return record


@app.put(
    "/language-records/{record_id}"
)
def update_language_record(
    record_id: int,
    body: LanguageRecordUpdate
):

    updated = (
        agent.db_update_language_record(
            record_id,
            body.language_name,
            body.status,
            body.file_name,
            body.translation
        )
    )

    if not updated:

        raise HTTPException(
            404,
            "Language record not found"
        )

    return {
        "success": True
    }

# ============================================================
# Translation dataset updates (separate from firmware OTA)
# ============================================================

def _get_latest_translation_data():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise HTTPException(503, "Supabase translation data service is not configured")

    query = urllib.parse.urlencode({
        "select": "ID,VersionCode,Status,UpdateInfo,FileUrl,FileSha256,FileSize",
        "Status": "eq.Latest",
        "order": "VersionCode.desc",
        "limit": "1",
    })
    rows = _supabase_request("GET", "translationDataManagement", query)
    if not isinstance(rows, list):
        raise HTTPException(502, "Invalid translation release response from Supabase")
    if not rows:
        return None
    if not isinstance(rows[0], dict):
        raise HTTPException(502, "Invalid translation release record from Supabase")
    return rows[0]


def _translation_release_positive_int(value, field):
    # Accept integer database values and numeric strings, without truncating
    # floats or silently treating missing metadata as zero.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise HTTPException(502, f"Invalid translation release {field}")
    text = str(value).strip()
    if not text or not text.isascii() or not text.isdigit():
        raise HTTPException(502, f"Invalid translation release {field}")
    try:
        number = int(text)
    except ValueError:
        raise HTTPException(502, f"Invalid translation release {field}")
    if number <= 0:
        raise HTTPException(502, f"Invalid translation release {field}")
    return number


@app.get("/api/translation-data/latest")
def latest_translation_data(
    current_version: int = Query(default=0, ge=0),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)
    data = _get_latest_translation_data()
    if data is None:
        raise HTTPException(404, "No translation data release is available")

    latest_version = _translation_release_positive_int(data.get("VersionCode"), "VersionCode")
    file_size = _translation_release_positive_int(data.get("FileSize"), "FileSize")
    file_url = str(data.get("FileUrl") or "").strip()
    sha256 = str(data.get("FileSha256") or "").strip().lower()
    try:
        parsed_url = urllib.parse.urlsplit(file_url)
        valid_url = (
            parsed_url.scheme == "https" and bool(parsed_url.hostname)
            and parsed_url.username is None and parsed_url.password is None
            and not any(char.isspace() for char in file_url)
        )
        # Accessing port also validates malformed ports in stored URLs.
        parsed_url.port
    except ValueError:
        valid_url = False
    if not valid_url:
        raise HTTPException(502, "Translation release FileUrl must be a valid HTTPS URL")
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise HTTPException(502, "Invalid translation release SHA-256")

    return {
        "success": True,
        "update_available": latest_version > current_version,
        "current_version": current_version,
        "latest_version": latest_version,
        "update_info": str(data.get("UpdateInfo") or ""),
        "file_url": file_url,
        "sha256": sha256,
        "file_size": file_size,
    }


# Successful ESP32 translations, counted by target language.
# Cumulative snapshots + atomic SQL MAX avoid double counting retries.
TranslationCount = conint(strict=True, ge=0, le=4294967295)


class DeviceTranslationCountsRequest(BaseModel):
    device_id: str
    installation_id: str
    english: TranslationCount
    tagalog: TranslationCount
    kapampangan: TranslationCount


@app.post("/api/device/translation-counts")
def sync_device_translation_counts(
    body: DeviceTranslationCountsRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)
    device_id = body.device_id.strip()
    installation_id = body.installation_id.strip().lower()
    if not re.fullmatch(r"SALINGO-[0-9A-Fa-f]{8}", device_id):
        raise HTTPException(400, "Invalid SALINGO device ID")
    if not re.fullmatch(r"[0-9a-f]{16}", installation_id):
        raise HTTPException(400, "Invalid installation ID")
    result = _supabase_request("POST", "rpc/salingo_sync_translation_counts", body={
        "p_device_id": device_id.upper(),
        "p_installation_id": installation_id,
        "p_english": body.english,
        "p_tagalog": body.tagalog,
        "p_kapampangan": body.kapampangan,
    })
    if not isinstance(result, dict) or result.get("success") is not True:
        raise HTTPException(502, "Translation count save was not confirmed")
    return {"success": True}


@app.get("/api/translation-stats/summary")
def translation_stats_summary(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(device_security),
):
    _require_device_auth(credentials)
    result = _supabase_request("POST", "rpc/salingo_translation_counts_summary", body={})
    counts = result.get("counts") if isinstance(result, dict) else None
    if not isinstance(counts, dict) or any(
        type(counts.get(language)) is not int or counts[language] < 0
        for language in ("english", "tagalog", "kapampangan")
    ):
        raise HTTPException(502, "Invalid translation counts response")
    return {"success": True, "counts": counts, "total": sum(counts.values())}
