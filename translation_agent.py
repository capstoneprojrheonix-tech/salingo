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

# Translation memory (RAG embeddings) used to live on local disk under
# BASE_DIR/vectorstores/<pair>/{embeddings.npy,metadata.json,info.json}.
# That's what caused trained datasets to "disappear" after the Render
# service slept/redeployed/restarted: Render's free-tier filesystem is
# EPHEMERAL — a new container is spun up from the last deployed image on
# every restart, so any files written to local disk after deploy (i.e.
# every uploaded dataset) are wiped, exactly like the audio recordings
# used to be before those were moved to Supabase's "recordingManagement"
# table. Translation memory now lives in Supabase's "translationMemory"
# table for the same reason — see salingo_supabase_migration.sql for the
# CREATE TABLE statement.
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
