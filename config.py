"""
config.py
─────────
Single place where ALL environment variables are loaded and validated.
Every other module imports from here — never from os.environ directly.

Usage
-----
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, GROQ_API_KEY, GROQ_MODEL
"""

import os
from dotenv import load_dotenv

# Load .env file if present (does nothing if missing — safe for production)
load_dotenv()


def _require(key: str) -> str:
    """Read an env var, raise a clear error if it's missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your values."
        )
    return value


# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY   = _require("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "talent-rag")

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = _require("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")