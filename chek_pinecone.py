"""
diagnose_ollama.py
──────────────────
Run this FIRST to check your Ollama setup before running the pipeline.

    python diagnose_ollama.py

Checks:
    1. Is Ollama reachable?
    2. What models are pulled?
    3. Is OLLAMA_MODEL in .env actually available?
    4. Does a test generate call work?
"""

import requests
import sys
import os
from pathlib import Path

# ── Load .env manually (no dotenv dependency needed) ─────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

OLLAMA_BASE  = os.getenv("OLLAMA_BASE",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

print(f"\n{'='*55}")
print(f"  Ollama Diagnostics")
print(f"{'='*55}")
print(f"  Base  : {OLLAMA_BASE}")
print(f"  Model : {OLLAMA_MODEL}")
print(f"{'='*55}\n")


# ── 1. Reachability ───────────────────────────────────────────────────────────
print("[ 1/4 ] Checking Ollama is reachable...")
try:
    r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    r.raise_for_status()
    print("        ✅ Ollama is running\n")
except requests.exceptions.ConnectionError:
    print("        ❌ Cannot reach Ollama")
    print(f"           Fix: run `ollama serve` in a separate terminal")
    sys.exit(1)
except Exception as e:
    print(f"        ❌ Unexpected error: {e}")
    sys.exit(1)


# ── 2. List pulled models ─────────────────────────────────────────────────────
print("[ 2/4 ] Checking pulled models...")
models = r.json().get("models", [])
if not models:
    print("        ⚠️  No models pulled yet")
    print(f"           Fix: run `ollama pull {OLLAMA_MODEL}`")
    sys.exit(1)

print("        Available models:")
for m in models:
    print(f"          - {m['name']}")
print()


# ── 3. Check configured model exists ─────────────────────────────────────────
print(f"[ 3/4 ] Checking '{OLLAMA_MODEL}' is pulled...")
names      = [m["name"] for m in models]
normalised = [n.split(":")[0] for n in names]

if OLLAMA_MODEL.split(":")[0] not in normalised:
    print(f"        ❌ '{OLLAMA_MODEL}' is NOT pulled")
    print(f"           Pulled models : {names}")
    print(f"           Fix           : run `ollama pull {OLLAMA_MODEL}`")
    sys.exit(1)

actual_name = next(n for n in names if n.split(":")[0] == OLLAMA_MODEL.split(":")[0])
print(f"        ✅ Found as '{actual_name}'\n")


# ── 4. Test generate call ─────────────────────────────────────────────────────
print(f"[ 4/4 ] Testing a generate call with '{actual_name}'...")
try:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model":   actual_name,
            "prompt":  "Reply with exactly: OK",
            "stream":  False,
            "options": {"temperature": 0, "num_predict": 5},
        },
        timeout=60,
    )

    if not resp.ok:
        try:
            err = resp.json().get("error", resp.text)
        except Exception:
            err = resp.text
        print(f"        ❌ HTTP {resp.status_code}: {err}")
        print(f"\n        Your OLLAMA_MODEL in .env is '{OLLAMA_MODEL}'")
        print(f"        but Ollama has it stored as  '{actual_name}'")
        print(f"        Fix: update your .env → OLLAMA_MODEL={actual_name}")
        sys.exit(1)

    reply = resp.json().get("response", "").strip()
    print(f"        ✅ Generate works — model replied: '{reply}'\n")

except requests.exceptions.Timeout:
    print("        ❌ Timed out — model may still be loading, try again in 30s")
    sys.exit(1)


# ── Summary ───────────────────────────────────────────────────────────────────
print("="*55)
print("  ✅ All checks passed — Ollama is ready")
print(f"  Add this to your .env if not already there:")
print(f"    OLLAMA_BASE={OLLAMA_BASE}")
print(f"    OLLAMA_MODEL={actual_name}")
print("="*55 + "\n")