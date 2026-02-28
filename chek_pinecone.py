"""
diagnose_groq.py
────────────────
Run this to check your Groq setup before running the pipeline.

    python chek_pinecone.py

Checks:
    1. Is GROQ_API_KEY set?
    2. Can we reach the Groq API?
    3. Does a test chat completion call work?
"""

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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

print(f"\n{'='*55}")
print(f"  Groq Diagnostics")
print(f"{'='*55}")
print(f"  Model : {GROQ_MODEL}")
print(f"  Key   : {GROQ_API_KEY[:10]}..." if GROQ_API_KEY else "  Key   : NOT SET")
print(f"{'='*55}\n")


# ── 1. Check API key is set ──────────────────────────────────────────────────
print("[ 1/2 ] Checking GROQ_API_KEY is set...")
if not GROQ_API_KEY:
    print("        ❌ GROQ_API_KEY is not set")
    print("           Fix: add GROQ_API_KEY=gsk_... to your .env file")
    sys.exit(1)
print("        ✅ GROQ_API_KEY is set\n")


# ── 2. Test a chat completion call ───────────────────────────────────────────
print(f"[ 2/2 ] Testing a chat completion call with '{GROQ_MODEL}'...")
try:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0,
        max_tokens=5,
    )

    reply = response.choices[0].message.content.strip()
    print(f"        ✅ Groq API works — model replied: '{reply}'\n")

except ImportError:
    print("        ❌ groq package not installed")
    print("           Fix: run `pip install groq`")
    sys.exit(1)
except Exception as e:
    print(f"        ❌ Groq API error: {e}")
    sys.exit(1)


# ── Summary ──────────────────────────────────────────────────────────────────
print("="*55)
print("  ✅ All checks passed — Groq is ready")
print(f"  Your .env should contain:")
print(f"    GROQ_API_KEY={GROQ_API_KEY[:10]}...")
print(f"    GROQ_MODEL={GROQ_MODEL}")
print("="*55 + "\n")