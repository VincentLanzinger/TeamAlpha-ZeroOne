"""Probe Featherless: list models + test a tiny chat call to find one this key can use."""
from __future__ import annotations
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

KEY = os.environ.get("FEATHERLESS_API_KEY", "").strip()
if not KEY:
    print("No FEATHERLESS_API_KEY in env"); sys.exit(2)

print(f"key prefix: {KEY[:6]}... (len {len(KEY)})\n")

# 1) GET /v1/models
print("=== GET /v1/models ===")
UA = "hedge-decision-agent/0.1 (+python-urllib)"
req = urllib.request.Request(
    "https://api.featherless.ai/v1/models",
    headers={"Authorization": f"Bearer {KEY}", "User-Agent": UA, "Accept": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    items = data.get("data") or data.get("models") or []
    print(f"got {len(items)} models")
    # Pick a few candidates by name
    candidates = []
    for m in items:
        mid = m.get("id") or m.get("name") or ""
        if any(k in mid.lower() for k in ("llama-3.1-8b", "llama-3.2-3b", "qwen", "mistral")):
            candidates.append(mid)
    for c in candidates[:15]:
        print(f"  candidate: {c}")
    # Also just show the first few raw ids
    print("\nfirst 10 model ids:")
    for m in items[:10]:
        print(f"  {m.get('id')}")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read()!r}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# 2) Tiny chat probe on a tiny model
print("\n=== POST /v1/chat/completions (tiny probe) ===")
for model_id in [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]:
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "say OK only"}],
        "max_tokens": 10,
    }).encode()
    req = urllib.request.Request(
        "https://api.featherless.ai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "User-Agent": UA,
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]["content"]
        print(f"  OK  {model_id}  ->  {msg[:60]!r}")
        break
    except urllib.error.HTTPError as e:
        body = e.read()
        print(f"  {e.code}  {model_id}  ->  {body[:200]!r}")
    except Exception as e:
        print(f"  ERR  {model_id}  ->  {type(e).__name__}: {e}")
