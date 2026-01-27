from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def gemini_enabled() -> bool:
    # Default behavior: enabled if GEMINI_API_KEY exists.
    # Optional opt-out: set GEMINI_POLISH=0/false/off.
    api_key_present = bool((os.getenv("GEMINI_API_KEY") or "").strip())
    if not api_key_present:
        return False
    if os.getenv("GEMINI_POLISH") is None:
        return True
    return _env_bool("GEMINI_POLISH", default=True)


def _gemini_model() -> str:
    return (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash").strip()


def _timeout_seconds() -> float:
    try:
        return float(os.getenv("GEMINI_TIMEOUT", "10"))
    except Exception:
        return 10.0


def _endpoint(api_key: str, model: str) -> str:
    # Gemini API (Google AI Studio / Generative Language)
    # https://ai.google.dev/api/rest/v1beta/models/generateContent
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"


def _build_prompt(title: str, reasons: List[str]) -> str:
    reasons_text = "\n".join([f"- {r}" for r in (reasons or [])][:8])

    return (
        "You are a helpful academic writing assistant. "
        "Polish the following topic for readability WITHOUT changing its meaning. "
        "Do NOT change rank, score, or invent new facts.\n\n"
        f"Title: {title}\n\n"
        f"Reasons:\n{reasons_text}\n\n"
        "Return STRICT JSON with keys: "
        "polished_title, polished_explanation, short_description. "
        "short_description must be 1-2 lines max."
    )


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t

    # Handle ```json\n{...}\n``` as well as bare ```\n...\n```
    lines = t.splitlines()
    if not lines:
        return ""

    # Drop opening fence line
    if lines[0].lstrip().startswith("```"):
        lines = lines[1:]

    # Drop closing fence line if present
    while lines and lines[-1].strip() == "````":
        lines = lines[:-1]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from model output.

    Gemini sometimes returns:
    - raw JSON
    - fenced JSON (```json ... ```)
    - JSON preceded/followed by commentary
    """

    t = _strip_code_fence(text)
    if not t:
        return None

    # Fast path: whole string is JSON.
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find first balanced {...} region and parse it.
    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = t[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None

    return None


@lru_cache(maxsize=256)
def _polish_cached(title: str, reasons_json: str) -> Optional[Dict[str, str]]:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    model = _gemini_model()
    url = _endpoint(api_key=api_key, model=model)

    try:
        reasons = json.loads(reasons_json)
        if not isinstance(reasons, list):
            reasons = []
    except Exception:
        reasons = []

    prompt = _build_prompt(title=title, reasons=[str(r) for r in reasons])

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 300,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=_timeout_seconds())
        if resp.status_code >= 400:
            return None
        data = resp.json()

        # Extract text from response (concatenate all parts)
        text = ""
        for cand in data.get("candidates", []) or []:
            content = cand.get("content") or {}
            parts = content.get("parts") or []
            if not isinstance(parts, list):
                continue
            chunks: List[str] = []
            for p in parts:
                if isinstance(p, dict) and (p.get("text") is not None):
                    chunks.append(str(p.get("text") or ""))
            joined = "".join(chunks).strip()
            if joined:
                text = joined
                break
        if not text:
            return None

        obj = _extract_first_json_object(text)
        if not obj:
            return None

        polished_title = str(obj.get("polished_title") or "").strip()
        polished_explanation = str(obj.get("polished_explanation") or "").strip()
        short_description = str(obj.get("short_description") or "").strip()

        # Guardrails: do not allow empty or wildly long values
        out = {
            "polished_title": polished_title[:180],
            "polished_explanation": polished_explanation[:320],
            "short_description": short_description[:220],
        }
        if not out["polished_title"] and not out["polished_explanation"] and not out["short_description"]:
            return None
        return out
    except Exception:
        return None


def polish_topics_inplace(recommended_topics: List[Dict[str, Any]]) -> None:
    """Attach Gemini polish fields to each topic dict (read-only enhancement).

    Adds: polished_title, polished_explanation, short_description.
    Never modifies ranking inputs (title, scores, etc.).
    """

    if not recommended_topics or not gemini_enabled():
        return

    try:
        top_n = int(os.getenv("GEMINI_POLISH_TOP_N", "3"))
    except Exception:
        top_n = 3
    top_n = max(0, min(10, top_n))

    for i, t in enumerate(recommended_topics):
        if top_n and i >= top_n:
            break
        title = str(t.get("title") or "").strip()
        reasons = t.get("reasons") or []
        if not title:
            continue
        try:
            reasons_json = json.dumps(reasons, ensure_ascii=False)
        except Exception:
            reasons_json = "[]"

        polished = _polish_cached(title, reasons_json)
        if not polished:
            continue

        # Attach only (do not overwrite canonical fields)
        if polished.get("polished_title"):
            t["polished_title"] = polished["polished_title"]
        if polished.get("polished_explanation"):
            t["polished_explanation"] = polished["polished_explanation"]
        if polished.get("short_description"):
            t["short_description"] = polished["short_description"]
