"""Validate a Hugging Face token and report what it can be used for.

This script checks:
- whether the token is syntactically present
- whether Hugging Face accepts the token
- whether the current multilingual translation settings are compatible
- any rate-limit / credit-style signals returned by the API

Important limitation:
Hugging Face does not expose an exact "credits left" value for every token through a simple public endpoint.
This script reports the HTTP status, headers, and response details that are available, and it can optionally
probe the translation endpoint with a small request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any

import httpx

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
}

DEFAULT_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
DEFAULT_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


@dataclass
class TokenCheckResult:
    token_present: bool
    token_valid: bool
    whoami_ok: bool
    username: str | None
    account_type: str | None
    orgs: list[str]
    multilingual_ready: bool
    multilingual_reason: str
    translation_model: str
    chat_endpoint: str
    target_language: str
    target_language_name: str
    http_status: int | None
    rate_limit_headers: dict[str, str]
    response_preview: str
    probe_requested: bool
    probe_ok: bool | None
    probe_error: str | None


def _mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 12:
        return token[:4] + "..."
    return f"{token[:6]}...{token[-4:]}"


def _extract_rate_limit_headers(headers: httpx.Headers) -> dict[str, str]:
    wanted = [
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "retry-after",
        "x-request-id",
        "x-powered-by",
    ]
    result: dict[str, str] = {}
    for name in wanted:
        value = headers.get(name)
        if value:
            result[name] = value
    return result


def _load_env_value(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _check_whoami(client: httpx.Client, token: str) -> tuple[bool, dict[str, Any], int | None, dict[str, str], str]:
    try:
        response = client.get(
            DEFAULT_WHOAMI_URL,
            headers={"Authorization": f"Bearer {token}"},
        )
    except httpx.HTTPError as exc:
        return False, {}, None, {}, f"whoami request failed: {exc}"

    preview = response.text[:400]
    headers = _extract_rate_limit_headers(response.headers)

    if response.status_code != 200:
        return False, {}, response.status_code, headers, preview

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    return True, payload, response.status_code, headers, preview


def _probe_translation_endpoint(
    client: httpx.Client,
    token: str,
    chat_url: str,
    model: str,
    source_lang: str,
    target_lang: str,
    timeout_seconds: int,
) -> tuple[bool | None, str | None, int | None, dict[str, str], str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise academic translator. Preserve meaning, technical terms, "
                    "numbers, citations, section names, and named entities. Return only the translation."
                ),
            },
            {
                "role": "user",
                "content": f"Translate the following text from {SUPPORTED_LANGUAGES.get(source_lang, source_lang)} to {SUPPORTED_LANGUAGES.get(target_lang, target_lang)}:\n\nHello world. This is a tiny multilingual probe.",
            },
        ],
    }

    try:
        response = client.post(
            chat_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        return None, str(exc), None, {}, ""

    headers = _extract_rate_limit_headers(response.headers)
    preview = response.text[:400]

    if response.status_code != 200:
        return False, f"HTTP {response.status_code}: {preview}", response.status_code, headers, preview

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return False, f"Unexpected response format: {exc}; preview={preview}", response.status_code, headers, preview

    return True, content[:400], response.status_code, headers, preview


def build_report(args: argparse.Namespace) -> TokenCheckResult:
    token = args.token or _load_env_value("HF_TOKEN")
    token_present = bool(token)
    translation_model = args.model or _load_env_value("TRANSLATION_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    chat_endpoint = args.chat_url or _load_env_value("HF_CHAT_URL", DEFAULT_CHAT_URL)
    target_language = (args.target_language or "te").strip().lower()
    target_language_name = SUPPORTED_LANGUAGES.get(target_language, target_language)

    multilingual_ready = True
    multilingual_reason = ""
    if target_language not in SUPPORTED_LANGUAGES:
        multilingual_ready = False
        multilingual_reason = f"Unsupported target language: {target_language}"
    elif not token_present:
        multilingual_ready = False
        multilingual_reason = "HF_TOKEN is missing"

    if multilingual_ready and target_language in {"hi", "te"}:
        multilingual_reason = (
            "This app can attempt multilingual output for Hindi/Telugu, but actual success depends on "
            "the Hugging Face model/API being reachable and having enough quota/credits."
        )
    elif multilingual_ready and not multilingual_reason:
        multilingual_reason = "Target language is supported by the app configuration."

    if not token_present:
        return TokenCheckResult(
            token_present=False,
            token_valid=False,
            whoami_ok=False,
            username=None,
            account_type=None,
            orgs=[],
            multilingual_ready=multilingual_ready,
            multilingual_reason=multilingual_reason,
            translation_model=translation_model,
            chat_endpoint=chat_endpoint,
            target_language=target_language,
            target_language_name=target_language_name,
            http_status=None,
            rate_limit_headers={},
            response_preview="HF_TOKEN is missing",
            probe_requested=bool(args.probe),
            probe_ok=None,
            probe_error="HF_TOKEN is missing",
        )

    with httpx.Client(timeout=args.timeout_seconds) as client:
        whoami_ok, whoami_payload, status_code, headers, preview = _check_whoami(client, token)

        username = None
        account_type = None
        orgs: list[str] = []
        if whoami_ok:
            username = whoami_payload.get("name") or whoami_payload.get("user", {}).get("name")
            account_type = whoami_payload.get("type") or whoami_payload.get("user", {}).get("type")
            raw_orgs = whoami_payload.get("orgs") or []
            for item in raw_orgs:
                if isinstance(item, dict):
                    org_name = item.get("name") or item.get("displayName")
                    if org_name:
                        orgs.append(str(org_name))
                elif item:
                    orgs.append(str(item))

        probe_ok = None
        probe_error = None
        if args.probe and whoami_ok:
            probe_ok, probe_result, probe_status, probe_headers, probe_preview = _probe_translation_endpoint(
                client=client,
                token=token,
                chat_url=chat_endpoint,
                model=translation_model,
                source_lang="en",
                target_lang=target_language,
                timeout_seconds=args.probe_timeout_seconds,
            )
            headers = {**headers, **probe_headers}
            if probe_ok is False and probe_result:
                probe_error = probe_result
            elif probe_ok is None:
                probe_error = probe_result
            elif probe_result:
                probe_error = None
            if probe_status is not None and status_code is None:
                status_code = probe_status
            if probe_preview:
                preview = probe_preview
        elif args.probe and not whoami_ok:
            probe_error = "Skipped probe because token validation failed."

    return TokenCheckResult(
        token_present=True,
        token_valid=whoami_ok,
        whoami_ok=whoami_ok,
        username=username,
        account_type=account_type,
        orgs=orgs,
        multilingual_ready=multilingual_ready,
        multilingual_reason=multilingual_reason,
        translation_model=translation_model,
        chat_endpoint=chat_endpoint,
        target_language=target_language,
        target_language_name=target_language_name,
        http_status=status_code,
        rate_limit_headers=headers,
        response_preview=preview,
        probe_requested=bool(args.probe),
        probe_ok=probe_ok,
        probe_error=probe_error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a Hugging Face token is valid and whether it can be used for multilingual translation.",
    )
    parser.add_argument("--token", help="Hugging Face token. If omitted, HF_TOKEN from environment is used.")
    parser.add_argument(
        "--target-language",
        default="te",
        help="Target language to validate against. Default: te",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Translation model name. If omitted, TRANSLATION_MODEL or the default app model is used.",
    )
    parser.add_argument(
        "--chat-url",
        default="",
        help="Chat completions endpoint. If omitted, HF_CHAT_URL or the default router endpoint is used.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="Timeout for the whoami validation call. Default: 20",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Also send a tiny translation probe request. This may consume quota/credits.",
    )
    parser.add_argument(
        "--probe-timeout-seconds",
        type=int,
        default=45,
        help="Timeout for the optional probe request. Default: 45",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON only.",
    )

    args = parser.parse_args()
    result = build_report(args)
    payload = asdict(result)
    payload["masked_token"] = _mask_token(args.token or _load_env_value("HF_TOKEN"))

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Hugging Face token report")
        print(f"- token_present: {payload['token_present']}")
        print(f"- token_valid: {payload['token_valid']}")
        print(f"- whoami_ok: {payload['whoami_ok']}")
        print(f"- username: {payload['username']}")
        print(f"- account_type: {payload['account_type']}")
        print(f"- orgs: {payload['orgs']}")
        print(f"- target_language: {payload['target_language']} ({payload['target_language_name']})")
        print(f"- multilingual_ready: {payload['multilingual_ready']}")
        print(f"- multilingual_reason: {payload['multilingual_reason']}")
        print(f"- translation_model: {payload['translation_model']}")
        print(f"- chat_endpoint: {payload['chat_endpoint']}")
        print(f"- http_status: {payload['http_status']}")
        print(f"- rate_limit_headers: {json.dumps(payload['rate_limit_headers'], ensure_ascii=False)}")
        print(f"- probe_requested: {payload['probe_requested']}")
        print(f"- probe_ok: {payload['probe_ok']}")
        print(f"- probe_error: {payload['probe_error']}")
        print(f"- response_preview: {payload['response_preview'][:300]}")
        print(f"- masked_token: {payload['masked_token']}")

    if not result.token_present:
        return 2
    if not result.token_valid:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
