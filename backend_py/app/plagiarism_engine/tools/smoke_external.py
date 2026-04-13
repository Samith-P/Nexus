from __future__ import annotations

import json
import sys
import time

import requests


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/plagiarism/check"
    text = sys.argv[2] if len(sys.argv) > 2 else "A Survey on Machine Learning"

    t0 = time.time()
    r = requests.post(
        url,
        files={
            "user_id": (None, "smoke"),
            "check_type": (None, "full"),
            "text": (None, text),
        },
        timeout=120,
    )
    dt = time.time() - t0

    print("status:", r.status_code)
    print("seconds:", round(dt, 2))
    data = r.json()
    dbg = data.get("debug") or {}
    print("external_docs_count:", dbg.get("external_docs_count"))
    print("semantic_chunk_limit:", dbg.get("semantic_chunk_limit"))
    print("providers:")
    print(json.dumps(dbg.get("external_providers_enabled"), indent=2))

    ext_dbg = dbg.get("external_debug") or {}
    for k in ["semantic_scholar", "semantic_scholar_errors", "openalex", "openalex_errors", "arxiv", "arxiv_errors", "tavily", "tavily_errors"]:
        v = ext_dbg.get(k)
        if v:
            print(k + ":")
            try:
                print(json.dumps(v[:2] if isinstance(v, list) else v, indent=2))
            except Exception:
                print(str(v)[:500])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
