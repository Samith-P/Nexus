from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


@dataclass
class TokenState:
    token: str
    unhealthy_until: float = 0.0
    last_status: int | None = None


class HFTokenPool:
    """Thread-safe Hugging Face token pool with cooldown support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states = [TokenState(token=t) for t in self._load_tokens()]
        self.primary_count = max(1, int(os.getenv("HF_PRIMARY_TOKEN_COUNT", "3")))
        self.fallback_count = max(0, int(os.getenv("HF_FALLBACK_TOKEN_COUNT", "2")))
        self.cooldown_seconds = max(1, int(os.getenv("HF_TOKEN_COOLDOWN_SECONDS", "180")))

    @staticmethod
    def _load_tokens() -> list[str]:
        tokens: list[str] = []

        for idx in range(1, 6):
            token = os.getenv(f"HF_TOKEN_{idx}", "").strip()
            if token:
                tokens.append(token)

        legacy_token = os.getenv("HF_TOKEN", "").strip()
        if legacy_token:
            tokens.append(legacy_token)

        csv_tokens = os.getenv("HF_TOKENS", "").strip()
        if csv_tokens:
            tokens.extend([item.strip() for item in csv_tokens.split(",") if item.strip()])

        return list(dict.fromkeys(tokens))

    def has_tokens(self) -> bool:
        return bool(self._states)

    def token_count(self) -> int:
        return len(self._states)

    def _is_healthy(self, state: TokenState, now: float) -> bool:
        return state.unhealthy_until <= now

    def _healthy_tokens(self) -> list[str]:
        now = time.time()
        healthy = [s.token for s in self._states if self._is_healthy(s, now)]
        if healthy:
            return healthy
        return [s.token for s in self._states]

    def get_primary_tokens(self) -> list[str]:
        with self._lock:
            healthy = self._healthy_tokens()
            return healthy[: self.primary_count]

    def get_fallback_tokens(self, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        with self._lock:
            healthy = [t for t in self._healthy_tokens() if t not in exclude]
            return healthy[: self.fallback_count]

    def mark_result(self, token: str, status_code: int | None) -> None:
        if status_code is None:
            return
        with self._lock:
            for state in self._states:
                if state.token != token:
                    continue
                state.last_status = status_code
                if status_code in {401, 402, 403, 429, 500, 502, 503, 504}:
                    state.unhealthy_until = time.time() + self.cooldown_seconds
                else:
                    state.unhealthy_until = 0.0
                return
