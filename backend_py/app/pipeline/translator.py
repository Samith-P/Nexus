# app/pipeline/translator.py
# Output translation layer for user-facing multilingual responses.
# Keeps the analysis pipeline English-first and translates only the final payload.

from __future__ import annotations

import os
import re
import threading
from itertools import count
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from app.pipeline.multilingual import SUPPORTED_LANGUAGES
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HFChatTranslator:
    """Translate text through the Hugging Face OpenAI-compatible chat endpoint."""

    def __init__(self, target_lang: str = "en"):
        self.hf_tokens = self._load_hf_tokens()
        self.primary_tokens = self.hf_tokens[:3]
        self.fallback_tokens = self.hf_tokens[3:5]
        self.translation_model = os.getenv(
            "TRANSLATION_MODEL",
            "Qwen/Qwen2.5-72B-Instruct",
        ).strip()
        self.translation_max_chars = int(os.getenv("TRANSLATION_MAX_CHARS", "2000"))
        self.hf_chat_url = os.getenv(
            "HF_CHAT_URL",
            "https://router.huggingface.co/v1/chat/completions",
        ).strip()
        self.timeout_seconds = self._resolve_timeout_seconds(target_lang)
        self._request_counter = count(1)
        self._request_counter_lock = threading.Lock()

        if not self.hf_tokens:
            raise RuntimeError(
                "No Hugging Face token found. Configure HF_TOKEN_1..HF_TOKEN_5 or HF_TOKEN in Nexus-journal/backend_py/.env."
            )

        logger.info(
            "HF chat translator initialized with model=%s endpoint=%s token_count=%s primary=%s fallback=%s",
            self.translation_model,
            self.hf_chat_url,
            len(self.hf_tokens),
            len(self.primary_tokens),
            len(self.fallback_tokens),
        )
        logger.debug(
            "HF translator config target_language=%s timeout_seconds=%s max_chars=%s",
            target_lang,
            self.timeout_seconds,
            self.translation_max_chars,
        )

    def _load_hf_tokens(self) -> list[str]:
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

        # Keep token order but remove duplicates.
        return list(dict.fromkeys(tokens))

    def _resolve_timeout_seconds(self, target_lang: str) -> int:
        base_timeout = int(os.getenv("HF_CHAT_TIMEOUT_SECONDS", "60"))
        if target_lang == "te":
            return int(os.getenv("HF_CHAT_TIMEOUT_TE_SECONDS", "180"))
        if target_lang == "hi":
            return int(os.getenv("HF_CHAT_TIMEOUT_HI_SECONDS", "120"))
        return base_timeout

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or not text.strip() or source_lang == target_lang:
            return text

        chunks = self._split_text(text)
        translated_chunks: list[str] = []

        logger.info(
            "Translating text source=%s target=%s chars=%s chunks=%s",
            source_lang,
            target_lang,
            len(text),
            len(chunks),
        )

        for index, chunk in enumerate(chunks, start=1):
            logger.debug(
                "Translating chunk %s/%s target=%s chunk_chars=%s preview=%r",
                index,
                len(chunks),
                target_lang,
                len(chunk),
                chunk[:120],
            )
            try:
                translated_chunks.append(
                    self._translate_chunk(
                        text=chunk,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        chunk_index=index,
                        total_chunks=len(chunks),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Chunk translation failed source=%s target=%s chunk_index=%s/%s; using English chunk. error=%s",
                    source_lang,
                    target_lang,
                    index,
                    len(chunks),
                    exc,
                )
                translated_chunks.append(chunk)

        logger.debug(
            "Translation finished source=%s target=%s translated_chunks=%s",
            source_lang,
            target_lang,
            len(translated_chunks),
        )
        return "\n\n".join(piece for piece in translated_chunks if piece.strip()).strip()

    def _translate_chunk(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        chunk_index: int,
        total_chunks: int,
    ) -> str:
        source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

        payload = {
            "model": self.translation_model,
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
                    "content": (
                        f"Translate the following academic text from {source_name} to {target_name}.\n\n"
                        f"{text}"
                    ),
                },
            ],
        }

        primary_errors: list[str] = []
        if self.primary_tokens:
            executor = ThreadPoolExecutor(max_workers=len(self.primary_tokens))
            futures = {
                executor.submit(
                    self._translate_with_token,
                    token,
                    payload,
                    source_lang,
                    target_lang,
                    chunk_index,
                    total_chunks,
                ): token
                for token in self.primary_tokens
            }
            try:
                for future in as_completed(futures):
                    try:
                        translated = future.result()
                        for pending in futures:
                            if pending is not future:
                                pending.cancel()
                        logger.info(
                            "Chunk translated via primary race winner chunk=%s/%s; skipping remaining primary waits.",
                            chunk_index,
                            total_chunks,
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
                        return translated
                    except Exception as exc:
                        primary_errors.append(str(exc))
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        fallback_errors: list[str] = []
        for token in self.fallback_tokens:
            try:
                return self._translate_with_token(
                    token,
                    payload,
                    source_lang,
                    target_lang,
                    chunk_index,
                    total_chunks,
                )
            except Exception as exc:
                fallback_errors.append(str(exc))

        errors = primary_errors + fallback_errors
        raise RuntimeError(
            "All translation token attempts failed for chunk. "
            f"errors={errors[:5]}"
        )

    def _translate_with_token(
        self,
        token: str,
        payload: dict,
        source_lang: str,
        target_lang: str,
        chunk_index: int,
        total_chunks: int,
    ) -> str:
        token_hint = token[:8]
        with self._request_counter_lock:
            request_id = next(self._request_counter)

        logger.info(
            "[req=%s] HF translation request start source=%s target=%s chunk=%s/%s token=%s",
            request_id,
            source_lang,
            target_lang,
            chunk_index,
            total_chunks,
            token_hint,
        )

        client = httpx.Client(
            timeout=self.timeout_seconds,
            headers={"Authorization": f"Bearer {token}"},
        )

        try:
            response = client.post(self.hf_chat_url, json=payload)
        except httpx.HTTPError as exc:
            logger.warning(
                "[req=%s] HF translation request error chunk=%s/%s token=%s error=%s",
                request_id,
                chunk_index,
                total_chunks,
                token_hint,
                exc,
            )
            logger.debug(
                "HF translation HTTP error source=%s target=%s token=%s error=%s",
                source_lang,
                target_lang,
                token_hint,
                exc,
            )
            raise RuntimeError(f"request_failed token={token_hint} error={exc}") from exc
        finally:
            client.close()

        logger.info(
            "[req=%s] HF translation response chunk=%s/%s token=%s status=%s",
            request_id,
            chunk_index,
            total_chunks,
            token_hint,
            response.status_code,
        )

        if response.status_code in {401, 403}:
            logger.debug(
                "HF translation unauthorized source=%s target=%s token=%s status=%s",
                source_lang,
                target_lang,
                token_hint,
                response.status_code,
            )
            raise RuntimeError(f"unauthorized token={token_hint} status={response.status_code}")

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.debug(
                "HF translation status error source=%s target=%s token=%s status=%s body_preview=%r",
                source_lang,
                target_lang,
                token_hint,
                response.status_code,
                response.text[:200],
            )
            raise RuntimeError(
                f"status_error token={token_hint} status={response.status_code}"
            ) from exc

        try:
            translated = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.debug(
                "HF translation parse error source=%s target=%s token=%s response_preview=%r",
                source_lang,
                target_lang,
                token_hint,
                response.text[:200],
            )
            raise RuntimeError(f"parse_error token={token_hint}") from exc

        logger.debug(
            "HF translation succeeded source=%s target=%s token=%s translated_chars=%s",
            source_lang,
            target_lang,
            token_hint,
            len(translated),
        )
        logger.info(
            "[req=%s] HF translation completed chunk=%s/%s token=%s translated_chars=%s",
            request_id,
            chunk_index,
            total_chunks,
            token_hint,
            len(translated),
        )
        return self._clean_response(translated)

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.translation_max_chars:
            return [text]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for paragraph in text.split("\n\n"):
            piece = paragraph.strip()
            if not piece:
                continue

            piece_len = len(piece)
            if piece_len > self.translation_max_chars:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0

                chunks.extend(self._split_oversized_piece(piece))
                continue

            if current and current_len + piece_len + 2 > self.translation_max_chars:
                chunks.append("\n\n".join(current))
                current = [piece]
                current_len = piece_len
            else:
                current.append(piece)
                current_len += piece_len + 2

        if current:
            chunks.append("\n\n".join(current))

        return chunks or [text]

    def _split_oversized_piece(self, text: str) -> list[str]:
        max_chars = self.translation_max_chars
        sentence_parts = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentence_parts) == 1:
            return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentence_parts:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_len = 0
                chunks.extend([sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)])
                continue

            projected_len = current_len + len(sentence) + (1 if current else 0)
            if current and projected_len > max_chars:
                chunks.append(" ".join(current).strip())
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len = projected_len

        if current:
            chunks.append(" ".join(current).strip())

        return [chunk for chunk in chunks if chunk.strip()]

    @staticmethod
    def _clean_response(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()
        return cleaned


class TranslationOutputLayer:
    """Translate the final literature review payload to the requested language."""

    def __init__(self, target_lang: str = "en"):
        self.target_lang = target_lang
        self.translator = None
        self.parallel_workers = max(1, int(os.getenv("TRANSLATION_PARALLEL_WORKERS", "4")))

        if target_lang != "en":
            if target_lang not in SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Unsupported target language '{target_lang}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
                )
            logger.info("Translation output layer active: target=%s", target_lang)
            self.translator = HFChatTranslator(target_lang=target_lang)
        else:
            logger.info("Translation output layer: English (pass-through)")

    def _translate_texts_parallel(self, texts: list[str], label: str) -> list[str]:
        if self.target_lang == "en" or self.translator is None:
            return texts
        if not texts:
            return texts

        logger.info(
            "Parallel translation start label=%s items=%s workers=%s",
            label,
            len(texts),
            self.parallel_workers,
        )

        translated = list(texts)
        with ThreadPoolExecutor(max_workers=min(self.parallel_workers, len(texts))) as executor:
            futures = {
                executor.submit(self._translate_text, text): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    translated[idx] = future.result()
                except Exception as exc:
                    logger.warning(
                        "Parallel translation failed label=%s item_index=%s; preserving English. error=%s",
                        label,
                        idx,
                        exc,
                    )

        logger.info("Parallel translation end label=%s items=%s", label, len(texts))
        return translated

    def _translate_mapping_values_parallel(self, items: dict[str, str], label: str) -> dict[str, str]:
        if not items:
            return items
        keys = list(items.keys())
        translated_values = self._translate_texts_parallel([items[key] for key in keys], label=label)
        return {key: translated_values[idx] for idx, key in enumerate(keys)}

    def _translate_text(self, text: str) -> str:
        if not text or not text.strip():
            return text
        if self.target_lang == "en" or self.translator is None:
            return text
        return self.translator.translate_text(text=text, source_lang="en", target_lang=self.target_lang)

    def _translate_list(self, items: list[str]) -> list[str]:
        return self._translate_texts_parallel(items, label="list_items")

    def _translate_mapping_values(self, items: dict[str, str]) -> dict[str, str]:
        return self._translate_mapping_values_parallel(items, label="mapping_values")

    def translate_analysis(self, analysis) -> None:
        if self.target_lang == "en":
            return

        logger.debug(
            "Translating analysis payload target=%s title_chars=%s summary_chars=%s sections=%s",
            self.target_lang,
            len(analysis.title or ""),
            len(analysis.summary or ""),
            len(analysis.sections or {}),
        )

        logger.info("Translating analysis field=title")
        analysis.title = self._translate_text(analysis.title)

        logger.info("Translating analysis field=sections count=%s", len(analysis.sections or {}))
        analysis.sections = self._translate_mapping_values_parallel(
            analysis.sections,
            label="analysis.sections",
        )

        logger.info("Translating analysis field=summary")
        analysis.summary = self._translate_text(analysis.summary)

        logger.info("Translating analysis field=section_summaries count=%s", len(analysis.section_summaries or {}))
        analysis.section_summaries = self._translate_mapping_values_parallel(
            analysis.section_summaries,
            label="analysis.section_summaries",
        )

        logger.info("Translating analysis field=insights.contributions count=%s", len(analysis.insights.contributions))
        analysis.insights.contributions = self._translate_texts_parallel(
            analysis.insights.contributions,
            label="analysis.insights.contributions",
        )

        logger.info("Translating analysis field=insights.methods count=%s", len(analysis.insights.methods))
        analysis.insights.methods = self._translate_texts_parallel(
            analysis.insights.methods,
            label="analysis.insights.methods",
        )

        logger.info("Translating analysis field=insights.results count=%s", len(analysis.insights.results))
        analysis.insights.results = self._translate_texts_parallel(
            analysis.insights.results,
            label="analysis.insights.results",
        )

        logger.info("Translating analysis field=gaps count=%s", len(analysis.gaps))
        analysis.gaps = self._translate_texts_parallel(analysis.gaps, label="analysis.gaps")

        logger.info("Translating analysis field=evidence_spans count=%s", len(analysis.evidence_spans))
        evidence_texts = [evidence.text for evidence in analysis.evidence_spans]
        translated_evidence = self._translate_texts_parallel(evidence_texts, label="analysis.evidence_spans")
        for idx, evidence in enumerate(analysis.evidence_spans):
            evidence.text = translated_evidence[idx]

        logger.debug("Analysis translation complete target=%s", self.target_lang)

    def translate_review_result(self, result) -> None:
        if self.target_lang == "en":
            return

        logger.debug(
            "Translating review result target=%s papers=%s has_comparison_matrix=%s related_works=%s",
            self.target_lang,
            len(result.papers),
            bool(result.comparison_matrix),
            len(result.related_works),
        )

        logger.info("Translation phase start target=%s papers=%s", self.target_lang, len(result.papers))
        for paper in result.papers:
            self.translate_analysis(paper)

        if result.comparison_matrix:
            for entry in result.comparison_matrix.entries:
                entry.paper_title = self._translate_text(entry.paper_title)
                entry.methods = self._translate_list(entry.methods)
                entry.results = self._translate_list(entry.results)
                entry.gaps = self._translate_list(entry.gaps)

            result.comparison_matrix.common_methods = self._translate_list(
                result.comparison_matrix.common_methods
            )
            result.comparison_matrix.differing_methods = self._translate_list(
                result.comparison_matrix.differing_methods
            )

        result.common_themes = self._translate_list(result.common_themes)
        result.research_gaps = self._translate_list(result.research_gaps)

        for related_work in result.related_works:
            related_work.title = self._translate_text(related_work.title)
            if related_work.abstract:
                related_work.abstract = self._translate_text(related_work.abstract)

        logger.debug("Review result translation complete target=%s", self.target_lang)
        logger.info("Translation phase complete target=%s", self.target_lang)
