"""Prompt compression + tiktoken-based token accounting.

compress(): deterministic, lossless-ish text reduction:
  - collapse whitespace runs, strip trailing/leading space
  - drop exact-duplicate sentences (context dedup)
  - cap per-call context length to a token budget (head-truncation of long docs)
"""

from __future__ import annotations

import re

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text or ""))


def _dedupe_sentences(text: str) -> tuple[str, int]:
    """Remove exact duplicate sentences. Returns (text, n_removed)."""
    sentences = _SENT_SPLIT.split(text)
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for s in sentences:
        key = s.strip().lower()
        if not key:
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(s.strip())
    return " ".join(kept), removed


def compress(text: str, *, max_tokens: int = 4000) -> dict:
    """Compress a prompt. Returns dict with text/tokens_before/tokens_after/saved."""
    original = text or ""
    before = count_tokens(original)

    # 1. whitespace collapse
    out = re.sub(r"[ \t]+", " ", original)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()

    # 2. duplicate sentence removal
    out, dupes_removed = _dedupe_sentences(out)

    # 3. hard cap: truncate tail beyond budget (keep the head = most relevant)
    tokens = _ENCODING.encode(out)
    truncated = False
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        out = _ENCODING.decode(tokens)
        truncated = True

    after = count_tokens(out)
    return {
        "text": out,
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": max(0, before - after),
        "dupes_removed": dupes_removed,
        "truncated": truncated,
    }
