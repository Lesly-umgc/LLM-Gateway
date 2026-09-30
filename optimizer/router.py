"""Tiered model routing: cheap heuristic picks the model tier per request.

Simple factual questions go to the small tier, multi-part / analytical /
code / debugging questions go to the large tier. Deterministic and
explainable — routing itself must cost ~nothing, so there is deliberately
no LLM call here. Returns (tier, reason) with tier in {"small", "large"}.
"""

from __future__ import annotations

import re

from optimizer.compression import count_tokens

SMALL = "small"
LARGE = "large"

# signals that a question needs the larger model (each worth +2)
_COMPLEX = [
    "explain why", "compare", "contrast", "analyze", "analyse",
    "pros and cons", "step by step", "step-by-step", "debug", "troubleshoot",
    "write a", "write me", "draft a", "summarize", "essay", "code",
    "function", "script", "traceback", "error:", "what if", "how would",
    "recommend", "which plan", "dispute", "why does", "why is",
    "integrate", "migrate", "optimize",
]

# signals that a question is a simple lookup (each worth -2)
_SIMPLE = [
    "what is your", "what are your", "where is", "how do i reset",
    "define", "translate", "capital of", "do you offer", "is there a",
]

_THRESHOLD = 2


def route(user_text: str) -> tuple[str, str]:
    t = (user_text or "").lower()
    score = 0
    reasons: list[str] = []

    tokens = count_tokens(user_text)
    if tokens > 120:
        score += 2
        reasons.append("long prompt")
    elif tokens > 60:
        score += 1
        reasons.append("medium prompt")

    for sig in _COMPLEX:
        if sig in t:
            score += 2
            reasons.append(f"complex signal: {sig!r}")
            break
    for sig in _SIMPLE:
        if sig in t:
            score -= 2
            reasons.append(f"simple signal: {sig!r}")
            break

    if t.count("?") > 1:
        score += 1
        reasons.append("multi-part question")
    if "`" in t or "traceback" in t:
        score += 2
        reasons.append("code content")

    tier = LARGE if score >= _THRESHOLD else SMALL
    reason = "; ".join(reasons) if reasons else "short factual question"
    return tier, reason
