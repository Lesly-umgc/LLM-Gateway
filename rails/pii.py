"""Deterministic PII detection and redaction for input scan + output rails."""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("PHONE", re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

_PLACEHOLDER = "[REDACTED:{kind}]"


def scan(text: str) -> list[dict]:
    """Return list of {"kind", "match"} found in text (values masked in logs)."""
    hits: list[dict] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text or ""):
            # Avoid flagging years / plain numbers as credit cards
            if kind == "CREDIT_CARD" and not _looks_like_card(m.group(0)):
                continue
            if kind == "PHONE" and not _looks_like_phone(m.group(0)):
                continue
            hits.append({"kind": kind, "length": len(m.group(0))})
    return hits


def _looks_like_card(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    if len(digits) not in (13, 14, 15, 16):
        return False
    # Luhn check
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _looks_like_phone(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return len(digits) in (10, 11)


def redact(text: str) -> tuple[str, list[str]]:
    """Redact PII in place. Returns (redacted_text, [kinds_redacted])."""
    kinds: list[str] = []
    out = text or ""
    for kind, pattern in _PATTERNS:
        def _repl(m, kind=kind):
            if kind == "CREDIT_CARD" and not _looks_like_card(m.group(0)):
                return m.group(0)
            if kind == "PHONE" and not _looks_like_phone(m.group(0)):
                return m.group(0)
            kinds.append(kind)
            return _PLACEHOLDER.format(kind=kind)

        out = pattern.sub(_repl, out)
    return out, sorted(set(kinds))
