"""Deterministic heuristic pre-filter for prompt-injection / jailbreak defense.

Fast, no-LLM, zero false-positive-tolerant patterns. Anything matched here is
blocked without invoking the LLM judge. Anything not matched but suspicious goes
to the judge (see rails/engine.py)."""

from __future__ import annotations

import base64
import binascii
import re

# (regex, reason) — curated from real-world jailbreak / injection techniques.
_BLOCK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(all\s+|your\s+|any\s+)?(previous|prior|earlier|above)\s+instructions?", re.I),
     "instruction-override"),
    (re.compile(r"disregard\s+(all\s+|your\s+)?(previous|prior|earlier|above|the)\s+(instructions?|rules?|guidelines?)", re.I),
     "instruction-override"),
    (re.compile(r"\bDAN\b"), "dan-jailbreak"),
    (re.compile(r"do\s+anything\s+now", re.I), "dan-jailbreak"),
    (re.compile(r"jailbreak", re.I), "jailbreak-mention"),
    (re.compile(r"(reveal|show|print|output|repeat|disclose)\s+(your|the|this)\s+(system\s+prompt|instructions?|initial\s+prompt|developer\s+message)", re.I),
     "prompt-leakage"),
    (re.compile(r"what\s+(is|are)\s+your\s+(system\s+prompt|instructions|hidden\s+rules)", re.I),
     "prompt-leakage"),
    (re.compile(r"(developer\s+mode|god\s*mode|unrestricted\s+mode|evil\s+mode)", re.I),
     "mode-escalation"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(unrestricted|evil|amoral|uncensored)", re.I),
     "roleplay-escalation"),
    (re.compile(r"you\s+are\s+now\s+(an?\s+)?(unrestricted|evil|amoral|uncensored|jailbroken)", re.I),
     "roleplay-escalation"),
    (re.compile(r"roleplay\s+as\s+(an?\s+)?(unrestricted|evil|amoral|uncensored)", re.I),
     "roleplay-escalation"),
    (re.compile(r"(bypass|circumvent|disable|turn\s+off|remove)\s+(your|the|all)\s+(safety|guardrails?|filters?|restrictions?|policies?)", re.I),
     "safety-bypass"),
    (re.compile(r"(hypothetically|in\s+a\s+fictional|for\s+educational\s+purposes).{0,60}(bomb|weapon|malware|ransomware|poison|explosive)", re.I),
     "harmful-preface"),
    (re.compile(r"how\s+(do|can)\s+i\s+make\s+(a\s+)?(bomb|explosive|bioweapon|chemical\s+weapon|gun|malware|ransomware|virus)", re.I),
     "harmful-request"),
    (re.compile(r"write\s+(malware|ransomware|a\s+virus|a\s+keylogger|a\s+trojan)", re.I),
     "harmful-request"),
    (re.compile(r"help\s+me\s+(hack|break\s+into|phish|dox)", re.I), "harmful-request"),
    (re.compile(r"\bsudo\b.{0,40}(mode|prompt|override)", re.I), "sudo-escalation"),
    (re.compile(r"\n\s*(system|developer|admin)\s*:\s*", re.I), "role-spoofing"),
    (re.compile(r"<\|(system|im_start\|system)\|>", re.I), "token-injection"),
    (re.compile(r"###\s*instruction\s*###", re.I), "instruction-spoofing"),
    (re.compile(r"\[INST\]", re.I), "instruction-spoofing"),
    (re.compile(r"(from\s+now\s+on|starting\s+now),?\s+you\s+(will|must|are)", re.I), "instruction-override"),
    (re.compile(r"forget\s+(everything|all|your)\s+(you\s+know|instructions|rules)", re.I), "instruction-override"),
]

# Suspicious-but-not-conclusive markers -> route to the LLM judge instead of
# blocking outright. Keeps precision high on ambiguous phrasing.
_SUSPICIOUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore", re.I), "word-ignore"),
    (re.compile(r"override", re.I), "word-override"),
    (re.compile(r"pretend", re.I), "word-pretend"),
    (re.compile(r"hypothetically", re.I), "word-hypothetically"),
    (re.compile(r"base64|rot13|caesar", re.I), "encoding-mention"),
    (re.compile(r"system prompt", re.I), "word-system-prompt"),
]

_MIN_BASE64_LEN = 40


def _contains_obfuscated_payload(text: str) -> bool:
    """Detect long base64 blobs that decode to instruction-like text."""
    for m in re.finditer(r"[A-Za-z0-9+/=]{%d,}" % _MIN_BASE64_LEN, text):
        blob = m.group(0).strip("=")
        if len(blob) < _MIN_BASE64_LEN:
            continue
        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4)).decode(
                "utf-8", errors="strict"
            )
        except (binascii.Error, UnicodeDecodeError):
            continue
        lowered = decoded.lower()
        if any(
            kw in lowered
            for kw in ("ignore", "system", "instruction", "jailbreak", "bypass", "disregard")
        ):
            return True
    return False


def prefilter(text: str) -> dict:
    """Classify input text.

    Returns {"decision": "block"|"allow"|"judge", "reason": str|None}.
    """
    if not text or not text.strip():
        return {"decision": "allow", "reason": None}

    for pattern, reason in _BLOCK_PATTERNS:
        if pattern.search(text):
            return {"decision": "block", "reason": reason}

    if _contains_obfuscated_payload(text):
        return {"decision": "block", "reason": "obfuscated-base64-payload"}

    for pattern, reason in _SUSPICIOUS_PATTERNS:
        if pattern.search(text):
            return {"decision": "judge", "reason": reason}

    return {"decision": "allow", "reason": None}
