"""Rails engine: input rails + output rails per rails/config.yml.

Pipeline:
  check_input(text)
      1. heuristic prefilter (rails/prefilter.py)
         - block  -> BLOCKED (deterministic)
         - allow  -> ALLOWED
         - judge  -> LLM judge (rails/actions.py); block iff judge says BLOCK
  check_output(text)
      1. PII redaction (rails/pii.py) -> redact in place, record kinds
      2. policy judge on the redacted text -> BLOCKED iff judge says BLOCK
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

from rails import prefilter, pii
from rails import actions

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")

REFUSAL_INPUT = (
    "I can't help with that. Your request was blocked by the security gateway "
    "because it looks like a prompt-injection or jailbreak attempt."
)
REFUSAL_OUTPUT = (
    "I can't help with that. The response was blocked by the security gateway's "
    "content-policy rail."
)


@dataclass
class Verdict:
    blocked: bool
    reason: str | None = None
    method: str = "heuristic"  # heuristic | llm-judge | pii
    redactions: list[str] = field(default_factory=list)
    text: str = ""  # possibly-redacted text for outputs


class RailsEngine:
    def __init__(self, config_path: str = _CONFIG_PATH):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        rails = (self.config.get("rails") or {})
        self.input_flows = list((rails.get("input") or {}).get("flows", []))
        self.output_flows = list((rails.get("output") or {}).get("flows", []))
        thresholds = self.config.get("thresholds") or {}
        self.faithfulness_threshold = float(
            thresholds.get("faithfulness_flag_below", 0.70)
        )

    # ---- input rail ----
    def check_input(self, text: str) -> Verdict:
        pf = prefilter.prefilter(text)
        if pf["decision"] == "block":
            return Verdict(blocked=True, reason=pf["reason"], method="heuristic")
        if pf["decision"] == "judge":
            blocked, reason = actions.judge_input_injection(text)
            if blocked:
                return Verdict(blocked=True, reason="injection-judge", method="llm-judge")
            return Verdict(blocked=False, reason=pf["reason"], method="llm-judge")
        return Verdict(blocked=False, reason=None, method="heuristic")

    # ---- output rails ----
    def check_output(self, text: str) -> Verdict:
        redacted, kinds = pii.redact(text)
        blocked, reason = actions.judge_output_policy(redacted)
        if blocked:
            return Verdict(
                blocked=True, reason="policy-judge", method="llm-judge",
                redactions=kinds, text="",
            )
        return Verdict(
            blocked=False, reason=None, method="pii" if kinds else "heuristic",
            redactions=kinds, text=redacted,
        )

    def scan_input_pii(self, text: str) -> list[dict]:
        return pii.scan(text)
