"""Real NeMo Guardrails runtime for the live request path.

Builds nemoguardrails' LLMRails from rails/config.yml + rails/rails.co,
with ChatOllama as the underlying LLM, and registers this repo's rail
logic as NeMo actions:

  nemo_input_judge   input rail  - LLM injection/jailbreak judge
                                   (rails/actions.py); memoized so the two
                                   parallel input flows cost one LLM call
  nemo_redact_output output rail - PII redaction (rails/pii.py), applied to
                                   the bot message via a context update
  nemo_policy_check  output rail - LLM policy judge (rails/actions.py)

The gateway delegates generation to generate() here instead of calling
the backend directly, so NeMo's runtime executes the input/output rails
on every request. The deterministic prefilter (rails/prefilter.py) stays
in the gateway as a cheap first layer in front of NeMo.
"""

from __future__ import annotations

import os
import threading

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.integrations.langchain.llm_adapter import (
    LangChainLLMAdapter,
)

from rails import pii
from rails.actions import judge_input_injection, judge_output_policy

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Must match the `define bot ...` canonical forms in rails/rails.co —
# the gateway maps these back to 403 rail blocks.
REFUSAL_INPUT = (
    "I can't help with that. Your request was blocked by the security gateway "
    "because it looks like a prompt-injection or jailbreak attempt."
)
REFUSAL_OUTPUT = (
    "I can't help with that. The response was blocked by the security "
    "gateway's content-policy rail."
)

_judge_cache: dict[str, tuple[bool, str]] = {}
_judge_lock = threading.Lock()


def _judge_once(text: str) -> tuple[bool, str]:
    # Both input flows invoke the judge for the same message; the lock
    # serializes them so the LLM is only called once per unique text.
    with _judge_lock:
        if text not in _judge_cache:
            _judge_cache[text] = judge_input_injection(text)
            if len(_judge_cache) > 512:
                _judge_cache.clear()
        return _judge_cache[text]


def _input_judge_action(context: dict) -> dict:
    blocked, _reason = _judge_once(context.get("user_message", "") or "")
    return {"blocked": blocked}


def _redact_output_action(context: dict):
    text = context.get("bot_message", "") or ""
    redacted, kinds = pii.redact(text)
    if kinds:
        return ActionResult(
            return_value={"changed": True, "kinds": kinds},
            context_updates={"bot_message": redacted},
        )
    return {"changed": False, "kinds": []}


def _policy_check_action(context: dict) -> dict:
    blocked, _reason = judge_output_policy(context.get("bot_message", "") or "")
    return {"blocked": blocked}


_rails: LLMRails | None = None
_build_lock = threading.Lock()


def _build() -> LLMRails:
    from langchain_ollama import ChatOllama

    config = RailsConfig.from_path(_CONFIG_DIR)
    llm = LangChainLLMAdapter(ChatOllama(
        model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        base_url=os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        temperature=0.2,
    ))
    rails = LLMRails(config, llm=llm)
    rails.register_action(_input_judge_action, "nemo_input_judge")
    rails.register_action(_redact_output_action, "nemo_redact_output")
    rails.register_action(_policy_check_action, "nemo_policy_check")
    return rails


def get_rails() -> LLMRails:
    global _rails
    if _rails is None:
        with _build_lock:
            if _rails is None:
                _rails = _build()
    return _rails


def generate(messages: list[dict]) -> str:
    """Run one request through NeMo's runtime; return the final text."""
    res = get_rails().generate(messages=messages)
    if isinstance(res, dict):
        return res.get("content", "") or ""
    return str(res)


def reset_for_tests():
    """Drop the cached runtime (tests only)."""
    global _rails
    with _build_lock:
        _rails = None
    with _judge_lock:
        _judge_cache.clear()
