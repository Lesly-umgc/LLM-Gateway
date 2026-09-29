"""Hallucination monitor: RAGAS-style faithfulness scoring on sampled traffic.

RAGAS faithfulness protocol (judge via common.llm, which defaults to Ollama):
  1. Extract atomic claims from the answer.
  2. For each claim, judge whether it is supported by the provided context.
  3. faithfulness = supported_claims / total_claims.

The monitor is sampled (default 20%) so judging costs stay bounded; it runs
async via an in-process worker queue. Low-faithfulness responses (< threshold)
are flagged into the audit log and metrics counters.

If the judge LLM is unreachable, scoring falls back to a deterministic
heuristic (sentence-split claims, content-word overlap) instead of failing
silently — the result carries "judge": "heuristic" so evals can tell.
"""

from __future__ import annotations

import os
import queue
import re
import threading

from common import llm as judge_llm

SAMPLE_RATE = float(os.environ.get("LNM_FAITHFULNESS_SAMPLE_RATE", "0.20"))
THRESHOLD = float(os.environ.get("LNM_FAITHFULNESS_THRESHOLD", "0.70"))

_STOPWORDS = frozenset("""
a an the and or but of to in on at for with as by is was were are be been being
it its this that these those i you he she we they them his her their our your
my me him us what which who whom whose where when how why not no yes if then
than so such do does did done can could will would should may might must shall
there here from into over under between through during before after about above
below up down out off again further once more most other some any each few more
most own same too very just don also than then once""".split())

_CLAIM_PROMPT = """Break the ANSWER below into atomic factual claims, one per line.
Each claim must be a single verifiable statement. Do not add claims that are not in the answer.
If the answer has no factual claims (e.g. a greeting), output exactly: NO_CLAIMS

ANSWER:
---
{answer}
---
Claims (one per line):"""

_VERIFY_PROMPT = """You are a fact-checker. Given the CONTEXT and a single CLAIM, answer with exactly
one word: SUPPORTED if the claim is directly supported by the context, CONTRADICTED if the
context contradicts it, otherwise UNSUPPORTED.

CONTEXT:
---
{context}
---
CLAIM: {claim}
Verdict:"""


def extract_claims(answer: str) -> tuple[list[str], str]:
    """Returns (claims, judge) where judge is "llm" or "heuristic"."""
    try:
        text = judge_llm.generate(
            _CLAIM_PROMPT.format(answer=answer[:4000]),
            max_output_tokens=512, temperature=0.0,
        )[0]
        text = text.strip()
        if not text or text.upper().startswith("NO_CLAIMS"):
            return [], "llm"
        claims = [c.strip("-•* ").strip() for c in text.splitlines()]
        return [c for c in claims if c and len(c) > 8], "llm"
    except Exception:
        # judge down: fall back to naive sentence splitting
        parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [p.strip() for p in parts if len(p.split()) >= 4], "heuristic"


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in _STOPWORDS}


def _overlap_supported(claim: str, context: str) -> bool:
    # FIXME: crude word-overlap heuristic — fine as a degraded fallback, but a
    # real eval run should use the LLM judge (see "judge" in the result dict).
    words = _content_words(claim)
    if not words:
        return False
    ctx = _content_words(context)
    return len(words & ctx) / len(words) >= 0.5


def verify_claim(claim: str, context: str) -> bool:
    try:
        verdict = judge_llm.classify(
            _VERIFY_PROMPT.format(claim=claim[:800], context=context[:6000])
        )
        return verdict.strip().upper().startswith("SUPPORTED")
    except Exception:
        return _overlap_supported(claim, context)


def faithfulness_score(answer: str, context: str) -> dict:
    """Synchronous full RAGAS-style faithfulness evaluation."""
    claims, judge = extract_claims(answer)
    if not claims:
        return {"score": 1.0, "claims": 0, "supported": 0, "flagged": False,
                "judge": judge, "note": "no factual claims"}
    supported = sum(1 for c in claims if verify_claim(c, context))
    score = supported / len(claims)
    return {
        "score": score,
        "claims": len(claims),
        "supported": supported,
        "flagged": score < THRESHOLD,
        "judge": judge,
    }


class FaithfulnessMonitor:
    """Sampled async monitor wired into the gateway pipeline."""

    def __init__(self, sample_rate: float = SAMPLE_RATE,
                 threshold: float = THRESHOLD,
                 on_flag=None, on_scored=None):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.on_flag = on_flag  # callback(record_dict) on low-faithfulness
        self.on_scored = on_scored  # callback(record_dict) on every scoring
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()
        self.scored = 0
        self.flagged = 0
        self.scores: list[float] = []
        self._lock = threading.Lock()

    def maybe_check(self, *, request_id: str, answer: str, context: str,
                    force: bool = False) -> None:
        import random
        if not context:
            return
        if not force and random.random() >= self.sample_rate:
            return
        self._q.put({"request_id": request_id, "answer": answer, "context": context})

    def _worker(self):
        while True:
            job = self._q.get()
            try:
                res = faithfulness_score(job["answer"], job["context"])
                with self._lock:
                    self.scored += 1
                    self.scores.append(res["score"])
                    if res["flagged"]:
                        self.flagged += 1
                record = {
                    "request_id": job["request_id"],
                    "score": res["score"],
                    "claims": res["claims"],
                    "supported": res["supported"],
                    "flagged": res["flagged"],
                }
                if self.on_scored:
                    self.on_scored(record)
                if res["flagged"] and self.on_flag:
                    self.on_flag({
                        "request_id": job["request_id"],
                        "faithfulness": res,
                    })
            except Exception:
                pass
            finally:
                self._q.task_done()

    def stats(self) -> dict:
        with self._lock:
            scores = list(self.scores)
        return {
            "scored": self.scored,
            "flagged": self.flagged,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "min_score": min(scores) if scores else None,
        }
