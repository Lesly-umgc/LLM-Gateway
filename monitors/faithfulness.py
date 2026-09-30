"""Hallucination monitor: real RAGAS faithfulness scoring on sampled traffic.

Scoring uses the `ragas` package (metrics.collections.Faithfulness) with an
Ollama model as the judge, reached through ragas' llm_factory over Ollama's
OpenAI-compatible API. The monitor samples a subset of grounded requests and
scores them on an in-process worker queue, so judging never blocks responses.

Env:
  LNM_FAITHFULNESS_SAMPLE_RATE  default 0.20
  LNM_FAITHFULNESS_THRESHOLD    default 0.70
  LNM_JUDGE_MODEL               default llama3.1:8b (needs `ollama pull`)
  OLLAMA_BASE_URL               default http://localhost:11434

If the judge is unreachable, scoring falls back to a deterministic heuristic
(sentence-split claims, content-word overlap) and the result carries
"judge": "heuristic" so callers can tell it apart from "ragas".
"""

from __future__ import annotations

import os
import queue
import re
import threading

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


def _ensure_ragas_importable():
    # ragas 0.4.3 still does `from langchain_community.chat_models.vertexai
    # import ChatVertexAI`, but langchain-community>=0.4 moved VertexAI into
    # its own partner package. The classes are only used for isinstance
    # checks, so stub them instead of pinning old langchain versions.
    import sys
    import types

    for mod_name, cls_name in (
        ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
        ("langchain_community.llms.vertexai", "VertexAI"),
    ):
        try:
            __import__(mod_name)
        except ImportError:
            stub = types.ModuleType(mod_name)
            setattr(stub, cls_name, type(cls_name, (), {}))
            sys.modules[mod_name] = stub


_scorer = None
_scorer_lock = threading.Lock()


def _get_scorer():
    global _scorer
    if _scorer is None:
        with _scorer_lock:
            if _scorer is None:
                _ensure_ragas_importable()
                from openai import AsyncOpenAI

                from ragas.llms import llm_factory
                from ragas.metrics.collections import Faithfulness

                base_url = os.environ.get(
                    "OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
                model = os.environ.get("LNM_JUDGE_MODEL", "llama3.1:8b")
                client = AsyncOpenAI(base_url=base_url + "/v1",
                                    api_key="ollama")
                _scorer = Faithfulness(llm=llm_factory(model, client=client))
    return _scorer


def _statement_counts(res) -> tuple[int | None, int | None]:
    try:
        statements = res.traces["output"].statements
        claims = len(statements)
        supported = sum(1 for s in statements if s.verdict)
        return claims, supported
    except Exception:
        return None, None


def _ragas_score(answer: str, context: str, question: str) -> dict:
    res = _get_scorer().score(user_input=question, response=answer,
                              retrieved_contexts=[context])
    score = float(res.value)
    if score != score:  # NaN: ragas found no verifiable statements
        return {"score": 1.0, "claims": 0, "supported": 0, "flagged": False,
                "judge": "ragas", "note": "no factual claims"}
    claims, supported = _statement_counts(res)
    return {
        "score": score,
        "claims": claims,
        "supported": supported,
        "flagged": score < THRESHOLD,
        "judge": "ragas",
    }


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in _STOPWORDS}


def _heuristic_score(answer: str, context: str) -> dict:
    # Degraded fallback when the judge LLM is unreachable: split the answer
    # into sentences and count a claim supported when at least half of its
    # content words appear in the context. Crude, but deterministic and fast.
    claims = [p.strip() for p in re.split(r"(?<=[.!?])\s+", answer.strip())
              if len(p.split()) >= 4]
    if not claims:
        return {"score": 1.0, "claims": 0, "supported": 0, "flagged": False,
                "judge": "heuristic", "note": "no factual claims"}
    ctx_words = _content_words(context)

    def supported(claim: str) -> bool:
        words = _content_words(claim)
        return bool(words) and len(words & ctx_words) / len(words) >= 0.5

    n_supported = sum(1 for c in claims if supported(c))
    score = n_supported / len(claims)
    return {
        "score": score,
        "claims": len(claims),
        "supported": n_supported,
        "flagged": score < THRESHOLD,
        "judge": "heuristic",
    }


def faithfulness_score(answer: str, context: str, question: str = "") -> dict:
    """Synchronous RAGAS faithfulness evaluation (real `ragas` metric)."""
    try:
        return _ragas_score(answer, context, question)
    except Exception:
        return _heuristic_score(answer, context)


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
                    question: str = "", force: bool = False) -> None:
        import random
        if not context:
            return
        if not force and random.random() >= self.sample_rate:
            return
        self._q.put({"request_id": request_id, "answer": answer,
                      "context": context, "question": question})

    def _worker(self):
        while True:
            job = self._q.get()
            try:
                res = faithfulness_score(job["answer"], job["context"],
                                         job.get("question", ""))
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
