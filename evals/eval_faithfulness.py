"""Focused faithfulness eval: real `ragas` metric on the 20-sample grounded set.

Reuses the GROUNDED questions and the live gateway pipeline from
run_benchmark, but scores each answer with the real ragas Faithfulness
metric (Ollama judge) instead of the old RAGAS-style heuristic.

Two phases, because a small machine cannot hold both models at once:
  1. answers: generate one gateway answer per sample (3B backend),
     cached in evals/faithfulness_answers.json
  2. scores:  judge the cached answers with ragas + the 8B judge model,
     results in evals/faithfulness_ragas.json

Costs: one gateway answer per sample plus ragas judging (statement
extraction + per-statement NLI on the judge model).

Env:
  LNM_JUDGE_MODEL  judge model, default llama3.1:8b
  EVAL_MAX_TOKENS  default 160
  OLLAMA_BASE_URL  default http://localhost:11434
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("LNM_API_KEYS", "test-key")
os.environ.setdefault("LNM_ADMIN_KEY", "admin-key")
os.environ.setdefault("LNM_BACKEND", "ollama")
# score directly via fm.faithfulness_score; the gateway's sampled worker
# would double the judge load, so disable it for this run
os.environ.setdefault("LNM_FAITHFULNESS_SAMPLE_RATE", "0")

from fastapi.testclient import TestClient  # noqa: E402

from gateway.app import create_app  # noqa: E402
from gateway.auth import GatewayConfig  # noqa: E402
from monitors import faithfulness as fm  # noqa: E402
from run_benchmark import GROUNDED, _post  # noqa: E402

EVAL_MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "160"))
JUDGE_MODEL = os.environ.get("LNM_JUDGE_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL",
                                 "http://localhost:11434").rstrip("/")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__),
                            "faithfulness_answers.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__),
                            "faithfulness_ragas.json")


def generate_answers() -> list:
    """Phase 1: one live gateway answer per grounded sample (3B backend)."""
    config = GatewayConfig(api_keys={"test-key"}, admin_key="admin-key",
                           backend_name="ollama",
                           audit_path="evals/audit_faithfulness.jsonl",
                           rate_limit_per_min=1000)
    client = TestClient(create_app(config))
    assert client.get("/healthz").json()["status"] == "ok"

    answers = []
    for i, (ctx, q) in enumerate(GROUNDED, 1):
        res = _post(client, q, context=ctx)
        if res["status"] != 200:
            print(f"[{i:2d}/20] HTTP {res['status']} :: {q[:50]}", flush=True)
            continue
        answer = res["payload"]["choices"][0]["message"]["content"]
        answers.append({"question": q, "context": ctx, "answer": answer})
        print(f"[{i:2d}/20] answer={len(answer)} chars :: {q[:50]}", flush=True)
    with open(ANSWERS_PATH, "w") as f:
        json.dump(answers, f, indent=2)
    print(f"cached {len(answers)} answers in {ANSWERS_PATH}")
    return answers


def prime_judge():
    # On small-RAM machines the 8B weights only fit via mmap; a plain load
    # gets OOM-killed, so warm it through the native API with mmap forced.
    # Once loaded, the ragas judge calls below reuse the running instance.
    # stream=true so headers arrive immediately; a blocking (stream=false)
    # call would sit headerless through the multi-minute mmap load and hit
    # the client timeout.
    req = urllib.request.Request(
        OLLAMA_BASE_URL + "/api/generate",
        data=json.dumps({"model": JUDGE_MODEL, "prompt": "ok",
                         "stream": True, "keep_alive": "60m",
                         "options": {"use_mmap": True, "num_ctx": 2048}}
                        ).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.strip():
                continue
            if json.loads(line).get("done"):
                break
    print(f"judge {JUDGE_MODEL} warmed (mmap, keep_alive 60m)", flush=True)


def score_answers(answers: list) -> dict:
    """Phase 2: ragas faithfulness on the cached answers (8B judge).

    Resumable: each scored sample is flushed to RESULTS_PATH, and samples
    already present there are skipped on a rerun.
    """
    started = time.time()
    prime_judge()

    done = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                for d in json.load(f).get("detail", []):
                    done[d["question"]] = d
        except (json.JSONDecodeError, KeyError):
            pass

    scores, flagged, detail, judges = [], 0, [], {}
    for i, item in enumerate(answers, 1):
        key = item["question"][:60]
        if key in done:
            d = done[key]
            scores.append(d["score"])
            flagged += 1 if d["flagged"] else 0
            judges[d["judge"]] = judges.get(d["judge"], 0) + 1
            detail.append(d)
            print(f"[{i:2d}/{len(answers)}] cached score={d['score']:.3f}"
                  f" :: {item['question'][:50]}", flush=True)
            continue
        t0 = time.time()
        s = fm.faithfulness_score(item["answer"], item["context"],
                                  question=item["question"])
        dt = time.time() - t0
        judges[s["judge"]] = judges.get(s["judge"], 0) + 1
        scores.append(s["score"])
        flagged += 1 if s["flagged"] else 0
        d = {"question": key,
             "score": round(s["score"], 3),
             "claims": s["claims"], "supported": s["supported"],
             "flagged": s["flagged"], "judge": s["judge"],
             "score_seconds": round(dt, 1)}
        detail.append(d)
        print(f"[{i:2d}/{len(answers)}] score={s['score']:.3f}"
              f" judge={s['judge']} {dt:5.0f}s"
              f" :: {item['question'][:50]}", flush=True)
        _write_partial(detail, started)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge": "ragas",
        "judge_model": JUDGE_MODEL,
        "n": len(scores),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "flagged": flagged,
        "flag_rate": flagged / len(scores) if scores else 0,
        "threshold": fm.THRESHOLD,
        "judges_used": judges,
        "elapsed_seconds": round(time.time() - started, 1),
        "detail": detail,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


def _write_partial(detail: list, started: float):
    flagged = sum(1 for d in detail if d["flagged"])
    judges: dict = {}
    for d in detail:
        judges[d["judge"]] = judges.get(d["judge"], 0) + 1
    scores = [d["score"] for d in detail]
    partial = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge": "ragas",
        "judge_model": JUDGE_MODEL,
        "n": len(scores),
        "partial": True,
        "flagged": flagged,
        "judges_used": judges,
        "elapsed_seconds": round(time.time() - started, 1),
        "detail": detail,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(partial, f, indent=2)


def load_cached_answers() -> list | None:
    if not os.path.exists(ANSWERS_PATH):
        return None
    with open(ANSWERS_PATH) as f:
        answers = json.load(f)
    if len(answers) != len(GROUNDED):
        return None
    return answers


def run() -> dict:
    answers = load_cached_answers()
    if answers is None:
        print("phase 1: generating gateway answers (3B backend)")
        answers = generate_answers()
    else:
        print(f"phase 1: reusing {len(answers)} cached answers")
    print("phase 2: ragas judging with", JUDGE_MODEL)
    return score_answers(answers)


if __name__ == "__main__":
    r = run()
    print(f"n={r['n']} mean={r['mean_score']:.3f} min={r['min_score']:.3f} "
          f"flagged={r['flagged']} judges={r['judges_used']} "
          f"elapsed={r['elapsed_seconds']}s")
    print(f"wrote {RESULTS_PATH}")
