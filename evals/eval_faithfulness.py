"""Focused faithfulness eval: real `ragas` metric on the 20-sample grounded set.

Reuses the GROUNDED questions and the live gateway pipeline from
run_benchmark, but scores each answer with the real ragas Faithfulness
metric (Ollama judge) instead of the old RAGAS-style heuristic. Writes
evals/faithfulness_ragas.json.

Costs: one gateway answer per sample (3B) plus ragas judging
(statement extraction + per-statement NLI on the judge model).

Env:
  LNM_JUDGE_MODEL  judge model, default llama3.1:8b
  EVAL_MAX_TOKENS  default 160
"""

from __future__ import annotations

import json
import os
import sys
import time

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
RESULTS_PATH = os.path.join(os.path.dirname(__file__),
                            "faithfulness_ragas.json")


def run() -> dict:
    started = time.time()
    config = GatewayConfig(api_keys={"test-key"}, admin_key="admin-key",
                           backend_name="ollama",
                           audit_path="evals/audit_faithfulness.jsonl",
                           rate_limit_per_min=1000)
    client = TestClient(create_app(config))
    assert client.get("/healthz").json()["status"] == "ok"

    scores, flagged, detail, judges = [], 0, [], {}
    for i, (ctx, q) in enumerate(GROUNDED, 1):
        res = _post(client, q, context=ctx)
        if res["status"] != 200:
            detail.append({"question": q[:60], "status": res["status"],
                           "score": None})
            print(f"[{i:2d}/20] HTTP {res['status']} :: {q[:50]}", flush=True)
            continue
        answer = res["payload"]["choices"][0]["message"]["content"]
        t0 = time.time()
        s = fm.faithfulness_score(answer, ctx, question=q)
        dt = time.time() - t0
        judges[s["judge"]] = judges.get(s["judge"], 0) + 1
        scores.append(s["score"])
        flagged += 1 if s["flagged"] else 0
        detail.append({"question": q[:60], "score": round(s["score"], 3),
                       "claims": s["claims"], "supported": s["supported"],
                       "flagged": s["flagged"], "judge": s["judge"],
                       "score_seconds": round(dt, 1)})
        print(f"[{i:2d}/20] score={s['score']:.3f} judge={s['judge']}"
              f" {dt:5.0f}s :: {q[:50]}", flush=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge": "ragas",
        "judge_model": os.environ.get("LNM_JUDGE_MODEL", "llama3.1:8b"),
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


if __name__ == "__main__":
    r = run()
    print(f"n={r['n']} mean={r['mean_score']:.3f} min={r['min_score']:.3f} "
          f"flagged={r['flagged']} judges={r['judges_used']} "
          f"elapsed={r['elapsed_seconds']}s")
    print(f"wrote {RESULTS_PATH}")
