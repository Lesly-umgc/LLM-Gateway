# LNM Gateway

I built this as a portfolio project: a centralized security gateway that sits
in front of any LLM and enforces prompt-injection defense, PII redaction,
hallucination monitoring, and token optimization — with every claim backed by
a real benchmark I ran myself.

The idea is simple. Every request goes through one choke point:

```
client -> auth + rate limit -> input rails -> cache -> compress -> model
       -> output rails -> faithfulness monitor -> audit log + metrics
```

Everything is OpenAI-compatible (`POST /v1/chat/completions`), so you can
point existing tooling at it.

## Quickstart (Docker)

This is the easiest way. It starts the gateway, Redis (shared semantic
cache), and Ollama (local model server) together:

```bash
git clone https://github.com/Lesly-umgc/LLM-Gateway.git
cd LLM-Gateway
docker compose up --build
```

Wait until all three containers report healthy (the gateway logs
`Uvicorn running on http://0.0.0.0:8000`). Then, in a second terminal:

```bash
curl localhost:8000/healthz
./demo/run_demo.sh
```

The first start pulls `llama3.2:3b` (~2 GB) into a Docker volume; later
starts reuse it. On Apple Silicon this works as-is under Docker Desktop.
Everything runs locally — no API keys or accounts needed.

## Quickstart (native, no Docker)

You need Python 3.12+ and [Ollama](https://ollama.com) installed.

```bash
ollama serve &                      # in one terminal
ollama pull llama3.2:3b             # one-time model download

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export LNM_API_KEYS=demo-key LNM_ADMIN_KEY=admin-secret
uvicorn gateway.app:app --port 8000
```

Then try it:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer demo-key" -H "Content-Type: application/json" \
  -d '{"model":"demo","messages":[{"role":"user","content":"What is the capital of France?"}]}' \
  | python3 -m json.tool
```

## Replay the demo

`demo/run_demo.sh` runs the exact 6 scenarios from the demo video, in order,
against a local gateway and prints a verdict for each:

```bash
docker compose up --build
./demo/run_demo.sh
```

1. Normal chat — full pipeline, 200 OK.
2. Prompt injection — blocked at the input rail (403, model never sees it).
3. PII in model output — redacted in place (`[REDACTED:EMAIL]` etc.).
4. Repeat question — served from cache, prompt tokens saved.
5. Ungrounded answer — flagged by the faithfulness monitor (score < 0.70).
6. `GET /admin/metrics` — live counters for all of the above.

Exact model wording can vary slightly run to run (sampling), but the
verdicts — blocked / redacted / cached / flagged — are deterministic. The
script exits non-zero if any scenario misbehaves.

## Talk to it yourself

The gateway is a live OpenAI-compatible API; the demo is just scripted
calls to it. Send your own prompts any time the stack is up:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer demo-key" -H "Content-Type: application/json" \
  -d '{"model":"demo","messages":[{"role":"user","content":"Explain caching in one sentence."}],"max_tokens":256}' \
  | python3 -m json.tool
```

The answer is under `choices[0].message.content`. `demo-key` is the default
client key (change it via `LNM_API_KEYS` in `docker-compose.yml`).

Things worth trying, each shows a different gateway feature:

- an injection attempt (`Ignore all previous instructions...`) — blocked
  with 403 before the model ever sees it;
- the same question twice — the second comes back with `"cached": true`
  and no backend call;
- a message containing an email address — redacted in the reply as
  `[REDACTED:EMAIL]`;
- `GET /admin/metrics` with `-H "Authorization: Bearer admin-secret"` —
  live counters for requests, blocks, cache hits, and tokens saved.

## How the pieces fit

- `gateway/` — FastAPI app, API-key auth, per-key rate limiting, JSONL audit
  log, `/admin/metrics`.
- `rails/` — NeMo Guardrails config (`config.yml` + `rails.co`) plus the
  actual rail pipeline, now running on the **real NeMo `LLMRails` runtime**
  (`rails/nemo_runtime.py`): a deterministic regex prefilter for known
  injection/jailbreak patterns (fast, no LLM) in front, NeMo's self-check
  input rail (LLM judge) for ambiguous cases, and PII redaction on the way
  out.
- `monitors/` — faithfulness via the **real `ragas` package**
  (`ragas.metrics.faithfulness`), judged by `llama3.1:8b` through Ollama.
  Sampled (~20% of requests) on a background thread so it never blocks
  responses. If the judge LLM is down it falls back to a deterministic
  heuristic and says so in the result.
- `optimizer/` — tiered model router (small vs large), prefix cache for
  repeated static prompt prefixes, exact + semantic cache with TTL, prompt
  compression, tiktoken accounting. Redis-backed when `LNM_REDIS_URL` is
  set (that's what compose uses), in-process otherwise.
- `backends/` — `ollama` (default, everything measured runs on it),
  `gemini` (free-tier fallback), `vllm` (GPU production path).
- `evals/` — benchmark harnesses: `run_benchmark.py` (original 30-attack /
  12-benign / 20-Q&A suite, `results.json`); `run_cost_eval.py` + `evals/COST_EVAL.md`
  (cost-lever methodology); `ollama_cost_eval_chunked.py` (the 100-request
  Colab cost eval, `cost_results_llama_ollama.json`);
  `eval_faithfulness.py` (real-ragas eval, `faithfulness_ragas.json`).
- `serving/` — vLLM + FP8 (INT8-class) serving recipe for a real GPU deploy.

More detail: `docs/ARCHITECTURE.md`.

## Measured results

Every number below comes from a real run against real local models. The
old `evals/results.json` (24/30 injection, 0.654 RAGAS-style faithfulness,
2.2% tokens) is superseded by the three workstream evals.

### Injection defense — 46/50 blocked (92.0%)

Run 2026-09-30 with the real NeMo `LLMRails` in the request path
(`rails/nemo_runtime.py`), backend=ollama. 50 fixed adversarial prompts
(the original 30 plus 20 new jailbreak styles), 12 benign prompts. The
prompt set was not tuned after seeing results.

Layered defense, measured per layer:

| layer | blocked | rate |
|---|---|---|
| regex prefilter alone | 23/50 | 46.0% |
| NeMo self-check rail alone | 41/50 | 82.0% |
| **combined** | **46/50** | **92.0%** |

- Original 30 attacks: 29/30 (96.7%). New 20: 17/20 (85.0%).
- Benign false positives: 0/12.
- Remaining misses, stated plainly: emotional-manipulation framing,
  fictional-world framing, novel/screenplay framing, ROT13-encoded
  instructions.
- Cost of the stronger layer: the NeMo judge path is ~2.1x slower than
  prefilter-only.

### Faithfulness — real `ragas`, mean 0.285

Run 2026-09-30 with the actual `ragas` package
(`ragas.metrics.faithfulness`), judge=`llama3.1:8b` via Ollama, n=20
grounded Q&A samples (`evals/faithfulness_ragas.json`).

- Mean faithfulness 0.285, min 0.000 — 20/20 samples below the 0.70 flag
  threshold.
- Honest read: this is not a win, it's a baseline. The 8B judge found
  most sampled answers unsupported by their grounding context. The value
  here is that the metric is real and wired into the request path — the
  answers (and the judge's harshness) are what need work.

### Cost — 7.81% measured savings

Run 2026-09-30/10-01 on Colab (Ollama, `llama3.2:3b` + `llama3.1:8b`),
n=100 requests in 6 chunks with Ollama restarts between chunks (the T4
runner hung after ~40 sequential 8B calls). Notebook: "LNM Gateway - Llama
Cost Eval (Ollama)". Artifact: `evals/cost_results_llama_ollama.json`.

- Baseline: 100/100 requests on 8B → $0.002003.
- Gateway: 100/100 tier-routed (3B for simple, 8B for complex) with the
  repeated system prompt billed at the 50%-off cached-prefix rate →
  $0.001846.
- Saved $0.000156 → **7.81%**.
- Dollars are measured token counts × a documented public price book
  (3B: $0.10/1M in+out; 8B: $0.20/1M in+out; cached prefix input 50%
  off). Ollama itself charges nothing — the dollars proxy "what this
  traffic would cost on a small/frontier model pair".
- Honest read: the 33% cost-reduction target from the project aim is
  **not** met. 7.81% is the measured number on this workload; the 33%
  stays a design target until a production workload proves otherwise.

## What was NOT run or measured here

- **vLLM / INT8 on GPU.** The client (`backends/vllm.py`) is tested against
  a mock OpenAI-compatible server, and `serving/vllm_int8_example.yaml`
  documents the deploy recipe — but there is no GPU in this environment,
  so no throughput, latency, or quality numbers are claimed for it.
  (Ollama serves 4-bit quants, not INT8 — INT8 refers to the vLLM config
  only.)
- **The 33% figure** in the project aim is a design target, not a result.
- **Semantic-cache savings at scale.** A 90-request paraphrase stress test
  exists (`evals/ollama_cost_eval_semantic.py`) but the Colab CPU run was
  parked after repeated model-server instability — no result is claimed.

## Config

Copy `.env.example` to `.env` for the full list. The ones you'll touch most:

| Var | Default | What |
|---|---|---|
| `LNM_BACKEND` | `ollama` | `ollama` \| `gemini` \| `vllm` |
| `OLLAMA_MODEL` | `llama3.2:3b` | model for the Ollama backend |
| `LNM_REDIS_URL` | (unset) | set to use the shared Redis cache |
| `LNM_FAITHFULNESS_SAMPLE_RATE` | `0.20` | fraction of grounded requests scored |
| `LNM_FAITHFULNESS_THRESHOLD` | `0.70` | below this, the answer is flagged |
| `LNM_JUDGE_ENABLED` | `1` | `0` = heuristic-only (what unit tests use) |
