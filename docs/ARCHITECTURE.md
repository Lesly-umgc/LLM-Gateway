# LNM Gateway — Architecture

A centralized security gateway that every LLM request passes through.
OpenAI-compatible API, pluggable model backends, real measured evals.

## Request flow

```
Client app
  │  POST /v1/chat/completions   (Bearer API key)
  ▼
┌─ Gateway (FastAPI) ─────────────────────────────────────────────┐
│ 1. Auth: API key → 401 if unknown                               │
│ 2. Rate limit: per-key token bucket → 429 if exceeded            │
│ 3. Cache check: exact sha256 → semantic (MiniLM, cos ≥ 0.95)    │
│    HIT → return stored answer, count tokens saved, audit, done  │
│ 4. Input rails (NeMo Guardrails flows, rails/rails.co):         │
│      heuristic prefilter → BLOCK / ALLOW / JUDGE                 │
│      JUDGE → LLM-judge (Ollama; Gemini fallback)                │
│    BLOCK → 403 + audit                                          │
│ 5. Prompt compression: whitespace collapse, sentence dedup,     │
│    token-budget truncation → count tokens saved                  │
│ 6. Backend: Ollama (demo) | Gemini (free tier) | vLLM (prod GPU) │
│ 7. Output rails: PII redaction (in place) → policy LLM-judge    │
│    BLOCK → 403 + audit                                          │
│ 8. Faithfulness monitor (async, sampled ~20%): RAGAS protocol   │
│    extract claims → verify each vs context → score → flag<0.70   │
│ 9. Cache store + audit log (JSONL) + metrics                    │
└─────────────────────────────────────────────────────────────────┘
  │  OpenAI-style JSON response (+ lnm.cached flag)
  ▼
GET /admin/metrics → counters, token accounting, est. USD saved
```

## Components

| Dir | What | Key design choice |
|---|---|---|
| `gateway/` | FastAPI app, auth, rate limiting, audit, metrics | Sync endpoints; monitor runs on a daemon worker thread so judging never blocks responses |
| `rails/` | NeMo Guardrails config (`config.yml`, `rails.co`) + Python rail pipeline | Rails.co declares the flows; `engine.py` executes them: deterministic heuristic first (fast, no LLM), LLM judge only for ambiguous cases |
| `rails/prefilter.py` | Regex pre-filter of known jailbreak/injection techniques | Curated from real-world attacks; zero-LLM, ~µs |
| `rails/pii.py` | PII redaction (email, phone, SSN, credit card, API keys) | Redacts in place, records kinds; never blocks on PII alone |
| `rails/actions.py` | LLM-judge prompts (injection / policy) | Via `common/llm.py`: Ollama by default, Gemini fallback |
| `monitors/` | RAGAS faithfulness protocol | Implements the RAGAS faithfulness algorithm exactly (claim extraction → per-claim verification → supported/total) without importing the heavy `ragas` package at serve time, keeping the gateway light |
| `optimizer/` | Semantic cache + prompt compression + tiktoken accounting | Cache degrades to exact-match if embeddings unavailable; all savings measured in tokens, converted to USD with a documented blended rate |
| `backends/` | `ollama.py` (primary demo), `gemini.py` (free tier), `vllm.py` (GPU prod) | Common `Backend` interface; switching is `LNM_BACKEND=` |
| `evals/` | `run_benchmark.py` → `results.json` | The ONLY source of numbers in the README |

## Deployment: docker compose

`docker-compose.yml` runs the whole stack as three services:

- **gateway** — this repo's FastAPI app (rails, monitor, optimizer).
- **redis** — shared semantic-cache backend (`LNM_REDIS_URL=redis://redis:6379/0`).
- **ollama** — local model server; `docker/ollama-entrypoint.sh` pulls
  `${OLLAMA_MODEL:-llama3.2:3b}` on first start into the `ollama-models`
  volume so restarts are instant.

`ollama/ollama` publishes arm64 images, so the stack works on Apple Silicon
via Docker Desktop. The gateway falls back to its in-process cache if Redis
is unreachable, so a bad `LNM_REDIS_URL` degrades instead of crashing.

## Backends

- **Ollama** (default): local, quantized models, CPU-friendly. `ollama serve`
  + `ollama pull llama3.2:3b`. What the demo and all measured numbers use.
- **Gemini**: free-tier API via connector; used as judge fallback and optional backend.
- **vLLM**: OpenAI-compatible client for GPU production serving of
  FP8-quantized models. Code-complete and mock-tested; **never executed here
  (no GPU)** — see `serving/vllm_int8_example.yaml` for the deploy recipe.

## What "RAGAS" means here

RAGAS's faithfulness metric is: split the answer into atomic claims, check
each claim against the retrieved context, score = supported / total.
`monitors/faithfulness.py` implements exactly this protocol with an LLM judge.
We do not import the `ragas` package at serve time (it drags in langchain,
datasets, and torch overhead the gateway doesn't need); the algorithm and
the reported metric are the same. This is stated in the README too.

## Honesty ledger

- `evals/results.json` is the single source of measured numbers.
- The "33% expenditure reduction" in the project aim is a **design target**;
  the README reports the measured token-savings % from the eval suite.
- vLLM/INT8: config + client are real and tested against a mock; no
  throughput/latency numbers are claimed because no GPU run happened here.
- `LNM_COST_PER_1K_USD` is a documented blended estimate used ONLY to turn
  measured token savings into an indicative USD figure.
