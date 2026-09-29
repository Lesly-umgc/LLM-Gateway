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
docker compose up --build
./demo/run_demo.sh
```

The first start pulls `llama3.2:3b` (~2 GB) into a Docker volume; later
starts reuse it. On Apple Silicon this works as-is under Docker Desktop.

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

## How the pieces fit

- `gateway/` — FastAPI app, API-key auth, per-key rate limiting, JSONL audit
  log, `/admin/metrics`.
- `rails/` — NeMo Guardrails config (`config.yml` + `rails.co`) plus the
  actual rail pipeline: a deterministic regex prefilter for known
  injection/jailbreak patterns (fast, no LLM), an LLM judge for ambiguous
  cases, and PII redaction on the way out.
- `monitors/` — RAGAS-style faithfulness: split the answer into claims,
  check each against the grounding context, score = supported / total.
  Sampled (~20% of requests) on a background thread so it never blocks
  responses. If the judge LLM is down it falls back to a deterministic
  heuristic and says so in the result.
- `optimizer/` — exact + semantic (MiniLM) cache with TTL, prompt
  compression, tiktoken accounting. Redis-backed when `LNM_REDIS_URL` is
  set (that's what compose uses), in-process otherwise.
- `backends/` — `ollama` (default, everything measured runs on it),
  `gemini` (free-tier fallback), `vllm` (GPU production path).
- `evals/` — `run_benchmark.py`: 30 injection attacks, 12 benign
  false-positive checks, 20 grounded Q&A, cache/token tests. Writes
  `results.json` — the only source of numbers in this README.
- `serving/` — vLLM + FP8 (INT8-class) serving recipe for a real GPU deploy.

More detail: `docs/ARCHITECTURE.md`.

## Measured results

Pending — I run the eval suite with:

```bash
python evals/run_benchmark.py
```

and paste the real numbers here (attack block rate, false-positive rate,
cache hit rate, tokens saved %). I won't put the 33% cost-reduction target
from the project aim in this section until it's measured.

## What was NOT run or measured here

- **vLLM / INT8 on GPU.** The client (`backends/vllm.py`) is tested against
  a mock OpenAI-compatible server, and `serving/vllm_int8_example.yaml`
  documents the deploy recipe — but there is no GPU in this environment,
  so no throughput, latency, or quality numbers are claimed for it.
- **Cost savings in dollars.** `est_cost_saved_usd` in metrics is measured
  tokens × a documented blended per-token rate — indicative, not a bill.
- **The 33% figure** in the project aim is a design target, not a result.

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
