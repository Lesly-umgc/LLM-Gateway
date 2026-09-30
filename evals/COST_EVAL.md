# Cost-reduction eval

Measures what the gateway's cost levers actually save, in dollars, on a
realistic workload — run against real local models (Ollama), with public
per-token prices applied to the measured token counts.

## What it does

`run_cost_eval.py` simulates enterprise customer-support traffic:

- 300 requests (env `LNM_COST_N`), fixed seed `20260930`
- 40 base questions (26 simple factual, 14 complex analytical), each with
  3 phrasing variants (canonical + 2 paraphrases)
- Zipf-ish popularity: 45% exact repeats, 35% paraphrase A, 20% paraphrase B
- the same long Northwind Cloud system prompt on every request

Each request goes through the real gateway pipeline (`TestClient`) with
the tiered `RoutedOllamaBackend`:

1. **tiered routing** — a deterministic heuristic router sends simple
   questions to `llama3.2:3b` and complex ones to `llama3.1:8b`
   (`optimizer/router.py`)
2. **prefix caching** — the repeated system prompt is hashed; repeats are
   billed at 50% of the input rate (`optimizer/prefix_cache.py`)
3. **exact + semantic response cache** — repeat questions cost $0, no
   model call (already in the gateway)
4. **prompt compression** — already in the gateway path; value reported
   at the routed mix's average input rate

## The money math

"Direct" baseline: every request at the expensive tier, full input price,
no caching, no routing. Input tokens counted exactly per request;
completions use the measured mean from the gateway run, cross-checked
against `LNM_COST_VALIDATE_N` (default 40) **real direct `llama3.1:8b`
calls** whose mean $/request is reported in `direct_validation`.

Public price book (verified 2026-09-30; proxies, not Ollama charges):

| tier | input / 1M | output / 1M |
|---|---|---|
| expensive ("frontier", GPT-4o rates) | $2.50 | $10.00 |
| cheap ("small", GPT-4o-mini rates) | $0.15 | $0.60 |
| cached prefix input | 50% off the tier rate | — |
| exact/semantic cache hit | $0 (no model call) | $0 |

Lever contributions are marginal (each lever toggled off alone) and don't
sum exactly to total savings because the levers interact.

## Run it

```bash
# needs both models pulled: llama3.2:3b and llama3.1:8b
LNM_COST_N=300 venv/bin/python evals/run_cost_eval.py
```

Writes `evals/cost_results.json`. Small smoke test first:

```bash
LNM_COST_N=10 LNM_COST_VALIDATE_N=4 venv/bin/python evals/run_cost_eval.py
```

## Honest limitations

- The price book is public OpenAI rates applied to local Ollama token
  counts. Ollama itself charges nothing; the dollars are a proxy for
  "what this traffic would cost on a frontier/small model pair".
- Prefix "caching" is price-book modeling at OpenAI's prompt-caching
  convention (50% off repeated-prefix input). It does not prove Ollama
  skipped recomputation — the system prompt is still sent each call.
  Treat it as the saving a cache-capable serving provider would give.
- The direct baseline's completion tokens come from the gateway run's
  measured mean (mix of 3B/8B completions), not from real all-8B
  completions; the 40-call real-8B sample is the check on that assumption.
- The 33% figure is a target. This eval reports whatever it measures.
