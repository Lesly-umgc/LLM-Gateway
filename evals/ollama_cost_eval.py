"""
Colab cost eval via Ollama: measures token-based savings from tiered routing + prefix caching.

Uses Ollama with llama3.2:3b (small) and llama3.1:8b (large) — the exact
models from the LNM Gateway project.

Compares:
  - Baseline: all requests to 8B model.
  - Gateway: simple queries to 3B, complex to 8B, with prefix caching.

Cost model (public price book):
  - 8B: $0.20 / 1M input tokens, $0.20 / 1M output tokens
  - 3B: $0.10 / 1M input tokens, $0.10 / 1M output tokens
  - Cached prefix: 50% discount on input.
"""

import time
import random
import json
import urllib.request

# Price book (USD per 1M tokens)
PRICE_8B_IN = 0.20 / 1e6
PRICE_8B_OUT = 0.20 / 1e6
PRICE_3B_IN = 0.10 / 1e6
PRICE_3B_OUT = 0.10 / 1e6
CACHED_DISCOUNT = 0.5

SMALL_MODEL = "llama3.2:3b"
LARGE_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

QUESTIONS = [
    (True, "What is the capital of France?"),
    (True, "What is 2 + 2?"),
    (True, "Name three primary colors."),
    (True, "What year did World War II end?"),
    (True, "What is the largest planet?"),
    (False, "Explain the theory of relativity in simple terms."),
    (False, "Write a short essay on the causes of the French Revolution."),
    (False, "Describe the process of photosynthesis in detail."),
    (False, "What are the implications of quantum computing for cryptography?"),
    (False, "Analyze the themes in Shakespeare's Hamlet."),
]


def build_workload(n, seed=20260930):
    rng = random.Random(seed)
    workload = []
    for _ in range(n):
        is_simple = rng.random() < 0.45
        candidates = [q for q in QUESTIONS if q[0] == is_simple]
        _, prompt = rng.choice(candidates)
        workload.append((is_simple, prompt))
    return workload


def ollama_generate(model, prompt, max_tokens=96):
    """Call Ollama API, return (response_text, prompt_tokens, completion_tokens)."""
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    return (
        result.get("response", ""),
        result.get("prompt_eval_count", 0),
        result.get("eval_count", 0),
    )


def count_tokens(text):
    return max(1, len(text) // 4)


def run_eval(n=100):
    print("Checking Ollama...")
    # Verify Ollama is running
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            tags = json.loads(resp.read())
            models = [m["name"] for m in tags.get("models", [])]
            print(f"Available models: {models}")
    except Exception as e:
        print(f"Ollama not reachable: {e}")
        return None

    workload = build_workload(n)
    print(f"Workload: {n} requests")

    # Baseline: all to 8B
    print("\nRunning baseline (all 8B)...")
    baseline_cost = 0.0
    for i, (is_simple, prompt) in enumerate(workload):
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        _, in_tok, out_tok = ollama_generate(LARGE_MODEL, full_prompt)
        baseline_cost += in_tok * PRICE_8B_IN + out_tok * PRICE_8B_OUT
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n}")

    # Gateway: routed + prefix caching
    print("\nRunning gateway (routed + prefix cache)...")
    gateway_cost = 0.0
    prefix_cached = False
    sys_tokens = count_tokens(SYSTEM_PROMPT)
    for i, (is_simple, prompt) in enumerate(workload):
        model = SMALL_MODEL if is_simple else LARGE_MODEL
        in_rate = PRICE_3B_IN if is_simple else PRICE_8B_IN
        out_rate = PRICE_3B_OUT if is_simple else PRICE_8B_OUT

        if prefix_cached:
            cached_cost = sys_tokens * in_rate * (1 - CACHED_DISCOUNT)
        else:
            cached_cost = sys_tokens * in_rate
            prefix_cached = True

        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        _, in_tok, out_tok = ollama_generate(model, full_prompt)

        uncached_in = max(0, in_tok - sys_tokens)
        gateway_cost += cached_cost + uncached_in * in_rate + out_tok * out_rate

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n}")

    saved = baseline_cost - gateway_cost
    pct = (saved / baseline_cost * 100) if baseline_cost > 0 else 0

    results = {
        "n": n,
        "baseline_cost_usd": round(baseline_cost, 6),
        "gateway_cost_usd": round(gateway_cost, 6),
        "saved_usd": round(saved, 6),
        "savings_pct": round(pct, 2),
        "models": {"small": SMALL_MODEL, "large": LARGE_MODEL},
        "backend": "ollama",
        "price_book": {
            "8b_in_per_1m": PRICE_8B_IN * 1e6,
            "8b_out_per_1m": PRICE_8B_OUT * 1e6,
            "3b_in_per_1m": PRICE_3B_IN * 1e6,
            "3b_out_per_1m": PRICE_3B_OUT * 1e6,
            "cached_prefix_discount": CACHED_DISCOUNT,
        },
    }

    print("\n" + "="*50)
    print(f"Baseline (all 8B): ${baseline_cost:.4f}")
    print(f"Gateway (routed):  ${gateway_cost:.4f}")
    print(f"Saved: ${saved:.4f} ({pct:.1f}%)")
    print("="*50)

    with open("/tmp/ollama_cost_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/ollama_cost_results.json")
    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_eval(n=n)
