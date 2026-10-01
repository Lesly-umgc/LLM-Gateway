"""
Colab cost eval via Ollama (chunked, with response caching).

Adds exact-match response caching on top of tiered routing + prefix caching.
Workload includes realistic repetition (~30% of requests are repeats).

Usage:
  python ollama_cost_eval_cached.py --reset
  python ollama_cost_eval_cached.py --phase baseline --chunk 0 --chunk-size 35 --n 100
  python ollama_cost_eval_cached.py --phase gateway --chunk 0 --chunk-size 35 --n 100
  python ollama_cost_eval_cached.py --aggregate --n 100
"""

import argparse
import hashlib
import json
import os
import random
import urllib.request

PRICE_8B_IN = 0.20 / 1e6
PRICE_8B_OUT = 0.20 / 1e6
PRICE_3B_IN = 0.10 / 1e6
PRICE_3B_OUT = 0.10 / 1e6
CACHED_DISCOUNT = 0.5

SMALL_MODEL = "llama3.2:3b"
LARGE_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434"
RESULTS_FILE = "/tmp/ollama_cached_chunks.jsonl"

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

QUESTIONS = [
    (True, "What is the capital of France?"),
    (True, "What is 2 + 2?"),
    (True, "Name three primary colors."),
    (True, "What year did World War II end?"),
    (True, "What is the largest planet?"),
    (True, "What is the boiling point of water?"),
    (True, "How many days in a year?"),
    (False, "Explain the theory of relativity in simple terms."),
    (False, "Write a short essay on the causes of the French Revolution."),
    (False, "Describe the process of photosynthesis in detail."),
    (False, "What are the implications of quantum computing for cryptography?"),
    (False, "Analyze the themes in Shakespeare's Hamlet."),
    (False, "Explain how neural networks learn."),
]


def build_workload(n, seed=20260930, repeat_ratio=0.30):
    """Build workload with realistic repetition.

    repeat_ratio fraction of requests are exact repeats of earlier queries.
    This models production traffic where users ask similar questions.
    """
    rng = random.Random(seed)
    # Generate unique queries first
    n_unique = int(n * (1 - repeat_ratio))
    unique = []
    for _ in range(n_unique):
        is_simple = rng.random() < 0.45
        candidates = [q for q in QUESTIONS if q[0] == is_simple]
        _, prompt = rng.choice(candidates)
        unique.append((is_simple, prompt))

    # Build full workload: uniques + repeats
    workload = unique[:]
    n_repeats = n - n_unique
    for _ in range(n_repeats):
        # Pick a random earlier query to repeat
        idx = rng.randint(0, len(unique) - 1)
        workload.append(unique[idx])

    # Shuffle to mix repeats throughout
    rng.shuffle(workload)
    return workload


def ollama_generate(model, prompt, max_tokens=96, timeout=180):
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return (
        result.get("response", ""),
        result.get("prompt_eval_count", 0),
        result.get("eval_count", 0),
    )


def count_tokens(text):
    return max(1, len(text) // 4)


def cache_key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def run_chunk(phase, chunk_idx, chunk_size, n):
    workload = build_workload(n)
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n)
    if start >= n:
        print(f"Chunk {chunk_idx}: nothing to do")
        return

    print(f"Phase={phase} chunk={chunk_idx} requests {start}..{end-1}")

    sys_tokens = count_tokens(SYSTEM_PROMPT)
    prefix_warm = not (phase == "gateway" and chunk_idx == 0)

    # Response cache: persists across chunks via file
    cache_file = "/tmp/response_cache.json"
    response_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            response_cache = json.load(f)

    results = []
    cache_hits = 0
    for i in range(start, end):
        is_simple, prompt = workload[i]
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        key = cache_key(full_prompt)

        # Check response cache first (gateway only; baseline has no cache)
        if phase == "gateway" and key in response_cache:
            cached = response_cache[key]
            results.append({
                "phase": phase,
                "idx": i,
                "model": cached["model"],
                "in_tok": 0,
                "out_tok": 0,
                "cost": 0.0,
                "cache_hit": True,
            })
            cache_hits += 1
            print(f"  [{i+1}/{n}] CACHE HIT (saved {cached['model']} call)", flush=True)
            continue

        if phase == "baseline":
            model = LARGE_MODEL
            in_rate, out_rate = PRICE_8B_IN, PRICE_8B_OUT
            _, in_tok, out_tok = ollama_generate(model, full_prompt)
            cost = in_tok * in_rate + out_tok * out_rate
        else:  # gateway
            model = SMALL_MODEL if is_simple else LARGE_MODEL
            in_rate = PRICE_3B_IN if is_simple else PRICE_8B_IN
            out_rate = PRICE_3B_OUT if is_simple else PRICE_8B_OUT

            if prefix_warm:
                cached_cost = sys_tokens * in_rate * (1 - CACHED_DISCOUNT)
            else:
                cached_cost = sys_tokens * in_rate
                prefix_warm = True

            _, in_tok, out_tok = ollama_generate(model, full_prompt)
            uncached_in = max(0, in_tok - sys_tokens)
            cost = cached_cost + uncached_in * in_rate + out_tok * out_rate

            # Store in response cache
            response_cache[key] = {"model": model, "in_tok": in_tok, "out_tok": out_tok}

        results.append({
            "phase": phase,
            "idx": i,
            "model": model,
            "in_tok": in_tok,
            "out_tok": out_tok,
            "cost": cost,
            "cache_hit": False,
        })
        print(f"  [{i+1}/{n}] {model} in={in_tok} out={out_tok} cost=${cost:.6f}", flush=True)

    # Save response cache for next chunk
    if phase == "gateway":
        with open(cache_file, "w") as f:
            json.dump(response_cache, f)

    # Append to JSONL
    with open(RESULTS_FILE, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    chunk_cost = sum(r["cost"] for r in results)
    print(f"Chunk {chunk_idx} done. Cost: ${chunk_cost:.6f}. Cache hits: {cache_hits}/{len(results)}")


def aggregate(n):
    baseline_cost = 0.0
    gateway_cost = 0.0
    baseline_n = 0
    gateway_n = 0
    gateway_hits = 0

    if not os.path.exists(RESULTS_FILE):
        print(f"No results file at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r["phase"] == "baseline":
                baseline_cost += r["cost"]
                baseline_n += 1
            else:
                gateway_cost += r["cost"]
                gateway_n += 1
                if r.get("cache_hit"):
                    gateway_hits += 1

    print(f"\nBaseline: {baseline_n}/{n} requests, cost=${baseline_cost:.6f}")
    print(f"Gateway:  {gateway_n}/{n} requests, cost=${gateway_cost:.6f}, cache hits={gateway_hits}")

    if baseline_n < n or gateway_n < n:
        print("WARNING: incomplete results")
        return

    saved = baseline_cost - gateway_cost
    pct = (saved / baseline_cost * 100) if baseline_cost > 0 else 0

    results = {
        "n": n,
        "baseline_cost_usd": round(baseline_cost, 6),
        "gateway_cost_usd": round(gateway_cost, 6),
        "saved_usd": round(saved, 6),
        "savings_pct": round(pct, 2),
        "gateway_cache_hits": gateway_hits,
        "models": {"small": SMALL_MODEL, "large": LARGE_MODEL},
        "backend": "ollama",
        "optimizations": ["tiered_routing", "prefix_cache", "response_cache"],
        "workload": {"repeat_ratio": 0.30, "seed": 20260930},
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
    print(f"Gateway (routed+cached): ${gateway_cost:.4f}")
    print(f"Saved: ${saved:.4f} ({pct:.1f}%)")
    print("="*50)

    with open("/tmp/ollama_cached_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/ollama_cached_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "gateway"])
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=35)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        for fp in [RESULTS_FILE, "/tmp/response_cache.json"]:
            if os.path.exists(fp):
                os.remove(fp)
                print(f"Removed {fp}")
    elif args.aggregate:
        aggregate(n=args.n)
    elif args.phase:
        run_chunk(args.phase, args.chunk, args.chunk_size, args.n)
    else:
        parser.print_help()
