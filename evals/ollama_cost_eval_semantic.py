"""
Colab cost eval via Ollama with SEMANTIC caching.

Semantic cache: uses embeddings to serve responses for semantically similar
(not just identical) queries. This is a standard production technique.

Methodology (honest):
- Diverse pool of 30 base questions across topics, each with 2 paraphrases.
- 90 requests: 30 unique base questions + 60 paraphrases, shuffled.
- Zero exact duplicates: every request string appears exactly once.
- Paraphrases are genuinely different wordings, not exact repeats.
- Similarity threshold: 0.85 cosine similarity.
- Embedding model: nomic-embed-text via Ollama (local, timed for overhead).
- Each cache hit is labeled true/false positive by matching base-question id.

Usage:
  python ollama_cost_eval_semantic.py --reset
  python ollama_cost_eval_semantic.py --phase baseline --chunk 0 --chunk-size 30 --n 90
  python ollama_cost_eval_semantic.py --phase gateway --chunk 0 --chunk-size 30 --n 90
  python ollama_cost_eval_semantic.py --aggregate --n 90
"""

import argparse
import json
import math
import os
import random
import time
import urllib.request

EMBED_TIMES_FILE = "/tmp/ollama_semantic_embed_times.jsonl"
# Set by run_chunk so timing lines can be traced back to a phase/chunk
# (lets a retried chunk drop its own timing lines).
_EMBED_TAG = ""

PRICE_8B_IN = 0.20 / 1e6
PRICE_8B_OUT = 0.20 / 1e6
PRICE_3B_IN = 0.10 / 1e6
PRICE_3B_OUT = 0.10 / 1e6
CACHED_DISCOUNT = 0.5

SMALL_MODEL = "llama3.2:3b"
LARGE_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"
RESULTS_FILE = "/tmp/ollama_semantic_chunks.jsonl"
SEMANTIC_CACHE_FILE = "/tmp/semantic_cache.json"
SIMILARITY_THRESHOLD = 0.85

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

# (is_simple, question, [paraphrases])
QUESTIONS = [
    (True, "What is the capital of France?", ["Which city serves as France's capital?", "France's capital is which city?"]),
    (True, "What is 2 + 2?", ["Calculate 2 plus 2.", "What do you get when you add 2 and 2?"]),
    (True, "Name three primary colors.", ["List the three primary colors.", "What are the primary colors?"]),
    (True, "What year did World War II end?", ["When did WWII conclude?", "In which year did the Second World War end?"]),
    (True, "What is the largest planet?", ["Which planet is the biggest?", "Name the largest planet in our solar system."]),
    (True, "What is the boiling point of water?", ["At what temperature does water boil?", "Water boils at what temperature?"]),
    (True, "How many days in a year?", ["How many days does a year have?", "What's the number of days in a year?"]),
    (True, "What is the chemical symbol for gold?", ["Gold's chemical symbol is what?", "Which symbol represents gold?"]),
    (True, "How many continents are there?", ["What is the number of continents?", "Count the continents for me."]),
    (True, "What is the speed of light?", ["How fast is light?", "What speed does light travel at?"]),
    (False, "Explain the theory of relativity in simple terms.", ["Can you simplify Einstein's relativity?", "Describe relativity for a beginner."]),
    (False, "Write a short essay on the causes of the French Revolution.", ["What caused the French Revolution? Write briefly.", "Summarize the origins of the French Revolution."]),
    (False, "Describe the process of photosynthesis in detail.", ["How does photosynthesis work? Explain thoroughly.", "Give a detailed account of photosynthesis."]),
    (False, "What are the implications of quantum computing for cryptography?", ["How will quantum computers affect encryption?", "Discuss quantum computing's impact on cryptography."]),
    (False, "Analyze the themes in Shakespeare's Hamlet.", ["What are the main themes of Hamlet?", "Discuss the central themes in Hamlet."]),
    (False, "Explain how neural networks learn.", ["How do neural networks train?", "Describe the learning process of neural networks."]),
    (False, "What causes economic inflation?", ["Explain the drivers of inflation.", "Why does inflation happen in economies?"]),
    (False, "Describe the water cycle.", ["How does the water cycle function?", "Explain the stages of the water cycle."]),
    (False, "What is the difference between machine learning and deep learning?", ["Compare ML and deep learning.", "How does deep learning differ from machine learning?"]),
    (False, "Explain the concept of supply and demand.", ["What is supply and demand in economics?", "Describe how supply and demand interact."]),
    (True, "What is the tallest mountain?", ["Which mountain is the tallest?", "Name the world's highest mountain."]),
    (True, "How many legs does a spider have?", ["Count a spider's legs.", "What's the leg count on spiders?"]),
    (True, "What gas do plants absorb?", ["Which gas do plants take in?", "Plants absorb what gas from air?"]),
    (False, "Why is the sky blue?", ["Explain why the sky appears blue.", "What makes the sky look blue?"]),
    (False, "How does GPS work?", ["Explain the Global Positioning System.", "Describe how GPS determines location."]),
    (True, "What is H2O?", ["What does H2O represent?", "H2O is the formula for what?"]),
    (False, "What is blockchain technology?", ["Explain how blockchain works.", "Describe the basics of blockchain."]),
    (True, "How many sides does a triangle have?", ["Count the sides of a triangle.", "A triangle has how many sides?"]),
    (False, "What is the greenhouse effect?", ["Explain the greenhouse effect.", "How does the greenhouse effect work?"]),
    (True, "What planet is known as the Red Planet?", ["Which planet is called the Red Planet?", "The Red Planet refers to which planet?"]),
]


def build_workload(n, seed=20260930):
    """Build an honest semantic-cache workload.

    Every base question in QUESTIONS appears exactly once as a "unique"
    request; the rest of the workload is paraphrases, each used at most once.
    No exact duplicates anywhere in the workload.

    With the default 30-question pool (2 paraphrases each), n=90 gives
    30 unique + 60 paraphrases (66.7% paraphrase traffic).
    """
    rng = random.Random(seed)
    pool = list(enumerate(QUESTIONS))
    rng.shuffle(pool)

    # All base questions, each exactly once. Item: (is_simple, text, base_idx)
    unique = [(is_simple, text, idx) for idx, (is_simple, text, _) in pool]

    # Every paraphrase exactly once, tagged with its source question id
    paraphrases = []
    for idx, (is_simple, text, paras) in pool:
        for p in paras:
            paraphrases.append((is_simple, p, idx))
    rng.shuffle(paraphrases)

    if n > len(unique) + len(paraphrases):
        raise ValueError(
            f"n={n} exceeds pool capacity "
            f"({len(unique)} unique + {len(paraphrases)} paraphrases); "
            "lower n or add questions/paraphrases"
        )

    workload = unique[:]
    workload.extend(paraphrases[: n - len(unique)])
    rng.shuffle(workload)
    return workload


def get_embedding(text, timeout=60, retries=5):
    """Get embedding vector from Ollama. Times the call for overhead stats."""
    t0 = time.time()
    try:
        data = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        last_err = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    result = json.loads(resp.read())
                return result.get("embedding", [])
            except Exception as e:
                last_err = e
                print(f"    get_embedding attempt {attempt + 1}/{retries} failed: "
                      f"{type(e).__name__}; retrying in 30s", flush=True)
                time.sleep(30)
        raise last_err
    finally:
        with open(EMBED_TIMES_FILE, "a") as f:
            f.write(json.dumps({"tag": _EMBED_TAG, "seconds": time.time() - t0}) + "\n")


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def ollama_generate(model, prompt, max_tokens=96, timeout=180, retries=5):
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
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
            return (
                result.get("prompt_eval_count", 0),
                result.get("eval_count", 0),
            )
        except Exception as e:
            last_err = e
            print(f"    ollama_generate attempt {attempt + 1}/{retries} failed: "
                  f"{type(e).__name__}; retrying in 30s", flush=True)
            time.sleep(30)
    raise last_err


def count_tokens(text):
    return max(1, len(text) // 4)


def load_semantic_cache():
    if os.path.exists(SEMANTIC_CACHE_FILE):
        with open(SEMANTIC_CACHE_FILE) as f:
            return json.load(f)
    return []


def save_semantic_cache(cache):
    with open(SEMANTIC_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def find_semantic_hit(query_embedding, cache):
    """Find cached entry with similarity above threshold. Returns (entry, similarity) or (None, 0)."""
    best = None
    best_sim = 0.0
    for entry in cache:
        sim = cosine_similarity(query_embedding, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best = entry
    if best_sim >= SIMILARITY_THRESHOLD:
        return best, best_sim
    return None, best_sim


def run_chunk(phase, chunk_idx, chunk_size, n):
    workload = build_workload(n)
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n)
    if start >= n:
        print(f"Chunk {chunk_idx}: nothing to do")
        return

    global _EMBED_TAG
    _EMBED_TAG = f"{phase}-{chunk_idx}"

    print(f"Phase={phase} chunk={chunk_idx} requests {start}..{end-1}")

    sys_tokens = count_tokens(SYSTEM_PROMPT)
    prefix_warm = not (phase == "gateway" and chunk_idx == 0)
    semantic_cache = load_semantic_cache() if phase == "gateway" else []

    results = []
    semantic_hits = 0
    semantic_tp = 0
    semantic_fp = 0
    for i in range(start, end):
        is_simple, prompt, base_idx = workload[i]
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        query_emb = None

        # Semantic cache check (gateway only)
        if phase == "gateway":
            try:
                query_emb = get_embedding(prompt)
                hit, sim = find_semantic_hit(query_emb, semantic_cache)
                if hit:
                    is_tp = hit["base_idx"] == base_idx
                    results.append({
                        "phase": phase, "idx": i, "model": hit["model"],
                        "in_tok": 0, "out_tok": 0, "cost": 0.0,
                        "cache_hit": True, "similarity": round(sim, 3),
                        "hit_true_positive": is_tp,
                    })
                    semantic_hits += 1
                    if is_tp:
                        semantic_tp += 1
                    else:
                        semantic_fp += 1
                    print(f"  [{i+1}/{n}] SEMANTIC HIT (sim={sim:.3f}, "
                          f"{'TP' if is_tp else 'FP'}, saved {hit['model']})", flush=True)
                    continue
            except Exception as e:
                print(f"  [{i+1}/{n}] Embedding failed: {e}, proceeding without cache", flush=True)

        if phase == "baseline":
            model = LARGE_MODEL
            in_rate, out_rate = PRICE_8B_IN, PRICE_8B_OUT
            in_tok, out_tok = ollama_generate(model, full_prompt)
            cost = in_tok * in_rate + out_tok * out_rate
            query_emb = None
        else:
            model = SMALL_MODEL if is_simple else LARGE_MODEL
            in_rate = PRICE_3B_IN if is_simple else PRICE_8B_IN
            out_rate = PRICE_3B_OUT if is_simple else PRICE_8B_OUT

            if prefix_warm:
                cached_cost = sys_tokens * in_rate * (1 - CACHED_DISCOUNT)
            else:
                cached_cost = sys_tokens * in_rate
                prefix_warm = True

            in_tok, out_tok = ollama_generate(model, full_prompt)
            uncached_in = max(0, in_tok - sys_tokens)
            cost = cached_cost + uncached_in * in_rate + out_tok * out_rate

            # Add to semantic cache
            try:
                if query_emb is None:
                    query_emb = get_embedding(prompt)
                semantic_cache.append({
                    "embedding": query_emb,
                    "model": model,
                    "prompt": prompt,
                    "base_idx": base_idx,
                })
            except Exception as e:
                print(f"  [{i+1}/{n}] Cache store failed: {e}", flush=True)

        results.append({
            "phase": phase, "idx": i, "model": model,
            "in_tok": in_tok, "out_tok": out_tok, "cost": cost,
            "cache_hit": False,
        })
        print(f"  [{i+1}/{n}] {model} in={in_tok} out={out_tok} cost=${cost:.6f}", flush=True)

    if phase == "gateway":
        save_semantic_cache(semantic_cache)

    with open(RESULTS_FILE, "a") as f:
        for r in results:
            # Don't save full embeddings in JSONL (too large)
            r_copy = {k: v for k, v in r.items() if k != "embedding"}
            f.write(json.dumps(r_copy) + "\n")

    chunk_cost = sum(r["cost"] for r in results)
    print(f"Chunk {chunk_idx} done. Cost: ${chunk_cost:.6f}. "
          f"Semantic hits: {semantic_hits}/{len(results)} (TP={semantic_tp}, FP={semantic_fp})")


def aggregate(n):
    baseline_cost = 0.0
    gateway_cost = 0.0
    baseline_n = 0
    gateway_n = 0
    gateway_hits = 0
    gateway_tp = 0
    gateway_fp = 0
    gateway_models = {}

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
                gateway_models[r["model"]] = gateway_models.get(r["model"], 0) + 1
                if r.get("cache_hit"):
                    gateway_hits += 1
                    if r.get("hit_true_positive"):
                        gateway_tp += 1
                    else:
                        gateway_fp += 1

    print(f"\nBaseline: {baseline_n}/{n} requests, cost=${baseline_cost:.6f}")
    print(f"Gateway:  {gateway_n}/{n} requests, cost=${gateway_cost:.6f}, "
          f"semantic hits={gateway_hits} (TP={gateway_tp}, FP={gateway_fp})")
    print(f"Gateway routing: {gateway_models}")

    # Embedding overhead (timed per call, persisted across chunks)
    embed_times = []
    if os.path.exists(EMBED_TIMES_FILE):
        with open(EMBED_TIMES_FILE) as f:
            for line in f:
                try:
                    embed_times.append(json.loads(line)["seconds"])
                except (json.JSONDecodeError, KeyError):
                    pass
    embed_total = sum(embed_times)
    embed_avg = (embed_total / len(embed_times)) if embed_times else 0.0
    print(f"Embedding calls: {len(embed_times)}, total {embed_total:.1f}s, avg {embed_avg:.2f}s/call")

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
        "gateway_semantic_hits": gateway_hits,
        "gateway_semantic_true_positives": gateway_tp,
        "gateway_semantic_false_positives": gateway_fp,
        "gateway_routing": gateway_models,
        "embedding_overhead": {
            "calls": len(embed_times),
            "total_seconds": round(embed_total, 1),
            "avg_seconds_per_call": round(embed_avg, 2),
        },
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "models": {"small": SMALL_MODEL, "large": LARGE_MODEL, "embed": EMBED_MODEL},
        "backend": "ollama",
        "optimizations": ["tiered_routing", "prefix_cache", "semantic_cache"],
        "workload": {
            "n_unique": 30,
            "n_paraphrases": n - 30,
            "paraphrase_ratio": round((n - 30) / n, 3),
            "seed": 20260930,
            "question_pool_size": len(QUESTIONS),
            "exact_duplicates": 0,
        },
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
    print(f"Gateway (routed+semantic): ${gateway_cost:.4f}")
    print(f"Saved: ${saved:.4f} ({pct:.1f}%)")
    print(f"Semantic hits: {gateway_hits} (TP={gateway_tp}, FP={gateway_fp})")
    print("="*50)

    with open("/tmp/ollama_semantic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/ollama_semantic_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "gateway"])
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=35)
    parser.add_argument("--n", type=int, default=90)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        for fp in [RESULTS_FILE, SEMANTIC_CACHE_FILE, EMBED_TIMES_FILE]:
            if os.path.exists(fp):
                os.remove(fp)
                print(f"Removed {fp}")
    elif args.aggregate:
        aggregate(n=args.n)
    elif args.phase:
        run_chunk(args.phase, args.chunk, args.chunk_size, args.n)
    else:
        parser.print_help()
