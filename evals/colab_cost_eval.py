"""
Colab cost eval: measures token-based savings from tiered routing + prefix caching.

Runs on Colab GPU using HuggingFace transformers (4-bit quantized).
Compares:
  - Baseline: all requests to 8B model.
  - Gateway: simple queries to 3B, complex to 8B, with prefix caching.

Cost model (public price book, e.g. Together AI / Fireworks):
  - 8B: $0.20 / 1M input tokens, $0.20 / 1M output tokens (example rates)
  - 3B: $0.10 / 1M input tokens, $0.10 / 1M output tokens
  - Cached prefix: 50% discount on input.

This is a simplified standalone version for Colab. It does NOT use the full
LNM Gateway stack (no NeMo, no FastAPI). It measures the core optimization:
tiered routing + prefix caching.
"""

import time
import random
import json

# Price book (USD per 1M tokens) - public rates, adjust as needed
PRICE_8B_IN = 0.20 / 1e6
PRICE_8B_OUT = 0.20 / 1e6
PRICE_3B_IN = 0.10 / 1e6
PRICE_3B_OUT = 0.10 / 1e6
CACHED_DISCOUNT = 0.5  # 50% off for cached prefix

# Workload: (is_simple, prompt)
# Simple queries go to 3B, complex to 8B.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely."
)

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
        # 45% simple, 55% complex (Zipf-ish)
        is_simple = rng.random() < 0.45
        candidates = [q for q in QUESTIONS if q[0] == is_simple]
        _, prompt = rng.choice(candidates)
        workload.append((is_simple, prompt))
    return workload


def count_tokens(text):
    # Rough estimate: 1 token ~ 4 chars
    return max(1, len(text) // 4)


def run_eval(n=100, use_gpu=True):
    print("Loading models...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Use 4-bit quantization to fit 8B on T4
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    # Small model (3B equivalent)
    print("Loading 3B model...")
    small_id = "meta-llama/Llama-3.2-3B-Instruct"
    try:
        small_tok = AutoTokenizer.from_pretrained(small_id)
        small_model = AutoModelForCausalLM.from_pretrained(
            small_id, quantization_config=quant_config,
            device_map="auto", trust_remote_code=True,
        )
    except Exception as e:
        print(f"Llama 3B failed ({e}), trying Qwen 2.5 3B...")
        small_id = "Qwen/Qwen2.5-3B-Instruct"
        small_tok = AutoTokenizer.from_pretrained(small_id)
        small_model = AutoModelForCausalLM.from_pretrained(
            small_id, quantization_config=quant_config,
            device_map="auto", trust_remote_code=True,
        )

    # Large model (8B equivalent)
    print("Loading 8B model...")
    large_id = "meta-llama/Llama-3.1-8B-Instruct"
    try:
        large_tok = AutoTokenizer.from_pretrained(large_id)
        large_model = AutoModelForCausalLM.from_pretrained(
            large_id, quantization_config=quant_config,
            device_map="auto", trust_remote_code=True,
        )
    except Exception as e:
        print(f"Llama 8B failed ({e}), trying Qwen 2.5 7B...")
        large_id = "Qwen/Qwen2.5-7B-Instruct"
        large_tok = AutoTokenizer.from_pretrained(large_id)
        large_model = AutoModelForCausalLM.from_pretrained(
            large_id, quantization_config=quant_config,
            device_map="auto", trust_remote_code=True,
        )

    def generate(model, tok, prompt, max_tokens=96):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        response = text[len(prompt):].strip()
        in_tokens = len(inputs["input_ids"][0])
        out_tokens = len(tok.encode(response))
        return response, in_tokens, out_tokens

    workload = build_workload(n)
    print(f"Workload: {n} requests")

    # Baseline: all to 8B
    print("\nRunning baseline (all 8B)...")
    baseline_cost = 0.0
    for i, (is_simple, prompt) in enumerate(workload):
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        _, in_tok, out_tok = generate(large_model, large_tok, full_prompt)
        baseline_cost += in_tok * PRICE_8B_IN + out_tok * PRICE_8B_OUT
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n}")

    # Gateway: routed + prefix caching
    print("\nRunning gateway (routed + prefix cache)...")
    gateway_cost = 0.0
    prefix_cache = {}  # system prompt -> cached
    for i, (is_simple, prompt) in enumerate(workload):
        # Route: simple -> 3B, complex -> 8B
        model, tok = (small_model, small_tok) if is_simple else (large_model, large_tok)
        in_rate, out_rate = (PRICE_3B_IN, PRICE_3B_OUT) if is_simple else (PRICE_8B_IN, PRICE_8B_OUT)

        # Prefix caching: system prompt is cached after first use
        sys_tokens = count_tokens(SYSTEM_PROMPT)
        if SYSTEM_PROMPT in prefix_cache:
            # Cached: 50% discount on prefix
            cached_cost = sys_tokens * in_rate * (1 - CACHED_DISCOUNT)
            prefix_hit = True
        else:
            cached_cost = sys_tokens * in_rate
            prefix_cache[SYSTEM_PROMPT] = True
            prefix_hit = False

        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        _, in_tok, out_tok = generate(model, tok, full_prompt)

        # Cost: cached prefix + uncached rest + output
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
        "models": {"small": small_id, "large": large_id},
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

    with open("/tmp/colab_cost_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/colab_cost_results.json")

    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_eval(n=n)
