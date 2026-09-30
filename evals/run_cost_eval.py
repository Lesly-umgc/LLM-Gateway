"""Cost-reduction eval: realistic workload + public price-book money math.

Simulates enterprise customer-support traffic against the REAL gateway
pipeline (TestClient) with the tiered routed backend injected via app state
(the gateway reads the backend from app state, so no gateway/ changes needed).

Workload (fixed seed 20260930, reproducible):
  - 300 requests, one long shared system prompt on EVERY call
  - 40 base questions (26 simple factual / 14 complex analytical),
    each with 3 phrasing variants
  - Zipf-ish popularity: a few questions asked many times, like real
    support traffic (45% exact repeats, 35%/20% paraphrase variants)

Levers measured:
  1. tiered routing (llama3.2:3b vs llama3.1:8b)
  2. prefix caching of the repeated system prompt
  3. exact + semantic response cache (already in the gateway)
  4. prompt compression (already in the gateway path)

Price book (public prices, stated exactly):
  - expensive tier ("frontier", our 8B): GPT-4o rates,
    $2.50 / $10.00 per 1M input / output tokens
  - cheap tier ("small", our 3B): GPT-4o-mini rates,
    $0.15 / $0.60 per 1M input / output tokens
  - cached prefix input tokens: 50% off the tier input rate
    (OpenAI's published prompt-caching discount: gpt-4o-mini cached
    input $0.075 vs $0.15 per 1M)
  - exact/semantic cache hit: $0 (no model call happens)

Baseline ("direct"): every request sent to the expensive tier at full
input price, no caching, no routing. Input tokens are counted exactly
per request; completion tokens use the measured mean from the gateway
run (documented assumption), cross-checked against VALIDATE_N REAL
direct llama3.1:8b calls whose mean $/request is reported in
direct_validation alongside the baseline's implied mean $/request.

Writes evals/cost_results.json. Reports whatever the number is.

Env: LNM_COST_N (default 300), LNM_COST_MAX_TOKENS (default 96),
     LNM_COST_VALIDATE_N (default 40), OLLAMA_BASE_URL.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LNM_API_KEYS", "test-key")
os.environ.setdefault("LNM_ADMIN_KEY", "admin-key")

from fastapi.testclient import TestClient  # noqa: E402

from gateway.app import create_app  # noqa: E402
from gateway.auth import GatewayConfig  # noqa: E402
from backends.routed import RoutedOllamaBackend  # noqa: E402
from backends.ollama import OllamaBackend  # noqa: E402
from optimizer import router as _router  # noqa: E402
from optimizer.compression import count_tokens  # noqa: E402

N = int(os.environ.get("LNM_COST_N", "300"))
MAX_TOKENS = int(os.environ.get("LNM_COST_MAX_TOKENS", "96"))
VALIDATE_N = int(os.environ.get("LNM_COST_VALIDATE_N", "40"))
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "cost_results.json")
AUDIT_PATH = os.path.join(os.path.dirname(__file__), "cost_audit.jsonl")

HEADERS = {"Authorization": "Bearer test-key"}
ADMIN_HEADERS = {"Authorization": "Bearer admin-key"}

# ---------------------------------------------------------------- price book
# Public per-token prices, verified 2026-09-30:
#   GPT-4o-mini  $0.15 / $0.60  per 1M in/out; cached input $0.075 (50% off)
#   GPT-4o       $2.50 / $10.00 per 1M in/out
PRICE_BOOK = {
    "expensive_tier": "frontier (GPT-4o public rates)",
    "expensive_input_per_1m": 2.50,
    "expensive_output_per_1m": 10.00,
    "cheap_tier": "small (GPT-4o-mini public rates)",
    "cheap_input_per_1m": 0.15,
    "cheap_output_per_1m": 0.60,
    "cached_prefix_discount": 0.50,  # cached prefix billed at 50% of input rate
    "cache_hit_cost_usd": 0.0,
    "msg_overhead_tokens": 8,  # per-request framing overhead (assumption)
}

SYSTEM_PROMPT = (
    "You are Northwind Cloud's customer support assistant. Northwind Cloud "
    "is a SaaS project management platform for teams. Always be concise, "
    "friendly, and accurate. Key facts you must use: refund policy — full "
    "refunds within 30 days of purchase, no questions asked; after 30 days, "
    "prorated credit only. Support hours: Monday to Friday, 9am to 6pm "
    "Eastern, with 24/7 emergency support for enterprise plans. SLA: 99.9% "
    "uptime on team and enterprise plans, 99.5% on starter. Data centers in "
    "Toronto and Virginia; data residency options for enterprise. Security: "
    "SOC 2 Type II certified, encryption at rest and in transit, SSO/SAML "
    "on team and enterprise, audit logs on enterprise only. Plans: Starter "
    "$12/user/month (up to 10 users, 100GB storage), Team $24/user/month "
    "(SSO, 1TB storage, priority support), Enterprise custom pricing (audit "
    "logs, dedicated CSM, data residency, 99.99% uptime option). Free 14-day "
    "trial, no credit card required. Integrations: Slack, GitHub, Jira, "
    "Google Workspace, Microsoft Teams. API rate limits: 1000 requests/minute "
    "on team, 5000 on enterprise. For billing disputes, explain the charge "
    "first, then offer the dispute form at billing@northwindcloud.example. "
    "Never invent policy details not listed here; if unsure, say you will "
    "escalate to a human agent. Keep answers under 120 words."
)

# (kind, [canonical, paraphrase, paraphrase]); kind in {"simple","complex"}
QUESTIONS: list[tuple[str, list[str]]] = [
    ("simple", ["What is your refund policy?",
                "What's your refund policy?",
                "Can I get a refund?"]),
    ("simple", ["How do I reset my password?",
                "I forgot my password, how do I reset it?",
                "Password reset steps?"]),
    ("simple", ["What are your support hours?",
                "When is support available?",
                "Support hours?"]),
    ("simple", ["Where can I download my invoices?",
                "How do I find my invoices?",
                "Where are my invoices?"]),
    ("simple", ["How do I cancel my subscription?",
                "Cancel my plan",
                "How to cancel subscription?"]),
    ("simple", ["What is your SLA for the team plan?",
                "What's the uptime SLA?",
                "Team plan SLA?"]),
    ("simple", ["Do you offer SSO?",
                "Is SSO available?",
                "Do you support single sign-on?"]),
    ("simple", ["How much does the team plan cost?",
                "Team plan pricing?",
                "What does the team plan cost per month?"]),
    ("simple", ["How do I export my data?",
                "Can I export my projects?",
                "Data export options?"]),
    ("simple", ["Is there a status page?",
                "Where is your status page?",
                "How do I check service status?"]),
    ("simple", ["How do I invite a teammate?",
                "Invite team member",
                "Adding users to my workspace?"]),
    ("simple", ["What is the storage limit on the pro plan?",
                "Pro plan storage limit?",
                "How much storage do I get?"]),
    ("simple", ["Do you have a mobile app?",
                "Is there an iOS app?",
                "Mobile app availability?"]),
    ("simple", ["How do I change my billing email?",
                "Update billing email",
                "Change email for billing?"]),
    ("simple", ["What payment methods do you accept?",
                "Accepted payment methods?",
                "Can I pay by invoice?"]),
    ("simple", ["How do I enable two-factor authentication?",
                "Turn on 2FA",
                "Set up two-factor auth?"]),
    ("simple", ["What is your data retention policy?",
                "How long do you keep data?",
                "Data retention?"]),
    ("simple", ["Can I get a demo?",
                "How do I book a demo?",
                "Request a product demo?"]),
    ("simple", ["Where are your servers located?",
                "Data center locations?",
                "Where is my data hosted?"]),
    ("simple", ["How do I delete my account?",
                "Delete account",
                "Close my account?"]),
    ("simple", ["What integrations do you support?",
                "Do you integrate with Slack?",
                "Slack integration?"]),
    ("simple", ["How do I change my plan?",
                "Upgrade my subscription",
                "Switch plans?"]),
    ("simple", ["Is there a free trial?",
                "How long is the free trial?",
                "Free trial length?"]),
    ("simple", ["How do I contact support?",
                "Support contact",
                "How to reach support?"]),
    ("simple", ["What browsers are supported?",
                "Supported browsers?",
                "Does it work in Safari?"]),
    ("simple", ["How do I set up notifications?",
                "Notification settings",
                "Configure email notifications?"]),
    ("complex", ["Compare the pro and enterprise plans for a 50-person team "
                 "that needs SSO and audit logs, and explain why we should "
                 "pick one.",
                 "We are a 50-person team needing SSO and audit logs — compare "
                 "pro vs enterprise and recommend one with reasons.",
                 "Which plan fits a 50-person team with SSO and audit log "
                 "requirements? Compare pro and enterprise and justify your "
                 "recommendation."]),
    ("complex", ["Explain why my invoice shows two charges this month and "
                 "how to dispute the second one.",
                 "My invoice has a duplicate charge — why did this happen "
                 "and what are the steps to dispute it?",
                 "I was charged twice — explain why and walk me through "
                 "disputing the duplicate charge step by step."]),
    ("complex", ["Debug this error: getting 403 on API key rotation — explain "
                 "why it happens step by step and how to fix it.",
                 "API key rotation returns 403 — why? Debug it step by step "
                 "and give the fix.",
                 "Why does rotating my API key cause 403 errors? Explain the "
                 "cause and the fix in order."]),
    ("complex", ["We had a usage spike last Tuesday — analyze what could have "
                 "caused it and how to investigate.",
                 "Help me analyze a sudden usage spike from last Tuesday and "
                 "recommend investigation steps.",
                 "What would cause a usage spike last Tuesday? Walk me through "
                 "analyzing it."]),
    ("complex", ["Draft an escalation email to your support team about our "
                 "SLA breach with the details I should include.",
                 "Write an escalation email about an SLA breach — what details "
                 "must I include?",
                 "I need to escalate an SLA breach — draft the email and list "
                 "the required details."]),
    ("complex", ["How would migrating 200 projects from our old tool work? "
                 "Give me a step-by-step plan.",
                 "We need to migrate 200 projects — explain the migration "
                 "process step by step.",
                 "What's the best way to migrate 200 projects? Lay out each "
                 "step."]),
    ("complex", ["Summarize the security whitepaper's key points for our "
                 "compliance review.",
                 "What are the main security controls in your whitepaper? "
                 "Summarize for compliance.",
                 "For our compliance review, summarize your key security "
                 "practices."]),
    ("complex", ["Our webhook deliveries are failing intermittently — "
                 "troubleshoot with me.",
                 "Webhooks fail sometimes — help me troubleshoot the "
                 "intermittent failures.",
                 "Intermittent webhook failures — what should I check, in "
                 "order?"]),
    ("complex", ["Write a Python script that exports all our projects via "
                 "your API with pagination.",
                 "I need code to export projects via the API — write a Python "
                 "script with pagination.",
                 "Give me a Python function that pages through the projects "
                 "API and exports everything."]),
    ("complex", ["What if we exceed our API rate limit during a product "
                 "launch? Explain what happens and how to prepare.",
                 "Explain what happens if we hit the API rate limit at launch "
                 "and how to prepare for it.",
                 "We're worried about rate limits during launch — what happens "
                 "if we exceed them and how do we prepare?"]),
    ("complex", ["Recommend an onboarding plan for 30 new hires across three "
                 "departments.",
                 "We are onboarding 30 people in three departments — recommend "
                 "a rollout plan.",
                 "Draft an onboarding plan for 30 new hires split across three "
                 "teams."]),
    ("complex", ["Analyze our invoice from last quarter and explain each line "
                 "item.",
                 "Break down last quarter's invoice line by line and explain "
                 "the charges.",
                 "I don't understand last quarter's invoice — explain every "
                 "line item."]),
    ("complex", ["How do I integrate your API with our CI pipeline? Walk me "
                 "through it.",
                 "Explain step by step how to integrate the API into our CI/CD "
                 "pipeline.",
                 "CI integration guide — how do I connect your API to our "
                 "pipeline?"]),
    ("complex", ["Compare building an integration ourselves versus using your "
                 "native connectors — pros and cons.",
                 "Should we build our own integration or use native "
                 "connectors? Compare pros and cons.",
                 "Native connectors vs custom integration — analyze the "
                 "tradeoffs for us."]),
]


def build_workload(n: int, seed: int = 20260930) -> list[tuple[str, str]]:
    """(kind, user_text) requests with Zipf-ish popularity, fixed seed."""
    rng = random.Random(seed)
    order = list(range(len(QUESTIONS)))
    rng.shuffle(order)
    weights = [1.0 / (rank + 1) for rank in range(len(QUESTIONS))]
    out = []
    for _ in range(n):
        qi = rng.choices(order, weights=weights, k=1)[0]
        kind, variants = QUESTIONS[qi]
        r = rng.random()
        variant = variants[0] if r < 0.45 else (variants[1] if r < 0.80
                                               else variants[2])
        out.append((kind, variant))
    return out


def _tier_rates(tier: str) -> tuple[float, float]:
    if tier == _router.SMALL:
        return (PRICE_BOOK["cheap_input_per_1m"] / 1e6,
                PRICE_BOOK["cheap_output_per_1m"] / 1e6)
    return (PRICE_BOOK["expensive_input_per_1m"] / 1e6,
            PRICE_BOOK["expensive_output_per_1m"] / 1e6)


def gateway_cost(rec: dict) -> float:
    """Cost of one non-cache-hit gateway request from its backend record."""
    in_rate, out_rate = _tier_rates(rec["tier"])
    prompt_tokens = rec["prompt_tokens"]
    cached_prefix = min(rec["prefix_tokens"], prompt_tokens) \
        if rec["prefix_hit"] else 0
    cached_rate = in_rate * (1 - PRICE_BOOK["cached_prefix_discount"])
    return (cached_prefix * cached_rate
            + (prompt_tokens - cached_prefix) * in_rate
            + rec["completion_tokens"] * out_rate)


def run() -> dict:
    started = time.time()
    workload = build_workload(N)
    n_simple = sum(1 for k, _ in workload if k == "simple")
    print(f"workload: {N} requests ({n_simple} simple / {N - n_simple} "
          f"complex), {len(QUESTIONS)} base questions", flush=True)

    config = GatewayConfig(api_keys={"test-key"}, admin_key="admin-key",
                           backend_name="routed-ollama",
                           audit_path=AUDIT_PATH, rate_limit_per_min=10000)
    app = create_app(config)
    routed = RoutedOllamaBackend()
    conn = routed.check_connection()
    print(f"backend: {conn['detail']}", flush=True)
    assert conn["ok"], "both tiers must be up"
    app.state.lnm["backend"] = routed  # the gateway reads backend from state
    # Bypass NeMo rails for the cost measurement: the savings come from
    # tiered routing + prefix caching, not from the safety rails. Use the
    # routed backend directly so request_log populates for the money math.
    def _routed_generate(messages: list[dict]) -> str:
        text, _ = routed.generate(messages, max_tokens=MAX_TOKENS,
                                  temperature=0.2)
        return text
    app.state.lnm["nemo_generate"] = _routed_generate
    client = TestClient(app)
    assert client.get("/healthz").json()["status"] == "ok"

    kinds: list[str] = []
    cached: list[bool] = []   # gateway response-cache hit
    served: list[bool] = []   # produced a backend call (has a request_log entry)
    errors = 0
    for i, (kind, text) in enumerate(workload):
        kinds.append(kind)
        body = {"model": "routed-ollama",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": text}],
                "max_tokens": MAX_TOKENS, "temperature": 0.2}
        ok, hit = False, False
        try:
            r = client.post("/v1/chat/completions", json=body, headers=HEADERS)
        except Exception:
            time.sleep(5)
            try:
                r = client.post("/v1/chat/completions", json=body,
                                headers=HEADERS)
            except Exception:
                r = None
        if r is not None and r.status_code == 200:
            try:
                hit = bool(r.json().get("lnm", {}).get("cached"))
            except Exception:
                r = None
        if r is None or r.status_code not in (200, 403):
            errors += 1
        elif r.status_code == 200 and not hit:
            ok = True  # backend call happened; request_log grew by one
        # 403 (rail block): no model call, excluded from both arms
        cached.append(hit)
        served.append(ok)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{N}", flush=True)

    assert sum(served) == len(routed.request_log), \
        f"log mismatch: {sum(served)} served vs {len(routed.request_log)} logged"

    # ---- money math ----
    mean_completion = (sum(r["completion_tokens"] for r in routed.request_log)
                       / len(routed.request_log))
    gw_cost, direct_cost = 0.0, 0.0
    no_routing_cost, no_prefix_cost, no_cache_cost = 0.0, 0.0, 0.0
    log_idx = 0
    exp_in = PRICE_BOOK["expensive_input_per_1m"] / 1e6
    exp_out = PRICE_BOOK["expensive_output_per_1m"] / 1e6
    for kind, text in workload:
        in_tokens = (count_tokens(SYSTEM_PROMPT) + count_tokens(text)
                     + PRICE_BOOK["msg_overhead_tokens"])
        direct_cost += in_tokens * exp_in + mean_completion * exp_out
    for i, (kind, text) in enumerate(workload):
        if cached[i]:
            # counterfactual: billed as a full routed request, no response cache
            tier = _router.route(text)[0]
            in_rate, out_rate = _tier_rates(tier)
            prefix_tokens = count_tokens(SYSTEM_PROMPT)
            in_tokens = (prefix_tokens + count_tokens(text)
                         + PRICE_BOOK["msg_overhead_tokens"])
            cached_rate = in_rate * (1 - PRICE_BOOK["cached_prefix_discount"])
            no_cache_cost += (prefix_tokens * cached_rate
                              + (in_tokens - prefix_tokens) * in_rate
                              + mean_completion * out_rate)
            continue
        if not served[i]:
            continue  # error/block: contributes to neither arm
        rec = routed.request_log[log_idx]
        log_idx += 1
        gw_cost += gateway_cost(rec)
        # no routing: same request at the expensive tier
        no_routing_cost += gateway_cost(dict(rec, tier=_router.LARGE))
        # no prefix cache: prefix at full rate
        in_rate, out_rate = _tier_rates(rec["tier"])
        no_prefix_cost += (rec["prompt_tokens"] * in_rate
                           + rec["completion_tokens"] * out_rate)
        # no response cache: actual cost stands
        no_cache_cost += gateway_cost(rec)

    # ---- validation: real direct 8B calls on a sample ----
    # Runs VALIDATE_N workload requests straight at llama3.1:8b (no gateway)
    # to cross-check the price-book baseline's completion-token assumption.
    direct = OllamaBackend(model="llama3.1:8b")
    real_costs, real_completions = [], []
    for kind, text in workload[:VALIDATE_N]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
        try:
            _, usage = direct.generate(msgs, max_tokens=MAX_TOKENS,
                                       temperature=0.2)
        except Exception:
            time.sleep(5)
            try:
                _, usage = direct.generate(msgs, max_tokens=MAX_TOKENS,
                                           temperature=0.2)
            except Exception:
                continue
        real_costs.append(usage["prompt_tokens"] * exp_in
                          + usage["completion_tokens"] * exp_out)
        real_completions.append(usage["completion_tokens"])
    mean_real_direct = (sum(real_costs) / len(real_costs)) if real_costs else 0
    mean_real_completion = (sum(real_completions) / len(real_completions)
                            if real_completions else 0)
    print(f"validation: {len(real_costs)}/{VALIDATE_N} real direct 8B calls, "
          f"mean ${mean_real_direct:.6f}/req "
          f"(baseline assumes ${direct_cost / N:.6f}/req)", flush=True)

    m = client.get("/admin/metrics", headers=ADMIN_HEADERS).json()
    comp_saved = m.get("tokens", {}).get("saved_compression", 0)
    # compression value at the routed mix's average input rate (approximate)
    avg_in_rate = (routed.routed_small * PRICE_BOOK["cheap_input_per_1m"]
                   + routed.routed_large * PRICE_BOOK["expensive_input_per_1m"]) \
        / max(1, routed.routed_small + routed.routed_large) / 1e6
    compression_value = comp_saved * avg_in_rate

    savings = direct_cost - gw_cost
    levers = {
        "routing": no_routing_cost - gw_cost,
        "prefix_caching": no_prefix_cost - gw_cost,
        "response_cache": no_cache_cost - gw_cost,
        "compression": compression_value,
    }

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": {
            "n_requests": N,
            "seed": 20260930,
            "n_simple": n_simple,
            "n_complex": N - n_simple,
            "base_questions": len(QUESTIONS),
            "variants_per_question": 3,
            "repeat_mix": "45% exact repeat / 35% paraphrase A / 20% paraphrase B",
            "popularity": "zipf-ish 1/(rank+1) over shuffled question order",
            "system_prompt_tokens": count_tokens(SYSTEM_PROMPT),
            "max_tokens_per_response": MAX_TOKENS,
        },
        "price_book": PRICE_BOOK,
        "gateway": {
            "model_calls": len(routed.request_log),
            "routed_small": routed.routed_small,
            "routed_large": routed.routed_large,
            "cache_hits": sum(cached),
            "prefix_hits": routed.prefixes.hits,
            "prefix_misses": routed.prefixes.misses,
            "errors": errors,
            "mean_completion_tokens": round(mean_completion, 1),
            "tokens_saved_compression": comp_saved,
        },
        "cost_usd": {
            "direct_baseline": round(direct_cost, 6),
            "gateway": round(gw_cost, 6),
            "saved": round(savings, 6),
            "savings_pct": round(savings / direct_cost, 4) if direct_cost else 0,
        },
        "lever_contribution_usd": {k: round(v, 6) for k, v in levers.items()},
        "direct_validation": {
            "sample_n": len(real_costs),
            "attempted_n": VALIDATE_N,
            "model": "llama3.1:8b",
            "mean_cost_usd_per_request": round(mean_real_direct, 6),
            "mean_completion_tokens": round(mean_real_completion, 1),
            "baseline_mean_cost_usd_per_request": round(direct_cost / N, 6),
        },
        "notes": [
            "Direct baseline: every request to the expensive tier at full "
            "input price, no caching/routing. Input tokens counted exactly; "
            "completions use the measured mean from the gateway run.",
            "Lever values are marginal (each toggled off alone); they do not "
            "sum exactly to total savings because the levers interact.",
            "Compression value is approximate (routed-mix average input rate).",
            "Cache hits cost $0: no model call happens.",
            "Errored/blocked requests cost $0 in the gateway arm (no model "
            "call) but are billed in the direct arm, which has no rails.",
            "Prefix cache prices repeated prompt prefixes at 50% of the "
            "input rate (OpenAI prompt-caching convention); this is "
            "price-book modeling, not measured compute saved in Ollama.",
        ],
        "elapsed_seconds": round(time.time() - started, 1),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results



def summarize(r: dict) -> str:
    c = r["cost_usd"]
    g = r["gateway"]
    lines = [
        "== LNM Gateway cost eval ==",
        f"workload: {r['workload']['n_requests']} requests "
        f"({g['routed_small']} small / {g['routed_large']} large, "
        f"{g['cache_hits']} cache hits, {g['prefix_hits']} prefix hits, "
        f"{g['errors']} errors)",
        f"direct baseline: ${c['direct_baseline']:.4f}",
        f"gateway:         ${c['gateway']:.4f}",
        f"saved:           ${c['saved']:.4f} ({c['savings_pct']:.1%})",
        "per-lever $:",
    ]
    for k, v in r["lever_contribution_usd"].items():
        lines.append(f"  {k}: ${v:.4f}")
    lines.append(f"elapsed={r['elapsed_seconds']}s wrote {RESULTS_PATH}")
    return "\n".join(lines)


if __name__ == "__main__":
    res = run()
    print(summarize(res))
