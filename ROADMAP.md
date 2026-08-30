# ModelOps Roadmap

Derived from `MODELOPS_IMPLEMENTATION_PLAN.md`, with the build order adjusted for one dependency the original plan didn't sequence: routing quality scores are meaningless until the eval framework produces them.

---

## Build order

### Phase 1 — Provider abstraction (Week 5)

`backend/app/providers/`

- `base.py` — `ModelProvider` ABC: `complete()`, `count_tokens()`, `estimate_cost()`
- `anthropic_.py`, `openai_.py`, `gemini_.py`
- A pricing table, versioned and dated, because prices change and a stale one silently mis-routes
- `CompletionResult`: content, tokens in/out, latency, cost, provider, model

**Done when:** the same prompt runs through all three providers and returns a comparable `CompletionResult`, with costs matching each vendor's published rate.

### Phase 2 — Evaluation framework (Weeks 6–7)

`backend/app/eval/`

Built *before* routing, not after. Routing scores multiply a quality term; without evals that term is a guess, and a plausible-looking router that optimizes against a made-up number is worse than no router.

- `EvalSet` / `EvalExample`, loadable from JSONL
- Graders: exact match, fuzzy, JSON-schema, LLM-as-judge
- Parallel runner (target: 100 examples across 3 providers in under 5 minutes)
- Per-provider report: accuracy, p50/p95 latency, cost per query

**Done when:** `router.evaluate(evals, task_type=...)` produces a table you'd actually use to pick a model.

### Phase 3 — Routing (Weeks 7–8)

`backend/app/core/`

- Weighted scoring over cost, latency and measured quality
- Hard constraints (`cost_limit`, `latency_budget_ms`, `min_quality`) exclude rather than penalize
- **Fallback chain** — on provider error or timeout, escalate; never fail the customer's request to save money
- Every decision written to `routing_decisions`

**Done when:** a task with a real eval set routes to a cheaper model and the eval report shows quality held.

### Phase 4 — API and persistence (Weeks 8–9)

- FastAPI: `/complete`, `/evals`, `/metrics`, `/alerts`
- SQLAlchemy models + Alembic migrations
- API-key auth, per-key rate limiting
- Redis for provider health and rolling latency stats

### Phase 5 — Dashboard (Weeks 9–10)

`frontend/`

- Cost over time, by provider and task type
- Latency distribution (p50/p95/p99 — averages hide the incidents)
- Eval comparison table
- Alert feed

### Phase 6 — Monitoring and alerts (Week 11)

- Cost spike, latency regression, accuracy regression
- Slack and email delivery
- Threshold config per task type

### Phase 7 — Beta (Week 12)

- Python SDK published
- Onboarding docs
- 5 design partners

---

## Decisions to make before writing code

**Sync or async provider calls?** Async — evaluation is embarrassingly parallel, and a synchronous runner makes Phase 2's five-minute target unreachable.

**Where does quality come from between eval runs?** Either the last eval result (stale but honest) or online sampling (fresh but costly). Start with the former and label it as such in the dashboard, so nobody reads a stale number as a live one.

**What happens when a provider is down?** Fallback chain, ordered by the same scoring with the failed provider excluded. Health tracked in Redis with a short TTL, so a recovered provider comes back automatically.

**Do we store prompts?** No, not by default. Token counts, cost and latency only. Prompt storage becomes an explicit per-customer opt-in with retention settings, because the companies most likely to pay for cost optimization are the ones least able to send production data to a third party that logs it.

---

## Open risks

**Multi-model routing is a crowded space.** OpenRouter, Martian, Portkey and the LLM gateways all overlap. The differentiator here has to be the eval framework — routing you can *trust* rather than routing you can *do* — and that needs to be the story from day one, not a Phase 2 feature.

**Cost savings are workload-dependent.** A team already using a small model saves nothing. Qualify design partners on their current model mix before promising a number.

**Quality scores are only as good as the eval set.** A customer with a 20-example eval set gets routing decisions built on 20 examples. The onboarding flow has to make building a real eval set the first thing that happens, not an optional step.
