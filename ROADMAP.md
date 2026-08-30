# ModelOps Roadmap

Derived from `MODELOPS_IMPLEMENTATION_PLAN.md`, with the build order adjusted for one dependency the original plan didn't sequence: routing quality scores are meaningless until the eval framework produces them.

**Status: Phases 1â€“4 and 6 are built and tested.** Tested end to end with no network and no vendor keys; `ruff`/`black`/`mypy` clean. What remains is the dashboard (Phase 5) and everything that needs real customers (Phase 7).

---

## Build order

### Phase 1 â€” Provider abstraction âœ…

`backend/app/providers/`

- `base.py` â€” `ModelProvider` ABC: `complete()`, `count_tokens()`, `estimate_cost()`
- `anthropic_.py`, `openai_.py`, `gemini_.py`
- `pricing.py` â€” versioned and dated table, because prices change and a stale one silently mis-routes
- `CompletionResult`: content, tokens in/out, latency, cost, provider, model
- `registry.py` â€” instantiates only the models whose vendor key is configured

**Done when:** the same prompt runs through all three providers and returns a comparable `CompletionResult`, with costs matching each vendor's published rate.

*Met.* `TestCostAgreementAcrossVendors` sends one token split through all three adapters and asserts each result equals `pricing.cost(...)` for its own model.

**Added beyond the plan:**

- **Errors are classified, not just raised.** `ProviderBadRequest` is never retried and never failed over â€” a prompt over the context window fails identically at every vendor, and retrying it across three turns one client error into three bills. It also never counts against provider health, so one customer's oversized prompt cannot take a healthy vendor out of rotation for everyone.
- **Retries with full jitter**, honouring `Retry-After` and capping it at 30s. Synchronised retries rebuild the burst that caused the 429.
- **Vendor SDK retry loops disabled**, so retries happen once, uniformly, in one place.
- **Missing usage data is a hard failure.** OpenAI and Gemini both raise rather than reporting a fabricated zero cost.
- **Defensive response parsing.** `content[0].text` breaks the first time a model emits a tool-use or thinking block; a Gemini safety block makes `response.text` raise. Both are handled, so one odd response cannot abort a 300-call eval run.
- **`assert_fresh()`** warns once the pricing table is over 90 days old, and its age is exposed at `/health` and in `modelops pricing`.

### Phase 2 â€” Evaluation framework âœ…

`backend/app/eval/`

Built *before* routing, not after. Routing scores multiply a quality term; without evals that term is a guess, and a plausible-looking router that optimizes against a made-up number is worse than no router.

- `EvalSet` / `EvalExample`, loadable from JSONL (errors name the line number)
- Graders: `exact_match`, `contains`, `fuzzy`, `regex`, `numeric`, `json_match`, `json_schema`, `llm_judge`
- Parallel runner, bounded by `EVAL_CONCURRENCY`
- Per-model report: accuracy, pass rate, error rate, p50/p95/p99, cost per query

**Done when:** `router.evaluate(evals, task_type=...)` produces a table you'd actually use to pick a model.

*Met,* via `EvalRunner.run()` and `EvalReport.table()`.

**Added beyond the plan:**

- **A provider error is an availability fact, not a wrong answer.** Errors are excluded from `accuracy` and reported as `error_rate`. Averaging them together would let an outage look like a quality regression, and those lead to opposite actions.
- **`cost_per_correct_answer`** â€” the number that actually decides a downgrade. A model at half the price that gets a third fewer answers right is more expensive per correct answer.
- **Percentiles by nearest rank**, so a reported p95 is always a latency that was observed rather than an average of two that were not.
- **Partial credit on `json_match`**, because wrong on one field of four is a different failure from wrong on everything.
- **Fence and prose tolerance** in JSON graders. Models wrap JSON in ```json even when told not to; failing a correct extraction over that measures instruction-following instead.
- **Outputs are not retained by default** â€” an eval report is stored and shared, and model output can carry the customer's data.
- **Deterministic `sample(n, seed=...)`**, so two runs compare like with like.
- **A judge failure is not a model failure.** An unreachable `llm_judge` records "judge unavailable" rather than a quality regression that never happened.

### Phase 3 â€” Routing âœ…

`backend/app/core/`

- Weighted scoring over cost, latency and measured quality, normalised within the candidate set
- Hard constraints (`cost_limit`, `latency_budget_ms`, `min_quality`) exclude rather than penalize
- **Fallback chain** â€” on provider error or timeout, escalate; never fail the customer's request to save money
- Every decision written to `routing_decisions`

**Done when:** a task with a real eval set routes to a cheaper model and the eval report shows quality held.

*Met.* `EvalReport.savings_vs()` and `/evals/run`'s `recommendation` produce exactly that comparison, and `TestScoring` pins the behaviour.

**Added beyond the plan:**

- **Unmeasured quality is labelled, never defaulted.** A model with no eval result for a task is penalised and marked `unmeasured` in the decision record. `allow_unmeasured=False` turns it into a hard exclusion once eval coverage exists.
- **Scores normalise within the candidate set.** With three models between $0.001 and $0.002, dividing by a $0.05 ceiling would make all three look identically cheap and hand the decision entirely to quality.
- **Unmeasured latency scores neutral (0.5), not best.** A new model has not earned the top slot and should not be punished for being new either.
- **`NoEligibleModel` carries per-model reasons.** "No model available" with no explanation is the least actionable error a router can produce.
- **A fallback chain cannot bypass a hard constraint** â€” it is an escalation preference, not a way around a cost ceiling.
- **Routing overhead is measured and asserted <100ms**, because a router that saves 30% and adds 200ms is a bad trade for interactive work.
- **`/route`** explains a decision for free, with no side effects, so a policy can be reviewed rather than discovered in production.

### Phase 4 â€” API and persistence âœ…

- FastAPI: `/complete`, `/route`, `/evals`, `/evals/run`, `/evals/quality`, `/metrics`, `/metrics/timeseries`, `/models`, `/alerts`, `/health`
- SQLAlchemy models + Alembic migration (upgrade *and* downgrade, both tested in CI against Postgres)
- API-key auth on SHA-256 hashes with constant-time comparison; per-key rate limiting in Redis
- Redis for provider health and rolling latency, degrading to per-process state when absent

**Added beyond the plan:**

- **Production configuration is validated at startup**, and the app refuses to boot with an empty key list, no provider key, `CORS_ORIGINS=*`, or a localhost database. A service that silently serves unauthenticated traffic is worse than one that does not start.
- **`/health` distinguishes a missing schema from a dead database.** A deploy that skipped its migration would otherwise report a healthy database and then fail every write.
- **Log redaction is enforced by a processor**, not by discipline at hundreds of call sites â€” one of those eventually logs a prompt.
- **Failures are recorded, not only successes.** A table of successes cannot answer "how often does the cheap model fall over".
- **`modelops` CLI** â€” `hash-key`, `pricing`, `check-config`, `migrate`, `eval` (with `--min-accuracy` for CI gating, so a quality regression fails a build).

### Phase 5 â€” Dashboard (not started)

`frontend/`

The API is in place: `/metrics`, `/metrics/timeseries`, `/evals/history`, `/evals/quality`, `/alerts` return everything the UI needs.

- Cost over time, by provider and task type
- Latency distribution (p50/p95/p99 â€” averages hide the incidents)
- Eval comparison table
- Alert feed

**One requirement carried over from a design note:** the dashboard must label the quality score's measurement date. `/evals/quality` returns `measured_at` for exactly this reason â€” a stale number rendered as a live one is worse than no number.

### Phase 6 â€” Monitoring and alerts âœ…

- Cost spike, latency regression, accuracy regression â€” each against a baseline window rather than a fixed number
- Slack delivery; failures logged and swallowed, since the alert is already durable in the database
- Per-threshold config via `AlertThresholds`

**Added beyond the plan:**

- **Cost is compared per request, not in total.** A doubling of traffic is not a cost regression, and alerting on it trains people to ignore the alert.
- **Thin windows produce no alert.** Comparing three requests to two is arithmetic, not signal.
- **Dedupe keys, hourly.** An alerting system that repeats itself gets muted, and a muted alert is worse than none.
- **An accuracy drop across an eval-set version bump is downgraded to `info`**, because different test data means the two numbers do not measure the same thing.

### Phase 7 â€” Beta (blocked on customers)

- Python SDK âœ… (async + sync, typed errors, tested against the real ASGI app)
- Onboarding docs âœ… (`README.md`, `docs/api.md`, `docs/deployment.md`)
- 5 design partners â€” needs outbound, not code

---

## Decisions made

**Sync or async provider calls?** Async. Evaluation is embarrassingly parallel and a synchronous runner makes the five-minute target unreachable.

**Where does quality come from between eval runs?** The last eval result â€” stale but honest â€” surfaced through `/evals/quality` with its `measured_at`. Online sampling is fresher and costs a real fraction of production spend; it can come later if customers ask.

**What happens when a provider is down?** Fallback chain, ordered by the same scoring with the failed provider excluded. Health tracked in Redis with a short TTL, so a recovered provider returns automatically without an operator.

**Do we store prompts?** No, not by default. Token counts, cost and latency only. `STORE_PROMPTS=true` is an explicit per-deployment opt-in, because the companies most likely to pay for cost optimization are the ones least able to send production data to a third party that logs it.

---

## Still to decide

**Retention policy.** `purge_routing_decisions()` exists but nothing calls it on a schedule. Silently deleting the table that justifies every routing decision would be a bad default, so it stays manual until there is a customer with an actual retention requirement to encode.

**Multi-tenancy.** `routing_decisions.api_key_hash` scopes rows to a key, but `/metrics` does not filter by it â€” every key sees the whole deployment's numbers. Fine for a single-team install, wrong for shared SaaS. This is the largest single piece of work before the product can be sold as hosted rather than deployed.

**Task-type inference.** `task_type` is supplied by the caller. Inferring it from the prompt would be more convenient and would make routing non-deterministic in a way that is very hard to debug. Staying explicit for now.

---

## Open risks

**Multi-model routing is a crowded space.** OpenRouter, Martian, Portkey and the LLM gateways all overlap. The differentiator has to be the eval framework â€” routing you can *trust* rather than routing you can *do* â€” and that needs to be the story from day one, not a Phase 2 feature.

**Cost savings are workload-dependent.** A team already using a small model saves nothing. Qualify design partners on their current model mix before promising a number.

**Quality scores are only as good as the eval set.** A customer with a 20-example eval set gets routing decisions built on 20 examples. The onboarding flow has to make building a real eval set the first thing that happens, not an optional step.

**The pricing table is a maintenance liability.** Every cost figure derives from it. `assert_fresh()` and the `/health` field mitigate this; nothing removes it short of scraping vendor pricing pages, which trades a stale number for an unpredictable one.
