<div align="center">

# ModelOps

**Multi-model routing and evaluation for AI teams.**

Route each query to the cheapest model that still meets your quality bar — and prove it didn't break anything.

[![test](https://github.com/4ahul/modelops/actions/workflows/test.yml/badge.svg)](https://github.com/4ahul/modelops/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)](backend/app/main.py)
[![Postgres](https://img.shields.io/badge/Postgres-16-336791)](docker-compose.yml)
[![license](https://img.shields.io/badge/license-MIT-black)](LICENSE)

</div>

```python
from modelops import ModelOps

async with ModelOps(api_url="https://modelops.internal", api_key="mo_...") as client:
    result = await client.complete(
        "Classify this ticket: my card was charged twice",
        task_type="classification",
    )

# billing · gemini-flash · $0.0000021 · 182ms · routing overhead 2.1ms
```

---

## The problem

Teams pick one frontier model, wire it into everything, and pay frontier prices for work a small model does identically. Classification, extraction, routing, short summarization — none of it needs the biggest model.

But nobody knows which tasks are safe to downgrade, because nobody has the data.

So two things are missing at once: a way to route by cost and latency, and **a way to know that routing didn't break anything.**

## Why the evals are the product

Routing is easy. Half a dozen gateways will route for you. What none of them tell you is whether the cheap model was *good enough for your workload* — the only question that matters when deciding what to downgrade.

So the build order here was deliberate: **evals came before the router.** A routing score multiplies a quality term. Without measurement that term is a guess, and a plausible-looking router optimising against a made-up number is worse than no router at all.

Three consequences, enforced in code:

```python
#  A model with no eval result is marked unmeasured and penalised —
#  never silently assigned a default quality.
{"model_id": "gpt-4o-mini", "quality": None, "unmeasured": true}

#  min_quality EXCLUDES, it does not penalise. A cost ceiling expressed as a
#  score penalty can always be outvoted by a high enough quality term — which
#  is exactly how a "cost-aware" router ends up expensive.
{"claude-opus": "estimated $0.0021 over cost_limit $0.0010"}

#  Quality is strictly per task type. A model measured on classification
#  tells you nothing about its reasoning.
{"classification": {"gemini-flash": 0.94}, "reasoning": {"gemini-flash": 0.61}}
```

## The number that decides a downgrade

```bash
modelops eval evals/classification.jsonl
```

```
test_classification v1 — 100 examples, grader=exact_match, 41.2s
MODEL               ACC   PASS    ERR    P50 ms    P95 ms    $/QUERY   $/CORRECT
------------------------------------------------------------------------------
gemini-flash      94.0%  94.0%   0.0%       180       410   0.000042    0.000045
gpt-4o-mini       95.0%  95.0%   0.0%       240       520   0.000078    0.000082
claude-sonnet     97.0%  97.0%   0.0%       610      1180   0.001850    0.001907
```

`$/CORRECT` is the last column for a reason. **A model at half the price that gets a third fewer answers right is more expensive per correct answer, not cheaper.** No gateway shows you that.

Add `--min-accuracy 0.92` and it exits non-zero — a quality regression fails the build.

## How a request flows

```
   POST /complete  { prompt, task_type }
          │
          ▼
   ┌─────────────────────────────────────────────────────┐
   │  rank candidates                                     │
   │    exclude  cost_limit · latency_budget · min_quality│
   │    score    cost · measured p95 · measured quality   │
   └────────────────────────┬────────────────────────────┘
                            │ cheapest survivor
                            ▼
                    ┌───────────────┐   error / timeout
                    │  provider #1  │─────────────┐
                    └───────┬───────┘             ▼
                            │ ok           ┌───────────────┐
                            │              │  provider #2  │  ← escalate,
                            │              └───────┬───────┘    never fail
                            │                      │            the request
                            ▼                      │
   ┌─────────────────────────────────────────────────────┐
   │  record: tokens · cost · latency · overhead ·        │
   │          fallback count · success                   │
   └─────────────────────────────────────────────────────┘
```

Ask what it *would* do — free, no side effects, no model call:

```python
decision = await client.route("...", task_type="classification")

decision.chosen     # 'gemini-flash'
decision.reason     # 'gemini-flash: score 0.871, est $0.000042, quality 0.940, p95 410ms'
decision.excluded   # {'claude-opus': 'estimated $0.002100 over cost_limit $0.001000'}
```

`excluded` is the point. "No model available" with no explanation is the least actionable error a router can produce, so every exclusion names the constraint to relax.

## Quick start

```bash
docker compose up -d                       # postgres + redis
cp .env.example .env                       # add one provider key
pip install -e ".[dev,providers]"
alembic upgrade head
uvicorn app.main:app --reload --app-dir backend
```

Measure, then route:

```python
await client.upload_eval_set(
    "classification",
    [{"input": "my card was charged twice", "expected": "billing"}, ...],
    task_type="classification",
)

report = await client.run_eval("classification")
print(report.table())
print(report.cheapest_above(0.92))   # the cheapest model you can defend switching to
print(report.recommendation)
# {'from': 'claude-sonnet', 'to': 'gemini-flash',
#  'cost_reduction_pct': 97.7, 'accuracy_delta': -0.03, 'p95_latency_delta_ms': -770.0}
```

An eval run refreshes routing quality immediately — no restart.

## API

| | Endpoint | Purpose |
|---|---|---|
| `POST` | `/complete` | Route and run. Returns content, cost, latency, routing overhead, reason. |
| `POST` | `/route` | Explain the decision without running it. Free. |
| `POST` | `/evals` | Upload an eval set. Versioned on change. |
| `POST` | `/evals/run` | Run across models. Refreshes routing quality. |
| `GET` | `/evals/quality` | Scores routing uses, **with their measurement date**. |
| `GET` | `/evals/history` | Past results, newest first. |
| `GET` | `/metrics` | Spend, volume, failure rate, p50/p95/p99. Scoped per key. |
| `GET` | `/metrics/timeseries` | Cost bucketed over time. |
| `GET` | `/models` | Configured models, pricing, measured p95. |
| `GET` | `/alerts` · `POST /alerts/check` | Regression detection and delivery. |
| `GET` | `/health` | Unauthenticated. Dependency + config status. |

Full request/response shapes: **[docs/api.md](docs/api.md)**

## Graders

| Grader | Measures | Honest about |
|---|---|---|
| `exact_match` | Normalised string equality | Brittle on free text |
| `contains` | Substring | Blind to a hedged answer that contains the label then contradicts it |
| `fuzzy` | Character similarity | Produces a plausible number for any two strings — easiest way to fool yourself |
| `regex` | Pattern match | Only as good as the pattern |
| `numeric` | Parsed number within tolerance | Tolerance scales with magnitude |
| `json_match` | Parsed JSON, partial credit on key overlap | Tolerates code fences and surrounding prose |
| `json_schema` | Required keys and types | Shape, not values |
| `llm_judge` | Another model's 0–5 verdict | Costs a call per example, inherits the judge's bias |

A provider error is recorded as an **availability** failure and excluded from accuracy. Mixing the two would let an outage look like a quality regression — and those two facts lead to opposite actions.

## Decisions worth knowing about

<details open>
<summary><b>Never fail a customer's request to save money</b></summary>

If the chosen model errors or times out, the router escalates through the remaining candidates. A cost optimiser that converts a saving into an availability incident is a bad trade at any price.

Fallbacks are counted in `routing_decisions.fallback_count`, so the availability cost of cheap routing is **visible rather than assumed**. Track it against total spend — a saving bought with a rising fallback count is not a saving.
</details>

<details>
<summary><b>A bad request is not a provider failure</b></summary>

A prompt over the context window or a content-filter rejection fails identically at every vendor. Retrying it across three turns one client error into three bills.

So `ProviderBadRequest` is never retried, never failed over, and **never counts against provider health** — one customer's oversized prompt must not take a healthy vendor out of rotation for everyone else.
</details>

<details>
<summary><b>Prompt bodies are not stored</b></summary>

Token counts, cost and latency are. `STORE_PROMPTS=true` exists and is off by default, because the companies most likely to pay for cost optimisation are the least able to send production data to a third party that logs it.

The log pipeline enforces it too: a structlog processor redacts `prompt`, `content`, `api_key` and friends regardless of what any call site passes. Relying on discipline across hundreds of log statements means one of them eventually logs a prompt.
</details>

<details>
<summary><b>API keys are stored as hashes</b></summary>

Only SHA-256 hashes are ever configured (`modelops hash-key`), compared in constant time against every candidate. A leaked environment dump hands over nothing usable.
</details>

<details>
<summary><b>Metrics are scoped to the calling key</b></summary>

`METRICS_SCOPE=key` (default) means one tenant cannot read another's spend, volume or model mix. `METRICS_SCOPE=deployment` opts a single team sharing staging/prod/CI keys into one combined picture.

The scope is a read-time filter — switching it changes what is visible, never what was stored.
</details>

<details>
<summary><b>The pricing table is dated, and its age is a first-class fact</b></summary>

Every cost figure derives from `backend/app/providers/pricing.py`. A stale table keeps routing, keeps rendering dashboards, and is quietly wrong about everything.

So `PRICING_AS_OF` is exposed at `/health`, printed by `modelops pricing`, and warned about after 90 days.
</details>

<details>
<summary><b>Redis degrades, never blocks</b></summary>

It backs provider health and rate limiting across replicas. Unreachable, both fall back to per-process state and `/health` says so. A monitoring dependency that can take down the request path is a worse design than an approximate rate limit.

Startup caps the Redis connect at 2s, so an unreachable Redis doesn't stall every replica in a rolling deploy.
</details>

<details>
<summary><b>Percentiles, not averages — and routing overhead is published</b></summary>

Every latency figure is p50/p95/p99, computed by nearest rank so a reported p95 is always a latency that actually happened. An average hides exactly the tail that causes incidents.

A router that saves 30% and adds 200ms is a bad trade for interactive work, so `routing_overhead_ms` is on every response and a test asserts it stays under 100ms. Health and latency for every candidate come back in **one** Redis pipeline — per-model lookups would be two round trips each, which on a seven-model deployment is a quarter of the budget spent waiting on Redis.
</details>

<details>
<summary><b>Aggregation happens in the database</b></summary>

`hours` accepts up to 90 days. Percentiles are fetched by rank (`ORDER BY latency_ms LIMIT 1 OFFSET k`) and the cost chart is grouped by a computed bucket index, so neither loads a window into memory to produce three numbers or a 24-point chart.
</details>

## Operations

```bash
modelops check-config --production   # validate before deploying; exits 1 on problems
modelops hash-key mo_live_xxx        # generate an API_KEY_HASHES entry
modelops pricing                     # pricing table and its age
modelops eval evals/x.jsonl          # run evals with no server or database
modelops migrate                     # alembic upgrade head
```

In production, **startup refuses to continue** if `API_KEY_HASHES` is empty, no provider key is set, CORS is `*`, or the database URL still points at localhost. A service that silently serves unauthenticated traffic is worse than one that does not start.

Deployment guide, health-probe wiring, scheduled work and retention: **[docs/deployment.md](docs/deployment.md)**

## Layout

```
backend/
  app/
    api/          routes, auth, rate limiting, schemas
    core/         routing policy, scoring, execution, health, alerts, config
    db/           SQLAlchemy models, sessions, aggregation queries
    eval/         datasets, graders, parallel runner, reports
    providers/    Anthropic, OpenAI, Gemini adapters + dated pricing table
  migrations/     Alembic (upgrade and downgrade, both tested in CI)
sdk/python/       async + sync client, typed errors
evals/            two example eval sets
tests/            no network, no vendor keys, runs in seconds
docs/
```

## Data model

| Table | Holds |
|---|---|
| `routing_decisions` | Every routed call: task type, chosen model, tokens, cost, latency, routing overhead, fallback count, success |
| `eval_sets` / `eval_examples` | Versioned test sets |
| `eval_results` | Per-model accuracy, pass rate, error rate, percentile latency, cost per query |
| `alerts` | Cost spikes, latency and accuracy regressions, with dedupe keys |

`routing_decisions` is the asset. Once it holds real traffic, the routing algorithm stops being a heuristic and becomes a fit to your actual workload — the only way the product's claim gets verified rather than asserted.

## Alerts

| Kind | Compares | Suppressed when |
|---|---|---|
| `cost_spike` | Cost **per request** vs the previous window | Either window under 20 requests |
| `latency_regression` | p95 per model vs previous window | Under 20 samples |
| `accuracy_regression` | Newest eval result vs the one before | Either run under 10 examples |

Cost is compared per request, not in total: a doubling of traffic is not a cost regression, and alerting on it trains people to ignore the alert. Alerts carry an hourly dedupe key — an alerting system that repeats itself gets muted, and a muted alert is worse than none.

An accuracy drop across an eval-set version bump is downgraded to `info`, because different test data means the two numbers do not measure the same thing.

## Relationship to agenticpolicy

The sibling repo, [`agenticpolicy`](../agenticpolicy), guards what an agent may **do**. ModelOps decides which model **runs** it and whether that choice held up. They share a customer and compose cleanly — an agent can be policy-guarded and cost-routed at once — but neither depends on the other, and each is useful alone.

## Open risks

**Multi-model routing is a crowded space.** OpenRouter, Martian, Portkey and the LLM gateways all overlap. The differentiator has to be the eval framework — routing you can *trust* rather than routing you can *do*.

**Cost savings are workload-dependent.** A team already on a small model saves nothing. Qualify design partners on their current model mix before promising a number.

**Quality scores are only as good as the eval set.** A customer with 20 examples gets routing decisions built on 20 examples. Onboarding has to make building a real eval set the first thing that happens, not an optional step.

**The pricing table is a maintenance liability.** `assert_fresh()` and the `/health` field mitigate it; nothing removes it short of scraping vendor pricing, which trades a stale number for an unpredictable one.

## Status

| Phase | State |
|---|---|
| 1 · Provider abstraction | Complete |
| 2 · Evaluation framework | Complete |
| 3 · Routing | Complete |
| 4 · API and persistence | Complete |
| 5 · Dashboard | **Not started** — every endpoint it needs exists |
| 6 · Monitoring and alerts | Complete |
| 7 · Beta | SDK and docs done; design partners need outbound, not code |

See [ROADMAP.md](ROADMAP.md) for what was added beyond the original plan and why.

## License

MIT
