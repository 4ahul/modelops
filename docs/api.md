# API reference

Base URL is your deployment. All endpoints except `/health` and `/` require an API key:

```
Authorization: Bearer mo_live_xxx
```
or
```
X-API-Key: mo_live_xxx
```

Errors use FastAPI's `detail`. Where the reason is actionable, `detail` is an object:

```json
{
  "detail": {
    "detail": "No model satisfies the policy for task 'classification'. ...",
    "kind": "no_eligible_model",
    "context": {"excluded": {"claude-opus": "estimated $0.0021 over cost_limit $0.0010"}}
  }
}
```

`kind` is stable and safe to branch on. The SDK maps it onto typed exceptions.

| Status | `kind` | Meaning |
|---|---|---|
| 400 | `bad_request` | The request itself is invalid — prompt over the context window, bad parameter, content filter. Not retried, not failed over. |
| 401 | — | Missing or invalid key. No hint about which, because that tells an attacker which guesses were close. |
| 422 | `no_eligible_model` | The policy excluded every model. `context.excluded` says why, per model. |
| 429 | — | Rate limit. `Retry-After` header set. |
| 502 | `all_providers_failed` | Every candidate was tried and failed. `context.attempts` lists each. |
| 503 | — | No providers configured, or a dependency is unavailable. |
| 500 | `internal` | Generic on purpose. Details go to the log, not the response body. |

---

## POST /complete

Route a prompt to the cheapest model that meets the policy, and run it.

**Request**

| Field | Type | Notes |
|---|---|---|
| `prompt` | string | Required, non-blank. |
| `task_type` | string? | Selects the task policy. An unknown value falls back to the default rather than failing. |
| `max_tokens` | int? | Defaults to the task policy's limit. |
| `temperature` | float | Default `0.0`. Zero by default because the product's claim — a cheaper model gives comparable output — is only measurable if output is reproducible. |
| `system` | string? | System prompt. |
| `model_id` | string? | Pin a model and skip routing. For A/B comparisons and reproducing a stored decision. |

```bash
curl -X POST "$URL/complete" -H "Authorization: Bearer $KEY" \
  -d '{"prompt": "Classify: my card was charged twice", "task_type": "classification"}'
```

**Response**

```json
{
  "content": "billing",
  "model_id": "gemini-flash",
  "provider": "gemini",
  "input_tokens": 14,
  "output_tokens": 3,
  "cost_usd": 0.0000021,
  "latency_ms": 182.4,
  "routing_overhead_ms": 2.1,
  "routing_reason": "gemini-flash: score 0.871, est $0.000042, quality 0.940, p95 410ms",
  "fallbacks": [],
  "task_type": "classification"
}
```

`fallbacks` is non-empty when a provider failed and the router escalated. `routing_reason` says `quality unmeasured` when no eval result exists for that task and model — a guess is never presented as a measurement.

Every call is recorded in `routing_decisions`, including failures. A table of only successes cannot answer "how often does the cheap model fall over", which is the question that decides whether a saving was real.

---

## POST /route

The same decision, without running anything. Free and side-effect free, so a policy can be reviewed rather than discovered in production.

**Request**: `prompt`, `task_type?`, `expected_output_tokens` (default 512).

**Response**

```json
{
  "chosen": "gemini-flash",
  "reason": "gemini-flash: score 0.871, est $0.000042, quality 0.940",
  "overhead_ms": 1.8,
  "candidates": [
    {
      "model_id": "gemini-flash",
      "provider": "gemini",
      "estimated_cost": 0.000042,
      "quality": 0.94,
      "p95_latency_ms": 410.0,
      "score": 0.871,
      "unmeasured": false
    }
  ],
  "excluded": {
    "claude-opus": "estimated $0.002100 over cost_limit $0.001000"
  }
}
```

`candidates` are ordered best-first. `excluded` names every model that failed a hard constraint and which one — that is the difference between a debuggable router and a black box.

---

## POST /evals

Upload an eval set. Returns 201.

```json
{
  "name": "classification",
  "task_type": "classification",
  "grader": "exact_match",
  "examples": [
    {"input": "my card was charged twice", "expected": "billing", "tags": ["short"]},
    {"input": "how do I reset my password", "expected": "account", "grader": "contains"}
  ]
}
```

Per-example `grader`, `weight` and `id` are optional. Unknown graders are rejected here rather than at run time, so the mistake surfaces on upload instead of after paying for 300 model calls.

Re-uploading identical examples is a no-op. Changed examples create a **new version**, so a stored result always refers to the exact data it measured.

---

## POST /evals/run

Run a stored set across models.

**Request**: `eval_set`, `model_ids?` (defaults to all configured), `max_tokens`, `system?`, `keep_outputs` (default false — an eval report is stored and shared, and model output can carry customer data).

**Response**

```json
{
  "eval_set": "classification",
  "eval_version": 1,
  "task_type": "classification",
  "grader": "exact_match",
  "examples": 100,
  "duration_s": 41.2,
  "models": [
    {
      "model_id": "gemini-flash",
      "provider": "gemini",
      "examples": 100,
      "errors": 0,
      "accuracy": 0.94,
      "pass_rate": 0.94,
      "error_rate": 0.0,
      "p50_latency_ms": 180.0,
      "p95_latency_ms": 410.0,
      "p99_latency_ms": 890.0,
      "cost_per_query": 0.000042,
      "total_cost_usd": 0.0042,
      "cost_per_correct_answer": 0.0000447
    }
  ],
  "recommendation": {
    "from": "claude-sonnet",
    "to": "gemini-flash",
    "cost_reduction_pct": 97.7,
    "accuracy_delta": -0.03,
    "p95_latency_delta_ms": -770.0
  }
}
```

`accuracy` covers answered examples only; `error_rate` carries provider failures separately. Averaging an outage into accuracy would let a down provider look like a quality regression, and those two facts lead to opposite actions.

`cost_per_correct_answer` is `null` when nothing passed. It is the number that actually decides a downgrade.

A run refreshes the router's quality table immediately, so the measurement affects routing without a restart.

---

## GET /evals/quality

The scores routing is currently using, with their provenance.

```json
{
  "scores": {"classification": {"gemini-flash": 0.94, "gpt-4o-mini": 0.95}},
  "source": "most recent eval result per (task_type, model)",
  "measured_at": "2026-08-29T02:14:55+00:00",
  "note": "Scores are the last measurement, not a live estimate. Models with no score route with a penalty and are marked unmeasured."
}
```

Stale but honest, and labelled as such. The alternative — online sampling — is fresher and costs a real fraction of production spend.

---

## GET /evals, /evals/history, /evals/graders

- `GET /evals` — stored sets with versions and example counts.
- `GET /evals/history?name=&limit=` — recent results, newest first.
- `GET /evals/graders` — available graders and one-line descriptions.

---

## GET /metrics

```
?hours=24&task_type=classification
```

```json
{
  "window_hours": 24,
  "requests": 14203,
  "failures": 7,
  "failure_rate": 0.0005,
  "total_cost_usd": 4.71,
  "avg_cost_usd": 0.00033,
  "avg_latency_ms": 214.8,
  "avg_routing_overhead_ms": 2.3,
  "input_tokens": 1840221,
  "output_tokens": 402113,
  "fallbacks": 12,
  "by_model": [{"model_id": "gemini-flash", "provider": "gemini", "requests": 13980, "cost_usd": 0.59, "avg_latency_ms": 186.2}],
  "latency": {"p50": 180.0, "p95": 430.0, "p99": 1120.0, "samples": 14196}
}
```

`fallbacks` is the availability cost of cheap routing. Track it against `total_cost_usd` — a saving bought with a rising fallback count is not a saving.

`GET /metrics/timeseries?hours=&bucket_minutes=` returns the same spend bucketed over time.

---

## GET /models

Configured models with pricing and measured latency, cheapest first, plus the full catalog so you can see what adding a vendor key would unlock.

---

## Alerts

- `GET /alerts?limit=&unacknowledged_only=`
- `POST /alerts/check?window_hours=1` — run detection now; delivers to Slack when configured. Call this from a scheduler, not on a timer inside the service, or every replica alerts independently.
- `POST /alerts/{id}/acknowledge`

Three kinds:

| Kind | Compares | Suppressed when |
|---|---|---|
| `cost_spike` | Cost **per request** now vs the previous window | Either window has under 20 requests |
| `latency_regression` | p95 per model now vs previous window | Under 20 samples |
| `accuracy_regression` | Newest eval result vs the one before | Either run has under 10 examples |

Cost is compared per request, not in total: a doubling of traffic is not a cost regression, and alerting on it trains people to ignore the alert.

An `accuracy_regression` across an eval-set version bump is downgraded to `info`, because different test data means the two numbers do not measure the same thing.

---

## GET /health

Unauthenticated. See [deployment.md](deployment.md#health-checks).
