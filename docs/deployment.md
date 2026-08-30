# Deployment

## Before you deploy

```bash
modelops check-config --production
```

Exits non-zero and lists every problem. Startup applies the same checks and refuses to boot in production if any fail:

| Check | Why it is fatal |
|---|---|
| `API_KEY_HASHES` is non-empty | An empty list means every endpoint is open. A service that silently serves unauthenticated traffic is worse than one that does not start. |
| At least one provider key | Otherwise every completion returns 503 and the deployment is decorative. |
| `CORS_ORIGINS` is not `*` | A wildcard with credentials allowed lets any origin spend your API budget. |
| `DATABASE_URL` is not localhost | A container pointing at its own loopback silently writes to a database nobody reads. |

## Configuration

Everything comes from the environment. See [`.env.example`](../.env.example) for the full list with commentary.

The four that matter most:

```bash
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/modelops
API_KEY_HASHES=$(modelops hash-key mo_live_xxx)   # comma-separate several
ANTHROPIC_API_KEY=sk-ant-...                       # and/or OPENAI_API_KEY, GOOGLE_API_KEY
```

### API keys

Only hashes are ever configured or stored:

```bash
$ modelops hash-key mo_live_2f9a...
8c7e1b0f4a2d...   # put this in API_KEY_HASHES
```

Rotate by adding the new hash, deploying, moving clients over, then removing the old one. Comparison is constant-time against every configured hash, so key checking does not leak timing.

### Metrics scope

`/metrics` and `/metrics/timeseries` are scoped to the calling key by default:

```bash
METRICS_SCOPE=key          # default — each key sees only its own numbers
METRICS_SCOPE=deployment   # every key sees the whole deployment
```

Leave it at `key` for anything multi-tenant; one tenant reading another's spend, volume and model mix is a data leak. Set `deployment` when a single team shares staging, production and CI keys and wants one combined picture.

`routing_decisions.api_key_hash` records the caller on every row, so the scope is a read-time filter — switching it changes what is visible, never what was stored.

### Redis is optional

It backs provider health and rate limiting *across replicas*. Without it both fall back to per-process state:

- Rate limits become approximate — each replica enforces its own counter.
- A provider marked unhealthy by one replica is not avoided by the others.

`/health` reports `redis: false` and a warning. This is a deliberate degradation: a monitoring dependency that can take down the request path is worse than an approximate rate limit.

## Migrations

Run before the new code serves traffic:

```bash
DATABASE_URL=postgresql+asyncpg://... alembic upgrade head
```

The same image should run the migration and serve traffic, so a deploy cannot apply a migration built from different code than the one it runs:

```bash
docker run --rm -e DATABASE_URL=... modelops:TAG alembic upgrade head
```

`downgrade` is implemented and tested in CI. A migration that cannot be reverted is one you cannot deploy on a Friday.

## Docker

```bash
docker build -t modelops:1.0.0 .
docker run -d -p 8000:8000 --env-file .env modelops:1.0.0
```

The image is multi-stage (no compilers in the runtime layer), runs as a non-root user, and its `HEALTHCHECK` hits `/health` rather than opening a TCP socket — a port check calls a container healthy while every request fails on a dead database.

One uvicorn worker per container. Scale with replicas rather than in-process workers, so a crash loses one request's worth of work and the orchestrator can see it.

## Health checks

`/health` is unauthenticated, because a load balancer cannot hold a credential. It reports dependency and configuration status and never a secret or a hostname.

```json
{
  "status": "ok",
  "environment": "production",
  "version": "0.1.0",
  "database": true,
  "redis": true,
  "models": ["claude-haiku", "claude-sonnet", "gpt-4o-mini"],
  "pricing_as_of": "2026-08-30",
  "pricing_age_days": 0,
  "warnings": []
}
```

Wire probes like this:

- **Liveness** — `GET /health`, accept any 200. The endpoint returns 200 while degraded on purpose, so a Redis blip does not restart a working container.
- **Readiness** — `GET /health` and require `database: true`.

`status` is `degraded` whenever `warnings` is non-empty. Two warnings deserve a page:

- `"Database is reachable but the schema is missing. Run: alembic upgrade head"` — the deploy skipped its migration. Distinguished from an unreachable database because the fix is completely different.
- `"Pricing table is N days old"` — every cost figure and routing decision derives from that table.

## Scheduled work

Two things run on a schedule, and both are HTTP endpoints rather than internal timers, so the schedule lives in one place instead of firing independently on every replica.

**Regression checks** — every 15 minutes:

```bash
curl -fsS -X POST "$MODELOPS_URL/alerts/check?window_hours=1" \
     -H "Authorization: Bearer $MODELOPS_KEY"
```

Detects cost spikes, latency regressions and accuracy regressions, writes them to `alerts`, and delivers to Slack if `SLACK_WEBHOOK_URL` is set. Alerts carry an hourly `dedupe_key`, so a persistent problem re-alerts once an hour rather than every run — an alerting system that repeats itself gets muted, and a muted alert is worse than none.

**Eval runs** — nightly or per deploy:

```bash
curl -fsS -X POST "$MODELOPS_URL/evals/run" \
     -H "Authorization: Bearer $MODELOPS_KEY" \
     -d '{"eval_set": "classification"}'
```

Routing quality is refreshed immediately after a run, so a measurement affects routing without waiting for a restart.

In CI, gate a deploy on quality instead:

```bash
modelops eval evals/classification.jsonl --min-accuracy 0.92
```

Exits 1 if no model clears the bar.

## Retention

`routing_decisions` grows with traffic. There is no automatic deletion — silently dropping the table that justifies every routing decision would be a bad default. Purge deliberately:

```python
from datetime import datetime, timedelta, timezone
from app.db.crud import purge_routing_decisions

await purge_routing_decisions(
    session, before=datetime.now(timezone.utc) - timedelta(days=90)
)
```

Keep at least a couple of eval cycles' worth: the accuracy-regression check compares consecutive `eval_results`, and the cost and latency checks compare a window against the one before it.

## Sizing

- **Routing overhead** is target <100ms and asserted <100ms in tests. It is scoring arithmetic plus at most a few Redis reads; the provider call dominates every request.
- **Eval runs** are bounded by `EVAL_CONCURRENCY` (default 8). Unbounded concurrency turns a 300-call run into a wall of 429s. 100 examples × 3 models inside five minutes is the target.
- **Database connections**: pool of 10 with 20 overflow per replica, recycled at 30 minutes — below the typical managed-Postgres idle timeout, so a connection is never handed out after the server closed it.

## What is deliberately not here

**No prompt storage by default.** `STORE_PROMPTS=true` exists and writes the first 200 characters to `routing_decisions.prompt_preview`. Leaving it off is the right default for the customers most likely to buy this.

**No automatic model downgrade.** The eval report recommends; a human decides and changes the policy. A system that silently reroutes production traffic based on last night's eval is a system nobody will approve.

**No job queue for evals.** A 100-example run across three models finishes inside a request. A queue would add operational surface for no benefit until eval sets are much larger.
