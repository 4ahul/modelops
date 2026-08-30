# ModelOps

**Multi-model routing and evaluation for AI teams.** Route each query to the cheapest model that still meets your quality bar, and catch quality regressions before your users do.

> **Status: scaffold.** The structure, data model and interfaces are laid out; the implementation starts in Week 5. See [ROADMAP.md](ROADMAP.md). The SDK in `sdk/python/` is the intended public API — treat it as a design document until the backend lands.

---

## The problem

Teams pick one frontier model, wire it into everything, and pay frontier prices for work a small model does identically. Classification, extraction, routing, summarization of short text — none of it needs the biggest model, but nobody knows which tasks are safe to downgrade because nobody has the data.

So two things are missing at once: a way to route by cost and latency, and a way to know that routing didn't break anything.

## The intended shape

```python
from modelops import Router, RoutingPolicy

router = Router(
    providers=["claude-sonnet", "gpt-4o-mini", "gemini-flash"],
    policy=RoutingPolicy(
        tasks={
            "classification": {"cost_limit": 0.001, "latency_budget_ms": 800,
                               "min_quality": 0.92},
            "drafting":       {"cost_limit": 0.05,  "latency_budget_ms": 5000,
                               "min_quality": 0.85},
        }
    ),
)

result = await router.complete(prompt="...", task_type="classification")
print(result.content, result.provider, result.cost, result.latency_ms)
```

The routing decision is only trustworthy if `min_quality` means something, which is why the eval framework is not a separate feature — it's the thing that makes routing safe:

```python
from modelops import EvalSet

evals = EvalSet.from_jsonl("evals/classification.jsonl")
report = await router.evaluate(evals, task_type="classification")
report.print_table()   # accuracy, p50/p95 latency, cost per query, per provider
```

## Repository layout

```
backend/
  app/
    api/          FastAPI routes
    core/         routing algorithm, scoring
    db/           SQLAlchemy models, migrations
    providers/    Anthropic / OpenAI / Gemini adapters
    eval/         evaluation framework
sdk/python/       the client library above
frontend/         React dashboard
docs/
```

## Local development

```bash
docker compose up -d              # postgres + redis
cp .env.example .env              # add your API keys
```

## Data model

Five tables carry the product:

| Table | Holds |
|---|---|
| `routing_decisions` | Every routed call: task type, chosen provider, cost, latency, tokens |
| `eval_sets` / `eval_examples` | Versioned test sets |
| `eval_results` | Per-provider accuracy, latency, cost for a given eval set |
| `alerts` | Cost spikes, latency spikes, accuracy regressions |

`routing_decisions` is the asset. Once it has real traffic in it, the routing algorithm stops being a heuristic and starts being a fit to your actual workload.

## Relationship to agenticpolicy

The sibling repo, [`agenticpolicy`](../agenticpolicy), guards what an agent may *do*. ModelOps decides which model *runs* it and whether that choice held up. They share a customer and compose cleanly — an agent can be policy-guarded and cost-routed at once — but neither depends on the other, and each is useful alone.

## Design notes worth settling early

**Routing without evals is guesswork.** Cost and latency are measurable per call; quality is not. Ship the eval framework alongside routing, or the routing scores end up multiplying a quality number nobody measured.

**Never fail a customer's request to save money.** Routing needs a fallback chain: if the chosen provider errors or times out, escalate rather than fail. A cost optimizer that adds an availability incident is a bad trade at any price.

**Never log prompt contents by default.** `routing_decisions` stores token counts, cost and latency — not the prompt. Customers send production data through this; logging it by default makes the product unadoptable at exactly the companies that would pay for it.

**Measure the overhead honestly.** A router that saves 30% on API spend but adds 200ms is a bad trade for interactive workloads. Target under 100ms for the routing decision, and publish the real number.

## License

MIT
