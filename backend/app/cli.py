"""Command-line interface: ``modelops <command>``.

Commands:

``hash-key``
    Turn an API key into the hash to put in ``API_KEY_HASHES``. Plaintext keys
    are never configured or stored.

``pricing``
    Print the pricing table and its age. A stale table mis-routes every request
    while looking perfectly healthy, so its age is a first-class thing to check.

``check-config``
    Validate the environment for a target deployment, without starting a server.

``migrate``
    Run Alembic migrations.

``eval``
    Run an eval set from a JSONL file against the configured models and print the
    comparison table. No server or database required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from app import __version__
from app.core.config import Settings
from app.eval.dataset import EvalSet
from app.eval.runner import EvalRunner
from app.providers.pricing import MODEL_CATALOG, PRICING_AS_OF, pricing_age_days
from app.providers.registry import ProviderRegistry


def _cmd_hash_key(args: argparse.Namespace) -> int:
    from app.api.auth import hash_key

    print(hash_key(args.key))
    return 0


def _cmd_pricing(args: argparse.Namespace) -> int:
    age = pricing_age_days()
    if args.json:
        print(
            json.dumps(
                {
                    "as_of": PRICING_AS_OF.isoformat(),
                    "age_days": age,
                    "models": {
                        spec.id: {
                            "provider": spec.provider,
                            "model": spec.model,
                            "input_per_mtok": spec.pricing.input_per_mtok,
                            "output_per_mtok": spec.pricing.output_per_mtok,
                            "context_window": spec.context_window,
                        }
                        for spec in MODEL_CATALOG.values()
                    },
                },
                indent=2,
            )
        )
        return 0

    print(f"Pricing verified {PRICING_AS_OF.isoformat()} ({age} days ago)")
    if age > 90:
        print("  WARNING: stale. Every cost and routing decision derives from this table.")
    print()
    print(f"{'MODEL':<16} {'PROVIDER':<11} {'$/Mtok IN':>10} {'$/Mtok OUT':>11} {'CONTEXT':>10}")
    print("-" * 62)
    for spec in sorted(MODEL_CATALOG.values(), key=lambda s: s.blended_cost_per_mtok):
        print(
            f"{spec.id:<16} {spec.provider:<11} {spec.pricing.input_per_mtok:>10.3f} "
            f"{spec.pricing.output_per_mtok:>11.3f} {spec.context_window:>10,}"
        )
    return 0


def _cmd_check_config(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.production:
        settings = Settings(environment="production")

    print(f"environment:  {settings.environment}")
    print(f"database:     {_redact_url(settings.database_url)}")
    print(f"redis:        {_redact_url(settings.redis_url)}")
    print(f"providers:    {sorted(settings.configured_providers()) or '(none)'}")
    print(f"api keys:     {len(settings.api_key_hash_set)} configured")
    print(f"rate limit:   {settings.rate_limit_per_minute}/min per key")
    print(f"store prompts:{settings.store_prompts}")
    print(f"pricing age:  {pricing_age_days()} days")

    problems = settings.validate_for_production()
    if problems:
        print("\nProblems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nConfiguration OK")
    return 0


def _redact_url(url: str) -> str:
    """Strip credentials before printing a connection string."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}"


def _cmd_migrate(args: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    if not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = Settings().database_url
    config = Config(args.config)
    command.upgrade(config, args.revision)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    settings = Settings()
    registry = ProviderRegistry.from_settings(settings)
    if not len(registry):
        print(
            "No provider API keys configured. Set at least one of ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY or GOOGLE_API_KEY.",
            file=sys.stderr,
        )
        return 1

    eval_set = EvalSet.from_jsonl(args.path, grader=args.grader, task_type=args.task_type)
    if args.sample:
        eval_set = eval_set.sample(args.sample)

    runner = EvalRunner(registry, concurrency=settings.eval_concurrency, max_tokens=args.max_tokens)
    report = asyncio.run(runner.run(eval_set, model_ids=args.models or None))

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        report.print_table()

    if args.min_accuracy is not None:
        best = report.best_by_accuracy()
        if best is None or best.accuracy < args.min_accuracy:
            actual = f"{best.accuracy:.1%}" if best else "no result"
            print(
                f"FAIL: no model reached {args.min_accuracy:.1%} (best: {actual})",
                file=sys.stderr,
            )
            return 1
        print(f"\nOK: {best.model_id} reached {best.accuracy:.1%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelops", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"modelops {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    hash_cmd = sub.add_parser("hash-key", help="hash an API key for API_KEY_HASHES")
    hash_cmd.add_argument("key")
    hash_cmd.set_defaults(func=_cmd_hash_key)

    pricing = sub.add_parser("pricing", help="show the pricing table and its age")
    pricing.add_argument("--json", action="store_true")
    pricing.set_defaults(func=_cmd_pricing)

    check = sub.add_parser("check-config", help="validate the environment")
    check.add_argument(
        "--production", action="store_true", help="apply production rules regardless of ENVIRONMENT"
    )
    check.set_defaults(func=_cmd_check_config)

    migrate = sub.add_parser("migrate", help="run database migrations")
    migrate.add_argument("--config", default="alembic.ini")
    migrate.add_argument("--revision", default="head")
    migrate.set_defaults(func=_cmd_migrate)

    ev = sub.add_parser("eval", help="run an eval set from JSONL against configured models")
    ev.add_argument("path", help="path to a JSONL eval set")
    ev.add_argument("--grader", default="exact_match")
    ev.add_argument("--task-type", default=None)
    ev.add_argument("--models", nargs="*", default=None, help="model ids; defaults to all")
    ev.add_argument("--max-tokens", type=int, default=1024)
    ev.add_argument("--sample", type=int, default=None, help="run a deterministic subset")
    ev.add_argument("--json", action="store_true")
    ev.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="exit 1 unless some model reaches this accuracy (for CI)",
    )
    ev.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
