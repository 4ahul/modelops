"""Graders: turn a model's output into a score between 0 and 1.

The grader is the load-bearing part of the whole product. A routing decision is
only as trustworthy as the number that justifies it, so each grader here is
explicit about what it measures and where it lies:

``exact_match``
    Strict after normalisation. Honest, and brittle on free text.

``contains``
    Substring. Good for classification labels, blind to a hedged answer that
    contains the label and then contradicts it.

``fuzzy``
    Character-level similarity. Produces a plausible number for any pair of
    strings, which makes it the easiest way to fool yourself.

``json_match`` / ``json_schema``
    Structural. The right choice for extraction tasks, and the only ones here
    that are robust to formatting noise.

``regex``
    Pattern match, for outputs with a defined shape.

``numeric``
    Parses a number and compares within a tolerance.

``llm_judge``
    Another model scores the answer. Necessary for open-ended output, and
    expensive: it costs an extra call per example and inherits the judge's bias.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class GradeResult:
    """A score with the reason behind it.

    The explanation exists so a 0.0 in a report can be understood without
    re-running the eval by hand.
    """

    score: float
    passed: bool
    explanation: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


class Grader(Protocol):
    """Callable that scores one output against one expectation."""

    def __call__(self, output: str, expected: Any) -> GradeResult: ...


def _normalise(text: str) -> str:
    """Collapse whitespace and case, strip surrounding punctuation.

    Applied before comparison so a model that answers ``"Spam."`` is not marked
    wrong against ``"spam"`` — that difference is formatting, not accuracy.
    """
    return re.sub(r"\s+", " ", str(text)).strip().strip(".,;:!\"'").lower()


def exact_match(output: str, expected: Any) -> GradeResult:
    """1.0 only if the normalised strings are identical."""
    if expected is None:
        return GradeResult(0.0, False, "no expected value supplied")
    hit = _normalise(output) == _normalise(expected)
    return GradeResult(
        1.0 if hit else 0.0,
        hit,
        "" if hit else f"expected {_normalise(expected)!r}, got {_normalise(output)[:120]!r}",
    )


def contains(output: str, expected: Any) -> GradeResult:
    """1.0 if the expected text appears anywhere in the output."""
    if expected is None:
        return GradeResult(0.0, False, "no expected value supplied")
    hit = _normalise(expected) in _normalise(output)
    return GradeResult(
        1.0 if hit else 0.0, hit, "" if hit else f"{_normalise(expected)!r} not found in output"
    )


def fuzzy(output: str, expected: Any, *, threshold: float = 0.85) -> GradeResult:
    """Character-similarity ratio, passing above ``threshold``.

    The raw ratio is the score, so a report shows how close a near-miss was
    rather than flattening everything to pass/fail.
    """
    if expected is None:
        return GradeResult(0.0, False, "no expected value supplied")
    ratio = SequenceMatcher(None, _normalise(output), _normalise(expected)).ratio()
    return GradeResult(ratio, ratio >= threshold, f"similarity {ratio:.3f}")


def regex_match(output: str, expected: Any) -> GradeResult:
    """1.0 if the output matches the expected pattern."""
    if not expected:
        return GradeResult(0.0, False, "no pattern supplied")
    try:
        hit = re.search(str(expected), output, re.IGNORECASE | re.DOTALL) is not None
    except re.error as exc:
        return GradeResult(0.0, False, f"invalid pattern: {exc}")
    return GradeResult(1.0 if hit else 0.0, hit, "" if hit else "pattern did not match")


def numeric(output: str, expected: Any, *, tolerance: float = 1e-6) -> GradeResult:
    """Parse the first number in the output and compare within ``tolerance``.

    Relative tolerance above 1, absolute below, so ``1_000_000`` is not judged
    by the same yardstick as ``0.5``.
    """
    if expected is None:
        return GradeResult(0.0, False, "no expected value supplied")
    found = re.search(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", str(output))
    if not found:
        return GradeResult(0.0, False, "no number found in output")
    try:
        actual = float(found.group().replace(",", ""))
        target = float(expected)
    except (TypeError, ValueError):
        return GradeResult(0.0, False, f"could not parse {expected!r} as a number")
    scale = max(abs(target), 1.0)
    close = abs(actual - target) <= tolerance * scale
    return GradeResult(1.0 if close else 0.0, close, f"got {actual}, expected {target}")


def _extract_json(text: str) -> Any:
    """Pull a JSON value out of a possibly chatty response.

    Models wrap JSON in prose and fences even when told not to. Failing an
    otherwise-correct extraction because of a ```json fence would measure
    instruction-following, not extraction accuracy.
    """
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in output")


def json_match(output: str, expected: Any) -> GradeResult:
    """Compare parsed JSON, scoring partial credit on dict key overlap.

    Key order and whitespace are irrelevant; a dict that gets three of four
    fields right scores 0.75 rather than 0, because "wrong on one field" and
    "wrong on everything" are different failures.
    """
    if expected is None:
        return GradeResult(0.0, False, "no expected value supplied")
    try:
        actual = _extract_json(output)
    except ValueError as exc:
        return GradeResult(0.0, False, str(exc))

    target = expected
    if isinstance(target, str):
        try:
            target = _extract_json(target)
        except ValueError:
            return GradeResult(0.0, False, "expected value is not valid JSON")

    if actual == target:
        return GradeResult(1.0, True, "")

    if isinstance(actual, dict) and isinstance(target, dict):
        if not target:
            return GradeResult(0.0, False, "expected an empty object")
        correct = sum(1 for k, v in target.items() if k in actual and actual[k] == v)
        score = correct / len(target)
        missing = [k for k in target if k not in actual]
        wrong = [k for k in target if k in actual and actual[k] != target[k]]
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if wrong:
            detail.append(f"wrong {wrong}")
        return GradeResult(score, score == 1.0, "; ".join(detail))

    return GradeResult(0.0, False, f"expected {target!r}, got {actual!r}")


def json_schema(output: str, expected: Any) -> GradeResult:
    """Check the output parses and carries the required keys and types.

    ``expected`` is ``{"required": [...], "types": {"field": "str"}}``. This
    validates shape, not values — the right grader when a task must produce
    well-formed structure and the content varies legitimately.
    """
    try:
        actual = _extract_json(output)
    except ValueError as exc:
        return GradeResult(0.0, False, str(exc))
    if not isinstance(expected, dict):
        return GradeResult(0.0, False, "json_schema expects a schema object")
    if not isinstance(actual, dict):
        return GradeResult(0.0, False, f"output is {type(actual).__name__}, not an object")

    required = list(expected.get("required", []))
    types: dict[str, str] = dict(expected.get("types", {}))
    checks = len(required) + len(types)
    if checks == 0:
        return GradeResult(1.0, True, "empty schema; parsed successfully")

    problems: list[str] = []
    passed = 0
    for key in required:
        if key in actual:
            passed += 1
        else:
            problems.append(f"missing {key!r}")

    type_map: dict[str, type | tuple[type, ...]] = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": (int, float),
        "number": (int, float),
        "bool": bool,
        "boolean": bool,
        "list": list,
        "array": list,
        "dict": dict,
        "object": dict,
    }
    for key, type_name in types.items():
        wanted = type_map.get(type_name.lower())
        if wanted is None:
            problems.append(f"unknown type {type_name!r} in schema")
            continue
        if key in actual and isinstance(actual[key], wanted):
            passed += 1
        else:
            problems.append(f"{key!r} is not {type_name}")

    score = passed / checks
    return GradeResult(score, score == 1.0, "; ".join(problems))


#: Async graders take a judge callable, so they cannot share the sync signature.
AsyncGrader = Callable[[str, Any], Awaitable[GradeResult]]

_JUDGE_PROMPT = """You are grading a model's answer against a reference answer.

Question:
{question}

Reference answer:
{reference}

Model's answer:
{answer}

Reply with a single digit 0-5 and nothing else, where 0 means completely wrong \
and 5 means equivalent to the reference. Judge meaning, not wording."""


def make_llm_judge(
    judge_call: Callable[[str], Awaitable[str]],
    *,
    threshold: float = 0.8,
) -> Callable[[str, Any, str], Awaitable[GradeResult]]:
    """Build an LLM-as-judge grader over a completion callable.

    The judge is injected rather than constructed here, so the eval runner
    controls which model grades and the tests can grade deterministically.

    A 0–5 integer scale is used rather than asking for a float: models are
    markedly more consistent on a small ordinal scale, and an unparseable reply
    scores 0 with the reason recorded rather than silently becoming 0.5.
    """

    async def grade(output: str, expected: Any, question: str = "") -> GradeResult:
        if expected is None:
            return GradeResult(0.0, False, "no reference answer supplied")
        prompt = _JUDGE_PROMPT.format(
            question=question or "(not supplied)", reference=expected, answer=output
        )
        try:
            verdict = await judge_call(prompt)
        except Exception as exc:
            # A judge failure is not a model failure. Scoring 0 here would
            # record a quality regression that never happened.
            log.warning("llm_judge_failed", error=str(exc))
            return GradeResult(0.0, False, f"judge unavailable: {exc}")

        found = re.search(r"[0-5]", verdict)
        if not found:
            return GradeResult(0.0, False, f"unparseable judge reply: {verdict[:80]!r}")
        score = int(found.group()) / 5
        return GradeResult(score, score >= threshold, f"judge scored {found.group()}/5")

    return grade


#: Graders addressable by name from an eval set or an API request.
GRADERS: dict[str, Grader] = {
    "exact_match": exact_match,
    "contains": contains,
    "fuzzy": fuzzy,
    "regex": regex_match,
    "numeric": numeric,
    "json_match": json_match,
    "json_schema": json_schema,
}


def get_grader(name: str) -> Grader:
    """Look up a grader by name.

    Raises:
        KeyError: listing the valid names. ``llm_judge`` is excluded because it
            needs a judge callable and is wired up by the runner.
    """
    try:
        return GRADERS[name]
    except KeyError:
        known = ", ".join(sorted(GRADERS))
        raise KeyError(
            f"Unknown grader {name!r}. Available: {known}. "
            "For llm_judge, set judge_model on the eval run."
        ) from None


__all__ = [
    "GRADERS",
    "GradeResult",
    "Grader",
    "contains",
    "exact_match",
    "fuzzy",
    "get_grader",
    "json_match",
    "json_schema",
    "make_llm_judge",
    "numeric",
    "regex_match",
]
