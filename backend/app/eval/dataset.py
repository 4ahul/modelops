"""Eval sets: the test data routing decisions are justified by.

An eval set is a list of ``(input, expected)`` pairs with optional tags and a
per-example grader override. Loadable from JSONL so it lives in the customer's
repository next to their code, reviewable in a pull request, rather than in a
database nobody diffs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalExample:
    """One graded case.

    Args:
        input: The prompt to send.
        expected: What a correct answer looks like. Interpretation depends on
            the grader — exact text, a substring, or a JSON schema.
        tags: Free-form labels, for slicing a report by subset.
        grader: Overrides the eval set's default grader for this example.
        weight: Relative importance when averaging. A rare-but-critical case can
            be weighted above a common one.
        id: Stable identifier, so a regression can be traced to a case.
    """

    input: str
    expected: Any = None
    tags: tuple[str, ...] = ()
    grader: str | None = None
    weight: float = 1.0
    id: str = ""

    def __post_init__(self) -> None:
        if not self.input:
            raise ValueError("EvalExample.input cannot be empty")
        if self.weight <= 0:
            raise ValueError(f"EvalExample.weight must be positive, got {self.weight}")


@dataclass
class EvalSet:
    """A named, versioned collection of examples.

    Args:
        name: Identifies the set in reports and in the database.
        examples: The cases.
        grader: Default grader name for examples that do not override it.
        version: Bumped when examples change, so an old result is never
            compared against a new set as though they measured the same thing.
        task_type: The routing task this set is evidence for. Quality scores are
            per task type; a set that measures classification says nothing about
            drafting.
    """

    name: str
    examples: list[EvalExample] = field(default_factory=list)
    grader: str = "exact_match"
    version: int = 1
    task_type: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError(f"EvalSet {self.name!r} has no examples")
        # Fill in ids so a report can name a failing case even when the source
        # file did not supply one.
        for index, example in enumerate(self.examples):
            if not example.id:
                object.__setattr__(example, "id", f"{self.name}-{index}")

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[EvalExample]:
        return iter(self.examples)

    @property
    def total_weight(self) -> float:
        return sum(e.weight for e in self.examples)

    def filter(self, *, tag: str | None = None) -> EvalSet:
        """A new set containing only examples carrying ``tag``."""
        selected = [e for e in self.examples if tag is None or tag in e.tags]
        if not selected:
            raise ValueError(f"No examples in {self.name!r} carry tag {tag!r}")
        return EvalSet(
            name=f"{self.name}[{tag}]",
            examples=selected,
            grader=self.grader,
            version=self.version,
            task_type=self.task_type,
            description=self.description,
        )

    def sample(self, n: int, *, seed: int = 0) -> EvalSet:
        """A deterministic subset, for a quick smoke run.

        Seeded so two runs compare like with like — a random subset would make
        every re-run a different measurement.
        """
        import random

        if n >= len(self.examples):
            return self
        rng = random.Random(seed)
        selected = rng.sample(self.examples, n)
        return EvalSet(
            name=f"{self.name}[sample={n}]",
            examples=selected,
            grader=self.grader,
            version=self.version,
            task_type=self.task_type,
        )

    # ------------------------------------------------------------ loading

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        grader: str = "exact_match",
        task_type: str | None = None,
    ) -> EvalSet:
        """Load from newline-delimited JSON.

        Each line is an object with at least ``input``. ``expected``, ``tags``,
        ``grader``, ``weight`` and ``id`` are optional::

            {"input": "Is this spam? ...", "expected": "spam", "tags": ["short"]}

        Errors name the line number, because a 500-example file with one typo is
        otherwise painful to debug.
        """
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"Eval set not found: {file}")

        examples: list[EvalExample] = []
        for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file}:{lineno}: invalid JSON: {exc.msg}") from None
            if not isinstance(row, dict):
                raise ValueError(
                    f"{file}:{lineno}: expected a JSON object, got {type(row).__name__}"
                )
            if "input" not in row:
                raise ValueError(f"{file}:{lineno}: missing required field 'input'")
            examples.append(
                EvalExample(
                    input=row["input"],
                    expected=row.get("expected"),
                    tags=tuple(row.get("tags", ())),
                    grader=row.get("grader"),
                    weight=float(row.get("weight", 1.0)),
                    id=str(row.get("id", "")),
                )
            )

        if not examples:
            raise ValueError(f"{file} contains no examples")
        return cls(
            name=name or file.stem,
            examples=examples,
            grader=grader,
            task_type=task_type,
        )

    @classmethod
    def from_records(
        cls, name: str, records: Sequence[dict[str, Any]], *, grader: str = "exact_match"
    ) -> EvalSet:
        """Build from already-parsed dicts — the API's ingestion path."""
        return cls(
            name=name,
            examples=[
                EvalExample(
                    input=r["input"],
                    expected=r.get("expected"),
                    tags=tuple(r.get("tags", ())),
                    grader=r.get("grader"),
                    weight=float(r.get("weight", 1.0)),
                    id=str(r.get("id", "")),
                )
                for r in records
            ],
            grader=grader,
        )

    def to_jsonl(self, path: str | Path) -> int:
        """Write the set back out. Returns the number of lines written."""
        lines = [
            json.dumps(
                {
                    "id": e.id,
                    "input": e.input,
                    "expected": e.expected,
                    "tags": list(e.tags),
                    **({"grader": e.grader} if e.grader else {}),
                    **({"weight": e.weight} if e.weight != 1.0 else {}),
                },
                default=str,
            )
            for e in self.examples
        ]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return len(lines)


__all__ = ["EvalExample", "EvalSet"]
