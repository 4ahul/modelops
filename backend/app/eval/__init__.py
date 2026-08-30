"""Evaluation framework.

Built before the router, not after. Routing scores multiply a quality term;
without evals that term is a guess, and a plausible-looking router optimising
against a made-up number is worse than no router at all.
"""

from __future__ import annotations

from app.eval.dataset import EvalExample, EvalSet
from app.eval.graders import GRADERS, GradeResult, get_grader
from app.eval.runner import EvalReport, EvalRunner, ExampleResult, ModelReport

__all__ = [
    "GRADERS",
    "EvalExample",
    "EvalReport",
    "EvalRunner",
    "EvalSet",
    "ExampleResult",
    "GradeResult",
    "ModelReport",
    "get_grader",
]
