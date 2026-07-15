"""Shared result type + backend registry for optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class OptimizerResult:
    """Backend-agnostic optimization result."""

    x_best: list[float]
    f_best: float
    n_evaluations: int
    history: dict[str, list] = field(default_factory=dict)  # e.g. {"f": [...], "x": [...]}
    raw: Any = None  # backend-specific extra info, kept for debugging/plotting


# Signature every backend implements:
#   optimize(cf, eqs, ineqs, x0, lb, ub, constants, options) -> OptimizerResult
OptimizerFn = Callable[..., OptimizerResult]
