"""Optimizer backend registry.

``DFTR`` (derivative-free trust-region SQP-filter) is the default. Everything
else lives in :mod:`petlab.optimization.extras` and is looked up lazily so
that importing this package never requires the optional packages those
backends need.
"""

from __future__ import annotations

from typing import Callable

from petlab.config import OptimizationConfig
from petlab.optimization.base import OptimizerResult
from petlab.optimization import dftr

DEFAULT_BACKEND = "DFTR"


def get_optimizer(name: str) -> Callable:
    if name == "DFTR":
        return dftr.optimize
    from petlab.optimization import extras

    try:
        return extras.BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown optimizer {name!r}. Available: DFTR, {', '.join(extras.BACKENDS)}") from None


def optimize(
    cf: Callable,
    eqs: list[Callable],
    ineqs: list[Callable],
    x0: list[float],
    lb: list[float],
    ub: list[float],
    config: OptimizationConfig,
) -> OptimizerResult:
    """Dispatch to ``config.optimizer``'s backend."""
    return get_optimizer(config.optimizer)(cf, eqs, ineqs, x0, lb, ub, config)
