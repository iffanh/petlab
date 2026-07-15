"""Default optimizer: the derivative-free trust-region filter method
("DFTR" in the thesis; ``py_trsqp``), model-based and typically needing far
fewer simulations than the other backends -- see
``petlab.optimization.extras`` for the rest.

``py_trsqp`` is developed separately from this repo and its API has moved on
since the numbers in most existing config JSON files
(``constants``/``options``) were tuned: the class is now ``DFTRFilter`` (not
``TrustRegionSQPFilter``), constants are a typed ``Constants`` dataclass with
different field names, and points are tracked in normalized coordinates
internally. ``_translate_constants`` maps what it can from the old
``gamma_0``/``eta_1``/``L_threshold``-style keys onto the new
``gamma0``/``epsilon_tol``/``lambda_th``-style ones so existing configs keep
working, rather than raising on an upstream rename; anything with no clear
new-side equivalent is dropped (there's no way to re-derive a value that
represents a different algorithmic step altogether).
"""

from __future__ import annotations

from typing import Callable

from petlab.config import OptimizationConfig
from petlab.optimization.base import OptimizerResult

# old key -> new key. Old keys with no equivalent step in the current
# algorithm (`gamma_2`, `eta_1`, `kappa_vartheta`, `kappa_radius`,
# `kappa_mu`, `kappa_tmd`) are intentionally left out.
_CONSTANTS_RENAME = {
    "gamma_0": "gamma0",
    "gamma_1": "gamma1",
    "eta_2": "eta2",
    "mu": "mu",
    "gamma_vartheta": "gamma_vartheta",
    "init_radius": "init_radius",
    "stopping_radius": "min_radius",
    "L_threshold": "lambda_th",
}


def _translate_constants(constants: dict) -> dict:
    return {_CONSTANTS_RENAME[k]: v for k, v in constants.items() if k in _CONSTANTS_RENAME}


def optimize(
    cf: Callable,
    eqs: list[Callable],
    ineqs: list[Callable],
    x0: list[float],
    lb: list[float],
    ub: list[float],
    config: OptimizationConfig,
) -> OptimizerResult:
    from py_trsqp import DFTRFilter

    dftr = DFTRFilter(x0, cf=cf, ub=ub, lb=lb, eqcs=list(eqs), ineqcs=list(ineqs), constants=_translate_constants(config.constants))
    dftr.optimize(max_iter=config.max_iter, budget=config.options.get("budget"))

    # `IterationRecord.x` (despite the internal parameter being named `xn`)
    # is already stored denormalized, in the original x0/lb/ub scale -- do
    # NOT run it through `problem.denormalize()` again (verified directly
    # against the installed package; an earlier version of this wrapper got
    # this wrong and produced badly out-of-bounds "best" points).
    x_best = list(dftr.best.x)

    history = {
        "f": [r.f for r in dftr.records],
        "x": [list(r.x) for r in dftr.records],
        "vartheta": [r.vartheta for r in dftr.records],
        "iteration_type": [r.it_type for r in dftr.records],
    }

    return OptimizerResult(
        x_best=x_best,
        f_best=dftr.best.f,
        n_evaluations=dftr.problem.n_cf_evals,
        history=history,
        raw=dftr,
    )
