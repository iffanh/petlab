"""Optional optimizer backends.

These are kept for people who want to compare against DFTR, but are not the
default and are not required by ``requirements.txt`` -- most of them need
their own (sometimes non-pip-installable) package, which used to be silently
missing from this repo's dependency list. Each backend now fails with a
clear message telling you what to install, instead of an ``ImportError``
deep inside a 1000-line file.
"""

from __future__ import annotations

import numpy as np

from petlab.config import OptimizationConfig
from petlab.optimization.base import OptimizerResult


def _require(module_name: str, pip_name: str):
    try:
        return __import__(module_name)
    except ImportError as e:
        raise ImportError(f"The {module_name!r} optimizer needs the optional '{pip_name}' package: pip install {pip_name}") from e


def optimize_cobyla(cf, eqs, ineqs, x0, lb, ub, config: OptimizationConfig) -> OptimizerResult:
    """COBYLA (scipy) -- linear-approximation trust region, no gradients."""
    from scipy.optimize import NonlinearConstraint, minimize

    constraints = [NonlinearConstraint(eq, 0, 0) for eq in eqs] + [NonlinearConstraint(ineq, 0, np.inf) for ineq in ineqs]
    bounds = list(zip(lb, ub))

    history = {"f": [], "x": []}

    def callback(x):
        history["f"].append(cf(x))
        history["x"].append(list(x))

    result = minimize(
        cf, x0, method="COBYLA", constraints=constraints, bounds=bounds,
        options={"maxiter": config.options.get("budget", 1000)}, callback=callback,
    )
    return OptimizerResult(x_best=list(result.x), f_best=float(result.fun), n_evaluations=result.nfev, history=history, raw=result)


def optimize_cobyqa(cf, eqs, ineqs, x0, lb, ub, config: OptimizationConfig) -> OptimizerResult:
    """COBYQA -- quadratic-approximation successor to COBYLA."""
    cobyqa = _require("cobyqa", "cobyqa")
    from scipy.optimize import NonlinearConstraint

    constraints = [NonlinearConstraint(eq, 0, 0) for eq in eqs] + [NonlinearConstraint(ineq, 0, np.inf) for ineq in ineqs]
    bounds = list(zip(lb, ub))

    history = {"f": [], "x": []}

    def callback(x):
        history["f"].append(cf(x))
        history["x"].append(list(x))

    c = config.constants
    result = cobyqa.minimize(
        cf, x0, constraints=constraints, bounds=bounds,
        options={
            "maxfev": config.options.get("budget", 1000),
            "low_ratio": c.get("eta_1", 0.1),
            "high_ratio": c.get("eta_2", 0.7),
            "very_low_ratio": 1e-16,
            "decrease_radius_factor": c.get("gamma_0", 0.5),
            "increase_radius_factor": c.get("gamma_2", 2.0),
            "store_history": True,
            "radius_init": c.get("init_radius", 1.0),
            "radius_final": c.get("stopping_radius", 1e-6),
            "nb_points": 2 * len(x0) + 1,
            "scale": True,
        },
        callback=callback,
    )
    return OptimizerResult(
        x_best=list(result.x), f_best=float(result.fun), n_evaluations=result.nfev,
        history={"f": list(result.fun_history), "x": history["x"], "maxcv": list(result.maxcv_history)}, raw=result,
    )


def optimize_nomad(cf, eqs, ineqs, x0, lb, ub, config: OptimizationConfig) -> OptimizerResult:
    """NOMAD -- mesh adaptive direct search (needs the PyNomad bindings,
    see https://github.com/bbopt/nomad)."""
    PyNomad = _require("PyNomad", "PyNomad")

    def bb(x):
        rawBBO = f"{cf(x)} " + " ".join(str(-ineq(x)) for ineq in ineqs)
        x.setBBO(rawBBO.encode("UTF-8"))
        return 1

    params = [
        f"DIMENSION {len(x0)}",
        f"BB_OUTPUT_TYPE OBJ {'EB ' * len(ineqs)}",
        f"MAX_BB_EVAL {config.options.get('budget', 1000)}",
        "DISPLAY_DEGREE 2",
        "DISPLAY_ALL_EVAL true",
        "DISPLAY_STATS BBE BBO",
    ]
    result = PyNomad.optimize(bb, x0, lb, ub, params)

    return OptimizerResult(x_best=list(result["x_best"]), f_best=result["f_best"], n_evaluations=result["nb_evals"], raw=result)


def optimize_bo(cf, eqs, ineqs, x0, lb, ub, config: OptimizationConfig) -> OptimizerResult:
    """Bayesian Optimization (needs the 'bayesian-optimization' package)."""
    bayes_opt = _require("bayes_opt", "bayesian-optimization")
    from scipy.optimize import NonlinearConstraint

    pbounds = {f"x_{i}": (lb[i], ub[i]) for i in range(len(x0))}

    def as_array(kwargs):
        return np.array([kwargs[f"x_{i}"] for i in range(len(x0))])

    def constraint_fn(**kwargs):
        return np.array([ineq(as_array(kwargs)) for ineq in ineqs])

    constraint = NonlinearConstraint(constraint_fn, np.zeros(len(ineqs)), np.full(len(ineqs), np.inf)) if ineqs else None

    optimizer = bayes_opt.BayesianOptimization(
        f=lambda **kwargs: -cf(as_array(kwargs)), constraint=constraint, pbounds=pbounds, verbose=1, random_state=1
    )
    optimizer.maximize(init_points=len(x0), n_iter=config.options.get("budget", 100))

    x_best = [optimizer.max["params"][f"x_{i}"] for i in range(len(x0))]
    return OptimizerResult(x_best=x_best, f_best=-optimizer.max["target"], n_evaluations=len(optimizer.res), history={"x": [list(as_array(r["params"])) for r in optimizer.res], "f": [-r["target"] for r in optimizer.res]}, raw=optimizer)


def optimize_stosag(cf, eqs, ineqs, x0, lb, ub, config: OptimizationConfig) -> OptimizerResult:
    """StoSAG needs a fundamentally different evaluator (per-ensemble-member
    controls plus a well-covariance structure, not a single scalar
    objective per control vector), so it doesn't fit this module's
    ``optimize(cf, eqs, ineqs, x0, lb, ub, config)`` interface, and it
    depends on a private ``stosag`` package that isn't published on PyPI. The
    original, working implementation (tied directly to
    ``optimize_ensemble.cost_function_stosag``) is preserved in git history
    (see ``git log --all --full-history -- src/optimize_ensemble.py`` before
    this refactor) if you need to resurrect it.
    """
    raise NotImplementedError(
        "STOSAG is not available through the standard optimizer interface; "
        "see this function's docstring for how to find the previous implementation."
    )


BACKENDS = {
    "COBYLA": optimize_cobyla,
    "COBYQA": optimize_cobyqa,
    "NOMAD": optimize_nomad,
    "BO": optimize_bo,
    "STOSAG": optimize_stosag,
}
