"""Objective (NPV / cashflow / immobile CO2) and constraint evaluation.

This is the "black box" from the thesis's Fig. 1.2: given a control vector
``x``, run the ensemble and return the objective plus equality/inequality
constraint values, all as robustness measures over the ensemble (expected
value by default; other measures -- percentile/min/max -- are configured
per-constraint, matching ``ci(x) = R^ci[Ci(x, w)]``).

Ported from ``optimize_ensemble.py``'s ``calculate_npv``/``cost_function``,
with the one-off CCS-study hacks (a CSV path loaded at import time, hardcoded
control-index pairs) turned into a declarative, config-driven "snap controls
to the nearest valid location" hook (``snap_controls_to_valid_locations``)
instead of being hardcoded into the general evaluator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from petlab import ensemble
from petlab.config import ControlSpec, EnsembleConfig
from petlab.deck import DeckParser


# ---------------------------------------------------------------------------
# Units / cost functions
# ---------------------------------------------------------------------------

def get_unit(realization_path: str) -> str:
    is_metric, _ = DeckParser().keyword_search(realization_path, keyword="METRIC")
    return "METRIC" if is_metric else "FIELD"


NPV_PRICES = {
    "FIELD": dict(oil=80.0, gas=1.5, water_prod=10.0, water_inj=5.0, gas_inj=2.0),
    # $/sm3, converted from the FIELD $/stb, $/Mscf prices via 1 stb = 0.15899 sm3,
    # 1 Mscf = 28.316 sm3 (same conversion the original code used)
    "METRIC": dict(oil=503.185, gas=0.0529, water_prod=62.9, water_inj=31.5, gas_inj=0.353),
}


def calculate_npv(summary: dict[str, dict[str, str]], is_success: list[bool], unit: str, discount_rate: float = 0.05) -> np.ndarray:
    """Per-realization NPV time series (``time x realization``), discounted
    daily. Any of FOPR/FWPR/FGPR/FGIR/FWIR missing from ``summary`` is
    treated as zero (some decks don't declare gas vectors, for instance)."""

    prices = NPV_PRICES[unit]
    real_names = list(summary.keys())

    tN = min(len(np.load(summary[r]["YEARS"])) for r in real_names)
    years_ref = np.load(summary[real_names[0]]["YEARS"])[:tN]

    npv_arr = []
    for real_name in real_names:
        years = np.load(summary[real_name]["YEARS"])
        cashflow = np.zeros_like(years)

        for key, price, sign in [
            ("FOPR", prices["oil"], +1),
            ("FWPR", prices["water_prod"], -1),
            ("FGPR", prices["gas"], +1),
            ("FGIR", prices["gas_inj"], -1),
            ("FWIR", prices["water_inj"], -1),
        ]:
            if key in summary[real_name]:
                cashflow = cashflow + sign * price * np.load(summary[real_name][key])

        dy = np.diff(years, prepend=0)
        cashflow = cashflow * dy * 365

        tM = len(years)
        npv = []
        for tn in range(tN):
            tm = tM * tn / tN
            w1 = tm - np.floor(tm)
            w2 = np.ceil(tm) - tm
            if w1 == 0.0:
                npv_t = cashflow[int(np.floor(tm))] / (1 + discount_rate) ** tm
            else:
                npv_t = (w1 * cashflow[int(np.floor(tm))] + w2 * cashflow[int(np.ceil(tm))]) / (1 + discount_rate) ** tm
            npv.append(npv_t)
        npv_arr.append(npv)

    return np.array(npv_arr).T  # shape: (tN, n_realizations)


def calculate_cashflow_last(summary: dict[str, dict[str, str]], is_success: list[bool], unit: str) -> np.ndarray:
    """Total net cash flow at the end of the run, per realization (used by
    the CO2-storage case studies where CAPEX/tax terms are lump sums, not
    rates -- see ``optimize_ensemble.calculate_net_cash_flow`` for the
    original, case-specific constants)."""

    cf = []
    for real_name, keys in summary.items():
        fopt = np.load(keys["FOPT"])[-1] if "FOPT" in keys else 0.0
        fwpt = np.load(keys["FWPT"])[-1] if "FWPT" in keys else 0.0
        cf.append(fopt * NPV_PRICES[unit]["oil"] - fwpt * NPV_PRICES[unit]["water_prod"])
    return np.array(cf)


def calculate_immobile_co2(summary: dict[str, dict[str, str]], is_success: list[bool], unit: str) -> np.ndarray:
    """Total immobile CO2 (tonnes) at the end of the run, from the ``FCGMI``
    summary vector (see thesis Ch. 5.2 / Article V)."""

    values = []
    for real_name, keys in summary.items():
        fcgmi = np.load(keys["FCGMI"])[-1]
        values.append(fcgmi * 30 * 44 / 1000)  # sm3 -> tonne
    return np.array(values)


COST_FUNCTIONS = {
    "NPV": calculate_npv,
    "NetCashFlow-Last": calculate_cashflow_last,
    "ImmobileCO2": calculate_immobile_co2,
}


def robustness_measure(values: np.ndarray, robustness: dict) -> float:
    kind = robustness.get("type", "average")
    if kind == "average":
        return float(np.mean(values))
    elif kind == "percentile":
        return float(np.percentile(values, robustness["value"]))
    elif kind == "minimum":
        return float(np.min(values))
    elif kind == "maximum":
        return float(np.max(values))
    raise NotImplementedError(f"Robustness measure {kind!r} is not implemented")


def objective_value(config: EnsembleConfig, summary: dict[str, dict[str, str]], is_success: list[bool], unit: str) -> float:
    """The scalar value the optimizer minimizes: the negative robustness
    measure of the configured cost function (NPV etc. is to be *maximized*,
    so the sign is flipped for the minimizer)."""

    cost_fn = COST_FUNCTIONS[config.optimization.cost_function]
    values = cost_fn(summary, is_success, unit)
    if values.ndim == 2:  # a (time, realization) trajectory -> cumulative total per realization
        values = np.cumsum(values, axis=0)[-1, :]
    values = values[np.array(is_success)] if any(is_success) else values
    return -float(np.mean(values)) if len(values) else float("nan")


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def _pick_timestep(vector: np.ndarray, spec: dict) -> float:
    if spec["timestep"] == "all":
        return float(np.max(vector))
    elif spec["timestep"] == "last":
        return float(vector[-1])
    raise ValueError(f"'timestep' must be 'all' or 'last', found {spec['timestep']!r}")


def _summary_key_for(constraint_name: str, spec: dict) -> str:
    if "wellname" in spec:
        base = constraint_name.split(":")[0] if ":" in constraint_name else constraint_name
        return f"{base}:{spec['wellname']}"
    return constraint_name


def evaluate_inequality(constraint_name: str, spec: dict, summary: dict[str, dict[str, str]], is_success: list[bool]) -> float:
    """Evaluate one field/well summary-based inequality constraint (e.g.
    ``FWPT <= 1e6``, ``WWCT:PROD1 <= 0.95``), normalized so that ``>= 0``
    means feasible: ``(target - measured) / |target|``.

    ``FOPT``-style "must be at least this much" constraints are declared
    with ``spec["minimum"] = True`` and get the sign flipped before
    normalizing.
    """

    key = _summary_key_for(constraint_name, spec)
    values = []
    for real_name, keys in summary.items():
        vector = np.load(keys[key])
        val = _pick_timestep(vector, spec)
        values.append(-val if spec.get("minimum") else val)

    values = np.array(values)
    if any(is_success):
        values = values[np.array(is_success)]
    measured = robustness_measure(values, spec["robustness"])
    target = -spec["value"] if spec.get("minimum") else spec["value"]
    return (target - measured) / abs(target)


def evaluate_constraints(config: EnsembleConfig, summary: dict[str, dict[str, str]], is_success: list[bool]) -> tuple[list[float], list[float]]:
    """Evaluate every active constraint in ``config.optimization.constraints``.
    Returns ``(equalities, inequalities)``, each normalized so the feasible
    region is ``== 0`` / ``>= 0``."""

    equalities: list[float] = []
    inequalities: list[float] = []

    for name, spec in config.optimization.constraints.items():
        if not spec.get("is_active", True):
            continue

        value = evaluate_inequality(name, spec, summary, is_success)

        if spec["type"] == "equality":
            equalities.append(value)
        elif spec["type"] == "inequality":
            inequalities.append(value)
        else:
            raise ValueError(f"Constraint type must be 'equality' or 'inequality', found {spec['type']!r}")

    return equalities, inequalities


def count_constraints(config: EnsembleConfig) -> tuple[int, int]:
    n_eq = sum(1 for s in config.optimization.constraints.values() if s.get("is_active", True) and s["type"] == "equality")
    n_ineq = sum(1 for s in config.optimization.constraints.values() if s.get("is_active", True) and s["type"] == "inequality")
    return n_eq, n_ineq


# ---------------------------------------------------------------------------
# Optional control post-processing hook (replaces the old hardcoded
# well-location-snapping/index-pair CCS hack)
# ---------------------------------------------------------------------------

def snap_controls_to_valid_locations(controls: list[ControlSpec], pairs: list[list[int]], csv_path: str) -> None:
    """For each ``(i, j)`` in ``pairs``, snap ``(controls[i].default,
    controls[j].default)`` to the nearest row in the 2-column CSV at
    ``csv_path`` (a discrete well-location grid). Mutates ``controls`` in
    place.

    This is a generalization of a hardcoded 2-pair snap that used to live
    directly in ``optimize_ensemble.cost_function``, for well-placement
    problems (e.g. ``data/Egg_CCS``) where the optimizer works with
    continuous coordinates but the deck only accepts a fixed set of valid
    grid locations.
    """

    if not pairs:
        return

    locations = pd.read_csv(csv_path).to_numpy()
    for i, j in pairs:
        point = np.array([controls[i].default, controls[j].default])
        nearest = locations[np.argmin(np.linalg.norm(locations - point, axis=1))]
        controls[i].default, controls[j].default = float(nearest[0]), float(nearest[1])


# ---------------------------------------------------------------------------
# Putting it together: control vector -> objective + constraints
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    objective: float
    equalities: list[float]
    inequalities: list[float]
    is_success: list[bool]
    summary: dict[str, dict[str, str]] = field(default_factory=dict)


def evaluate(
    x: np.ndarray,
    config: EnsembleConfig,
    study: "ensemble.Study",
    simulator_path: str,
    simfolder_path: str,
) -> EvaluationResult:
    """Apply control vector ``x`` (in the order of ``config.controls``),
    simulate the ensemble, extract results, and compute the objective +
    constraints. This is the function every optimizer backend calls."""

    controls = [ControlSpec(name=c.name, default=float(v), kind=c.kind, lb=c.lb, ub=c.ub) for c, v in zip(config.controls, x)]

    snap = getattr(config.optimization, "control_snap_pairs", None)
    if snap:
        csv_path = config.optimization.options.get("control_snap_csv", "")
        snap_controls_to_valid_locations(controls, snap, csv_path)

    controls_dicts = [c.to_dict() for c in controls]
    n_eq, n_ineq = count_constraints(config)
    failed = EvaluationResult(objective=float("nan"), equalities=[float("nan")] * n_eq, inequalities=[float("nan")] * n_ineq, is_success=[])

    try:
        realizations, is_success = ensemble.simulate(study.base_realizations, simulator_path, controls_dicts, simfolder_path, config.n_parallel)
    except (RuntimeError, TimeoutError):
        return failed

    ensemble.apply_simulation_result(study, realizations, is_success)

    if not any(is_success):
        failed.is_success = is_success
        return failed

    ensemble.extract(study, simfolder_path)

    unit = get_unit(next(iter(study.realizations.values())))
    obj = objective_value(config, study.summary, is_success, unit)
    equalities, inequalities = evaluate_constraints(config, study.summary, is_success)

    return EvaluationResult(objective=obj, equalities=equalities, inequalities=inequalities, is_success=is_success, summary=study.summary)


def make_evaluator(config: EnsembleConfig, study: "ensemble.Study", simulator_path: str, simfolder_path: str):
    """Build ``(cf, eqs, ineqs)`` closures for an optimizer backend: ``cf(x)``
    is the objective, ``eqs``/``ineqs`` are lists of per-constraint
    functions. All three share one cache keyed on ``x`` so that evaluating
    the objective and every constraint for the same control vector only
    simulates the ensemble once (mirrors ``optimize_ensemble.py``'s
    ``@np_cache``-wrapped ``cost_function``, without requiring ``config``
    and ``study`` themselves to be hashable).
    """

    cache: dict[tuple, EvaluationResult] = {}

    def cached_evaluate(x) -> EvaluationResult:
        key = tuple(float(v) for v in x)
        if key not in cache:
            cache[key] = evaluate(np.array(key), config, study, simulator_path, simfolder_path)
        return cache[key]

    n_eq, n_ineq = count_constraints(config)
    cf = lambda x: cached_evaluate(x).objective
    eqs = [(lambda x, i=i: cached_evaluate(x).equalities[i]) for i in range(n_eq)]
    ineqs = [(lambda x, i=i: cached_evaluate(x).inequalities[i]) for i in range(n_ineq)]

    return cf, eqs, ineqs
