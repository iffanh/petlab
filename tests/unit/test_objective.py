"""Objective/constraint evaluation against fabricated .npy fixtures -- no
simulator needed, since extraction just means "read a .npy file"."""

import numpy as np

from petlab import objective


def _write_summary(tmp_path, real_name, years, **vectors):
    real_dir = tmp_path / real_name
    real_dir.mkdir(parents=True, exist_ok=True)
    paths = {"YEARS": str(real_dir / "YEARS.npy")}
    np.save(paths["YEARS"], years)
    for key, values in vectors.items():
        paths[key] = str(real_dir / f"{key}.npy")
        np.save(paths[key], values)
    return paths


def test_calculate_npv_is_positive_for_pure_oil_production(tmp_path):
    years = np.array([0.5, 1.0])
    summary = {
        "R_1": _write_summary(tmp_path, "R_1", years, FOPR=np.array([100.0, 100.0])),
        "R_2": _write_summary(tmp_path, "R_2", years, FOPR=np.array([50.0, 50.0])),
    }
    is_success = [True, True]

    npv = objective.calculate_npv(summary, is_success, unit="METRIC")

    assert npv.shape == (2, 2)  # (time, n_realizations)
    assert np.all(npv[:, 0] > npv[:, 1])  # R_1 produces more oil -> higher NPV
    assert np.all(npv >= 0)


def test_objective_value_maximizes_npv_so_minimizer_gets_negative(tmp_path):
    from petlab.config import EnsembleConfig

    years = np.array([1.0])
    summary = {"R_1": _write_summary(tmp_path, "R_1", years, FOPR=np.array([100.0]))}
    config = EnsembleConfig.from_dict({
        "Name": "test", "Ne": 1, "root": "dummy.DATA",
        "optimization": {"parameters": {"costFunction": "NPV"}},
    })

    value = objective.objective_value(config, summary, is_success=[True], unit="METRIC")
    assert value < 0  # minimizer convention: negative of the (positive) NPV


def test_evaluate_inequality_feasible_when_within_target(tmp_path):
    years = np.array([1.0, 2.0])
    summary = {
        "R_1": _write_summary(tmp_path, "R_1", years, FWPT=np.array([100.0, 500.0])),
        "R_2": _write_summary(tmp_path, "R_2", years, FWPT=np.array([100.0, 400.0])),
    }
    spec = {"value": 1000.0, "timestep": "last", "robustness": {"type": "average"}, "type": "inequality", "is_active": True}

    value = objective.evaluate_inequality("FWPT", spec, summary, is_success=[True, True])
    assert value > 0  # (1000 - mean(500, 400)) / 1000 > 0 -> feasible


def test_evaluate_inequality_infeasible_when_exceeding_target(tmp_path):
    years = np.array([1.0])
    summary = {"R_1": _write_summary(tmp_path, "R_1", years, FWPT=np.array([2000.0]))}
    spec = {"value": 1000.0, "timestep": "last", "robustness": {"type": "average"}, "type": "inequality", "is_active": True}

    value = objective.evaluate_inequality("FWPT", spec, summary, is_success=[True])
    assert value < 0


def test_count_constraints_only_counts_active():
    from petlab.config import OptimizationConfig

    config = OptimizationConfig.from_dict({
        "parameters": {
            "constraints": {
                "FWPT": {"is_active": True, "type": "inequality", "value": 1, "robustness": {"type": "average"}, "timestep": "last"},
                "FGPT": {"is_active": False, "type": "inequality", "value": 1, "robustness": {"type": "average"}, "timestep": "last"},
                "SALT": {"is_active": True, "type": "equality", "value": 1, "robustness": {"type": "average"}, "timestep": "last"},
            }
        }
    })
    from petlab.config import EnsembleConfig
    ens = EnsembleConfig(name="t", ne=1, root="x", parameters=[], controls=[], optimization=config)

    n_eq, n_ineq = objective.count_constraints(ens)
    assert n_eq == 1
    assert n_ineq == 1
