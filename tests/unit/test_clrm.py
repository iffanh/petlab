"""End-to-end test of the CLRM stage loop's control flow (optimize -> apply
to truth -> truncate -> history-match -> regenerate), without a real
simulator.

``ensemble.create``/``objective.evaluate``/``optimization.optimize`` are the
only three places :mod:`petlab.clrm` touches a real simulator binary, so
those three are replaced with a small stub "reservoir": permeability drives
a synthetic, deterministic production response
(``rate * mean(permx) / 100``, an analytic stand-in for a real decline
curve -- *not* a physical model). Everything else -- the stage/freeze
bookkeeping, ES-MDA (with PLSR reduction) itself, truncating truth data to
each checkpoint, regenerating decks from a posterior field -- runs for real.
"""

from pathlib import Path

import numpy as np
import pytest

from petlab import clrm, ensemble, objective, optimization
from petlab.config import (
    CLRMConfig,
    CLRMStageConfig,
    ControlSpec,
    EnsembleConfig,
    HistoryMatchConfig,
    OptimizationConfig,
    ParameterSpec,
    VectorsConfig,
)

TRUE_PERMX_MEAN = 300.0
PRIOR_PERMX_MEAN = 150.0  # deliberately biased prior, so ES-MDA has something to correct
N_CELLS = 30


def _make_config(name: str, ne: int) -> EnsembleConfig:
    return EnsembleConfig(
        name=name,
        ne=ne,
        root="dummy.DATA",
        parameters=[ParameterSpec(name="$PERMFILE", type="IncrementalArray", distribution={"parameters": {"prefix": "unused_", "suffix": ".INC"}})],
        controls=[ControlSpec(name=f"$C{i}", default=1.0, kind="float", lb=0.0, ub=2.0) for i in range(4)],
        n_parallel=1,
        vectors=VectorsConfig(summary=["FOPR"], static3d=["PERMX"]),
        historymatching=HistoryMatchConfig(ncomponent=2),
        optimization=OptimizationConfig(optimizer="DFTR", cost_function="NPV", max_iter=1, constraints={}),
    )


def _fake_create(storage_root: Path):
    call_count = {"n": 0}

    def fake_create(config, storage_dir, case_numbers=None):
        call_count["n"] += 1
        case_numbers = list(case_numbers) if case_numbers is not None else list(range(1, config.ne + 1))
        study = ensemble.Study(name=config.name, config=config, case_numbers=case_numbers)

        static_dir = Path(storage_dir) / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1000 + call_count["n"])

        permx_param = next(p for p in config.parameters if p.name == "$PERMFILE")

        for case_number in case_numbers:
            name = f"{config.name}_{case_number}"
            study.base_realizations[name] = str(Path(storage_dir) / f"{name}.DATA")

            if permx_param.type == "IncrementalValue":
                # a real ES-MDA posterior, written by clrm.py itself
                p = permx_param.distribution["parameters"]
                permx = np.load(f"{p['prefix']}{case_number}{p['suffix']}")
            elif "truth" in config.name:
                permx = np.full(N_CELLS, TRUE_PERMX_MEAN)
            else:
                permx = np.clip(rng.normal(loc=PRIOR_PERMX_MEAN, scale=20.0, size=N_CELLS), 1.0, None)

            permx_path = static_dir / f"PERMX_{name}.npy"
            np.save(permx_path, permx)
            study.static3d[name] = {"PERMX": str(permx_path)}

        return study

    return fake_create


def _fake_evaluate(x, config, study, simulator_path, simfolder_path):
    names = list(study.static3d.keys())
    years = np.array([0.5, 1.0, 1.5, 2.0])

    for name in names:
        mean_permx = float(np.mean(np.load(study.static3d[name]["PERMX"])))
        total_rate = float(np.sum(x))
        fopr = total_rate * mean_permx / 100.0 * years  # a synthetic, monotone-in-time response

        real_dir = Path(simfolder_path) / name
        real_dir.mkdir(parents=True, exist_ok=True)
        years_path, fopr_path = real_dir / "YEARS.npy", real_dir / "FOPR.npy"
        np.save(years_path, years)
        np.save(fopr_path, fopr)
        study.summary[name] = {"YEARS": str(years_path), "FOPR": str(fopr_path)}

    is_success = [True] * len(names)
    total_fopr = np.mean([np.load(study.summary[n]["FOPR"])[-1] for n in names])
    return objective.EvaluationResult(objective=-float(total_fopr), equalities=[], inequalities=[], is_success=is_success, summary=study.summary)


def _fake_optimize(cf, eqs, ineqs, x0, lb, ub, config):
    f = cf(x0)  # exercise the evaluator once, just like a real (trivial) optimizer would
    return optimization.base.OptimizerResult(x_best=list(x0), f_best=f, n_evaluations=1, history={})


def test_clrm_stage_loop_reduces_permx_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble, "create", _fake_create(tmp_path))
    monkeypatch.setattr(objective, "evaluate", _fake_evaluate)
    monkeypatch.setattr(optimization, "optimize", _fake_optimize)

    prior = _make_config("stub_prior", ne=8)
    truth = _make_config("stub_truth", ne=1)

    config = CLRMConfig(
        name="stub_clrm",
        prior=prior,
        truth=truth,
        truth_case_number=100,
        stages=[
            CLRMStageConfig(checkpoint_years=1.0, control_indices=[0, 1]),
            CLRMStageConfig(checkpoint_years=2.0, control_indices=[2, 3]),
        ],
        updated_parameter="$PERMFILE",
        output_dir=str(tmp_path / "clrm_out"),
    )

    result = clrm.run(config, simulator_path="stub")

    assert len(result.stages) == 2
    assert len(result.applied_controls) == 4
    assert np.isfinite(result.final_truth_objective)

    for stage in result.stages:
        assert np.isfinite(stage.predicted_objective)
        assert np.isfinite(stage.truth_objective_so_far)
        assert stage.log_permx_rmse_before > 0  # prior is deliberately biased away from truth

    # ES-MDA should have pulled the (biased) prior mean permeability toward
    # the truth's, at least on average across the two stages.
    improved = [s.log_permx_rmse_after < s.log_permx_rmse_before for s in result.stages]
    assert sum(improved) >= 1

    assert Path(config.output_dir, "clrm_result.json").is_file()
