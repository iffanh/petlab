"""The closed-loop (optimize -> apply to truth -> history-match -> repeat)
receding-horizon driver -- the actual CLRM/MPC feature this package exists
for (thesis Ch. 7).

Per stage:

1. **Optimize** the *remaining* horizon's controls (DFTR by default) against
   the current ensemble, freeze this stage's slice of the result.
2. **Apply to truth**: run those controls (frozen so far + best-guess for
   the rest) on the held-out truth realization.
3. **Truncate & history-match**: truncate the truth's production data to
   this stage's checkpoint, then run ES-MDA (PLSR-reduced --
   ``historymatch.esmda.update_plsr``) on the ensemble's log-PERMX field
   against it.
4. **Regenerate** the ensemble's decks with the posterior PERMX field
   (``IncrementalValue``, reading the ``.npy`` files this step writes)
   instead of the original ``IncrementalArray`` pick, becoming the "current
   ensemble" for the next stage.

Re-simulation strategy: every stage reruns every realization (ensemble and
truth) from t=0 with the full control history so far, rather than an
Eclipse-RESTART warm start -- simpler and robust; see ``docs/CLRM.md``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace as dc_replace
from pathlib import Path

import numpy as np

from petlab import ensemble, objective, optimization
from petlab.config import CLRMConfig, EnsembleConfig, ParameterSpec
from petlab.historymatch.esmda import update_plsr
from petlab.io import resample, save_to_json


@dataclass
class StageDiagnostics:
    stage: int
    checkpoint_years: float
    x_best: list[float]
    predicted_objective: float
    truth_objective_so_far: float
    log_permx_rmse_before: float
    log_permx_rmse_after: float
    n_evaluations: int


@dataclass
class CLRMResult:
    name: str
    output_dir: str
    stages: list[StageDiagnostics] = field(default_factory=list)
    final_truth_objective: float | None = None
    applied_controls: list[float] = field(default_factory=list)


def _expand(x_free: np.ndarray, free_indices: list[int], frozen: dict[int, float], n_controls: int) -> np.ndarray:
    full = np.empty(n_controls)
    for i, v in frozen.items():
        full[i] = v
    for pos, i in enumerate(free_indices):
        full[i] = x_free[pos]
    return full


def _make_stage_evaluator(config: EnsembleConfig, study: ensemble.Study, simulator_path: str, simfolder_path: str, free_indices: list[int], frozen: dict[int, float]):
    n_controls = len(config.controls)
    cache: dict[tuple, objective.EvaluationResult] = {}

    def cached(x_free) -> objective.EvaluationResult:
        key = tuple(float(v) for v in x_free)
        if key not in cache:
            full_x = _expand(np.array(key), free_indices, frozen, n_controls)
            cache[key] = objective.evaluate(full_x, config, study, simulator_path, simfolder_path)
        return cache[key]

    n_eq, n_ineq = objective.count_constraints(config)
    cf = lambda x: cached(x).objective
    eqs = [(lambda x, i=i: cached(x).equalities[i]) for i in range(n_eq)]
    ineqs = [(lambda x, i=i: cached(x).inequalities[i]) for i in range(n_ineq)]
    return cf, eqs, ineqs


def _with_permx_from_posterior(config: EnsembleConfig, updated_parameter: str, npy_prefix: str) -> EnsembleConfig:
    """Return a copy of ``config`` whose ``updated_parameter`` (e.g.
    ``$PERMFILE``) now reads the ES-MDA posterior ``.npy`` files at
    ``{npy_prefix}{case_number}.npy`` instead of picking one of the original
    pre-generated realizations."""

    new_params = []
    for p in config.parameters:
        if p.name == updated_parameter:
            new_params.append(ParameterSpec(name=p.name, type="IncrementalValue", distribution={"parameters": {"prefix": npy_prefix, "suffix": ".npy"}}))
        else:
            new_params.append(p)
    return dc_replace(config, parameters=new_params)


def _log_permx_matrix(study: ensemble.Study) -> np.ndarray:
    names = list(study.static3d.keys())
    return np.log(np.array([np.load(study.static3d[n]["PERMX"]) for n in names]))


def _truncate_and_build_hm_data(prior_study: ensemble.Study, truth_study: ensemble.Study, checkpoint_years: float) -> tuple[np.ndarray, np.ndarray]:
    """Truncate the truth model's production data to ``checkpoint_years``
    and resample every prior-ensemble member's own production data onto
    that same truncated time axis. Returns ``(sim_array, hist_vector)``."""

    truth_name = next(iter(truth_study.summary))
    truth_years = np.load(truth_study.summary[truth_name]["YEARS"])
    mask = truth_years <= checkpoint_years + 1e-9
    truncated_years = truth_years[mask]

    match_keys = sorted(k for k in truth_study.summary[truth_name] if k != "YEARS")

    hist_vector: list[float] = []
    sim_columns = []
    for key in match_keys:
        hist_vector.extend(np.load(truth_study.summary[truth_name][key])[mask])

        column = []
        for real_name in prior_study.summary:
            real_years = np.load(prior_study.summary[real_name]["YEARS"])
            real_values = np.load(prior_study.summary[real_name][key])
            column.append(resample(truncated_years, real_years, real_values))
        sim_columns.append(np.array(column))  # (Ne, len(truncated_years))

    sim_array = np.concatenate(sim_columns, axis=1)  # (Ne, n_keys * len(truncated_years))
    return sim_array, np.array(hist_vector)


def run(config: CLRMConfig, simulator_path: str) -> CLRMResult:
    """Run the full closed-loop schedule described by ``config``."""

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    result = CLRMResult(name=config.name, output_dir=config.output_dir)

    prior_config = config.prior
    truth_config = config.truth

    # Each generation's base decks get their own directory: `ensemble.create`
    # writes to `{storage_dir}/BASE_{config.name}/...`, and `config.name`
    # doesn't change between generations (only `parameters` does, via
    # `_with_permx_from_posterior`), so reusing one `storage_dir` across
    # stages would silently alias stage N's file *paths* onto stage N+1's
    # (different-content) decks -- and `ensemble.simulate`'s cache is keyed
    # on those paths, not file content, so it would then return stale
    # results from an earlier generation.
    prior_study = ensemble.create(prior_config, storage_dir=os.path.join(config.output_dir, "stage_0"))
    truth_study = ensemble.create(truth_config, storage_dir=os.path.join(config.output_dir, "truth"), case_numbers=[config.truth_case_number])

    n_controls = len(prior_config.controls)
    frozen: dict[int, float] = {}

    for stage_idx, stage in enumerate(config.stages):
        free_indices = [i for i in range(n_controls) if i not in frozen]

        controls = prior_config.controls
        x0 = [controls[i].default for i in free_indices]
        lb = [controls[i].lb for i in free_indices]
        ub = [controls[i].ub for i in free_indices]

        simfolder = os.path.join(config.output_dir, f"stage_{stage_idx}", "ensemble")
        cf, eqs, ineqs = _make_stage_evaluator(prior_config, prior_study, simulator_path, simfolder, free_indices, frozen)

        opt_result = optimization.optimize(cf, eqs, ineqs, x0, lb, ub, prior_config.optimization)
        full_best_x = _expand(np.array(opt_result.x_best), free_indices, frozen, n_controls)

        # Make sure the study's recorded simulation/extraction matches the
        # reported best point exactly (the optimizer's last function call
        # isn't always at its own best point).
        eval_at_best = objective.evaluate(full_best_x, prior_config, prior_study, simulator_path, simfolder)

        # Freeze this stage's own slice of the decision for good; future
        # stages will only re-optimize what's left.
        for i in stage.control_indices:
            frozen[i] = float(full_best_x[i])

        # Apply the (frozen-so-far + best-guessed-future) control to the
        # truth model and truncate its production data to this checkpoint.
        truth_simfolder = os.path.join(config.output_dir, f"stage_{stage_idx}", "truth")
        truth_eval = objective.evaluate(full_best_x, truth_config, truth_study, simulator_path, truth_simfolder)

        m_prior = _log_permx_matrix(prior_study)
        truth_permx = _log_permx_matrix(truth_study)[0]
        rmse_before = float(np.sqrt(np.mean((np.mean(m_prior, axis=0) - truth_permx) ** 2)))

        sim_array, hist_vector = _truncate_and_build_hm_data(prior_study, truth_study, stage.checkpoint_years)
        m_post_log = update_plsr(m_prior, sim_array, hist_vector, ncomponent=prior_config.historymatching.ncomponent, alpha=1.0)
        rmse_after = float(np.sqrt(np.mean((np.mean(m_post_log, axis=0) - truth_permx) ** 2)))

        posterior_dir = os.path.join(config.output_dir, f"stage_{stage_idx}", "posterior")
        Path(posterior_dir).mkdir(parents=True, exist_ok=True)
        for row, case_number in zip(m_post_log, prior_study.case_numbers):
            np.save(os.path.join(posterior_dir, f"PERMX_{case_number}.npy"), np.exp(row))

        result.stages.append(
            StageDiagnostics(
                stage=stage_idx,
                checkpoint_years=stage.checkpoint_years,
                x_best=list(full_best_x),
                predicted_objective=-eval_at_best.objective,
                truth_objective_so_far=-truth_eval.objective,
                log_permx_rmse_before=rmse_before,
                log_permx_rmse_after=rmse_after,
                n_evaluations=opt_result.n_evaluations,
            )
        )

        # Next stage's ensemble: same wells/schedule, posterior permeability,
        # written to its own stage_{n+1} directory (see the comment above
        # the first `ensemble.create` call for why that matters).
        next_config = _with_permx_from_posterior(prior_config, config.updated_parameter, os.path.join(posterior_dir, "PERMX_"))
        next_storage_dir = os.path.join(config.output_dir, f"stage_{stage_idx + 1}")
        prior_study = ensemble.create(next_config, storage_dir=next_storage_dir, case_numbers=prior_study.case_numbers)
        prior_config = next_config

    result.applied_controls = [frozen[i] for i in range(n_controls)]
    final_simfolder = os.path.join(config.output_dir, "final", "truth")
    final_truth_eval = objective.evaluate(np.array(result.applied_controls), config.truth, truth_study, simulator_path, final_simfolder)
    result.final_truth_objective = -final_truth_eval.objective

    save_to_json(os.path.join(config.output_dir, "clrm_result.json"), {
        "name": result.name,
        "final_truth_objective": result.final_truth_objective,
        "applied_controls": result.applied_controls,
        "stages": [asdict(s) for s in result.stages],
    })

    return result
