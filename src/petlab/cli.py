"""The ``petlab`` command-line entrypoint.

Subcommands mirror the stages of the old pipeline
(``create-ensemble`` -> ``run`` -> ``extract`` -> ``optimize`` /
``historymatch`` -> ``evaluate``), plus the new ``clrm`` command. Each
stage's state is round-tripped through a small JSON "study" file (see
``ensemble.Study.to_dict``/``from_dict``), the same way the old
``create_ensemble.py``/``run_ensemble.py``/etc. scripts did, so existing
shell scripts built around that calling convention keep working -- see
``src/create_ensemble.py`` and friends, which are now thin shims that just
call into this module.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from petlab import clrm, ensemble, objective, optimization
from petlab.config import CLRMConfig, EnsembleConfig
from petlab.historymatch import get_backend as get_history_match_backend
from petlab.io import read_json, save_to_json

STORAGE_DIR = "./simulations/storage"
STUDIES_DIR = "./simulations/studies"


def _study_path(name: str) -> str:
    return os.path.join(STUDIES_DIR, f"{name}.json")


def cmd_create_ensemble(args: argparse.Namespace) -> None:
    config = EnsembleConfig.from_json(args.config)
    study = ensemble.create(config, storage_dir=STORAGE_DIR)

    Path(STUDIES_DIR).mkdir(parents=True, exist_ok=True)
    save_to_json(_study_path(config.name), study.to_dict())
    print(f"Created {len(study.base_realizations)} realization(s) -> {_study_path(config.name)}")


def cmd_run(args: argparse.Namespace) -> None:
    study = ensemble.Study.from_dict(read_json(args.study))
    simfolder = os.path.join(STORAGE_DIR, study.name)

    controls = [c.to_dict() for c in study.config.controls]
    ensemble.simulate_study(study, args.simulator, controls, simfolder, study.config.n_parallel)

    save_to_json(args.study, study.to_dict())
    n_ok = sum(study.is_success)
    print(f"Simulated {len(study.realizations)} realization(s), {n_ok} succeeded -> {args.study}")


def cmd_extract(args: argparse.Namespace) -> None:
    study = ensemble.Study.from_dict(read_json(args.study))
    simfolder = os.path.join(STORAGE_DIR, study.name)
    ensemble.extract(study, simfolder)
    save_to_json(args.study, study.to_dict())
    print(f"Extracted vectors for {len(study.summary)} realization(s) -> {args.study}")


def cmd_optimize(args: argparse.Namespace) -> None:
    study = ensemble.Study.from_dict(read_json(args.study))
    config = study.config
    simfolder = os.path.join(STORAGE_DIR, study.name)

    x0 = [c.default for c in config.controls]
    lb = [c.lb for c in config.controls]
    ub = [c.ub for c in config.controls]

    cf, eqs, ineqs = objective.make_evaluator(config, study, args.simulator, simfolder)
    result = optimization.optimize(cf, eqs, ineqs, x0, lb, ub, config.optimization)

    # Make sure the study reflects the reported best point exactly.
    objective.evaluate(np.array(result.x_best), config, study, args.simulator, simfolder)
    save_to_json(args.study, study.to_dict())

    print(f"Optimizer: {config.optimization.optimizer}")
    print(f"Best objective ({config.optimization.cost_function}): {-result.f_best}")
    print(f"Best controls: {dict(zip((c.name for c in config.controls), result.x_best))}")
    print(f"Function evaluations: {result.n_evaluations}")


def cmd_historymatch(args: argparse.Namespace) -> None:
    from petlab.historymatch.data import build_parameter_matrix, build_response_arrays

    study = ensemble.Study.from_dict(read_json(args.study))
    hm = study.config.historymatching
    method = args.method or hm.method

    m_prior = build_parameter_matrix(study, hm.model3d)
    darray, sim_array, hist_vector = build_response_arrays(study, hm.objectives, hm.timestep)

    if method == "ESMDA":
        m_post = get_history_match_backend("ESMDA")(m_prior, sim_array, hist_vector, hm.ncomponent, alpha=args.alpha or 1.0)
    elif method == "Spectral":
        m_post = get_history_match_backend("Spectral")(m_prior, darray, hm.ncomponent, hm.polynomial_order)
    elif method == "PCESMDA":
        m_post = get_history_match_backend("PCESMDA")(m_prior, sim_array, hist_vector, hm.ncomponent, hm.polynomial_order, hm.alphas)
    else:
        raise ValueError(f"Unknown history-match method {method!r}")

    Path(hm.updatepath).mkdir(parents=True, exist_ok=True)
    ncell = m_post.shape[1] // len(hm.model3d)
    for i, model in enumerate(hm.model3d):
        block = m_post[:, i * ncell:(i + 1) * ncell]
        if "PERM" in model:
            block = np.exp(block)
        for row, case_number in zip(block, study.case_numbers):
            np.save(os.path.join(hm.updatepath, f"{model}_{case_number}.npy"), row)

    print(f"History match ({method}) done -> posterior saved under {hm.updatepath}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    study = ensemble.Study.from_dict(read_json(args.study))
    config = study.config
    simfolder = os.path.join(STORAGE_DIR, study.name)

    with open(args.controls_csv) as f:
        x = np.array([float(row[0]) for row in csv.reader(f)])

    result = objective.evaluate(x, config, study, args.simulator, simfolder)
    save_to_json(args.study, study.to_dict())

    print(f"Objective ({config.optimization.cost_function}): {-result.objective}")
    print(f"Equality constraints: {result.equalities}")
    print(f"Inequality constraints: {result.inequalities}")


def cmd_clrm(args: argparse.Namespace) -> None:
    config = CLRMConfig.from_json(args.config)
    result = clrm.run(config, args.simulator)

    print(f"CLRM run '{result.name}' finished after {len(result.stages)} stage(s)")
    for s in result.stages:
        print(
            f"  stage {s.stage} (t<={s.checkpoint_years:.2f}y): predicted={s.predicted_objective:.3g} "
            f"truth_so_far={s.truth_objective_so_far:.3g} log-PERMX RMSE {s.log_permx_rmse_before:.3g} -> {s.log_permx_rmse_after:.3g}"
        )
    print(f"Final true objective (applying the fully-decided controls to the truth model): {result.final_truth_objective:.3g}")
    print(f"Results saved under {result.output_dir}/clrm_result.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="petlab", description="Closed-loop reservoir management toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-ensemble", help="Sample uncertain parameters and write one deck per realization")
    p.add_argument("config", help="Path to an EnsembleConfig JSON file")
    p.set_defaults(func=cmd_create_ensemble)

    p = sub.add_parser("run", help="Apply the configured default controls and simulate")
    p.add_argument("simulator", help="Path to the simulator binary (OPM 'flow' or 'eclrun')")
    p.add_argument("study", help="Path to the study JSON produced by create-ensemble")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("extract", help="Pull summary/3D vectors out of a simulated study")
    p.add_argument("study")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("optimize", help="Optimize controls (default: DFTR)")
    p.add_argument("simulator")
    p.add_argument("study")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("historymatch", help="Update uncertain parameters against observed data (default: ESMDA)")
    p.add_argument("study")
    p.add_argument("--method", choices=["ESMDA", "Spectral", "PCESMDA"], default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.set_defaults(func=cmd_historymatch)

    p = sub.add_parser("evaluate", help="Evaluate one specific control vector (from a CSV file)")
    p.add_argument("simulator")
    p.add_argument("study")
    p.add_argument("controls_csv")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("clrm", help="Run a full closed-loop (optimize -> apply to truth -> history-match) schedule")
    p.add_argument("simulator")
    p.add_argument("config", help="Path to a CLRMConfig JSON file")
    p.set_defaults(func=cmd_clrm)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
