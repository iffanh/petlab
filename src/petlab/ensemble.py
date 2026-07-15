"""Ensemble creation, simulation and result extraction.

This is the in-process replacement for
``create_ensemble.py`` + ``run_ensemble.py`` + ``extract_ensemble.py``: given
an :class:`~petlab.config.EnsembleConfig`, build one deck per realization,
run them through a simulator binary, and pull the requested summary/3D
vectors back out as ``.npy`` files (via ``resdata``).

The three stages are kept as separate functions (mirroring the old CLI
scripts) so the CLI can still expose them individually, but they now operate
on an in-memory :class:`Study` instead of round-tripping a big dict through
JSON on disk between every stage -- which matters for
:mod:`petlab.clrm`, where the same ensemble gets re-optimized, re-simulated
and re-extracted many times in a single process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from resdata.resfile import ResdataFile
from resdata.summary import Summary
from tqdm import tqdm
from wrapt_timeout_decorator import timeout

from petlab import deck
from petlab.config import ControlSpec, EnsembleConfig
from petlab.io import hashable_lru, run_bash_commands_in_parallel


@dataclass
class Study:
    """The state of one ensemble as it moves through create -> simulate ->
    extract. Realizations are keyed by name (``"<deck stem>_<case number>"``,
    e.g. ``"EGG_CLRM_7"``)."""

    name: str
    config: EnsembleConfig
    case_numbers: list[int]

    base_realizations: dict[str, str] = field(default_factory=dict)
    sampled_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    realizations: dict[str, str] = field(default_factory=dict)
    is_success: list[bool] = field(default_factory=list)

    summary: dict[str, dict[str, str]] = field(default_factory=dict)
    static3d: dict[str, dict[str, str]] = field(default_factory=dict)
    dynamic3d: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable snapshot, for the CLI to round-trip a study
        between separate ``create``/``run``/``extract``/``optimize``
        invocations. ``sampled_parameters`` is left out -- it can hold large
        arrays (e.g. a full permeability field) and nothing that reloads a
        study from disk needs it back (only :mod:`petlab.clrm`, which stays
        in one process and never round-trips through JSON, does)."""
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "case_numbers": self.case_numbers,
            "base_realizations": self.base_realizations,
            "realizations": self.realizations,
            "is_success": self.is_success,
            "summary": self.summary,
            "static3d": self.static3d,
            "dynamic3d": self.dynamic3d,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Study":
        return cls(
            name=d["name"],
            config=EnsembleConfig.from_dict(d["config"]),
            case_numbers=d["case_numbers"],
            base_realizations=d.get("base_realizations", {}),
            realizations=d.get("realizations", {}),
            is_success=d.get("is_success", []),
            summary=d.get("summary", {}),
            static3d=d.get("static3d", {}),
            dynamic3d=d.get("dynamic3d", {}),
        )


def realization_name(root_datafile_path: str, case_number: int) -> str:
    return f"{Path(root_datafile_path).stem}_{case_number}"


# ---------------------------------------------------------------------------
# Stage 1: create -- sample uncertain parameters, write one deck per realization
# ---------------------------------------------------------------------------

def create(config: EnsembleConfig, storage_dir: str, case_numbers: list[int] | None = None) -> Study:
    """Sample every :class:`~petlab.config.ParameterSpec` once per
    realization and write the resulting decks under
    ``{storage_dir}/BASE_{config.name}/``.

    ``case_numbers`` defaults to ``1..config.ne``; pass an explicit list to
    build a subset of realizations (e.g. ``[100]`` for a single held-out
    "truth" realization, as :mod:`petlab.clrm` does).
    """

    case_numbers = list(case_numbers) if case_numbers is not None else list(range(1, config.ne + 1))
    base_ens_path = os.path.join(storage_dir, f"BASE_{config.name}")

    study = Study(name=config.name, config=config, case_numbers=case_numbers)

    for case_number in tqdm(case_numbers, desc="Creating realizations", leave=False):
        real_name = realization_name(config.root, case_number)
        real_datafile_path = os.path.join(base_ens_path, real_name, real_name + ".DATA")

        sampled = deck.mutate_case(config.root, real_datafile_path, config.parameters, case_number)

        study.base_realizations[real_name] = real_datafile_path
        study.sampled_parameters[real_name] = sampled

    return study


# ---------------------------------------------------------------------------
# Stage 2: simulate -- apply controls, run the simulator
# ---------------------------------------------------------------------------

def _simulate_command(simulator_path: str, real_datafile_path: str) -> list[str]:
    if "flow" in simulator_path:
        return [simulator_path, "--enable-terminal-output=false", real_datafile_path]
    elif "ecl" in simulator_path:
        return [simulator_path, real_datafile_path[:-5]]  # eclrun does not want the .DATA suffix
    raise ValueError(f"Don't know how to invoke simulator {simulator_path!r} (expected 'flow' or 'eclrun' in the path)")


@timeout(1000)
@hashable_lru
def simulate(
    base_realizations: dict[str, str],
    simulator_path: str,
    controls: list[dict] | list[list[dict]],
    simfolder_path: str,
    n_parallel: int,
) -> tuple[dict[str, str], list[bool]]:
    """Apply ``controls`` to every realization's base deck (``{name: path}``,
    i.e. ``study.base_realizations``) and run the simulator on all of them
    (up to ``n_parallel`` at a time).

    ``controls`` is normally a single list of control dicts applied
    identically to every realization; pass a list-of-lists (one control list
    per realization, in the same order as ``base_realizations``) for
    per-realization controls (as StoSAG does).

    Cached (keyed on the JSON-serialized arguments) and time-boxed, since
    this is the expensive step that gets called once per optimizer
    iteration. Takes plain dicts rather than a :class:`Study` so the cache
    key only reflects what actually determines the simulation outcome --
    the previous implementation cached on the entire (much bigger) study
    dict, so unrelated bookkeeping changes could bust the cache.
    """

    control_specs = [ControlSpec.from_dict(c) for c in controls] if controls and not isinstance(controls[0], list) else None

    names = list(base_realizations.keys())
    commands = []
    realizations: dict[str, str] = {}

    for idx, real_name in enumerate(tqdm(names, desc="Preparing", leave=False)):
        real_path = os.path.join(simfolder_path, real_name)
        real_datafile_path = os.path.join(real_path, real_name + ".DATA")
        base_datafile_path = base_realizations[real_name]

        if control_specs is not None:
            this_controls = control_specs
        else:
            this_controls = [ControlSpec.from_dict(c) for c in controls[idx]]

        deck.apply_controls(base_datafile_path, real_datafile_path, this_controls)

        commands.append(_simulate_command(simulator_path, real_datafile_path))
        realizations[real_name] = real_datafile_path

    is_success = run_bash_commands_in_parallel(commands, max_tries=1, n_parallel=n_parallel)

    # A simulator may exit 0 without actually producing time steps (OPM flow
    # reports failures via exit code; Eclipse300 does not always). Treat "no
    # restart steps beyond the initial one" as a failure too.
    for idx, real_name in enumerate(names):
        unrst_path = realizations[real_name][:-5] + ".UNRST"
        if os.path.isfile(unrst_path):
            if len(ResdataFile(unrst_path).report_steps) == 1:
                is_success[idx] = False
        else:
            is_success[idx] = False

    return realizations, is_success


def simulate_study(
    study: Study,
    simulator_path: str,
    controls: list[dict] | list[list[dict]],
    simfolder_path: str,
    n_parallel: int,
) -> None:
    """Convenience wrapper: run :func:`simulate` against ``study``'s base
    realizations and record the result on ``study``."""
    realizations, is_success = simulate(study.base_realizations, simulator_path, controls, simfolder_path, n_parallel)
    apply_simulation_result(study, realizations, is_success)


def apply_simulation_result(study: Study, realizations: dict[str, str], is_success: list[bool]) -> None:
    """``simulate()`` is cached and pure, so it returns its result instead of
    mutating ``study`` -- call this to record that result on the study."""
    study.realizations = realizations
    study.is_success = is_success


# ---------------------------------------------------------------------------
# Stage 3: extract -- pull summary/3D vectors out as .npy files
# ---------------------------------------------------------------------------

def get_summary(realizations: dict[str, str], storage: str, sum_keys: list[str]) -> dict[str, dict[str, str]]:
    """Extract the requested summary vectors (e.g. ``FOPR``, ``WBHP``) for
    every realization and save each as a ``.npy`` file under
    ``{storage}/results/summary/{realization}/``. Returns
    ``{realization: {vector_name: npy_path}}``."""

    first = next(iter(realizations))
    summary_keys = Summary(realizations[first]).keys()
    available_keys = ["YEARS"] + [k for k in summary_keys if any(sk in k for sk in sum_keys)]

    data_dir = os.path.join(storage, "results", "summary")
    result: dict[str, dict[str, str]] = {}

    for key in tqdm(available_keys, desc="Extracting summary", leave=False, disable=True):
        for real_name, real_path in realizations.items():
            result.setdefault(real_name, {})
            vector = Summary(real_path).numpy_vector(key)

            real_dir = os.path.join(data_dir, real_name)
            Path(real_dir).mkdir(parents=True, exist_ok=True)
            filename = os.path.join(real_dir, f"{key}.npy")
            np.save(filename, vector)
            result[real_name][key] = filename

    return result


def get_3dprops(
    realizations: dict[str, str],
    storage: str,
    static3d_keys: list[str],
    dynamic3d_keys: list[str],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Extract requested static (``.INIT``) and dynamic (``.UNRST``) 3D
    properties for every realization, saved as ``.npy`` files. Returns
    ``(static3d, dynamic3d)``, each ``{realization: {property: npy_path}}``."""

    static3d: dict[str, dict[str, str]] = {}
    dynamic3d: dict[str, dict[str, str]] = {}

    if static3d_keys:
        first_dir = os.path.dirname(next(iter(realizations.values())))
        first_name = next(iter(realizations))
        static_keys = [k for k in ResdataFile(os.path.join(first_dir, first_name + ".INIT")).keys() if any(sk in k for sk in static3d_keys)]

        data_dir = os.path.join(storage, "results", "static3d")
        for key in tqdm(static_keys, desc="Extracting static3D", leave=False, disable=True):
            for real_name, real_path in realizations.items():
                static3d.setdefault(real_name, {})
                dirname = os.path.dirname(real_path)
                init_file = ResdataFile(os.path.join(dirname, real_name + ".INIT"))
                vector = init_file[key][0].numpy_view()

                real_dir = os.path.join(data_dir, real_name)
                Path(real_dir).mkdir(parents=True, exist_ok=True)
                filename = os.path.join(real_dir, f"{key}.npy")
                np.save(filename, vector)
                static3d[real_name][key] = filename

    if dynamic3d_keys:
        first_dir = os.path.dirname(next(iter(realizations.values())))
        first_name = next(iter(realizations))
        dyn_file = ResdataFile(os.path.join(first_dir, first_name + ".UNRST"))
        dynamic_keys = [k for k in dyn_file.keys() if any(dk in k for dk in dynamic3d_keys)]

        data_dir = os.path.join(storage, "results", "dynamic3d")
        for key in tqdm(dynamic_keys, desc="Extracting dynamic3D", leave=False, disable=True):
            for real_name, real_path in realizations.items():
                dirname = os.path.dirname(real_path)
                real_dir = os.path.join(data_dir, real_name)
                Path(real_dir).mkdir(parents=True, exist_ok=True)

                dyn_file = ResdataFile(os.path.join(dirname, real_name + ".UNRST"))
                if real_name not in dynamic3d:
                    dynamic3d[real_name] = {}
                    np.save(os.path.join(real_dir, "DATES.npy"), dyn_file.report_dates)

                matrix = np.array([dyn_file[key][t].numpy_view() for t in range(len(dyn_file.report_dates))])
                filename = os.path.join(real_dir, f"{key}.npy")
                np.save(filename, matrix)
                dynamic3d[real_name][key] = filename

    return static3d, dynamic3d


def extract(study: Study, storage_dir: str) -> None:
    """Populate ``study.summary``/``study.static3d``/``study.dynamic3d`` from
    the vectors declared in ``study.config.vectors``."""

    study.summary = get_summary(study.realizations, storage_dir, study.config.vectors.summary)
    if study.config.vectors.static3d or study.config.vectors.dynamic3d:
        study.static3d, study.dynamic3d = get_3dprops(
            study.realizations, storage_dir, study.config.vectors.static3d, study.config.vectors.dynamic3d
        )
