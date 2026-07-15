"""Build the (parameters, simulated response, observed response) arrays a
history-match backend needs, from a completed :class:`petlab.ensemble.Study`
and its ``historymatching`` config section.

This is only used by the *standalone* ``historymatch`` CLI command, matching
a real/recorded field-history dataset (as the old ``hm_ensemble.py`` did --
see e.g. ``data/SPE10/spe10model2``). :mod:`petlab.clrm` does not use this: it
builds its "truth" data directly from the held-out realization's own
extraction, in the same run.
"""

from __future__ import annotations

import numpy as np

from petlab.config import EnsembleConfig
from petlab.ensemble import Study
from petlab.io import resample


def build_parameter_matrix(study: Study, model3d: list[str]) -> np.ndarray:
    """Stack the requested static3d properties (e.g. ``PERMX``) across every
    realization into an (Ne, Nc) matrix, log-transforming permeability."""

    columns = []
    for prop in model3d:
        values = np.array([np.load(study.static3d[name][prop]) for name in study.static3d])
        if "PERM" in prop:
            values = np.log(values)
        columns.append(values)
    return np.concatenate(columns, axis=1)


def build_response_arrays(study: Study, objectives: dict[str, str], timestep_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(darray, sim_array, hist_vector)``:

    - ``darray``: (n_objectives, Ne, Nd_per_objective) normalized mismatch
      between history and simulation (used by the spectral/no-ESMDA method)
    - ``sim_array``: (Ne, Nd) simulated response, all objectives concatenated
      (used by ES-MDA)
    - ``hist_vector``: (Nd,) the historic/observed response
    """

    real_names = list(study.summary.keys())
    history_years = np.load(timestep_path)

    darrays, sim_columns, hist_vector = [], [], []
    for key, history_path in objectives.items():
        base_years = np.load(study.summary[real_names[0]]["YEARS"])
        history = resample(base_years, history_years, np.load(history_path))
        hist_vector.extend(history)

        sims = []
        for real_name in real_names:
            sim_years = np.load(study.summary[real_name]["YEARS"])
            sim = resample(base_years, sim_years, np.load(study.summary[real_name][key]))
            sims.append(sim)
        sims = np.array(sims)  # (Ne, Nd_per_objective)

        sim_columns.append(sims.T)
        variance = np.var(history - sims, axis=0)
        feasible = variance > 0.0
        darrays.append((history - sims))

    return np.array(darrays), np.concatenate(sim_columns, axis=0).T, np.array(hist_vector)
