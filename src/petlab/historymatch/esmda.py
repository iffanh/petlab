"""Ensemble Smoother with Multiple Data Assimilation (Emerick and Reynolds,
2013) -- the default history-match method.

``ESMDA`` itself (ported from ``src/utils/es_mda.py``) is a plain linear
update in whatever parameter space it's given. For a full-field parameter
(e.g. Egg's ~18.5k-cell log-PERMX, updated against a 16-member ensemble) that
space is far too wide for the update to be well-posed on its own -- Ne=16 <<
Ncell=18553 -- so :func:`update_plsr` wraps it with a PLSR dimensionality
reduction step (fit on parameters -> simulated production, update in the
reduced score space, inverse-transform back), exactly the "PLSR + ESMDA"
combination the thesis uses for the real (non-toy) CLRM experiments
(Ch. 4.4 / 7.2). This is *not* a new invention -- it's
``hm_ensemble.run_esmda``, cleaned up and generalized so it works whether the
parameter matrix is wide (a full field) or narrow (a handful of scalars).
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression


class ESMDA:
    """The linear ensemble-smoother update. Works in whatever space ``m``
    (parameters) and ``g`` (simulated response) are given in -- callers
    decide whether that's a full field, a PLSR score space, or a handful of
    scalars."""

    def __init__(self, m: np.ndarray, g_obs: np.ndarray, cd: np.ndarray):
        """
        m       : prior parameter matrix, shape (Ne, Nc)
        g_obs   : observed/"truth" response, shape (Nd,)
        cd      : per-observation measurement-error variance, shape (Nd,)
        """
        assert g_obs.shape[0] == cd.shape[0]
        self.m = m
        self.g_obs = g_obs
        self.cd = cd

    @staticmethod
    def _covariance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        da = a - np.mean(a, axis=0)
        db = b - np.mean(b, axis=0)
        return (da.T @ db) / (a.shape[0] - 1)

    def update(self, m: np.ndarray, g: np.ndarray, alpha: float) -> np.ndarray:
        """One ES-MDA correction step: ``m`` is the current parameter
        ensemble (Ne, Nc), ``g`` its corresponding simulated response
        (Ne, Nd), ``alpha`` the step size for this update (``1`` for a
        single-step "ensemble smoother"; a schedule across several calls for
        true multi-step MDA -- see ``historymatch.esmda`` module docstring).

        Kept numerically identical to the original ``utils/es_mda.py``:
        gain uses ``cov_gg + (1/alpha) * diag(cd)`` and perturbs the
        observation with std. dev. ``|cd|`` (not ``alpha``-scaled) -- this is
        the actual update rule the existing results in this repo were
        produced with, not the textbook Emerick & Reynolds formula.
        """
        cov_mg = self._covariance(m, g)
        cov_gg = self._covariance(g, g)
        gain = cov_mg @ np.linalg.pinv(cov_gg + (1.0 / alpha) * np.diag(self.cd))

        m_post = np.empty_like(m)
        for j in range(m.shape[0]):
            perturbed_obs = np.random.normal(self.g_obs, np.abs(self.cd))
            m_post[j, :] = m[j, :] + gain @ (perturbed_obs - g[j, :])
        return m_post

    def run(self, g_func, alphas: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Run a full multi-step MDA schedule, calling ``g_func(m_i)`` to
        re-simulate the response after every update."""
        m = self.m
        g = np.array([g_func(mi) for mi in m])
        for alpha in alphas:
            m = self.update(m, g, alpha)
            g = np.array([g_func(mi) for mi in m])
        return m, g


def clip_to_prior_range(posterior: np.ndarray, prior: np.ndarray) -> np.ndarray:
    lo, hi = np.min(prior, axis=0), np.max(prior, axis=0)
    return np.clip(posterior, lo, hi)


def update_plsr(
    m_prior: np.ndarray,
    sim_array: np.ndarray,
    hist_vector: np.ndarray,
    ncomponent: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """One PLSR-reduced ES-MDA correction step.

    m_prior     : prior parameter matrix, shape (Ne, Nc) -- e.g. log-PERMX
                  per active cell, or a handful of scalar parameters
    sim_array   : simulated production data per realization, shape (Ne, Nd)
    hist_vector : observed ("truth") production data, shape (Nd,)
    ncomponent  : number of PLSR components (clamped to what's actually
                  supported by the ensemble size / parameter count)
    alpha       : ES-MDA inflation factor for this step (``1.0`` for a
                  single correction; call repeatedly with a schedule
                  satisfying ``sum(1/alpha) == 1`` for multi-step MDA)

    Returns the posterior parameter matrix, clipped back to the prior's
    per-column min/max (physical realizability, e.g. no negative
    permeability).
    """

    ncomponent = max(1, min(ncomponent, m_prior.shape[0] - 1, m_prior.shape[1]))
    plsr = PLSRegression(ncomponent)
    scores_m, scores_d = plsr.fit_transform(m_prior, sim_array)
    residual = m_prior - plsr.inverse_transform(scores_m)

    _, hist_scores = plsr.transform(m_prior, np.array([hist_vector]))

    variance = np.var(scores_d, axis=0)
    feasible = variance > 0.0

    esmda = ESMDA(m=scores_m, g_obs=hist_scores[0, feasible], cd=1.0 / variance[feasible])
    scores_m_post = esmda.update(scores_m, scores_d[:, feasible], alpha=alpha)

    m_post = plsr.inverse_transform(scores_m_post) + residual
    return clip_to_prior_range(m_post, m_prior)
