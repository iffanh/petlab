"""Optional history-match backends.

- ``run_spectral``: Partial Least Squares / Independent Component / Principal
  Component regression + a Polynomial Chaos Expansion surrogate, with no
  ES-MDA step at all (the README's "new method", ported from
  ``hm_ensemble.run_history_matching`` -- Article IV).
- ``run_pcesmda``: PLSR reduction + a PCE surrogate of the reduced response,
  then ES-MDA through that surrogate (``hm_ensemble.run_pcesmda``) -- a
  heavier variant of the default ``historymatch.esmda.update_plsr``.

Neither is the default; both need ``chaospy`` (already a hard dependency)
plus, for ``run_spectral``, whichever scikit-learn reducer you choose.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.optimize import NonlinearConstraint
from sklearn.cross_decomposition import PLSRegression

from petlab.historymatch.esmda import ESMDA, clip_to_prior_range


def run_spectral(m_prior: np.ndarray, darray: np.ndarray, ncomponent: int, polynomial_order: int, reducer_cls=PLSRegression) -> np.ndarray:
    """PLSR/FastICA/PCA + PCE-regression history match, no ES-MDA (Article IV).

    m_prior : (Ne, Nc) parameter matrix
    darray  : (n_objectives, Ne, Nd_per_objective) history-minus-simulation
              mismatch, as built by ``historymatch.data.build_response_arrays``
    """
    import chaospy

    ofs = np.hstack(darray)
    std = np.std(ofs, axis=0)
    normalized = (ofs.T[std > 0] / std[std > 0][:, np.newaxis]).T

    method = reducer_cls(ncomponent)
    try:
        scores_m, scores_d = method.fit_transform(m_prior, normalized)
    except TypeError:  # FastICA/PCA don't take a `y` in fit_transform
        scores_m = method.fit_transform(m_prior, normalized)
    m_hat = method.inverse_transform(scores_m)
    residual = m_prior - m_hat

    nodes = scores_m[:, :ncomponent].T
    weights = np.ones(len(ofs))
    joint = chaospy.J(*[chaospy.GaussianKDE(scores_m[:, j], h_mat=1e1) for j in range(ncomponent)])
    expansion = chaospy.generate_expansion(polynomial_order, joint, rule="three_terms_recurrence")
    models = chaospy.fit_quadrature(expansion, nodes, weights, normalized)

    def objective(x):
        return np.sqrt(sum(model(*x) ** 2 for model in models))

    def transform_at(x, i):
        var = scores_m.copy()
        var[:, :ncomponent] = x
        return method.inverse_transform(var)[i, :]

    constraints = [
        NonlinearConstraint(lambda x, i=i: transform_at(x, i), np.min(m_hat, axis=0), np.max(m_hat, axis=0))
        for i in range(m_prior.shape[0])
    ]
    x0 = np.mean(scores_m[:, :ncomponent], axis=0)
    bounds = list(zip(np.min(nodes, axis=1), np.max(nodes, axis=1)))
    result = scipy.optimize.minimize(objective, x0, bounds=bounds, constraints=constraints)

    scores_m_post = scores_m.copy()
    scores_m_post[:, :ncomponent] = result.x
    m_post = residual + method.inverse_transform(scores_m_post)
    return clip_to_prior_range(m_post, m_prior)


def run_pcesmda(m_prior: np.ndarray, sim_array: np.ndarray, hist_vector: np.ndarray, ncomponent: int, polynomial_order: int, alphas: list[float]) -> np.ndarray:
    """PLSR reduction + PCE surrogate of the reduced response + a full
    multi-step ES-MDA through that surrogate.

    m_prior     : (Ne, Nc) parameter matrix
    sim_array   : (Ne, Nd) simulated response
    hist_vector : (Nd,) observed/"truth" response
    """
    import chaospy

    plsr = PLSRegression(ncomponent)
    scores_m, scores_d = plsr.fit_transform(m_prior, sim_array)

    nodes = scores_m[:, :ncomponent].T
    weights = np.ones(len(sim_array))
    joint = chaospy.J(*[chaospy.GaussianKDE(scores_m[:, j], h_mat=1e1) for j in range(ncomponent)])
    expansion = chaospy.generate_expansion(polynomial_order, joint, rule="three_terms_recurrence")
    models = chaospy.fit_quadrature(expansion, nodes, weights, scores_d)

    _, hist_scores = plsr.transform(m_prior, np.array([hist_vector]))

    def g_func(m):
        x = plsr.transform(np.array([m]))[0]
        return np.array([model(*x) for model in models])

    esmda = ESMDA(m=m_prior, g_obs=hist_scores[0], cd=np.ones(hist_scores.shape[1]))
    m_post, _ = esmda.run(g_func, alphas)
    return clip_to_prior_range(m_post, m_prior)


BACKENDS = {
    "Spectral": run_spectral,
    "PCESMDA": run_pcesmda,
}
