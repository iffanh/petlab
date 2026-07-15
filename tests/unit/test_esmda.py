"""ES-MDA math tests against small synthetic problems -- no simulator
needed."""

import numpy as np

from petlab.historymatch.esmda import ESMDA, clip_to_prior_range, update_plsr


def test_esmda_update_moves_toward_truth_linear_model():
    """A minimal linear inverse problem: g(m) = A @ m. The posterior mean
    should end up closer to the true parameter than the prior mean."""

    rng = np.random.default_rng(0)
    Nc, Nd, Ne = 3, 4, 30

    A = rng.normal(size=(Nd, Nc))
    m_true = np.array([2.0, -1.0, 0.5])
    g_obs = A @ m_true

    m_prior = rng.normal(loc=0.0, scale=2.0, size=(Ne, Nc))
    g_prior = m_prior @ A.T

    esmda = ESMDA(m=m_prior, g_obs=g_obs, cd=np.full(Nd, 0.01))
    m_post = esmda.update(m_prior, g_prior, alpha=1.0)

    prior_error = np.linalg.norm(np.mean(m_prior, axis=0) - m_true)
    post_error = np.linalg.norm(np.mean(m_post, axis=0) - m_true)
    assert post_error < prior_error


def test_esmda_run_with_schedule_further_reduces_error():
    rng = np.random.default_rng(1)
    Nc, Nd, Ne = 2, 3, 25

    A = rng.normal(size=(Nd, Nc))
    m_true = np.array([1.0, 3.0])
    g_obs = A @ m_true

    m_prior = rng.normal(loc=0.0, scale=3.0, size=(Ne, Nc))

    def g_func(m):
        return A @ m

    esmda = ESMDA(m=m_prior, g_obs=g_obs, cd=np.full(Nd, 0.01))
    m_post, g_post = esmda.run(g_func, alphas=[4.0, 4.0, 2.0, 1.0])

    prior_error = np.linalg.norm(np.mean(m_prior, axis=0) - m_true)
    post_error = np.linalg.norm(np.mean(m_post, axis=0) - m_true)
    assert post_error < prior_error


def test_update_plsr_reduces_wide_field_toward_truth():
    """A wide parameter matrix (Nc >> Ne), as with a full permeability
    field, updated through PLSR reduction."""

    rng = np.random.default_rng(2)
    Ne, Nc = 10, 40

    true_mean = 5.0
    m_prior = rng.normal(loc=2.0, scale=1.0, size=(Ne, Nc))

    # every "cell" responds to the same underlying mean shift, plus noise --
    # a stand-in for a permeability field driving a single production curve
    weights = rng.normal(scale=0.1, size=Nc)
    sim_array = (m_prior @ weights)[:, None] + rng.normal(scale=0.01, size=(Ne, 1))
    hist_vector = np.array([true_mean * np.sum(weights)])

    m_post = update_plsr(m_prior, sim_array, hist_vector, ncomponent=3, alpha=1.0)

    prior_gap = abs(np.mean(m_prior) - true_mean)
    post_gap = abs(np.mean(m_post) - true_mean)
    assert post_gap < prior_gap


def test_clip_to_prior_range():
    prior = np.array([[0.0, 1.0], [2.0, 3.0]])
    posterior = np.array([[-5.0, 10.0], [1.0, 2.0]])
    clipped = clip_to_prior_range(posterior, prior)

    np.testing.assert_allclose(clipped[0], [0.0, 3.0])
    np.testing.assert_allclose(clipped[1], [1.0, 2.0])
