"""Eigenfactor risk adjustment and round-7 registration."""
import numpy as np

from DataAnalysisPipeline2.v3 import gate
from DataAnalysisPipeline2.v3 import round7 as r7


def _F(k=12, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(k, k))
    return A @ A.T / k * 1e-4 + np.diag(np.linspace(1e-6, 1e-4, k))


def test_eigen_adjust_scale_zero_is_identity_and_psd():
    F = _F()
    assert np.allclose(r7.eigen_adjust(F, sims=20, T=100, scale=0.0), F, rtol=1e-8, atol=1e-14)
    Fa = r7.eigen_adjust(F, sims=50, T=100, scale=1.2)
    assert np.allclose(Fa, Fa.T) and np.linalg.eigvalsh(Fa).min() > 0


def test_eigen_adjust_inflates_smallest_eigenportfolios():
    """Sample eigenvalues are spread out: the smallest are under-estimated, so
    their adjusted variance must go UP (v_k > 1), the largest barely moves."""
    F = _F(k=15, seed=1)
    d0, U = np.linalg.eigh(F)
    Fa = r7.eigen_adjust(F, sims=200, T=60, scale=1.0, seed=3)
    da = np.einsum("ik,ij,jk->k", U, Fa, U)
    ratio = da / d0
    assert ratio[0] > 1.2 and ratio[0] > ratio[-1]
    assert abs(ratio[-1] - 1) < 0.25


def test_r7_registration():
    assert gate.N_TRIALS_R7 == 85 and len(gate.R7_CONFIGS) == 8
    perf = {"v2rec_tr/mv_abs_all_100m": {"ir_spy": 0.1, "max_dd": -0.30},
            "stack/lo30": {"ir_spy": 0.3, "max_dd": -0.40},
            "v2rec_tr/mv_abs_all_1m": {"ir_spy": 0.9, "max_dd": -0.10}}     # not at the primary AUM
    assert gate.choose_config_r7(perf) == "v2rec_tr/mv_abs_all_100m"
