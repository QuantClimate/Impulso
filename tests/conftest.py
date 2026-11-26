"""Shared test fixtures for Impulso."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso import ar1_residual_sd
from impulso._arviz_compat import get_group_dataset, make_idata
from impulso.data import VARData

# --------------- Raw data helpers ---------------


@pytest.fixture
def rng():
    """Deterministic RNG for test reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_endog(rng):
    """3 variables, 100 observations."""
    return rng.standard_normal((100, 3))


@pytest.fixture
def sample_index():
    return pd.date_range("2000-01-01", periods=100, freq="QS")


@pytest.fixture
def endog_names():
    return ["gdp", "inflation", "rate"]


# --------------- VARData fixtures ---------------


@pytest.fixture
def var_data_2v():
    """VAR(1) DGP with 2 endogenous variables, 200 obs.

    Used across fitted, identified, and lag_selection tests.
    """
    rng = np.random.default_rng(42)
    T, n = 200, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


@pytest.fixture
def var_data_3v_dgp2():
    """VAR(2) DGP with 3 endogenous variables, 200 obs.

    Used for lag selection tests.
    """
    rng = np.random.default_rng(42)
    T, n = 200, 3
    y = np.zeros((T, n))
    for t in range(2, T):
        y[t] = 0.5 * y[t - 1] + 0.2 * y[t - 2] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2", "y3"], index=index)


# --------------- Synthetic InferenceData for fast tests ---------------


@pytest.fixture
def synthetic_idata_2v():
    """Synthetic InferenceData mimicking a fitted 2-var VAR(1).

    No MCMC required. 2 chains, 50 draws, 2 variables, 1 lag.
    B shape: (2, 50, 2, 2) -- coefficient matrix
    intercept shape: (2, 50, 2)
    Sigma shape: (2, 50, 2, 2) -- positive definite covariance
    """
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_vars, n_lags = 2, 50, 2, 1

    B = rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.3
    intercept = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01

    # Positive definite Sigma via A @ A.T + I, plus its Cholesky factor L
    # so the fixture mirrors a real Constant-adapter posterior.
    sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
    L = np.zeros_like(sigma)
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars)) * 0.5
            sigma[c, d] = A @ A.T + np.eye(n_vars)
            L[c, d] = np.linalg.cholesky(sigma[c, d])

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(
            sigma,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        ),
        "L": xr.DataArray(L, dims=["chain", "draw", "var1", "var2"]),
    })
    return make_idata(posterior=posterior)


@pytest.fixture
def synthetic_idata_2v_t(synthetic_idata_2v):
    """`synthetic_idata_2v` plus a constant `nu` of 5.0, for Student-t tests.

    A *sibling* of `synthetic_idata_2v` rather than a modification of it, so
    Gaussian tests keep the exact posterior they have always had. `nu` is
    constant across (chain, draw) on purpose: it makes exact distributional
    assertions about the innovation draws possible. Draw-varying `nu` is
    covered by the unit tests in test_error_distributions.py.
    """
    posterior = get_group_dataset(synthetic_idata_2v, "posterior").copy()
    n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    posterior["nu"] = xr.DataArray(
        np.full((n_chains, n_draws), 5.0),
        dims=["chain", "draw"],
    )
    return make_idata(posterior=posterior)


@pytest.fixture
def permanent_transitory_2v():
    """Exact-arithmetic 2-var VAR(1) with a known long-run structure.

    Built backwards from the answer so the long-run identification has a
    closed form to be checked against:

        A_1 = [[0.5, 0.1], [0.2, 0.4]]   (eigenvalues 0.6, 0.3 — stable)
        M   = I - A_1                    (det 0.28)
        C1  = M^-1                       (long-run multiplier)
        G   = [[1.0, 0.0], [0.5, 0.8]]   (imposed cumulative impact, G[0, 1] = 0)
        P   = M @ G = [[0.45, -0.08], [0.10, 0.48]]
        Sigma = P @ P.T

    `P` is deliberately not lower-triangular, so it is distinguishable
    from the Cholesky factor of Sigma.

    Returns:
        Dict with keys `A1`, `M`, `C1`, `G`, `P_true`, `Sigma`, `L`
        (Cholesky factor of Sigma) and `idata` — an InferenceData with 2
        chains x 50 draws whose draws all equal the truth, laid out like
        `synthetic_idata_2v`.
    """
    n_chains, n_draws, n_vars = 2, 50, 2

    A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
    M = np.eye(n_vars) - A1
    C1 = np.linalg.inv(M)
    G = np.array([[1.0, 0.0], [0.5, 0.8]])
    P_true = M @ G
    Sigma = P_true @ P_true.T
    L = np.linalg.cholesky(Sigma)

    shape = (n_chains, n_draws, n_vars, n_vars)
    B = np.broadcast_to(A1, shape).copy()
    intercept = np.zeros((n_chains, n_draws, n_vars))
    sigma_draws = np.broadcast_to(Sigma, shape).copy()
    L_draws = np.broadcast_to(L, shape).copy()

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(
            sigma_draws,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        ),
        "L": xr.DataArray(L_draws, dims=["chain", "draw", "var1", "var2"]),
    })
    return {
        "A1": A1,
        "M": M,
        "C1": C1,
        "G": G,
        "P_true": P_true,
        "Sigma": Sigma,
        "L": L_draws,
        "idata": make_idata(posterior=posterior),
    }


@pytest.fixture
def identified_long_run_nan_draws(permanent_transitory_2v, var_data_2v):
    """Long-run-identified 2-var VAR whose first 5 draws of chain 0 are NaN.

    Setting `B = I` on those draws makes `M = I - A_1` exactly singular, so
    `on_undefined="nan"` (the default) blanks them. Callers must expect the
    "long-run multiplier" warning the first time identification runs.
    """
    from impulso.fitted import FittedVAR
    from impulso.identification import LongRunRestriction
    from impulso.volatility import Constant

    n_bad = 5
    # Rebuild rather than assigning into the group in place: the two ArviZ
    # lines disagree about in-place group mutation (a `DataTree` leaf will not
    # accept a bare `(dims, values)` tuple the way a `Dataset` does).
    posterior = get_group_dataset(permanent_transitory_2v["idata"], "posterior").copy(deep=True)
    B = posterior["B"].values.copy()
    B[0, :n_bad] = np.eye(2)
    posterior["B"] = (("chain", "draw", "var", "coeff"), B)

    fitted = FittedVAR(
        idata=make_idata(posterior=posterior),
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    scheme = LongRunRestriction(ordering=["y1", "y2"], shock_names=["permanent", "transitory"])
    return fitted.set_identification_strategy(scheme)


# --------------- SV fixtures ---------------


@pytest.fixture
def sv_series_rw():
    """1-D series simulated from RW log-volatility SV DGP.

    T=500, sigma_eta=0.1. Used by slow recovery tests. Returns a
    dict with 'y', 'h_true', 'mu_true', 'sigma_eta_true' so
    recovery tests can compare to the truth.
    """
    rng = np.random.default_rng(42)
    T = 500
    sigma_eta_true = 0.1
    mu_true = 0.0
    h_true = np.zeros(T)
    h_true[0] = 0.0
    for t in range(1, T):
        h_true[t] = h_true[t - 1] + sigma_eta_true * rng.standard_normal()
    y = mu_true + np.exp(0.5 * h_true) * rng.standard_normal(T)
    return {
        "y": y,
        "h_true": h_true,
        "mu_true": mu_true,
        "sigma_eta_true": sigma_eta_true,
    }


@pytest.fixture
def sv_data_rw(sv_series_rw):
    """SVData wrapping the RW SV DGP series."""
    from impulso.sv.data import SVData

    index = pd.date_range("1980-01-01", periods=len(sv_series_rw["y"]), freq="MS")
    return SVData(y=sv_series_rw["y"], name="sim", index=index)


@pytest.fixture
def synthetic_sv_idata():
    """Synthetic InferenceData mimicking a fitted random-walk SV posterior.

    No MCMC required. 2 chains, 50 draws, T=100.
    posterior["h"] shape: (2, 50, 100)
    posterior["mu"] shape: (2, 50)
    posterior["sigma_eta"] shape: (2, 50)
    """
    rng = np.random.default_rng(123)
    n_chains, n_draws, T = 2, 50, 100

    # Simulate a plausible posterior around a mild vol path
    h_mean = np.linspace(-0.5, 0.5, T)
    h = h_mean[None, None, :] + 0.1 * rng.standard_normal((n_chains, n_draws, T))
    mu = 0.01 * rng.standard_normal((n_chains, n_draws))
    sigma_eta = 0.1 + 0.02 * np.abs(rng.standard_normal((n_chains, n_draws)))

    posterior = xr.Dataset({
        "h": xr.DataArray(h, dims=["chain", "draw", "time"]),
        "mu": xr.DataArray(mu, dims=["chain", "draw"]),
        "sigma_eta": xr.DataArray(sigma_eta, dims=["chain", "draw"]),
    })
    return make_idata(posterior=posterior)


@pytest.fixture
def synthetic_sv_idata_2v():
    """Synthetic InferenceData mimicking a fitted multivariate SV model.

    Shape: 2 chains, 50 draws, T=20, n_vars=2.
    Contains: h (2, 50, 20, 2), R_chol (2, 50, 2, 2), v0_h, v1_h, v0_mu,
    v1_mu, v0_sigma_eta, v1_sigma_eta.

    Respects the composition `build_pymc_latent` actually registers under
    random-walk dynamics: `h[..., i] == v{i}_h + v{i}_mu`, with `v{i}_h`
    the level-free path and `v{i}_mu` a nonzero per-variable level. An
    earlier version of this fixture set `v{i}_h = h[..., i]` *and* a
    nonzero `v{i}_mu`, which asserts `mu_i == 0` — the exact assumption
    the forecast bug made, so every test built on it was blind to
    the whole family of level bugs.
    """
    rng = np.random.default_rng(0)
    n_chains, n_draws, T, n_vars = 2, 50, 20, 2

    # Level-free per-variable RW paths and their separate levels.
    v_h = rng.standard_normal((n_chains, n_draws, T, n_vars)) * 0.3
    v_mu = rng.standard_normal((n_chains, n_draws, n_vars)) - 1.0
    # The invariant: the stacked in-sample log-vol is path + level.
    h = v_h + v_mu[:, :, None, :]

    R_chol = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars))
            R_chol[c, d] = np.linalg.cholesky(A @ A.T + np.eye(n_vars))
    diag_inv = 1.0 / np.diagonal(R_chol, axis1=-2, axis2=-1)[:, :, :, None]
    R_chol = R_chol * diag_inv

    posterior = xr.Dataset({
        "h": (("chain", "draw", "time", "variable"), h),
        "R_chol": (("chain", "draw", "i", "j"), R_chol),
        "v0_h": (("chain", "draw", "time"), v_h[:, :, :, 0]),
        "v1_h": (("chain", "draw", "time"), v_h[:, :, :, 1]),
        "v0_mu": (("chain", "draw"), v_mu[:, :, 0]),
        "v1_mu": (("chain", "draw"), v_mu[:, :, 1]),
        "v0_sigma_eta": (("chain", "draw"), np.abs(rng.standard_normal((n_chains, n_draws)))),
        "v1_sigma_eta": (("chain", "draw"), np.abs(rng.standard_normal((n_chains, n_draws)))),
    })
    return make_idata(posterior=posterior)


# --------------- VAR compiled for NumPyro ---------------


@pytest.fixture
def numpyro():
    """The `numpyro` module. Skips the test unless JAX and NumPyro are installed."""
    pytest.importorskip("jax")
    return pytest.importorskip("numpyro")


@pytest.fixture
def two_series_endog(rng):
    """Endogenous data of the two observed series `y1` and `y2`: 40 standard-normal rows, drawn from `rng`."""
    return rng.standard_normal((40, 2))


@pytest.fixture
def build_in_model_arguments():
    """A function giving `VAR.build_in_model`'s arguments other than `endog`, for a VAR on `y1` and `y2`.

    The function takes `endog`, the two observed columns, and `n_lags`, and
    the keyword options:

    - `n_latent`: latent series `b0, b1, ...` ahead of `y1` and `y2`, each
      with scale 1.
    - `exogenous_regressors`: two regressors `x1` and `x2` with standard
      deviations 0.5 and 3, so the exogenous prior's scaling by each
      regressor's standard deviation (ADR-0012) matters. Every call draws the
      same values.
    - `intercept_equations`, passed on as it is.

    `VAR.compile_for_numpyro` takes the same arguments.
    """

    def arguments(
        endog: np.ndarray,
        n_lags: int,
        *,
        n_latent: int = 0,
        exogenous_regressors: bool = False,
        intercept_equations: Sequence[str] | None = None,
    ) -> dict:
        latent_names = [f"b{i}" for i in range(n_latent)]
        kwargs = {
            "exog": None,
            "n_lags": n_lags,
            "endog_names": [*latent_names, "y1", "y2"],
            "endog_scales": np.concatenate([np.ones(n_latent), ar1_residual_sd(endog)]),
        }
        if latent_names:
            kwargs["latent_names"] = latent_names
        if exogenous_regressors:
            exog_rng = np.random.default_rng(7)
            kwargs["exog"] = exog_rng.standard_normal((endog.shape[0], 2)) * np.array([0.5, 3.0])
            kwargs["exog_names"] = ["x1", "x2"]
        if intercept_equations is not None:
            kwargs["intercept_equations"] = list(intercept_equations)
        return kwargs

    return arguments


@pytest.fixture
def build_pymc_model():
    """A function that builds a VAR in PyMC with `build_in_model` and returns the `pymc.Model`.

    It takes the `VAR`, `endog` and `build_in_model`'s other arguments, and
    gives `build_in_model` a symbolic tensor holding `endog`.
    """

    def build(spec, endog: np.ndarray, kwargs: dict):
        import pymc as pm
        import pytensor

        with pm.Model() as model:
            spec.build_in_model(endog=pytensor.shared(endog), **kwargs)
        return model

    return build


@pytest.fixture
def observed_endog_model():
    """A function from a compiled NumPyro block to a NumPyro model function that passes observed data to the
    block as `endog`."""

    def numpyro_model_for(block):
        def model(endog):
            return block(endog)

        return model

    return numpyro_model_for
