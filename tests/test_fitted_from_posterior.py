"""Tests for `FittedVAR.from_posterior`.

`from_posterior` is the public, validated alternative to
`FittedVAR.model_construct`: it wraps a posterior produced elsewhere,
checking it against the schema `VAR.fit` and `ConjugateVAR.fit` both
produce before constructing. Neither estimator calls it yet.

Most tests build a hand-rolled posterior via `_build_posterior` below — no
MCMC needed, mirroring `conftest.synthetic_idata_2v`. A handful of tests
marked `slow` exercise real `VAR.fit` output (PyMC/NUTS); `ConjugateVAR.fit`
is closed-form and fast, so its acceptance test is not marked.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso._arviz_compat import get_group_dataset, make_idata
from impulso.conjugate import ConjugateVAR
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.observation import Gaussian, StudentT
from impulso.priors import NIWPrior
from impulso.samplers import NUTSSampler
from impulso.spec import VAR
from impulso.sv.spec import StochasticVolatility
from impulso.volatility import Constant

# --------------- Posterior builder ---------------


def _coefficients(rng, n_chains, n_vars, n_draws, n_lags, endog_names, *, labels, bad_labels):
    values = rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.2
    coords = {}
    if labels:
        expected = [f"L{lag}.{name}" for lag in range(1, n_lags + 1) for name in endog_names]
        coords["coeff"] = list(reversed(expected)) if bad_labels else expected
        coords["var"] = list(reversed(endog_names)) if bad_labels else list(endog_names)
    return xr.DataArray(values, dims=["chain", "draw", "var", "coeff"], coords=coords or None)


def _intercept(rng, n_chains, n_vars, n_draws, endog_names, *, labels, bad_labels):
    values = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01
    coords = {"var": list(reversed(endog_names)) if bad_labels else list(endog_names)} if labels else None
    return xr.DataArray(values, dims=["chain", "draw", "var"], coords=coords)


def _cholesky(rng, n_chains, n_vars, n_draws, dim_names=("var1", "var2")):
    L = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars)) * 0.3
            L[c, d] = np.linalg.cholesky(A @ A.T + np.eye(n_vars))
    return xr.DataArray(L, dims=["chain", "draw", *dim_names])


def _log_volatility(rng, n_chains, n_vars, n_draws, t, dim_names=("time", "var")):
    """`h`, the per-variable log-volatility path `StochasticVolatility.cholesky_at` reads.

    Dim *names* are arbitrary (the validator checks shape only — see
    `_posterior_validation`'s module docstring for why); trailing shape is
    `(t, n_vars)`, matching how `build_pymc_latent` builds it before
    chain/draw are prepended.
    """
    values = rng.standard_normal((n_chains, n_draws, t, n_vars)) * 0.1
    return xr.DataArray(values, dims=["chain", "draw", *dim_names])


def _mixing_cholesky(rng, n_chains, n_vars, n_draws, dim_names=("var1", "var2")):
    """`R_chol`, the unit-diagonal mixing factor `StochasticVolatility.cholesky_at` reads."""
    values = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            offdiag = rng.standard_normal((n_vars, n_vars)) * 0.1
            values[c, d] = np.eye(n_vars) + np.tril(offdiag, k=-1)
    return xr.DataArray(values, dims=["chain", "draw", *dim_names])


def _add_volatility_variables(variables, rng, n_chains, n_vars, n_draws, *, sv, sv_time, omit):
    """Add either `h`/`R_chol` (`sv=True`) or `L` (`sv=False`) — mutually exclusive."""
    if sv:
        if "h" not in omit:
            variables["h"] = _log_volatility(rng, n_chains, n_vars, n_draws, sv_time)
        if "R_chol" not in omit:
            variables["R_chol"] = _mixing_cholesky(rng, n_chains, n_vars, n_draws)
    elif "L" not in omit:
        variables["L"] = _cholesky(rng, n_chains, n_vars, n_draws)


def _add_exog_variable(variables, rng, n_chains, n_vars, n_draws, endog_names, *, n_exog, exog_names, exog_labels):
    """Add `B_exog` when `n_exog` is given."""
    if n_exog is None:
        return
    values = rng.standard_normal((n_chains, n_draws, n_vars, n_exog))
    coords = {}
    if exog_labels:
        coords["var"] = endog_names
        if exog_names is not None:
            coords["exog"] = list(exog_names)
    variables["B_exog"] = xr.DataArray(values, dims=["chain", "draw", "var", "exog"], coords=coords or None)


def _build_posterior(
    *,
    n_chains=2,
    n_draws=5,
    n_vars=2,
    n_lags=1,
    endog_names=("y1", "y2"),
    coeff_labels=False,
    bad_coeff_labels=False,
    var_labels=False,
    bad_var_labels=False,
    n_exog=None,
    exog_names=None,
    exog_labels=False,
    nu=None,
    sv=False,
    sv_time=40,
    omit=(),
    extra=None,
    seed=0,
):
    """A hand-built `posterior` group, mirroring what `VAR.fit`/`ConjugateVAR.fit` produce.

    `coeff_labels`/`var_labels` add explicit string coordinates (as
    `VAR.fit` does via `pm.Model(coords=...)`); leaving them `False`
    mirrors `ConjugateVAR.fit`, whose posterior carries no coordinate
    labels at all. `bad_*_labels` scrambles those labels to exercise the
    label-mismatch rejection path. `omit` drops named variables entirely;
    `extra` overrides or adds raw `xr.DataArray`s after the defaults are
    built, for constructing malformed shapes.

    `sv=True` builds an `h` / `R_chol` pair (the `StochasticVolatility`
    contract) instead of `L` (the `Constant` / `ConjugateVolatility`
    contract) — the two are mutually exclusive, matching how a real fit
    only ever produces one or the other. `sv_time` sets `h`'s in-sample
    time-axis length; callers pass `data.endog.shape[0] - n_lags` to match
    a real `VARData`.
    """
    rng = np.random.default_rng(seed)
    endog_names = list(endog_names)
    variables: dict[str, xr.DataArray] = {}

    if "B" not in omit:
        variables["B"] = _coefficients(
            rng, n_chains, n_vars, n_draws, n_lags, endog_names, labels=coeff_labels, bad_labels=bad_coeff_labels
        )
    if "intercept" not in omit:
        variables["intercept"] = _intercept(
            rng, n_chains, n_vars, n_draws, endog_names, labels=var_labels, bad_labels=bad_var_labels
        )
    _add_volatility_variables(variables, rng, n_chains, n_vars, n_draws, sv=sv, sv_time=sv_time, omit=omit)
    if "B_exog" not in omit:
        _add_exog_variable(
            variables,
            rng,
            n_chains,
            n_vars,
            n_draws,
            endog_names,
            n_exog=n_exog,
            exog_names=exog_names,
            exog_labels=exog_labels,
        )
    if nu is not None:
        variables["nu"] = xr.DataArray(np.full((n_chains, n_draws), nu), dims=["chain", "draw"])

    if extra:
        variables.update(extra)

    return make_idata(posterior=xr.Dataset(variables))


# --------------- Fixtures ---------------


@pytest.fixture
def var_data(var_data_2v):
    return var_data_2v


@pytest.fixture
def var_data_exog():
    rng = np.random.default_rng(3)
    t, n = 60, 2
    y = np.zeros((t, n))
    for i in range(1, t):
        y[i] = 0.5 * y[i - 1] + rng.standard_normal(n) * 0.1
    exog = rng.standard_normal((t, 2))
    index = pd.date_range("2000-01-01", periods=t, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], exog=exog, exog_names=["z1", "z2"], index=index)


# --------------- Defaults and pass-through ---------------


class TestFromPosteriorDefaults:
    def test_defaults_volatility_to_constant(self, var_data):
        idata = _build_posterior()
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1)
        assert isinstance(fitted.volatility, Constant)

    def test_defaults_error_dist_to_gaussian(self, var_data):
        idata = _build_posterior()
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1)
        assert isinstance(fitted.error_dist, Gaussian)

    def test_var_names_come_from_data_endog_names(self, var_data):
        idata = _build_posterior()
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1)
        assert fitted.var_names == var_data.endog_names

    def test_n_lags_and_data_are_stored(self, var_data):
        idata = _build_posterior()
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1)
        assert fitted.n_lags == 1
        assert fitted.data is var_data

    def test_accepts_explicit_volatility_and_error_dist(self, var_data):
        idata = _build_posterior()
        volatility = Constant(sigma_sd_beta=1.0)
        error_dist = Gaussian()
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1, volatility=volatility, error_dist=error_dist)
        assert fitted.volatility is volatility
        assert fitted.error_dist is error_dist

    def test_passes_through_pymc_model(self, var_data):
        idata = _build_posterior()
        sentinel = object()
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1, pymc_model=sentinel)
        assert fitted.pymc_model is sentinel

    def test_stores_the_whole_idata_including_extra_groups(self, var_data):
        posterior = get_group_dataset(_build_posterior(), "posterior")
        sample_stats = xr.Dataset({"diverging": xr.DataArray(np.zeros((2, 5), dtype=bool), dims=["chain", "draw"])})
        idata = make_idata(posterior=posterior, sample_stats=sample_stats)
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1)
        assert "diverging" in get_group_dataset(fitted.idata, "sample_stats")


# --------------- Rejections: missing variables ---------------


class TestFromPosteriorRejectsMissingVariables:
    @pytest.mark.parametrize("name", ["B", "intercept", "L"])
    def test_rejects_missing_required_variable(self, var_data, name):
        idata = _build_posterior(omit=(name,))
        with pytest.raises(ValueError, match=f"missing required variable '{name}'"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)

    def test_error_names_variable_and_expected_layout(self, var_data):
        idata = _build_posterior(omit=("B",))
        with pytest.raises(ValueError) as exc_info:
            FittedVAR.from_posterior(idata, var_data, n_lags=1)
        message = str(exc_info.value)
        assert "'B'" in message
        assert "lag-major" in message

    def test_rejects_missing_b_exog_when_data_has_exog(self, var_data_exog):
        idata = _build_posterior(n_exog=None)  # no B_exog at all
        with pytest.raises(ValueError, match="missing required variable 'B_exog'"):
            FittedVAR.from_posterior(idata, var_data_exog, n_lags=1)

    def test_rejects_missing_nu_under_student_t(self, var_data):
        idata = _build_posterior()  # no nu
        with pytest.raises(ValueError, match="missing required variable 'nu'"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1, error_dist=StudentT())


# --------------- Rejections: wrong dims ---------------


class TestFromPosteriorRejectsWrongDims:
    def test_rejects_b_missing_coeff_dim(self, var_data):
        rng = np.random.default_rng(0)
        bad_b = xr.DataArray(rng.standard_normal((2, 5, 2)), dims=["chain", "draw", "var"])
        idata = _build_posterior(extra={"B": bad_b})
        with pytest.raises(ValueError, match="'B' has dims"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)

    def test_rejects_intercept_with_extra_dim(self, var_data):
        rng = np.random.default_rng(0)
        bad_intercept = xr.DataArray(rng.standard_normal((2, 5, 2, 1)), dims=["chain", "draw", "var", "extra"])
        idata = _build_posterior(extra={"intercept": bad_intercept})
        with pytest.raises(ValueError, match="'intercept' has dims"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)

    def test_rejects_l_with_only_three_dims(self, var_data):
        rng = np.random.default_rng(0)
        bad_l = xr.DataArray(rng.standard_normal((2, 5, 2)), dims=["chain", "draw", "var1"])
        idata = _build_posterior(extra={"L": bad_l})
        with pytest.raises(ValueError, match="'L' has dims"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)


# --------------- Rejections: wrong column counts ---------------


class TestFromPosteriorRejectsWrongColumnCounts:
    def test_rejects_b_coeff_count_not_matching_n_vars_times_n_lags(self, var_data):
        # n_vars=2, n_lags=1 -> coeff should be 2, not 3.
        rng = np.random.default_rng(0)
        bad_b = xr.DataArray(rng.standard_normal((2, 5, 2, 3)), dims=["chain", "draw", "var", "coeff"])
        idata = _build_posterior(extra={"B": bad_b})
        with pytest.raises(ValueError, match="'B' dim 'coeff' has size 3, expected 2"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)

    def test_rejects_b_var_count_not_matching_n_vars(self, var_data):
        # "var" is a shared dimension across B/intercept/L, so every variable
        # in this posterior is built at n_vars=3 (internally consistent);
        # `var_data` (n_vars=2) is what disagrees with it.
        idata = _build_posterior(n_vars=3, endog_names=("y1", "y2", "y3"))
        with pytest.raises(ValueError, match="'B' dim 'var' has size 3, expected 2"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)

    def test_rejects_l_trailing_shape_not_matching_n_vars(self, var_data):
        rng = np.random.default_rng(0)
        bad_l = xr.DataArray(rng.standard_normal((2, 5, 3, 3)), dims=["chain", "draw", "var1", "var2"])
        idata = _build_posterior(extra={"L": bad_l})
        with pytest.raises(ValueError, match=r"'L' has trailing shape \[3, 3\]"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)

    def test_rejects_b_exog_column_count_not_matching_data_exog(self, var_data_exog):
        # var_data_exog has 2 exog columns; posterior only carries 1.
        idata = _build_posterior(n_exog=1)
        with pytest.raises(ValueError, match="'B_exog' dim 'exog' has size 1, expected 2"):
            FittedVAR.from_posterior(idata, var_data_exog, n_lags=1)


# --------------- Rejections: bad explicit labels ---------------


class TestFromPosteriorRejectsBadLabels:
    def test_rejects_non_lag_major_coeff_labels(self, var_data):
        idata = _build_posterior(coeff_labels=True, bad_coeff_labels=True)
        with pytest.raises(ValueError, match="'B' 'coeff' labels"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)

    def test_rejects_var_labels_disagreeing_with_endog_names(self, var_data):
        # "var" is a Dataset-wide coordinate shared by every var-dim variable
        # (intercept is the only one that declares it here), so the mismatch
        # surfaces on 'B' — the first var-dim variable `from_posterior` checks.
        idata = _build_posterior(var_labels=True, bad_var_labels=True)
        with pytest.raises(ValueError, match="'B' 'var' labels"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1)

    def test_rejects_var_major_coeff_labels_at_n_lags_2(self, var_data):
        """Lag-major is `L1.y1, L1.y2, L2.y1, L2.y2`; var-major (all of y1's lags,
        then all of y2's) is a plausible mistake and must be rejected too."""
        n_lags = 2
        rng = np.random.default_rng(0)
        var_major_b = xr.DataArray(
            rng.standard_normal((2, 5, 2, 4)) * 0.2,
            dims=["chain", "draw", "var", "coeff"],
            coords={"coeff": ["L1.y1", "L2.y1", "L1.y2", "L2.y2"], "var": ["y1", "y2"]},
        )
        idata = _build_posterior(n_lags=n_lags, extra={"B": var_major_b})
        with pytest.raises(ValueError, match="'B' 'coeff' labels"):
            FittedVAR.from_posterior(idata, var_data, n_lags=n_lags)


# --------------- Volatility-adapter dispatch: Constant vs StochasticVolatility ---------------


class _UnknownVolatilityAdapter:
    """Stands in for a volatility adapter `_posterior_validation` doesn't recognise.

    Deliberately not a `Constant`, `ConjugateVolatility`, or `StochasticVolatility`
    subclass (and not registered with any of them), so it exercises the
    "unrecognised adapter" branch of `_validate_volatility` rather than
    accidentally satisfying one of the known ones structurally.
    """


class TestFromPosteriorVolatilityDispatch:
    def test_accepts_sv_shaped_posterior(self, var_data):
        n_lags = 1
        expected_t = var_data.endog.shape[0] - n_lags
        idata = _build_posterior(sv=True, sv_time=expected_t, n_lags=n_lags)
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=n_lags, volatility=StochasticVolatility())
        assert isinstance(fitted.volatility, StochasticVolatility)

    def test_rejects_missing_h_under_sv(self, var_data):
        n_lags = 1
        expected_t = var_data.endog.shape[0] - n_lags
        idata = _build_posterior(sv=True, sv_time=expected_t, n_lags=n_lags, omit=("h",))
        with pytest.raises(ValueError, match="missing required variable 'h'"):
            FittedVAR.from_posterior(idata, var_data, n_lags=n_lags, volatility=StochasticVolatility())

    def test_rejects_missing_r_chol_under_sv(self, var_data):
        n_lags = 1
        expected_t = var_data.endog.shape[0] - n_lags
        idata = _build_posterior(sv=True, sv_time=expected_t, n_lags=n_lags, omit=("R_chol",))
        with pytest.raises(ValueError, match="missing required variable 'R_chol'"):
            FittedVAR.from_posterior(idata, var_data, n_lags=n_lags, volatility=StochasticVolatility())

    def test_rejects_h_with_wrong_time_length(self, var_data):
        n_lags = 1
        expected_t = var_data.endog.shape[0] - n_lags
        idata = _build_posterior(sv=True, sv_time=expected_t + 5, n_lags=n_lags)
        with pytest.raises(ValueError, match=rf"'h' has trailing shape \[{expected_t + 5}, 2\]"):
            FittedVAR.from_posterior(idata, var_data, n_lags=n_lags, volatility=StochasticVolatility())

    def test_rejects_unrecognised_volatility_adapter(self, var_data):
        idata = _build_posterior()
        with pytest.raises(TypeError, match="unrecognised volatility adapter '_UnknownVolatilityAdapter'"):
            FittedVAR.from_posterior(idata, var_data, n_lags=1, volatility=_UnknownVolatilityAdapter())


# --------------- Acceptance: exog and Student-t ---------------


class TestFromPosteriorAcceptsExogAndStudentT:
    def test_accepts_b_exog_matching_data_exog(self, var_data_exog):
        idata = _build_posterior(n_exog=2, exog_labels=True, exog_names=["z1", "z2"])
        fitted = FittedVAR.from_posterior(idata, var_data_exog, n_lags=1)
        assert fitted.has_exog is True

    def test_ignores_b_exog_when_data_has_no_exog(self, var_data):
        """An estimator may carry exog data it never consumed (see `dynamic_multiplier`'s
        own comment on `has_exog_block` vs `has_exog`); from_posterior does not reject it."""
        idata = _build_posterior(n_exog=2, exog_labels=False)
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1)
        assert fitted.has_exog is False

    def test_accepts_nu_under_student_t(self, var_data):
        idata = _build_posterior(nu=5.0)
        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1, error_dist=StudentT())
        assert isinstance(fitted.error_dist, StudentT)


# --------------- Acceptance: truncated latent own lag ---------------


class TestFromPosteriorAcceptsTheTruncatedOwnLagLayout:
    """`build_in_model` with one latent series and `n_lags=1` records `B` as a `Deterministic` assembled from
    the free block `B_free` and the truncated own lag `latent_own_lag`, both also in the posterior."""

    def test_coefficients_are_the_b_draws_and_the_extra_variables_are_ignored(self, var_data):
        rng = np.random.default_rng(5)
        own_lag = rng.uniform(-0.9, 0.9, size=(2, 5))
        free = 0.2 * rng.standard_normal((2, 5, 3))
        B = np.concatenate([own_lag[..., None], free], axis=-1).reshape(2, 5, 2, 2)
        idata = _build_posterior(
            extra={
                "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
                "B_free": xr.DataArray(free, dims=["chain", "draw", "B_free_dim_0"]),
                "latent_own_lag": xr.DataArray(own_lag, dims=["chain", "draw"]),
            }
        )

        fitted = FittedVAR.from_posterior(idata, var_data, n_lags=1)

        np.testing.assert_array_equal(fitted.coefficients, B)
        assert np.isfinite(fitted.forecast(steps=2).median().to_numpy()).all()


# --------------- Acceptance: real estimator posteriors ---------------


class TestFromPosteriorAcceptsEstimatorPosteriors:
    def test_accepts_conjugate_var_fit_posterior(self, var_data):
        model = ConjugateVAR(lags=1, prior=NIWPrior(), draws=30, tune=30, seed=0)
        original = model.fit(var_data)

        rewrapped = FittedVAR.from_posterior(
            original.idata,
            var_data,
            original.n_lags,
            volatility=original.volatility,
            error_dist=original.error_dist,
            evidence=original.evidence,
        )

        assert isinstance(rewrapped, FittedVAR)
        np.testing.assert_array_equal(rewrapped.coefficients, original.coefficients)
        assert rewrapped.evidence is original.evidence

    @pytest.mark.slow
    def test_accepts_var_fit_default_posterior(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=20, tune=20, chains=1, cores=1, random_seed=0)
        original = spec.fit(var_data, sampler=sampler)

        rewrapped = FittedVAR.from_posterior(original.idata, var_data, original.n_lags)

        assert isinstance(rewrapped, FittedVAR)
        np.testing.assert_array_equal(rewrapped.coefficients, original.coefficients)

    @pytest.mark.slow
    def test_accepts_var_fit_posterior_with_exog_and_student_t(self, var_data_exog):
        spec = VAR(lags=1, prior="minnesota", error_dist="student_t")
        sampler = NUTSSampler(draws=20, tune=20, chains=1, cores=1, random_seed=0)
        original = spec.fit(var_data_exog, sampler=sampler)

        rewrapped = FittedVAR.from_posterior(
            original.idata,
            var_data_exog,
            original.n_lags,
            volatility=original.volatility,
            error_dist=original.error_dist,
        )

        assert isinstance(rewrapped, FittedVAR)
        assert rewrapped.has_exog is True
        np.testing.assert_array_equal(rewrapped.coefficients, original.coefficients)

    @pytest.mark.slow
    def test_accepts_var_fit_posterior_with_stochastic_volatility(self, var_data):
        spec = VAR(lags=1, prior="minnesota", volatility="sv")
        sampler = NUTSSampler(draws=10, tune=10, chains=1, cores=1, random_seed=0)
        original = spec.fit(var_data, sampler=sampler)

        rewrapped = FittedVAR.from_posterior(original.idata, var_data, original.n_lags, volatility=original.volatility)

        assert isinstance(rewrapped, FittedVAR)
        assert isinstance(rewrapped.volatility, StochasticVolatility)
        np.testing.assert_array_equal(rewrapped.coefficients, original.coefficients)
