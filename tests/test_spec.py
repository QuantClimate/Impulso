"""Tests for VAR specification."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from impulso._arviz_compat import make_idata
from impulso._conjugate import ar1_residual_sd
from impulso.data import VARData
from impulso.observation import Gaussian, StudentT
from impulso.priors import MinnesotaPrior
from impulso.spec import VAR, _exog_prior_sigma
from impulso.sv.dynamics import AR1, RandomWalk
from impulso.sv.priors import SVDefaultPrior
from impulso.sv.spec import StochasticVolatility
from impulso.volatility import Constant, InnovationScalePrior


def _exog_data(endog: np.ndarray, exog: np.ndarray, exog_names: list[str]) -> VARData:
    """VARData wrapper for the exog-prior tests."""
    return VARData(
        endog=endog,
        endog_names=[f"y{i + 1}" for i in range(endog.shape[1])],
        exog=exog,
        exog_names=exog_names,
        index=pd.date_range("2000-01-01", periods=endog.shape[0], freq="QS"),
    )


def _captured_sigma(model, name: str) -> np.ndarray:
    """Evaluate the sigma tensor a `pm.Normal` RV was built with.

    `dist_params` maps the RV's op back onto its distribution parameters, so
    this survives the positional reshuffles PyMC makes to `owner.inputs`.
    """
    rv = model[name]
    _mu, sigma = rv.owner.op.dist_params(rv.owner)
    return np.asarray(sigma.eval())


class TestVARSpec:
    def test_fixed_lags(self):
        spec = VAR(lags=4, prior="minnesota")
        assert spec.lags == 4
        assert spec.max_lags is None

    @pytest.mark.parametrize("criterion", ["aic", "bic", "hq"])
    def test_string_lags(self, criterion):
        spec = VAR(lags=criterion, prior="minnesota")
        assert spec.lags == criterion

    def test_string_lags_with_max(self):
        spec = VAR(lags="aic", max_lags=12, prior="minnesota")
        assert spec.max_lags == 12

    def test_rejects_max_lags_with_fixed(self):
        with pytest.raises(ValueError, match="max_lags is only valid"):
            VAR(lags=4, max_lags=12, prior="minnesota")

    def test_prior_string_resolves(self):
        spec = VAR(lags=2, prior="minnesota")
        assert isinstance(spec.resolved_prior, MinnesotaPrior)

    def test_prior_object(self):
        prior = MinnesotaPrior(tightness=0.2)
        spec = VAR(lags=2, prior=prior)
        assert spec.resolved_prior.tightness == 0.2

    def test_frozen(self):
        spec = VAR(lags=2, prior="minnesota")
        with pytest.raises(ValidationError):
            spec.lags = 3

    def test_rejects_zero_lags(self):
        with pytest.raises(ValidationError):
            VAR(lags=0, prior="minnesota")

    def test_default_sampler_is_safe(self):
        """The sampler `VAR.fit` falls back to must pin cores=1 to dodge the segfault."""
        from impulso.samplers import NUTSSampler

        sampler = VAR._default_sampler()
        assert isinstance(sampler, NUTSSampler)
        assert sampler.cores == 1
        assert sampler.chains == 4
        assert sampler.target_accept == 0.8


class TestVolatilityParameter:
    def test_default_volatility_is_constant_string(self):
        spec = VAR(lags=2)
        assert spec.volatility == "constant"

    def test_volatility_string_resolves_to_constant(self):
        spec = VAR(lags=2)
        assert isinstance(spec.resolved_volatility, Constant)

    def test_volatility_object_pass_through(self):
        adapter = Constant(sigma_sd_beta=3.0)
        spec = VAR(lags=2, volatility=adapter)
        assert spec.resolved_volatility is adapter
        assert spec.resolved_volatility.sigma_sd_beta == 3.0

    def test_unknown_string_raises(self):
        with pytest.raises(ValidationError):
            VAR(lags=2, volatility="nonexistent")


class TestPyMCModelBuild:
    """Verify the PyMC model graph composition after the seam refactor."""

    def test_model_has_expected_unobserved_rvs(self, var_data_2v):
        """The same set of RVs must exist in the PyMC graph as before the seam."""
        # We don't run MCMC here; we just build the model and inspect.
        # Rebuild logic mirrors VAR.fit but stops before sampler.sample().
        import pymc as pm

        from impulso._lag_selection import select_lag_order  # noqa: F401 — import-side-effect parity

        spec = VAR(lags=1)
        prior_params = spec.resolved_prior.build_priors(
            n_vars=2,
            n_lags=1,
            sigma=np.ones(2),
        )
        volatility = spec.resolved_volatility

        y = var_data_2v.endog
        Y = y[1:]
        X_lag = y[:-1]

        with pm.Model() as model:
            pm.Normal("intercept", mu=0, sigma=1, shape=2)
            pm.Normal("B", mu=prior_params["B_mu"], sigma=prior_params["B_sigma"], shape=(2, 2))
            L = volatility.build_pymc_latent(n_vars=2, T=Y.shape[0])
            pm.Deterministic("Sigma", pm.math.dot(L, L.T))
            mu = pm.math.dot(X_lag, model.named_vars["B"].T)
            pm.MvNormal("obs", mu=mu, chol=L, observed=Y)

        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}
        assert "intercept" in rv_names
        assert "B" in rv_names
        assert "sigma_sd" in rv_names
        assert "tril_offdiag" in rv_names
        assert "Sigma" in det_names

    def test_var_fit_routes_through_volatility_seam(self, var_data_2v):
        """VAR.fit (intercepted before sampling) must register the canonical RV set.

        The companion test above mirrors the model graph by hand, so it would
        keep passing even if VAR.fit silently skipped the volatility delegation.
        This one captures the model from the production codepath via a sampler
        that aborts before MCMC, then asserts the graph shape.
        """
        import pymc as pm

        captured: dict[str, pm.Model] = {}

        class CapturingSampler:
            name = "capture"

            def sample(self, model: pm.Model):
                captured["model"] = model
                raise RuntimeError("stop before sampling")

        spec = VAR(lags=1)
        with pytest.raises(RuntimeError, match="stop before sampling"):
            spec.fit(var_data_2v, sampler=CapturingSampler())

        model = captured["model"]
        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}
        assert {"intercept", "B", "sigma_sd", "tril_offdiag"} <= rv_names
        assert "Sigma" in det_names
        assert {v.name for v in model.observed_RVs} == {"obs"}


class TestVarFitWithInnovationScalePriors:
    """`Constant.innovation_scale_priors` wired through `VAR.fit`."""

    def test_mismatched_length_raises_before_sampling(self, var_data_2v):
        """A length mismatch must surface as a `ValueError` from model building,
        before the sampler is ever invoked (`_build_pymc_model` runs first in
        `VAR.fit`, so a real sampler is never reached)."""
        from impulso.volatility import InnovationScalePrior

        bad_volatility = Constant(innovation_scale_priors=(InnovationScalePrior(family="halfnormal", scale=0.1),))
        spec = VAR(lags=1, volatility=bad_volatility)

        with pytest.raises(ValueError, match="innovation_scale_priors"):
            spec.fit(var_data_2v)

    def test_matched_length_registers_per_variable_rvs(self, var_data_2v):
        """A correctly-sized `innovation_scale_priors` builds cleanly through
        the real `VAR.fit` pipeline (intercepted before sampling)."""
        from impulso.volatility import InnovationScalePrior

        priors = (
            InnovationScalePrior(family="halfnormal", scale=0.2),
            InnovationScalePrior(family="halfcauchy", scale=2.5),
        )
        spec = VAR(
            lags=1,
            volatility=Constant(innovation_scale_priors=priors),
        )

        captured: dict[str, object] = {}

        class CapturingSampler:
            name = "capture"

            def sample(self, model):
                captured["model"] = model
                raise RuntimeError("stop before sampling")

        with pytest.raises(RuntimeError, match="stop before sampling"):
            spec.fit(var_data_2v, sampler=CapturingSampler())

        model = captured["model"]
        rv_names = {v.name for v in model.unobserved_RVs}
        assert {"sigma_sd_0", "sigma_sd_1", "tril_offdiag"} <= rv_names
        assert "sigma_sd" not in rv_names


class TestVarFitWithSV:
    def test_var_fit_with_sv_builds_3d_chol(self, var_data_2v):
        """VAR(volatility=StochasticVolatility(...)).fit(...) builds a model
        whose obs likelihood uses a 3D chol factor (per-t)."""
        import pymc as pm

        from impulso.spec import VAR
        from impulso.sv.spec import StochasticVolatility

        captured = {}

        class CapturingSampler:
            name = "capture"

            def sample(self, model: pm.Model):
                captured["model"] = model
                raise RuntimeError("stop before sampling")

        with pytest.raises(RuntimeError, match="stop before sampling"):
            VAR(lags=1, volatility=StochasticVolatility()).fit(var_data_2v, sampler=CapturingSampler())

        model = captured["model"]
        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}

        # Per-variable log-vol paths registered.
        assert "v0_mu" in rv_names and "v1_mu" in rv_names
        assert "v0_sigma_eta" in rv_names and "v1_sigma_eta" in rv_names
        # Shared correlation factor.
        assert "R_chol" in det_names or "R_chol" in rv_names
        # h is the stacked log-vol path deterministic.
        assert "h" in det_names
        # No Sigma deterministic for SV (skipped — too memory-heavy).
        assert "Sigma" not in det_names


class TestVarPassesResidualsToVolatility:
    """VAR computes OLS residuals once and passes them as `data` to
    volatility.build_pymc_latent — the multivariate SV adapter relies on
    this to seed per-variable priors."""

    def test_var_threads_ols_residuals_through_volatility_adapter(self, var_data_2v):
        import numpy as np
        import pymc as pm
        import pytensor.tensor as pt

        from impulso.protocols import VolatilityProcess

        captured = {}

        class CapturingVolatility:
            """VolatilityProcess stand-in that records the `data` it receives
            and returns a trivial constant L so the model still builds."""

            name = "capturing"
            is_time_varying = False

            def build_pymc_latent(self, n_vars, T, data=None):
                captured["n_vars"] = n_vars
                captured["T"] = T
                captured["data"] = None if data is None else data.copy()
                sd = pm.HalfCauchy("capturing_sd", beta=1.0, shape=n_vars)
                L = pt.zeros((n_vars, n_vars))
                L = pt.set_subtensor(L[np.diag_indices(n_vars)], sd)
                return pm.Deterministic("L", L)

            def cholesky_at(self, posterior, t):
                return posterior["L"].values

            def cholesky_path(self, posterior, T):
                L = self.cholesky_at(posterior, t=None)
                return np.broadcast_to(L[:, :, None, :, :], (*L.shape[:2], T, *L.shape[-2:])).copy()

            def forecast_cholesky_path(self, posterior, steps, rng):
                return self.cholesky_path(posterior, steps)

        assert isinstance(CapturingVolatility(), VolatilityProcess)

        class CapturingSampler:
            name = "capture"

            def sample(self, model):
                raise RuntimeError("stop before sampling")

        with pytest.raises(RuntimeError, match="stop before sampling"):
            VAR(lags=1, volatility=CapturingVolatility()).fit(var_data_2v, sampler=CapturingSampler())

        n_lags = 1
        T_eff = var_data_2v.endog.shape[0] - n_lags
        n_vars = var_data_2v.endog.shape[1]

        assert captured["n_vars"] == n_vars
        assert captured["T"] == T_eff
        assert captured["data"] is not None, "VAR did not pass `data` to volatility"
        assert captured["data"].shape == (T_eff, n_vars)
        # OLS residuals are demeaned by construction (intercept in X_full).
        assert np.allclose(captured["data"].mean(axis=0), 0.0, atol=1e-8)


class TestExogPriorScaleField:
    def test_default_is_loose(self):
        assert VAR(lags=1).exog_prior_scale == 100.0

    def test_custom_value_round_trips(self):
        assert VAR(lags=1, exog_prior_scale=2.5).exog_prior_scale == 2.5

    @pytest.mark.parametrize("bad", [0.0, -1.0, -1e-9])
    def test_rejects_non_positive(self, bad):
        with pytest.raises(ValidationError):
            VAR(lags=1, exog_prior_scale=bad)

    def test_frozen(self):
        spec = VAR(lags=1)
        with pytest.raises(ValidationError):
            spec.exog_prior_scale = 5.0

    @pytest.mark.parametrize("bad", [(1.0, 0.0), (1.0, -1.0), (1.0, np.inf), (np.nan,), np.inf])
    def test_rejects_non_positive_or_non_finite_entries(self, bad):
        with pytest.raises(ValidationError, match="exog_prior_scale must be positive and finite"):
            VAR(lags=1, exog_prior_scale=bad)

    def test_per_equation_scale_round_trips_through_json(self):
        spec = VAR(lags=1, exog_prior_scale=(1.0, 100.0, 5.0))
        restored = VAR.model_validate(spec.model_dump(mode="json"))
        assert restored == spec
        assert restored.exog_prior_scale == (1.0, 100.0, 5.0)
        assert VAR.model_validate_json(spec.model_dump_json()) == spec


class TestExogPriorSigma:
    """`_exog_prior_sigma` puts the B_exog prior in contribution space.

    `_exog_prior_sigma` takes the per-variable `sigma` directly —
    `_build_pymc_model` computes it once via `ar1_residual_sd` and shares it
    with `Prior.build_priors`, so these tests compute it the same way rather
    than passing raw `endog` in.
    """

    def test_matches_independent_formula(self, rng):
        endog = rng.standard_normal((120, 3))
        exog = rng.standard_normal((120, 2)) * np.array([0.01, 500.0])
        sigma = ar1_residual_sd(endog)

        got = _exog_prior_sigma(sigma, exog, 100.0)
        expected = 100.0 * np.outer(sigma, 1.0 / exog.std(axis=0, ddof=1))

        assert got.shape == (3, 2)
        np.testing.assert_allclose(got, expected)

    def test_scale_is_linear(self, rng):
        endog = rng.standard_normal((120, 2))
        exog = rng.standard_normal((120, 3))
        sigma = ar1_residual_sd(endog)

        np.testing.assert_allclose(
            _exog_prior_sigma(sigma, exog, 5.0),
            0.05 * _exog_prior_sigma(sigma, exog, 100.0),
        )

    def test_tiny_scale_regressor_gets_a_huge_prior_sd(self, rng):
        """The bug: sd 0.01 regressor needs a prior ~100x looser than sd 1.0."""
        endog = rng.standard_normal((120, 2))
        tiny = rng.standard_normal((120, 1)) * 0.01
        ordinary = tiny / 0.01
        sigma = ar1_residual_sd(endog)

        ratio = _exog_prior_sigma(sigma, tiny, 100.0) / _exog_prior_sigma(sigma, ordinary, 100.0)
        np.testing.assert_allclose(ratio, 100.0)

    def test_floor_fires_on_a_numerically_flat_column(self, rng):
        """A column with negligible spread falls back to a level-derived scale."""
        from impulso.spec import _EXOG_SD_FLOOR_FRACTION

        endog = rng.standard_normal((120, 2))
        col = (1.0 + 1e-13 * rng.standard_normal(120))[:, None]
        sigma = ar1_residual_sd(endog)

        got = _exog_prior_sigma(sigma, col, 100.0)
        floor = _EXOG_SD_FLOOR_FRACTION * np.abs(col).max()
        expected = 100.0 * np.outer(sigma, 1.0 / np.array([floor]))

        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, expected)
        # The floor is doing real work: the raw sd would give a prior ~1e10 wider.
        unfloored = 100.0 * np.outer(sigma, 1.0 / col.std(axis=0, ddof=1))
        assert (got < unfloored / 1e9).all()

    def test_floor_does_not_fire_on_a_step_dummy(self, rng):
        """A 0/1 break at 75% has real spread, so the raw sd is used."""
        endog = rng.standard_normal((120, 2))
        dummy = np.zeros((120, 1))
        dummy[90:] = 1.0
        sigma = ar1_residual_sd(endog)

        raw_sd = dummy.std(axis=0, ddof=1)
        floor = 1e-3 * np.abs(dummy).max(axis=0)
        assert raw_sd[0] > floor[0], "fixture no longer exercises the non-floored path"

        expected = 100.0 * np.outer(sigma, 1.0 / raw_sd)
        np.testing.assert_allclose(_exog_prior_sigma(sigma, dummy, 100.0), expected)

    def test_rejects_column_that_is_all_zero_after_lag_trimming(self, rng):
        endog = rng.standard_normal((120, 2))
        pulse = np.zeros((120, 1))
        sigma = ar1_residual_sd(endog)

        with pytest.raises(ValueError, match=r"constant over the estimation sample: 'pulse'"):
            _exog_prior_sigma(sigma, pulse, 100.0, ["pulse"])

    def test_rejects_column_that_is_constant_nonzero_after_lag_trimming(self, rng):
        """The floor must not rescue a column that is flat at a non-zero level.

        A 0/1 dummy that switches inside the initial conditions passes VARData's
        whole-sample check, then arrives here as a column of ones — collinear
        with the intercept. Keying the floor off its level would hand it a wide
        prior on an unidentified coefficient, which is the failure this guard
        exists to stop.
        """
        endog = rng.standard_normal((120, 2))
        flat = np.ones((120, 1))
        sigma = ar1_residual_sd(endog)

        with pytest.raises(ValueError, match=r"constant over the estimation sample: 'early_break'") as exc:
            _exog_prior_sigma(sigma, flat, 100.0, ["early_break"])
        assert "reduce `lags`" in str(exc.value)

    def test_constant_column_error_falls_back_to_column_index(self, rng):
        endog = rng.standard_normal((120, 2))
        sigma = ar1_residual_sd(endog)
        with pytest.raises(ValueError, match=r"'column 0'"):
            _exog_prior_sigma(sigma, np.zeros((120, 1)), 100.0)


class TestExogPriorWiredIntoModel:
    """VAR.fit must build B_exog with the scale-adaptive sigma, not sigma=1."""

    @staticmethod
    def _capture(data, spec):
        import pymc as pm

        captured: dict[str, pm.Model] = {}

        class CapturingSampler:
            name = "capture"

            def sample(self, model: pm.Model):
                captured["model"] = model
                raise RuntimeError("stop before sampling")

        with pytest.raises(RuntimeError, match="stop before sampling"):
            spec.fit(data, sampler=CapturingSampler())
        return captured["model"]

    def test_b_exog_sigma_matches_the_formula(self, rng):
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 2)) * np.array([0.01, 250.0])
        data = _exog_data(endog, exog, ["tiny", "huge"])

        model = self._capture(data, VAR(lags=1))

        expected = _exog_prior_sigma(ar1_residual_sd(endog), exog[1:], 100.0)
        np.testing.assert_allclose(_captured_sigma(model, "B_exog"), expected)
        # The old bug: a flat unit prior regardless of the regressor's units.
        assert not np.allclose(_captured_sigma(model, "B_exog"), 1.0)

    def test_knob_reaches_the_graph(self, rng):
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 1)) * 0.01
        data = _exog_data(endog, exog, ["tiny"])

        model = self._capture(data, VAR(lags=1, exog_prior_scale=5.0))

        np.testing.assert_allclose(
            _captured_sigma(model, "B_exog"),
            _exog_prior_sigma(ar1_residual_sd(endog), exog[1:], 5.0),
        )

    def test_per_equation_knob_reaches_the_graph(self, rng):
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 2)) * np.array([0.01, 250.0])
        data = _exog_data(endog, exog, ["tiny", "huge"])

        unit = _captured_sigma(self._capture(data, VAR(lags=1, exog_prior_scale=1.0)), "B_exog")
        got = _captured_sigma(self._capture(data, VAR(lags=1, exog_prior_scale=(1.0, 100.0))), "B_exog")

        np.testing.assert_allclose(got, np.array([[1.0], [100.0]]) * unit)

    def test_sigma_uses_lag_trimmed_rows(self, rng):
        """The prior keys off the rows the likelihood sees, not the raw column."""
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 1))
        exog[:4] *= 500.0  # outlying head, dropped by lags=4
        data = _exog_data(endog, exog, ["x"])

        model = self._capture(data, VAR(lags=4))
        sigma = ar1_residual_sd(endog)

        got = _captured_sigma(model, "B_exog")
        np.testing.assert_allclose(got, _exog_prior_sigma(sigma, exog[4:], 100.0))
        assert not np.allclose(got, _exog_prior_sigma(sigma, exog, 100.0))

    def test_no_b_exog_without_exog(self, var_data_2v):
        model = self._capture(var_data_2v, VAR(lags=1))
        assert "B_exog" not in {v.name for v in model.unobserved_RVs}

    def test_fit_rejects_a_dummy_that_only_switches_inside_the_initial_conditions(self, rng):
        """VARData sees variation; the estimation sample does not."""
        endog = rng.standard_normal((150, 2))
        dummy = np.ones((150, 1))
        dummy[:2] = 0.0  # switches at t=2, but lags=4 trims rows 0-3 away
        data = _exog_data(endog, dummy, ["early_break"])

        with pytest.raises(ValueError, match=r"constant over the estimation sample: 'early_break'"):
            VAR(lags=4).fit(data)

        # Same column with lags=1 keeps the switch, so it is estimable.
        self._capture(data, VAR(lags=1))

    def test_prior_predictive_graph_carries_the_same_scaled_sigma(self, rng):
        """The graph `prior_predictive` draws from is the graph `fit` samples.

        `VAR.prior_predictive` builds its graph through `_build_pymc_model`,
        the same seam `fit` uses. If the scale-adaptive exog prior lived in
        `fit` instead, a prior-predictive check would silently describe a
        unit-scale model that was never fitted.
        """
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 2)) * np.array([0.01, 250.0])
        data = _exog_data(endog, exog, ["tiny", "huge"])
        spec = VAR(lags=1)

        prior_predictive_model, _ = spec._build_pymc_model(data)
        got = _captured_sigma(prior_predictive_model, "B_exog")

        np.testing.assert_allclose(got, _exog_prior_sigma(ar1_residual_sd(endog), exog[1:], 100.0))
        assert not np.allclose(got, 1.0)
        # Belt and braces: identical to what the fit path builds, so the two
        # cannot drift apart.
        np.testing.assert_allclose(got, _captured_sigma(self._capture(data, spec), "B_exog"))

    def test_prior_predictive_graph_honours_the_knob(self, rng):
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 1)) * 0.01
        data = _exog_data(endog, exog, ["tiny"])

        model, _ = VAR(lags=1, exog_prior_scale=5.0)._build_pymc_model(data)

        np.testing.assert_allclose(
            _captured_sigma(model, "B_exog"),
            _exog_prior_sigma(ar1_residual_sd(endog), exog[1:], 5.0),
        )


class TestMinnesotaPriorSigmaWiredIntoModel:
    """`VAR._build_pymc_model` passes `ar1_residual_sd(data.endog)` to
    `MinnesotaPrior.build_priors`: the coefficient prior sampled
    in the real graph must match what `build_priors` returns for that sigma,
    not a unit-scale baseline.
    """

    def test_b_sigma_matches_ar1_residual_sd_scaling(self, rng):
        endog = rng.standard_normal((150, 3)) * np.array([1.0, 20.0, 0.05])
        data = VARData(
            endog=endog,
            endog_names=["y1", "y2", "y3"],
            index=pd.date_range("2000-01-01", periods=150, freq="QS"),
        )
        sigma = ar1_residual_sd(endog)
        spec = VAR(lags=2)

        model, _ = spec._build_pymc_model(data)
        got = _captured_sigma(model, "B")

        expected = MinnesotaPrior().build_priors(
            n_vars=3,
            n_lags=2,
            sigma=sigma,
        )["B_sigma"]
        np.testing.assert_allclose(got, expected)
        # The scaling is doing real work: heterogeneous sigma must not match
        # the sigma=1 (no cross-lag scaling) baseline.
        baseline = MinnesotaPrior().build_priors(
            n_vars=3,
            n_lags=2,
            sigma=np.ones(3),
        )["B_sigma"]
        assert not np.allclose(got, baseline)


class TestValidateSigmaIsUsable:
    """`_validate_sigma_is_usable` guards the shared `sigma` before it reaches
    `Prior.build_priors` or `_exog_prior_sigma`.
    """

    def test_rejects_zero_entry_and_names_the_column(self):
        from impulso.spec import _validate_sigma_is_usable

        sigma = np.array([1.0, 0.0, 2.0])
        with pytest.raises(ValueError, match=r"'y2'"):
            _validate_sigma_is_usable(sigma, ["y1", "y2", "y3"])

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite(self, bad_value):
        from impulso.spec import _validate_sigma_is_usable

        sigma = np.array([1.0, bad_value])
        with pytest.raises(ValueError, match="zero, negative, or non-finite"):
            _validate_sigma_is_usable(sigma, ["y1", "y2"])

    def test_accepts_all_positive_finite(self):
        from impulso.spec import _validate_sigma_is_usable

        _validate_sigma_is_usable(np.array([1.0, 1e-10, 5.0]), ["y1", "y2", "y3"])  # must not raise

    def test_default_source_keeps_ar1_residual_sd_message(self):
        """No explicit `source` is the data-derived path; its message is
        unchanged."""
        from impulso.spec import _validate_sigma_is_usable

        with pytest.raises(ValueError, match="ar1_residual_sd"):
            _validate_sigma_is_usable(np.array([1.0, 0.0]), ["y1", "y2"])

    def test_endog_scales_source_drops_ar1_residual_sd_mention(self):
        """`source="endog_scales"` names the actual source instead of blaming
        `ar1_residual_sd`, which never ran on this path."""
        from impulso.spec import _validate_sigma_is_usable

        with pytest.raises(ValueError, match=r"endog_scales") as exc_info:
            _validate_sigma_is_usable(
                np.array([1.0, 0.0]),
                ["y1", "y2"],
                source="endog_scales",
            )
        assert "ar1_residual_sd" not in str(exc_info.value)
        assert "'y2'" in str(exc_info.value)


class TestBuildPymcModelRejectsDegenerateSigma:
    """`_build_pymc_model` validates the `sigma` it computes before handing it to
    the prior or `_exog_prior_sigma`. A column need not be literally
    constant to trigger this: an exactly-determined AR(1) fit (a very short,
    noiseless sample) can also give `sigma == 0` for a column `VARData` accepts
    because it does vary.
    """

    def test_rejects_a_column_with_exactly_zero_ar1_residual_sd(self, rng, monkeypatch):
        """A column need not be literally constant to trigger this guard: any
        column `ar1_residual_sd` happens to report a zero (or non-finite) scale
        for is caught, whatever produced that scale.

        `ar1_residual_sd` is forced to return an exact zero here rather than
        relying on a short/noiseless sample landing on exact-zero round-off
        naturally: `np.linalg.lstsq`'s residual for such a sample is only
        *machine-epsilon-small*, and how small depends on the BLAS/LAPACK
        backend, so it is not bit-exact `0.0` on every platform or numpy
        version (observed as low as ~6e-15, not 0.0, on some CI jobs). Forcing
        the value keeps the test deterministic while still exercising the real
        `_build_pymc_model` guard path against the real (varying, non-constant)
        `endog` data.
        """
        import impulso._conjugate as conjugate

        endog = rng.standard_normal((150, 2))  # ordinary, genuinely varying data
        data = VARData(
            endog=endog,
            endog_names=["y1", "y2"],
            index=pd.date_range("2000-01-01", periods=150, freq="QS"),
        )
        spec = VAR(lags=1)

        monkeypatch.setattr(conjugate, "ar1_residual_sd", lambda y: np.array([1.0, 0.0]))

        with pytest.raises(ValueError, match=r"'y2'"):
            spec._build_pymc_model(data)

    def test_prior_predictive_accepts_a_near_constant_column(self, rng):
        """A column that varies, however little, is not rejected (criterion 2)."""
        near_constant = 1.0 + 1e-12 * rng.standard_normal(150)
        ordinary = rng.standard_normal(150)
        endog = np.column_stack([near_constant, ordinary])
        data = VARData(
            endog=endog,
            endog_names=["almost_flat", "y"],
            index=pd.date_range("2000-01-01", periods=150, freq="QS"),
        )
        spec = VAR(lags=1)

        model, _ = spec._build_pymc_model(data)
        got = _captured_sigma(model, "B")
        assert np.isfinite(got).all()
        # The near-constant column's cross-lag ratio really is huge -- accepted,
        # not floored (docs/adr/0015).
        assert got.max() > 1e6


def _capture_model(var_data, **var_kwargs):
    """Build the production PyMC graph via VAR.fit, aborting before MCMC."""
    import pymc as pm

    captured: dict[str, pm.Model] = {}

    class CapturingSampler:
        name = "capture"

        def sample(self, model: pm.Model):
            captured["model"] = model
            raise RuntimeError("stop before sampling")

    with pytest.raises(RuntimeError, match="stop before sampling"):
        VAR(**var_kwargs).fit(var_data, sampler=CapturingSampler())
    return captured["model"]


class TestErrorDistributionParameter:
    """The `error_dist=` seam on the VAR spec."""

    def test_default_is_gaussian_string(self):
        assert VAR(lags=2).error_dist == "gaussian"

    def test_string_resolves_to_adapter(self):
        from impulso.observation import Gaussian, StudentT

        assert isinstance(VAR(lags=2).resolved_error_dist, Gaussian)
        resolved = VAR(lags=2, error_dist="student_t").resolved_error_dist
        assert isinstance(resolved, StudentT)
        # The string shorthand takes the adapter's defaults, i.e. inferred nu.
        assert resolved.nu == "infer"

    def test_object_pass_through_is_identity(self):
        from impulso.observation import StudentT

        adapter = StudentT(nu=7.0)
        assert VAR(lags=2, error_dist=adapter).resolved_error_dist is adapter

    def test_unknown_string_raises(self):
        with pytest.raises(ValidationError):
            VAR(lags=2, error_dist="nonexistent")

    def test_volatility_adapter_rejected_as_error_dist(self):
        """Cross-seam accident: a VolatilityProcess is not an ErrorDistribution."""
        with pytest.raises(ValidationError):
            VAR(lags=2, error_dist=Constant())

    def test_error_dist_adapter_rejected_as_volatility(self):
        from impulso.observation import StudentT

        with pytest.raises(ValidationError):
            VAR(lags=2, volatility=StudentT())


class TestStudentTRejectedWithStochasticVolatility:
    """SV x Student-t is rejected at construction, not at fit."""

    def test_string_forms_raise(self):
        with pytest.raises(ValidationError, match="time-varying volatility"):
            VAR(lags=1, volatility="sv", error_dist="student_t")

    def test_object_forms_raise(self):
        from impulso.observation import StudentT
        from impulso.sv.spec import StochasticVolatility

        with pytest.raises(ValidationError, match="time-varying volatility"):
            VAR(lags=1, volatility=StochasticVolatility(), error_dist=StudentT(nu=5.0))

    def test_error_names_the_alternatives(self):
        with pytest.raises(ValidationError, match="volatility='constant'"):
            VAR(lags=1, volatility="sv", error_dist="student_t")

    def test_gaussian_with_sv_still_allowed(self):
        assert VAR(lags=1, volatility="sv").resolved_volatility.is_time_varying


class TestStudentTModelBuild:
    """The graph VAR.fit builds under `error_dist='student_t'`."""

    def test_gaussian_branch_is_unchanged(self, var_data_2v):
        import pymc as pm

        model = _capture_model(var_data_2v, lags=1)
        obs = model.observed_RVs[0]
        assert isinstance(obs.owner.op, pm.MvNormal.rv_type)
        assert "nu" not in {v.name for v in model.deterministics}
        assert "nu_excess" not in {v.name for v in model.free_RVs}

    def test_fixed_nu_registers_a_deterministic(self, var_data_2v):
        import pymc as pm

        from impulso.observation import StudentT

        model = _capture_model(var_data_2v, lags=1, error_dist=StudentT(nu=5.0))
        obs = model.observed_RVs[0]
        assert isinstance(obs.owner.op, pm.MvStudentT.rv_type)
        dets = {v.name: v for v in model.deterministics}
        assert "nu" in dets
        assert "nu_excess" not in {v.name for v in model.free_RVs}
        assert float(dets["nu"].eval()) == 5.0

    def test_inferred_nu_registers_a_free_excess_rv(self, var_data_2v):
        import pymc as pm

        model = _capture_model(var_data_2v, lags=1, error_dist="student_t")
        obs = model.observed_RVs[0]
        assert isinstance(obs.owner.op, pm.MvStudentT.rv_type)
        assert "nu_excess" in {v.name for v in model.free_RVs}
        dets = {v.name: v for v in model.deterministics}
        assert "nu" in dets
        # The prior is a *shift*, not a truncation: every prior draw of nu
        # exceeds 2, so nu/(nu-2) never diverges.
        nu_prior_draws = np.asarray(pm.draw(dets["nu"], draws=50, random_seed=0))
        assert (nu_prior_draws > 2.0).all()

    @pytest.mark.parametrize("error_dist", ["gaussian", "student_t"])
    def test_manual_cholesky_parameterisation_survives(self, var_data_2v, error_dist):
        """LKJCholeskyCov is broken upstream; the manual factor must persist."""
        model = _capture_model(var_data_2v, lags=1, error_dist=error_dist)
        rv_names = {v.name for v in model.free_RVs}
        det_names = {v.name for v in model.deterministics}
        assert {"sigma_sd", "tril_offdiag"} <= rv_names
        assert "Sigma" in det_names
        op_names = {type(v.owner.op).__name__ for v in model.free_RVs}
        assert not any("LKJ" in op for op in op_names), op_names

    @pytest.mark.parametrize("error_dist", ["gaussian", "student_t"])
    def test_logp_compiles_and_is_finite(self, var_data_2v, error_dist):
        model = _capture_model(var_data_2v, lags=1, error_dist=error_dist)
        logp = model.compile_logp()(model.initial_point())
        assert np.isfinite(logp)

    def test_exogenous_regressors_survive_the_t_likelihood(self, var_data_2v):
        import pymc as pm

        from impulso.data import VARData

        data = VARData(
            endog=var_data_2v.endog,
            endog_names=var_data_2v.endog_names,
            index=var_data_2v.index,
            exog=np.arange(len(var_data_2v.endog), dtype=float)[:, None],
            exog_names=["trend"],
        )
        model = _capture_model(data, lags=1, error_dist="student_t")
        assert "B_exog" in {v.name for v in model.free_RVs}
        assert isinstance(model.observed_RVs[0].owner.op, pm.MvStudentT.rv_type)


class TestErrorDistThreadsToFittedVAR:
    """The model_construct threading guard.

    Forgetting `error_dist=` in `FittedVAR.model_construct` is a *silent*
    wrong-numbers bug: a t-fitted model would forecast Gaussian tails with
    no error anywhere.
    """

    def test_fit_passes_the_resolved_adapter_to_fitted_var(self, var_data_2v):
        import xarray as xr

        from impulso.observation import StudentT

        adapter = StudentT(nu=6.0)

        class StubSampler:
            name = "stub"

            def sample(self, model):
                # Minimal posterior; only the threading is under test here.
                return make_idata(
                    posterior=xr.Dataset({
                        "B": (("chain", "draw", "var", "coeff"), np.zeros((1, 2, 2, 2))),
                        "intercept": (("chain", "draw", "var"), np.zeros((1, 2, 2))),
                        "L": (("chain", "draw", "var1", "var2"), np.tile(np.eye(2), (1, 2, 1, 1))),
                        "nu": (("chain", "draw"), np.full((1, 2), 6.0)),
                    })
                )

        fitted = VAR(lags=1, error_dist=adapter).fit(var_data_2v, sampler=StubSampler())
        assert fitted.error_dist is adapter
        assert fitted.error_dist.is_heavy_tailed

    def test_gaussian_fit_keeps_a_gaussian_adapter(self, var_data_2v):
        import xarray as xr

        from impulso.observation import Gaussian

        class StubSampler:
            name = "stub"

            def sample(self, model):
                return make_idata(
                    posterior=xr.Dataset({
                        "B": (("chain", "draw", "var", "coeff"), np.zeros((1, 2, 2, 2))),
                        "intercept": (("chain", "draw", "var"), np.zeros((1, 2, 2))),
                        "L": (("chain", "draw", "var1", "var2"), np.tile(np.eye(2), (1, 2, 1, 1))),
                    })
                )

        fitted = VAR(lags=1).fit(var_data_2v, sampler=StubSampler())
        assert isinstance(fitted.error_dist, Gaussian)


class TestVarFitValidatesPosteriorSchema:
    """VAR.fit constructs its result through `FittedVAR.from_posterior`,
    so a sampler whose posterior breaks the shared schema must
    surface `from_posterior`'s `ValueError` rather than silently producing a
    malformed `FittedVAR`."""

    def test_posterior_missing_volatility_seam_raises(self, var_data_2v):
        """A sampler posterior missing `L` (the volatility seam) must raise,
        not silently return a `FittedVAR` that breaks on first use."""
        import xarray as xr

        class StubSampler:
            name = "stub"

            def sample(self, model):
                return make_idata(
                    posterior=xr.Dataset({
                        "B": (("chain", "draw", "var", "coeff"), np.zeros((1, 2, 2, 2))),
                        "intercept": (("chain", "draw", "var"), np.zeros((1, 2, 2))),
                        # "L" deliberately omitted: malformed volatility seam.
                    })
                )

        with pytest.raises(ValueError, match="L"):
            VAR(lags=1).fit(var_data_2v, sampler=StubSampler())


def _captured_mu(model):
    """The `mu` tensor feeding the observed `obs` likelihood.

    `dist_params` maps the RV's op back onto its distribution parameters —
    `mu` is always the first one, for both `MvNormal` and `MvStudentT` — so
    this survives the positional reshuffles PyMC makes to `owner.inputs`.
    """
    rv = model["obs"]
    return rv.owner.op.dist_params(rv.owner)[0]


class TestLagStackingParity:
    """Pins the numeric contract of the lag-major design matrix `VAR` bakes
    into its PyMC graph.

    Deliberately does not call any private stacking helper: it reconstructs
    `X_lag`/`X_exog` by hand and compares against `mu`, the conditional-mean
    tensor the observed likelihood is built from. That makes it a parity
    test in the strict sense — it exercises only the public `VAR.fit`
    surface and passes unchanged whether the stacking is inlined in
    `_build_pymc_model` or delegated to a shared builder.
    """

    @pytest.mark.parametrize("n_lags", [1, 2, 3])
    def test_mu_matches_hand_stacked_design_matrix(self, rng, n_lags):
        import pymc as pm

        endog = rng.standard_normal((40, 3))
        index = pd.date_range("2000-01-01", periods=40, freq="QS")
        data = VARData(endog=endog, endog_names=["y1", "y2", "y3"], index=index)

        model = _capture_model(data, lags=n_lags)
        mu = _captured_mu(model)

        intercept_draw, b_draw, mu_draw = pm.draw([model["intercept"], model["B"], mu], random_seed=0)

        x_parts = [endog[n_lags - lag : -lag] for lag in range(1, n_lags + 1)]
        x_lag_expected = np.hstack(x_parts)
        expected_mu = intercept_draw + x_lag_expected @ b_draw.T
        np.testing.assert_allclose(mu_draw, expected_mu)

    def test_mu_matches_with_exog(self, rng):
        import pymc as pm

        n_lags = 2
        endog = rng.standard_normal((40, 2))
        exog = rng.standard_normal((40, 1))
        data = _exog_data(endog, exog, ["z"])

        model = _capture_model(data, lags=n_lags)
        mu = _captured_mu(model)

        intercept_draw, b_draw, b_exog_draw, mu_draw = pm.draw(
            [model["intercept"], model["B"], model["B_exog"], mu], random_seed=0
        )

        x_parts = [endog[n_lags - lag : -lag] for lag in range(1, n_lags + 1)]
        x_lag_expected = np.hstack(x_parts)
        x_exog_expected = exog[n_lags:]
        expected_mu = intercept_draw + x_lag_expected @ b_draw.T + x_exog_expected @ b_exog_draw.T
        np.testing.assert_allclose(mu_draw, expected_mu)


class TestVolatilityShorthandSV:
    def test_sv_string_resolves_to_stochastic_volatility(self):
        from impulso.spec import VAR
        from impulso.sv.spec import StochasticVolatility

        spec = VAR(lags=2, volatility="sv")
        assert isinstance(spec.resolved_volatility, StochasticVolatility)

    def test_unknown_string_still_raises(self):
        from pydantic import ValidationError

        from impulso.spec import VAR

        with pytest.raises(ValidationError):
            VAR(lags=2, volatility="not_a_real_adapter")


class TestVARSpecRoundTrip:
    """`VAR.model_validate(spec.model_dump())` must equal `spec`, in both dump
    modes, for every prior/volatility/error-distribution combination.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="all_defaults"),
            pytest.param({"prior": "minnesota"}, id="prior_string"),
            pytest.param({"prior": MinnesotaPrior()}, id="prior_object_default"),
            pytest.param(
                {"prior": MinnesotaPrior(tightness=0.05, decay="geometric", cross_shrinkage=0.9, own_lag_mean=0.4)},
                id="prior_object_custom",
            ),
            pytest.param(
                {"prior": MinnesotaPrior(own_lag_mean=(1.0, 0.0, 0.5))},
                id="prior_object_own_lag_mean_tuple",
            ),
            pytest.param({"volatility": "constant"}, id="volatility_string_constant"),
            pytest.param({"volatility": "sv"}, id="volatility_string_sv"),
            pytest.param({"volatility": Constant()}, id="volatility_constant_object_default"),
            pytest.param(
                {
                    "volatility": Constant(
                        innovation_scale_priors=(
                            InnovationScalePrior(family="halfnormal", scale=1.0),
                            InnovationScalePrior(family="exponential", scale=0.5),
                            InnovationScalePrior(family="halfcauchy", scale=2.0),
                        )
                    )
                },
                id="volatility_constant_innovation_scale_priors",
            ),
            pytest.param({"volatility": StochasticVolatility()}, id="volatility_sv_object_default"),
            pytest.param({"error_dist": "gaussian"}, id="error_dist_string_gaussian"),
            pytest.param({"error_dist": "student_t"}, id="error_dist_string_student_t"),
            pytest.param({"error_dist": Gaussian()}, id="error_dist_gaussian_object"),
            pytest.param({"error_dist": StudentT(nu=5.0)}, id="error_dist_student_t_object_fixed_nu"),
            pytest.param({"error_dist": StudentT()}, id="error_dist_student_t_object_inferred_nu"),
            pytest.param(
                {
                    "prior": MinnesotaPrior(own_lag_mean=0.7),
                    "volatility": Constant(
                        innovation_scale_priors=(InnovationScalePrior(family="halfnormal", scale=1.0),) * 3
                    ),
                    "error_dist": StudentT(nu=6.0),
                },
                id="combined_prior_volatility_error_dist_objects",
            ),
            pytest.param(
                {
                    "prior": MinnesotaPrior(own_lag_mean=(1.0, 0.0)),
                    "volatility": StochasticVolatility(),
                    "error_dist": Gaussian(),
                },
                id="combined_prior_sv_gaussian_objects",
            ),
            pytest.param(
                {"volatility": StochasticVolatility(dynamics=AR1())}, id="volatility_sv_nested_dynamics_ar1_object"
            ),
            pytest.param(
                {"volatility": StochasticVolatility(dynamics=RandomWalk())},
                id="volatility_sv_nested_dynamics_random_walk_object",
            ),
            pytest.param(
                {"volatility": StochasticVolatility(prior=SVDefaultPrior())}, id="volatility_sv_nested_prior_object"
            ),
        ],
    )
    def test_round_trips_through_model_dump(self, kwargs):
        spec = VAR(lags=2, **kwargs)
        assert VAR.model_validate(spec.model_dump()) == spec
        assert VAR.model_validate(spec.model_dump(mode="json")) == spec
