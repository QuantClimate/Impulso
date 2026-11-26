"""Tests for handing a NumPyro posterior of an embedded VAR to `FittedVAR.from_posterior`.

A user compiles a VAR with `VAR.compile_for_numpyro`, calls the compiled NumPyro
block inside their own NumPyro model, samples, and turns the samples into a
posterior with `CompiledNumPyroBlock.posterior_from_samples`. Every test goes
through those public seams and through `FittedVAR.from_posterior`, and compares
what the user sees with the same VAR built in PyMC by `build_in_model`.

The fast tests make NumPyro samples without MCMC: they draw each block site
around PyMC's initial point and run the NumPyro model at those draws with
`numpyro.infer.Predictive`, which returns the recorded sites with the same
`(chain, draw, ...)` layout as `MCMC.get_samples(group_by_chain=True)`. The one
MCMC test is marked `slow`.

The tests skip without JAX and NumPyro.
"""

import numpy as np
import pandas as pd
import pytest

from impulso import VAR, VARData, ar1_residual_sd
from impulso._posterior import posterior_dataset
from impulso.fitted import FittedVAR
from impulso.numpyro_block import LOG_DENSITY_SITE
from impulso.observation import StudentT

ENDOG_NAMES = ["y1", "y2"]
EXOG_NAMES = ["x1", "x2"]
N_ROWS = 40
N_CHAINS = 2
N_DRAWS = 3


def _require_numpyro():
    """Skip unless JAX and NumPyro are installed; return the `numpyro` module."""
    pytest.importorskip("jax")
    return pytest.importorskip("numpyro")


def _exog(n_rows: int) -> np.ndarray:
    """Two exogenous regressors with standard deviations 0.5 and 3."""
    return np.random.default_rng(7).standard_normal((n_rows, len(EXOG_NAMES))) * np.array([0.5, 3.0])


def _var_data(endog: np.ndarray, exog: np.ndarray | None = None) -> VARData:
    return VARData(
        endog=endog,
        endog_names=ENDOG_NAMES,
        exog=exog,
        exog_names=None if exog is None else EXOG_NAMES,
        index=pd.date_range("2000-01-01", periods=len(endog), freq="QS"),
    )


def _build_kwargs(endog: np.ndarray, n_lags: int, exog: np.ndarray | None = None, **extra) -> dict:
    """`build_in_model`'s arguments other than `endog`, for a VAR on two observed series."""
    return {
        "exog": exog,
        "exog_names": None if exog is None else EXOG_NAMES,
        "n_lags": n_lags,
        "endog_names": ENDOG_NAMES,
        "endog_scales": ar1_residual_sd(endog),
    } | extra


def _pymc_model(spec: VAR, endog: np.ndarray, kwargs: dict):
    """The same VAR built in PyMC by `build_in_model`, with `endog` a symbolic tensor holding `endog`."""
    import pymc as pm
    import pytensor

    with pm.Model() as model:
        spec.build_in_model(endog=pytensor.shared(endog), **kwargs)
    return model


def _parameter_draws(model, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draws of every PyMC value variable, shape `(N_CHAINS, N_DRAWS, *value_shape)`, scattered around PyMC's
    initial point, keyed by value-variable name."""
    return {
        name: value + 0.3 * rng.standard_normal((N_CHAINS, N_DRAWS, *np.shape(value)))
        for name, value in model.initial_point().items()
    }


def _pymc_values(model, draws: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    """PyMC's values of the variables `names` at every draw, shape `(N_CHAINS, N_DRAWS, ...)`."""
    outputs = model.replace_rvs_by_values([model[name] for name in names])
    function = model.compile_fn(outputs, inputs=model.value_vars, on_unused_input="ignore")
    values: dict[str, list] = {name: [] for name in names}
    for chain in range(N_CHAINS):
        for draw in range(N_DRAWS):
            point = {name: value[chain, draw] for name, value in draws.items()}
            for name, value in zip(names, function(point), strict=True):
                values[name].append(np.asarray(value))
    return {name: np.reshape(value, (N_CHAINS, N_DRAWS, *value[0].shape)) for name, value in values.items()}


def _numpyro_samples(block, numpyro_model, endog: np.ndarray, draws: dict[str, np.ndarray]) -> dict:
    """What `MCMC.get_samples(group_by_chain=True)` returns for `numpyro_model` had it sampled `draws`: the block's
    sites at `draws` (keyed by PyMC value-variable name) and every site the model records at them."""
    import jax
    from numpyro.infer import Predictive

    site_draws = {block.numpyro_site_names[name]: value for name, value in draws.items()}
    recorded = Predictive(numpyro_model, posterior_samples=site_draws, batch_ndims=2)(jax.random.key(0), endog)
    return {**site_draws, **{name: np.asarray(value) for name, value in recorded.items()}}


def _observed_endog_model(block):
    """A NumPyro model function that passes observed data to the block as `endog`."""

    def model(endog):
        return block(endog)

    return model


@pytest.mark.parametrize(
    ("spec", "exogenous_regressors"),
    [
        pytest.param(VAR(lags=1), False, id="gaussian, 1 lag"),
        pytest.param(VAR(lags=2), False, id="gaussian, 2 lags"),
        pytest.param(VAR(lags=1), True, id="exogenous regressors"),
        pytest.param(VAR(lags=1, error_dist="student_t"), False, id="student-t"),
        pytest.param(VAR(lags=1, error_dist=StudentT(nu=6.0)), False, id="student-t, fixed degrees of freedom"),
        pytest.param(VAR(lags=2, error_dist="student_t"), True, id="student-t, exogenous regressors"),
    ],
)
def test_posterior_is_accepted_and_holds_pymc_values(rng, spec, exogenous_regressors):
    """`FittedVAR.from_posterior` accepts the posterior, and the posterior holds PyMC's `B`, `intercept`,
    `Sigma`, and `B_exog` and `nu` when the VAR has them, at every draw."""
    _require_numpyro()

    endog = rng.standard_normal((N_ROWS, 2))
    exog = _exog(N_ROWS) if exogenous_regressors else None
    kwargs = _build_kwargs(endog, spec.lags, exog=exog)
    model = _pymc_model(spec, endog, kwargs)
    draws = _parameter_draws(model, rng)
    heavy_tailed = spec.resolved_error_dist.is_heavy_tailed
    names = ["B", "intercept", "Sigma"] + ["B_exog"] * exogenous_regressors + ["nu"] * heavy_tailed
    expected = _pymc_values(model, draws, names)
    block = spec.compile_for_numpyro(**kwargs)
    samples = _numpyro_samples(block, _observed_endog_model(block), endog, draws)

    fitted = FittedVAR.from_posterior(
        block.posterior_from_samples(samples),
        _var_data(endog, exog),
        n_lags=spec.lags,
        error_dist=spec.resolved_error_dist,
    )

    np.testing.assert_allclose(fitted.coefficients, expected["B"], rtol=1e-10)
    np.testing.assert_allclose(fitted.intercepts, expected["intercept"], rtol=1e-10)
    np.testing.assert_allclose(fitted.sigma(), expected["Sigma"], rtol=1e-10)
    posterior = posterior_dataset(fitted.idata)
    if exogenous_regressors:
        np.testing.assert_allclose(posterior["B_exog"].values, expected["B_exog"], rtol=1e-10)
    if heavy_tailed:
        np.testing.assert_allclose(posterior["nu"].values, expected["nu"], rtol=1e-10)


@pytest.mark.parametrize(
    "intercept_equations", [pytest.param(["y2"], id="only y2"), pytest.param([], id="no equation")]
)
def test_intercept_of_an_equation_without_one_is_zero(rng, intercept_equations):
    """With `intercept_equations` a strict subset, the block records an intercept for those equations alone; with
    it empty, the block records none. The posterior holds the intercept of every equation, as the schema requires:
    zero for an equation without one, the recorded value for the others."""
    _require_numpyro()

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 1, intercept_equations=intercept_equations)
    spec = VAR(lags=1)
    draws = _parameter_draws(_pymc_model(spec, endog, kwargs), rng)
    block = spec.compile_for_numpyro(**kwargs)
    samples = _numpyro_samples(block, _observed_endog_model(block), endog, draws)

    fitted = FittedVAR.from_posterior(block.posterior_from_samples(samples), _var_data(endog), n_lags=1)

    np.testing.assert_array_equal(fitted.intercepts[..., 0], 0.0)
    y2_intercept = draws["intercept"][..., 0] if intercept_equations else 0.0
    np.testing.assert_allclose(fitted.intercepts[..., 1], y2_intercept, rtol=1e-12)


def test_variables_of_the_users_own_model_are_kept_and_accepted(rng):
    """The user's own sample and deterministic sites stay in the posterior, under their own names and with
    PyMC's default dimension names, even when one has as many entries as there are equations."""
    numpyro = _require_numpyro()
    import numpyro.distributions as dist

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 1)
    spec = VAR(lags=1)
    draws = _parameter_draws(_pymc_model(spec, endog, kwargs), rng)
    block = spec.compile_for_numpyro(**kwargs)

    def numpyro_model(endog):
        values = block(endog)
        scale = numpyro.sample("user_scale", dist.HalfNormal(1.0))
        numpyro.deterministic("user_fitted", scale * endog[:, 0])
        numpyro.deterministic("user_weights", scale * np.ones(5))
        numpyro.deterministic("user_intercept_copy", values.intercept)
        return values

    samples = _numpyro_samples(block, numpyro_model, endog, draws)

    fitted = FittedVAR.from_posterior(block.posterior_from_samples(samples), _var_data(endog), n_lags=1)

    posterior = posterior_dataset(fitted.idata)
    assert posterior["user_scale"].dims == ("chain", "draw")
    assert posterior["user_fitted"].dims == ("chain", "draw", "user_fitted_dim_0")
    assert posterior["user_weights"].dims == ("chain", "draw", "user_weights_dim_0")
    assert posterior["user_intercept_copy"].dims == ("chain", "draw", "user_intercept_copy_dim_0")
    for name in ["user_scale", "user_fitted", "user_weights", "user_intercept_copy"]:
        np.testing.assert_array_equal(posterior[name].values, samples[name])
    # The posterior leaves out the block's own sites: its parameters on PyMC's unconstrained scale, and the
    # log-density site, which `Predictive` returns.
    assert not {*block.numpyro_site_names.values(), LOG_DENSITY_SITE} & set(posterior.data_vars)


def test_users_own_scoped_sites_are_accepted_with_each_slash_replaced_by_a_dot(rng):
    """`numpyro.handlers.scope` names each site it prefixes `{prefix}/{name}`, and ArviZ 1 refuses `/` in a
    variable name. The posterior holds such a site under its name with each `/` replaced by `.`, on every ArviZ
    version, so the names do not depend on which one is installed."""
    numpyro = _require_numpyro()
    import numpyro.distributions as dist
    from numpyro import handlers

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 1)
    spec = VAR(lags=1)
    draws = _parameter_draws(_pymc_model(spec, endog, kwargs), rng)
    block = spec.compile_for_numpyro(**kwargs)

    def numpyro_model(endog):
        values = block(endog)
        with handlers.scope(prefix="user"):
            scale = numpyro.sample("scale", dist.HalfNormal(1.0))
            numpyro.deterministic("fitted", scale * endog[:, 0])
        return values

    samples = _numpyro_samples(block, numpyro_model, endog, draws)
    assert {"user/scale", "user/fitted"} <= set(samples)

    fitted = FittedVAR.from_posterior(block.posterior_from_samples(samples), _var_data(endog), n_lags=1)

    posterior = posterior_dataset(fitted.idata)
    assert posterior["user.scale"].dims == ("chain", "draw")
    assert posterior["user.fitted"].dims == ("chain", "draw", "user.fitted_dim_0")
    np.testing.assert_array_equal(posterior["user.scale"].values, samples["user/scale"])
    np.testing.assert_array_equal(posterior["user.fitted"].values, samples["user/fitted"])


def test_users_own_sites_that_get_the_same_posterior_name_are_refused(rng):
    """`user/scale` and `user.scale` would both be `user.scale` in the posterior, so the helper refuses them, naming
    both, instead of dropping one."""
    numpyro = _require_numpyro()
    import numpyro.distributions as dist
    from numpyro import handlers

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 1)
    spec = VAR(lags=1)
    draws = _parameter_draws(_pymc_model(spec, endog, kwargs), rng)
    block = spec.compile_for_numpyro(**kwargs)

    def numpyro_model(endog):
        values = block(endog)
        numpyro.sample("user.scale", dist.HalfNormal(1.0))
        with handlers.scope(prefix="user"):
            numpyro.sample("scale", dist.HalfNormal(1.0))
        return values

    samples = _numpyro_samples(block, numpyro_model, endog, draws)

    with pytest.raises(ValueError, match=r"'user\.scale' and 'user/scale', which would both be 'user\.scale'"):
        block.posterior_from_samples(samples)


def _merge_chains(samples: dict) -> dict:
    """What `MCMC.get_samples()` returns without `group_by_chain=True`: the chains merged into one axis."""
    return {name: value.reshape(N_CHAINS * N_DRAWS, *value.shape[2:]) for name, value in samples.items()}


def _drop_coefficients(samples: dict) -> dict:
    """Samples of a model that records no `B`, as when the block is called inside a `numpyro.handlers.scope`."""
    return {name: value for name, value in samples.items() if name != "B"}


@pytest.mark.parametrize(
    ("damage", "message"),
    [(_merge_chains, r"have shape \(6, .*group_by_chain=True"), (_drop_coefficients, r"hold no 'B'")],
    ids=["chains merged into one axis", "no B recorded"],
)
def test_samples_not_laid_out_as_the_block_records_them_are_refused(rng, damage, message):
    """The helper refuses samples it cannot lay out, naming the quantity and how to get usable samples."""
    _require_numpyro()

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 1)
    spec = VAR(lags=1)
    draws = _parameter_draws(_pymc_model(spec, endog, kwargs), rng)
    block = spec.compile_for_numpyro(**kwargs)
    samples = _numpyro_samples(block, _observed_endog_model(block), endog, draws)

    with pytest.raises(ValueError, match=message):
        block.posterior_from_samples(damage(samples))


@pytest.mark.slow
def test_numpyro_nuts_posterior_gives_finite_impulse_responses(var_data_2v):
    """End to end: compile, call the block inside a NumPyro model that adds a variable of its own, sample with
    NumPyro's NUTS, convert, `FittedVAR.from_posterior`, Cholesky identification, impulse responses."""
    numpyro = _require_numpyro()
    import jax
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    from impulso.identification import Cholesky

    endog = var_data_2v.endog
    block = VAR(lags=1).compile_for_numpyro(
        exog=None, n_lags=1, endog_names=var_data_2v.endog_names, endog_scales=ar1_residual_sd(endog)
    )

    def numpyro_model(endog):
        values = block(endog)
        numpyro.sample("user_scale", dist.HalfNormal(1.0))
        return values

    mcmc = MCMC(
        NUTS(numpyro_model),
        num_warmup=200,
        num_samples=100,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.key(0), endog)
    posterior = block.posterior_from_samples(mcmc.get_samples(group_by_chain=True))

    fitted = FittedVAR.from_posterior(posterior, var_data_2v, n_lags=1)
    irf = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"])).impulse_response(horizon=8)

    assert np.all(np.isfinite(irf.median().to_numpy()))
    hdi = irf.hdi()
    assert np.all(np.isfinite(hdi.lower.to_numpy()))
    assert np.all(np.isfinite(hdi.upper.to_numpy()))


LAG_MAJOR_LABELS = ("L1.y1", "L1.y2", "L2.y1", "L2.y2")


@pytest.mark.parametrize(
    ("spec", "exogenous_regressors", "intercept_equations", "expected_dimension_names", "expected_coordinate_labels"),
    [
        pytest.param(
            VAR(lags=2),
            False,
            None,
            {"B": ("var", "coeff"), "intercept": ("var",), "L": ("L_dim_0", "L_dim_1")},
            {"var": ("y1", "y2"), "coeff": LAG_MAJOR_LABELS},
            id="every equation has an intercept",
        ),
        pytest.param(
            VAR(lags=2),
            False,
            ["y2"],
            {"B": ("var", "coeff"), "intercept": ("var_intercept",), "L": ("L_dim_0", "L_dim_1")},
            {"var": ("y1", "y2"), "coeff": LAG_MAJOR_LABELS, "var_intercept": ("y2",)},
            id="only y2 has an intercept",
        ),
        pytest.param(
            VAR(lags=2),
            False,
            [],
            {"B": ("var", "coeff"), "L": ("L_dim_0", "L_dim_1")},
            {"var": ("y1", "y2"), "coeff": LAG_MAJOR_LABELS},
            id="no intercept",
        ),
        pytest.param(
            VAR(lags=2, error_dist="student_t"),
            True,
            None,
            {
                "B": ("var", "coeff"),
                "intercept": ("var",),
                "B_exog": ("var", "exog"),
                "L": ("L_dim_0", "L_dim_1"),
                "nu": (),
            },
            {"var": ("y1", "y2"), "coeff": LAG_MAJOR_LABELS, "exog": ("x1", "x2")},
            id="student-t, exogenous regressors",
        ),
    ],
)
def test_block_exposes_dimension_names_and_labels_of_each_recorded_quantity(
    rng, spec, exogenous_regressors, intercept_equations, expected_dimension_names, expected_coordinate_labels
):
    """The dimension names and labels are those of the PyMC model. `L`, which PyMC gives no dimension names,
    gets PyMC's default names and no labels; the degrees of freedom `nu` are a scalar."""
    _require_numpyro()

    endog = rng.standard_normal((N_ROWS, 2))
    exog = _exog(N_ROWS) if exogenous_regressors else None
    block = spec.compile_for_numpyro(**_build_kwargs(endog, 2, exog=exog, intercept_equations=intercept_equations))

    assert dict(block.dimension_names) == expected_dimension_names
    assert dict(block.coordinate_labels) == expected_coordinate_labels
