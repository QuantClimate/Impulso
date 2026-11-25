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
from impulso.fitted import FittedVAR

ENDOG_NAMES = ["y1", "y2"]
N_ROWS = 40
N_CHAINS = 2
N_DRAWS = 3


def _require_numpyro():
    """Skip unless JAX and NumPyro are installed; return the `numpyro` module."""
    pytest.importorskip("jax")
    return pytest.importorskip("numpyro")


def _var_data(endog: np.ndarray) -> VARData:
    return VARData(
        endog=endog, endog_names=ENDOG_NAMES, index=pd.date_range("2000-01-01", periods=len(endog), freq="QS")
    )


def _build_kwargs(endog: np.ndarray, n_lags: int, **extra) -> dict:
    """`build_in_model`'s arguments other than `endog`, for a VAR on two observed series."""
    return {"exog": None, "n_lags": n_lags, "endog_names": ENDOG_NAMES, "endog_scales": ar1_residual_sd(endog)} | extra


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


@pytest.mark.parametrize("n_lags", [1, 2])
def test_posterior_is_accepted_and_holds_pymc_values(rng, n_lags):
    """`FittedVAR.from_posterior` accepts the posterior, and the `FittedVAR` reads PyMC's `B`, `intercept` and
    `Sigma` at every draw."""
    _require_numpyro()

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, n_lags)
    spec = VAR(lags=n_lags)
    model = _pymc_model(spec, endog, kwargs)
    draws = _parameter_draws(model, rng)
    expected = _pymc_values(model, draws, ["B", "intercept", "Sigma"])
    block = spec.compile_for_numpyro(**kwargs)
    samples = _numpyro_samples(block, _observed_endog_model(block), endog, draws)

    fitted = FittedVAR.from_posterior(block.posterior_from_samples(samples), _var_data(endog), n_lags=n_lags)

    np.testing.assert_allclose(fitted.coefficients, expected["B"], rtol=1e-10)
    np.testing.assert_allclose(fitted.intercepts, expected["intercept"], rtol=1e-10)
    np.testing.assert_allclose(fitted.sigma(), expected["Sigma"], rtol=1e-10)


def test_intercept_of_a_strict_subset_of_equations_is_zero_in_the_other_equations(rng):
    """With `intercept_equations=["y2"]` the block records an intercept for `y2` alone. The posterior holds the
    intercept of every equation, as the schema requires: zero for `y1`, the recorded value for `y2`."""
    _require_numpyro()

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 1, intercept_equations=["y2"])
    spec = VAR(lags=1)
    draws = _parameter_draws(_pymc_model(spec, endog, kwargs), rng)
    block = spec.compile_for_numpyro(**kwargs)
    samples = _numpyro_samples(block, _observed_endog_model(block), endog, draws)

    fitted = FittedVAR.from_posterior(block.posterior_from_samples(samples), _var_data(endog), n_lags=1)

    y2_intercept = draws["intercept"][..., 0]
    np.testing.assert_array_equal(fitted.intercepts[..., 0], 0.0)
    np.testing.assert_allclose(fitted.intercepts[..., 1], y2_intercept, rtol=1e-12)


def test_variables_of_the_users_own_model_are_kept_and_accepted(rng):
    """The user's own sample and deterministic sites stay in the posterior, under their own names and with
    PyMC's default dimension names, even when one has as many entries as there are equations."""
    numpyro = _require_numpyro()
    import numpyro.distributions as dist

    from impulso._posterior import posterior_dataset

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
    # The block's own sites hold its parameters on PyMC's unconstrained scale; the posterior leaves them out.
    assert not set(block.numpyro_site_names.values()) & set(posterior.data_vars)


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


@pytest.mark.parametrize(
    ("intercept_equations", "expected_dimension_names", "expected_coordinate_labels"),
    [
        (
            None,
            {"B": ("var", "coeff"), "intercept": ("var",), "L": ("L_dim_0", "L_dim_1")},
            {"var": ("y1", "y2"), "coeff": ("L1.y1", "L1.y2", "L2.y1", "L2.y2")},
        ),
        (
            ["y2"],
            {"B": ("var", "coeff"), "intercept": ("var_intercept",), "L": ("L_dim_0", "L_dim_1")},
            {"var": ("y1", "y2"), "coeff": ("L1.y1", "L1.y2", "L2.y1", "L2.y2"), "var_intercept": ("y2",)},
        ),
    ],
    ids=["every equation has an intercept", "only y2 has an intercept"],
)
def test_block_exposes_dimension_names_and_labels_of_each_recorded_quantity(
    rng, intercept_equations, expected_dimension_names, expected_coordinate_labels
):
    """The dimension names and labels are those of the PyMC model; `L`, which PyMC gives no dimension names,
    gets PyMC's default names and no labels."""
    _require_numpyro()

    endog = rng.standard_normal((N_ROWS, 2))
    block = VAR(lags=2).compile_for_numpyro(**_build_kwargs(endog, 2, intercept_equations=intercept_equations))

    assert dict(block.dimension_names) == expected_dimension_names
    assert dict(block.coordinate_labels) == expected_coordinate_labels
