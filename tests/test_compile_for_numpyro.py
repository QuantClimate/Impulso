"""Tests for `VAR.compile_for_numpyro`.

The compile step builds the VAR's PyMC model once, with `VAR.build_in_model`,
and returns a compiled NumPyro block that a user calls inside their own NumPyro
model. Every test goes through that public seam: it compiles a `VAR`, calls the
block inside a NumPyro model function, and compares what a NumPyro user sees
(NumPyro's own log-density, its gradient, the recorded `B`, `intercept`, `L` and
`latent`, the initial point) with the same VAR built in PyMC by `build_in_model`
with a symbolic `endog`.

The tests that need JAX and NumPyro skip without them. The missing-package
tests do not: they hide the packages themselves.
"""

import subprocess
import sys

import numpy as np
import pytest

from impulso import VAR, ar1_residual_sd

ENDOG_NAMES = ["y1", "y2"]
N_ROWS = 40
RECORDED_NAMES = ["B", "intercept", "L"]
MODEL_VARIANTS = [
    pytest.param(0, 1, id="observed series, 1 lag"),
    pytest.param(0, 2, id="observed series, 2 lags"),
    pytest.param(1, 1, id="1 latent series, 1 lag (truncated own lag)"),
    pytest.param(2, 2, id="2 latent series, 2 lags (stationarity Potential)"),
]
"""Model variants as `(n_latent, n_lags)`: the number of latent series and the lag order."""


def _require_numpyro():
    """Skip unless JAX and NumPyro are installed; return the `numpyro` module."""
    pytest.importorskip("jax")
    return pytest.importorskip("numpyro")


def _build_kwargs(endog: np.ndarray, n_lags: int, n_latent: int = 0) -> dict:
    """`build_in_model`'s arguments other than `endog`: `n_latent` latent series `b0, b1, ...` ahead of the two
    observed series of `endog`."""
    latent_names = [f"b{i}" for i in range(n_latent)]
    kwargs = {
        "exog": None,
        "n_lags": n_lags,
        "endog_names": [*latent_names, *ENDOG_NAMES],
        "endog_scales": np.concatenate([np.ones(n_latent), ar1_residual_sd(endog)]),
    }
    if latent_names:
        kwargs["latent_names"] = latent_names
    return kwargs


def _compile(spec: VAR, endog: np.ndarray, kwargs: dict):
    """The compile step on `build_in_model`'s arguments `kwargs`. With latent series it also takes the number of
    rows of the `endog` each call receives."""
    if kwargs.get("latent_names"):
        return spec.compile_for_numpyro(**kwargs, n_endog_rows=len(endog))
    return spec.compile_for_numpyro(**kwargs)


def _pymc_model(spec: VAR, endog: np.ndarray, kwargs: dict):
    """The same VAR built in PyMC by `build_in_model`, with `endog` a symbolic tensor holding `endog`."""
    import pymc as pm
    import pytensor

    with pm.Model() as model:
        spec.build_in_model(endog=pytensor.shared(endog), **kwargs)
    return model


def _pymc_point(model, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PyMC's initial point with every value moved by noise, so no parameter sits at its prior mean."""
    return {name: value + 0.3 * rng.standard_normal(value.shape) for name, value in model.initial_point().items()}


def _pymc_values(model, point: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    """PyMC's values of the variables `names` at `point`.

    `replace_rvs_by_values` rewrites each variable in terms of the value variables, undoing their transforms,
    so the compiled function reads `point` instead of drawing from the prior.
    """
    outputs = model.replace_rvs_by_values([model[name] for name in names])
    function = model.compile_fn(outputs, inputs=model.value_vars, on_unused_input="ignore")
    return dict(zip(names, (np.asarray(value) for value in function(point)), strict=True))


def _numpyro_params(block, point: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`point`, keyed by PyMC value-variable name, re-keyed by the NumPyro site that holds each value."""
    return {block.numpyro_site_names[name]: value for name, value in point.items()}


def _observed_endog_model(block):
    """A NumPyro model function that passes observed data to the block as `endog`."""

    def model(endog):
        return block(endog)

    return model


def _count_function_graphs(monkeypatch) -> list[int]:
    """Count, from now on, every PyTensor `FunctionGraph` built: every PyTensor compilation, to any backend,
    starts from one. Returns the list that grows by one entry per graph."""
    from pytensor.graph.fg import FunctionGraph

    built: list[int] = []
    original_init = FunctionGraph.__init__

    def counting_init(self, *args, **kwargs):
        built.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(FunctionGraph, "__init__", counting_init)
    return built


class TestLogDensityParity:
    """NumPyro's log-density of a model calling the block equals PyMC's log-density of the same VAR."""

    @pytest.mark.parametrize(("n_latent", "n_lags"), MODEL_VARIANTS)
    def test_observed_endog(self, rng, n_latent, n_lags):
        _require_numpyro()
        from numpyro.infer.util import log_density

        endog = rng.standard_normal((N_ROWS, 2))
        kwargs = _build_kwargs(endog, n_lags, n_latent)
        spec = VAR(lags=n_lags)
        model = _pymc_model(spec, endog, kwargs)
        point = _pymc_point(model, rng)
        pymc_log_density = float(model.compile_logp()(point))

        block = _compile(spec, endog, kwargs)
        numpyro_log_density, _ = log_density(_observed_endog_model(block), (endog,), {}, _numpyro_params(block, point))

        assert np.isfinite(pymc_log_density)
        assert float(numpyro_log_density) == pytest.approx(pymc_log_density, rel=1e-9)

    @pytest.mark.parametrize("n_lags", [1, 2])
    def test_endog_generated_by_the_numpyro_model(self, rng, n_lags):
        """The NumPyro model draws `endog` from a standard normal and passes the draw to the block, so its
        log-density is the standard-normal density of the draw plus PyMC's log-density of the VAR given it."""
        numpyro = _require_numpyro()
        import numpyro.distributions as dist
        from numpyro.infer.util import log_density
        from scipy.stats import norm

        endog = rng.standard_normal((N_ROWS, 2))
        kwargs = _build_kwargs(endog, n_lags)
        spec = VAR(lags=n_lags)
        model = _pymc_model(spec, endog, kwargs)
        point = _pymc_point(model, rng)
        block = spec.compile_for_numpyro(**kwargs)

        def generating_model():
            generated = numpyro.sample("endog", dist.Normal(0.0, 1.0).expand((N_ROWS, 2)).to_event(2))
            return block(generated)

        params = {**_numpyro_params(block, point), "endog": endog}
        numpyro_log_density, _ = log_density(generating_model, (), {}, params)

        expected = float(model.compile_logp()(point)) + norm.logpdf(endog).sum()
        assert float(numpyro_log_density) == pytest.approx(expected, rel=1e-9)


def test_log_density_is_minus_infinity_outside_the_latent_stationary_region(rng):
    """With the first latent series' own first lag at 1.2 and every other latent-on-latent coefficient at 0, the
    latent block's companion matrix has spectral radius 1.2, outside the stationary region (radius below 1)."""
    _require_numpyro()
    from numpyro.infer.util import log_density

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 2, n_latent=2)
    spec = VAR(lags=2)
    model = _pymc_model(spec, endog, kwargs)
    point = model.initial_point()
    point["B"][:2, :] = 0.0
    point["B"][0, 0] = 1.2

    block = _compile(spec, endog, kwargs)
    numpyro_log_density, _ = log_density(_observed_endog_model(block), (endog,), {}, _numpyro_params(block, point))

    assert float(model.compile_logp()(point)) == -np.inf
    assert float(numpyro_log_density) == -np.inf


@pytest.mark.parametrize(("n_latent", "n_lags"), MODEL_VARIANTS)
def test_recorded_and_returned_quantities_equal_pymc_values(rng, n_latent, n_lags):
    """A call records `B`, `intercept`, `L` and, with latent series, the generated `latent` paths as deterministic
    sites under their posterior-schema names, and returns them, with PyMC's values at the same parameter point. It
    also returns the latent series' names in the path's column order, as `build_in_model`'s handles do."""
    _require_numpyro()
    from numpyro import handlers

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, n_lags, n_latent)
    spec = VAR(lags=n_lags)
    model = _pymc_model(spec, endog, kwargs)
    point = _pymc_point(model, rng)
    recorded_names = [*RECORDED_NAMES, "latent"] if n_latent else RECORDED_NAMES
    expected = _pymc_values(model, point, recorded_names)

    block = _compile(spec, endog, kwargs)
    substituted = handlers.substitute(_observed_endog_model(block), data=_numpyro_params(block, point))
    trace = handlers.trace(substituted).get_trace(endog)
    returned = substituted(endog)

    for name in recorded_names:
        assert trace[name]["type"] == "deterministic"
        np.testing.assert_allclose(trace[name]["value"], expected[name], rtol=1e-10)
        np.testing.assert_allclose(getattr(returned, name), expected[name], rtol=1e-10)
    assert returned.latent_names == tuple(f"b{i}" for i in range(n_latent))
    if not n_latent:
        assert returned.latent is None
        assert "latent" not in trace


@pytest.mark.parametrize(("n_latent", "n_lags"), MODEL_VARIANTS)
def test_log_density_gradient_is_finite_and_equals_pymc_at_the_pymc_initial_point(rng, n_latent, n_lags):
    """With latent series and two lags the log-density holds the `latent_stationarity` Potential, whose
    eigendecomposition JAX cannot differentiate. The gradient is still finite, and it equals PyMC's, which passes
    that Potential no gradient."""
    _require_numpyro()
    import jax
    from numpyro.infer.util import log_density

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, n_lags, n_latent)
    spec = VAR(lags=n_lags)
    model = _pymc_model(spec, endog, kwargs)
    initial_point = model.initial_point()
    block = _compile(spec, endog, kwargs)

    gradient = jax.grad(lambda params: log_density(_observed_endog_model(block), (endog,), {}, params)[0])(
        _numpyro_params(block, initial_point)
    )

    assert set(gradient) == set(block.numpyro_site_names.values())
    for value in gradient.values():
        assert np.all(np.isfinite(value))
    # PyMC's gradient is one flat array: each free variable's gradient raveled, in `model.value_vars` order.
    pymc_gradient = model.compile_dlogp(vars=model.free_RVs)(initial_point)
    numpyro_gradient = [np.ravel(gradient[block.numpyro_site_names[value.name]]) for value in model.value_vars]
    np.testing.assert_allclose(np.concatenate(numpyro_gradient), pymc_gradient, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(("n_latent", "n_lags"), MODEL_VARIANTS)
def test_initial_point_starts_numpyro_at_the_pymc_initial_point(rng, n_latent, n_lags):
    """`init_to_value` with the block's initial point starts NumPyro at PyMC's initial point, where the log-density
    and its gradient are finite. Every variant uses the default latent own-lag prior mean of 1, outside the latent
    block's stationary region, so with two lags PyMC starts the latent own lags at 0.5 instead (ADR-0016)."""
    _require_numpyro()
    import jax
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, n_lags, n_latent)
    spec = VAR(lags=n_lags)
    block = _compile(spec, endog, kwargs)
    expected = _numpyro_params(block, _pymc_model(spec, endog, kwargs).initial_point())

    started = initialize_model(
        jax.random.key(0),
        _observed_endog_model(block),
        init_strategy=init_to_value(values=block.initial_point),
        model_args=(endog,),
    ).param_info

    assert set(block.initial_point) == set(expected)
    for site, value in expected.items():
        np.testing.assert_allclose(block.initial_point[site], value, rtol=1e-12)
        np.testing.assert_allclose(started.z[site], value, rtol=1e-12)
        assert np.all(np.isfinite(started.z_grad[site]))
    assert np.isfinite(started.potential_energy)


@pytest.mark.parametrize("n_latent", [0, 2], ids=["observed series", "2 latent series"])
def test_calls_do_not_rebuild_or_recompile_the_pymc_model(rng, monkeypatch, n_latent):
    """NumPyro traces a model many times. After the compile step, tracing, evaluating, differentiating and
    jit-compiling the NumPyro model never calls `build_in_model` and never builds a PyTensor `FunctionGraph`."""
    _require_numpyro()
    import jax
    from numpyro.infer.util import log_density

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 2, n_latent)
    spec = VAR(lags=2)
    initial_point = _pymc_model(spec, endog, kwargs).initial_point()

    build_calls = []
    original_build = VAR.build_in_model

    def counting_build(self, *args, **build_kwargs):
        build_calls.append(1)
        return original_build(self, *args, **build_kwargs)

    monkeypatch.setattr(VAR, "build_in_model", counting_build)
    block = _compile(spec, endog, kwargs)
    params = _numpyro_params(block, initial_point)
    assert len(build_calls) == 1
    graphs_built = _count_function_graphs(monkeypatch)

    def joint(params):
        return log_density(_observed_endog_model(block), (endog,), {}, params)[0]

    for _ in range(3):
        joint(params)
    jax.grad(joint)(params)
    jax.jit(joint)(params)

    assert len(build_calls) == 1
    assert graphs_built == []


@pytest.mark.parametrize(
    ("spec", "n_latent"),
    [(VAR(lags="aic"), 0), (VAR(lags=1, volatility="sv"), 0), (VAR(lags=1, error_dist="student_t"), 1)],
    ids=["string lags", "stochastic volatility", "Student-t errors with latent series"],
)
def test_raises_the_embedded_path_refusals_before_compiling(rng, monkeypatch, spec, n_latent):
    """The compile step refuses what `build_in_model` refuses for a symbolic `endog`, with the same message,
    and builds no PyTensor `FunctionGraph` first."""
    _require_numpyro()

    endog = rng.standard_normal((N_ROWS, 2))
    kwargs = _build_kwargs(endog, 1, n_latent)
    with pytest.raises(ValueError) as from_build_in_model:
        _pymc_model(spec, endog, kwargs)

    graphs_built = _count_function_graphs(monkeypatch)
    with pytest.raises(ValueError) as from_compile_step:
        _compile(spec, endog, kwargs)

    assert str(from_compile_step.value) == str(from_build_in_model.value)
    assert graphs_built == []


class TestNumberOfEndogRows:
    """With latent series JAX needs the length of the latent paths when it compiles them, so the compile step takes
    the number of rows of the `endog` each call receives."""

    def test_is_required_with_latent_series(self, rng):
        _require_numpyro()

        endog = rng.standard_normal((N_ROWS, 2))
        with pytest.raises(ValueError, match="n_endog_rows is required when latent_names is given"):
            VAR(lags=1).compile_for_numpyro(**_build_kwargs(endog, 1, n_latent=1))

    def test_a_call_with_another_number_of_rows_raises(self, rng):
        _require_numpyro()

        endog = rng.standard_normal((N_ROWS, 2))
        block = _compile(VAR(lags=1), endog, _build_kwargs(endog, 1, n_latent=1))

        with pytest.raises(ValueError, match=f"endog has {N_ROWS - 5} rows, but the block was compiled for {N_ROWS}"):
            _observed_endog_model(block)(endog[:-5])


@pytest.mark.parametrize(
    ("missing", "expected_start"),
    [
        (["jax"], "jax is required"),
        (["numpyro"], "numpyro is required"),
        (["jax", "numpyro"], "jax and numpyro are required"),
    ],
)
def test_missing_packages_are_named(rng, monkeypatch, missing, expected_start):
    """A `None` entry in `sys.modules` makes importing that module raise `ImportError`, as if it were absent.
    NumPyro imports JAX, so with JAX missing only JAX is named."""
    for module in missing:
        monkeypatch.setitem(sys.modules, module, None)
    endog = rng.standard_normal((N_ROWS, 2))

    with pytest.raises(ImportError) as raised:
        VAR(lags=1).compile_for_numpyro(**_build_kwargs(endog, 1))

    assert str(raised.value).startswith(expected_start)
    assert 'pip install "impulso[numpyro]"' in str(raised.value)


def test_impulso_imports_without_jax_or_numpyro():
    script = (
        "import sys\n"
        "sys.modules['jax'] = None\n"
        "sys.modules['numpyro'] = None\n"
        "import impulso\n"
        "import impulso.numpyro_block\n"
        "assert impulso.VAR(lags=1).lags == 1\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr
