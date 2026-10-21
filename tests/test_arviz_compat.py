"""Tests for the ArviZ 0.x / 1.x compatibility seam.

Regression coverage. These tests must pass unchanged on both
supported stacks (PyMC 5 + ArviZ 0 on Python 3.11, PyMC 6 + ArviZ 1 on
Python >= 3.12), so they assert on the *normalised* behaviour rather than on
one stack's container type — except where a test is explicitly parameterised
on `ARVIZ_V1` to pin the public container contract.
"""

import subprocess
import sys
import typing

import arviz as az
import numpy as np
import pytest
import xarray as xr

import impulso
from impulso._arviz_compat import (
    ARVIZ_V1,
    InferenceDataLike,
    get_group_dataset,
    hdi_bounds,
    make_idata,
)


@pytest.fixture
def groups(rng):
    """Two schema groups with distinguishable contents and attrs."""
    posterior = xr.Dataset(
        {"B": xr.DataArray(rng.standard_normal((2, 25, 3)), dims=["chain", "draw", "coeff"])},
        attrs={"inference_library": "impulso-test"},
    )
    posterior_predictive = xr.Dataset({
        "irf": xr.DataArray(rng.standard_normal((2, 25, 4, 3)), dims=["chain", "draw", "horizon", "variable"])
    })
    return posterior, posterior_predictive


# --------------- Version detection and active container ---------------


def test_arviz_v1_matches_installed_metadata():
    """`ARVIZ_V1` is derived from distribution metadata, not a runtime probe."""
    from importlib.metadata import version

    assert ARVIZ_V1 is (int(version("arviz").split(".")[0]) >= 1)


def test_inference_data_like_is_the_active_container():
    """The alias binds exactly one concrete class, so Pydantic stays strict."""
    assert isinstance(InferenceDataLike, type)
    if ARVIZ_V1:
        assert InferenceDataLike is xr.DataTree
    else:
        assert InferenceDataLike is az.InferenceData


def test_import_impulso_does_not_trigger_the_migration_shim():
    """`import impulso` must not touch `arviz.InferenceData`.

    On ArviZ 1 that attribute only survives behind a `MigrationWarning`-emitting
    shim. Evaluated `az.InferenceData` annotations therefore turn every
    `import impulso` into a warning.
    Run in a subprocess so the check is unaffected by earlier imports.
    """
    if not ARVIZ_V1:
        pytest.skip("arviz.MigrationWarning only exists on ArviZ 1.x")
    script = (
        "import warnings, arviz as az\n"
        "warnings.simplefilter('error', az.MigrationWarning)\n"
        "import impulso\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, f"import impulso emitted a MigrationWarning:\n{proc.stderr}"


# --------------- make_idata ---------------


def test_make_idata_returns_the_active_container(groups):
    posterior, posterior_predictive = groups
    idata = make_idata(posterior=posterior, posterior_predictive=posterior_predictive)
    assert isinstance(idata, InferenceDataLike)


def test_make_idata_round_trips_groups_values_and_attrs(groups):
    posterior, posterior_predictive = groups
    idata = make_idata(posterior=posterior, posterior_predictive=posterior_predictive)

    got_posterior = get_group_dataset(idata, "posterior")
    got_pp = get_group_dataset(idata, "posterior_predictive")

    xr.testing.assert_identical(got_posterior["B"], posterior["B"])
    xr.testing.assert_identical(got_pp["irf"], posterior_predictive["irf"])
    assert got_posterior.attrs["inference_library"] == "impulso-test"


def test_make_idata_exposes_groups_through_the_public_upstream_api(groups):
    """`idata` is the upstream-native container, so upstream introspection works."""
    posterior, posterior_predictive = groups
    idata = make_idata(posterior=posterior, posterior_predictive=posterior_predictive)
    names = set(idata.children) if ARVIZ_V1 else set(idata.groups())
    assert {"posterior", "posterior_predictive"} <= names


@pytest.mark.parametrize("bad", [xr.DataArray([1.0, 2.0], dims=["draw"], name="x"), {"B": 1}, None])
def test_make_idata_rejects_non_dataset_groups(bad):
    with pytest.raises(TypeError, match=r"must be an xarray\.Dataset"):
        make_idata(posterior=bad)


def test_make_idata_rejects_a_container_as_a_group(groups):
    """Guards the easy mistake of re-wrapping a container instead of a group."""
    posterior, _ = groups
    idata = make_idata(posterior=posterior)
    with pytest.raises(TypeError, match="get_group_dataset"):
        make_idata(posterior=idata)


# --------------- get_group_dataset ---------------


def test_get_group_dataset_returns_a_plain_dataset(groups):
    """Protocol implementations are typed `xr.Dataset`; a DataTree node is not one."""
    posterior, posterior_predictive = groups
    idata = make_idata(posterior=posterior, posterior_predictive=posterior_predictive)
    for name in ("posterior", "posterior_predictive"):
        assert type(get_group_dataset(idata, name)) is xr.Dataset


def test_attaching_a_predictive_group_keeps_the_intended_observed_data(groups):
    """Pin the merge semantics users are told to use in `posterior_predictive`.

    `InferenceData.extend` and `DataTree.update` have opposite conflict
    precedence, so the recipe is only safe when the incoming container owns
    the groups being attached. Assert the resulting groups explicitly on each
    stack rather than assuming the two APIs agree.
    """
    posterior, posterior_predictive = groups
    fit = make_idata(posterior=posterior)
    extra = make_idata(
        posterior_predictive=posterior_predictive,
        observed_data=xr.Dataset({"obs": xr.DataArray(np.zeros((4, 3)), dims=["horizon", "variable"])}),
    )

    if ARVIZ_V1:
        fit.update(extra)
        names = set(fit.children)
    else:
        fit.extend(extra)
        names = set(fit.groups())

    assert {"posterior", "posterior_predictive", "observed_data"} == names
    xr.testing.assert_identical(get_group_dataset(fit, "posterior_predictive")["irf"], posterior_predictive["irf"])
    # The pre-existing group must survive the merge on both stacks.
    xr.testing.assert_identical(get_group_dataset(fit, "posterior")["B"], posterior["B"])


def test_get_group_dataset_raises_keyerror_for_a_missing_group(groups):
    posterior, _ = groups
    idata = make_idata(posterior=posterior)
    with pytest.raises(KeyError, match="log_likelihood"):
        get_group_dataset(idata, "log_likelihood")


def test_get_group_dataset_result_is_reusable_as_a_stable_local(groups):
    """Successive calls yield equal-but-distinct objects.

    `DataTree.to_dataset()` rebuilds the Dataset each time, so identity-keyed
    memoisation (`_PosteriorCache`) only works if callers bind the group to a
    local once. This test documents that constraint rather than hiding it.
    """
    posterior, _ = groups
    idata = make_idata(posterior=posterior)
    first, second = get_group_dataset(idata, "posterior"), get_group_dataset(idata, "posterior")
    xr.testing.assert_identical(first, second)
    local = get_group_dataset(idata, "posterior")
    assert local is local


# --------------- hdi_bounds ---------------


@pytest.fixture
def asymmetric_draws():
    """Deterministic, strongly right-skewed draws with a known ordering."""
    rng = np.random.default_rng(0)
    values = rng.lognormal(mean=0.0, sigma=1.0, size=(4, 500, 3))
    # Make each `term` slice occupy a clearly different location.
    values = values + np.array([0.0, 10.0, 100.0])
    return xr.DataArray(values, dims=["chain", "draw", "term"], name="wald")


def test_hdi_bounds_returns_ordered_bounds_without_sampling_dims(asymmetric_draws):
    lower, upper = hdi_bounds(asymmetric_draws, prob=0.89)

    assert lower.dims == ("term",)
    assert upper.dims == ("term",)
    assert (lower.values < upper.values).all()
    # The bound coordinate is normalised away; downstream code never sees it.
    assert "hdi" not in lower.coords
    assert "ci_bound" not in lower.coords


def test_hdi_bounds_preserves_non_sampling_dim_order(rng):
    """Downstream code builds DataFrames positionally, so dim order is load-bearing."""
    da = xr.DataArray(rng.standard_normal((2, 200, 5, 3)), dims=["chain", "draw", "horizon", "response"], name="irf")
    lower, upper = hdi_bounds(da, prob=0.9)
    assert lower.dims == ("horizon", "response")
    assert upper.dims == ("horizon", "response")
    assert lower.shape == (5, 3)


def test_hdi_bounds_brackets_the_median(asymmetric_draws):
    """A sanity property that does not reimplement the HDI algorithm."""
    lower, upper = hdi_bounds(asymmetric_draws, prob=0.89)
    median = asymmetric_draws.median(dim=["chain", "draw"])
    assert (lower.values <= median.values).all()
    assert (median.values <= upper.values).all()


def test_hdi_bounds_widens_with_probability(asymmetric_draws):
    narrow_lo, narrow_hi = hdi_bounds(asymmetric_draws, prob=0.5)
    wide_lo, wide_hi = hdi_bounds(asymmetric_draws, prob=0.99)
    assert ((wide_hi - wide_lo).values > (narrow_hi - narrow_lo).values).all()


def test_hdi_bounds_reduces_only_chain_and_draw_for_a_leading_sample_layout():
    """Pins the exact behaviour that silently differs between the two stacks.

    Given a bare `(chain, draw, term)` ndarray, ArviZ 0 reduces the leading
    sample axes while ArviZ 1 reduces only the trailing axis. Routing through
    `hdi_bounds` must always reduce `chain` and `draw`.
    """
    rng = np.random.default_rng(7)
    da = xr.DataArray(rng.standard_normal((2, 500, 7)), dims=["chain", "draw", "term"], name="wald")
    lower, upper = hdi_bounds(da, prob=0.89)
    assert lower.shape == (7,)
    assert upper.shape == (7,)


def test_hdi_bounds_handles_sampling_dims_only():
    """Scalar case used by the conditional-forecast plausibility statistic."""
    rng = np.random.default_rng(3)
    da = xr.DataArray(rng.standard_normal((2, 400)), dims=["chain", "draw"], name="plausibility")
    lower, upper = hdi_bounds(da, prob=0.89)
    assert lower.shape == ()
    assert float(lower) < float(upper)


@pytest.mark.parametrize("prob", [0.0, 1.0, -0.1, 1.5])
def test_hdi_bounds_rejects_probability_outside_open_unit_interval(asymmetric_draws, prob):
    with pytest.raises(ValueError) as exc_info:
        hdi_bounds(asymmetric_draws, prob=prob)

    assert str(exc_info.value) == f"hdi_bounds() prob must be in (0, 1), got {prob}."


def test_hdi_bounds_rejects_an_unnamed_dataarray(rng):
    da = xr.DataArray(rng.standard_normal((2, 50, 3)), dims=["chain", "draw", "term"])
    with pytest.raises(ValueError, match="named DataArray"):
        hdi_bounds(da, prob=0.89)


@pytest.mark.parametrize("dims", [["chain", "term"], ["draw", "term"], ["horizon", "term"]])
def test_hdi_bounds_rejects_missing_sampling_dims(dims, rng):
    da = xr.DataArray(rng.standard_normal((2, 3)), dims=dims, name="wald")
    with pytest.raises(ValueError, match="chain"):
        hdi_bounds(da, prob=0.89)


@pytest.mark.parametrize("bad", [np.zeros((2, 50, 3)), xr.Dataset({"a": ("draw", [1.0])}), 1.0])
def test_hdi_bounds_rejects_non_dataarray_input(bad):
    with pytest.raises(TypeError, match=r"xarray\.DataArray"):
        hdi_bounds(bad, prob=0.89)


def test_hdi_bounds_matches_a_direct_arviz_call(asymmetric_draws):
    """The seam normalises the API without changing the numbers."""
    dataset = xr.Dataset({"wald": asymmetric_draws})
    if ARVIZ_V1:
        expected = az.hdi(dataset, prob=0.89, dim=["chain", "draw"])["wald"]
        exp_lo = expected.sel(ci_bound="lower").values
        exp_hi = expected.sel(ci_bound="upper").values
    else:
        expected = az.hdi(dataset, hdi_prob=0.89)["wald"]
        exp_lo = expected.sel(hdi="lower").values
        exp_hi = expected.sel(hdi="higher").values

    lower, upper = hdi_bounds(asymmetric_draws, prob=0.89)
    np.testing.assert_allclose(lower.values, exp_lo)
    np.testing.assert_allclose(upper.values, exp_hi)


# --------------- Integration with Pydantic, beartype, and the public API ---------------


def test_annotations_resolve_to_the_active_container():
    """`typing.get_type_hints` must not blow up on the compat alias."""
    from impulso.fitted import FittedVAR
    from impulso.results import VARResultBase

    for cls in (FittedVAR, VARResultBase):
        hints = typing.get_type_hints(cls)
        assert hints["idata"] is InferenceDataLike


def test_pydantic_validates_the_active_container(var_data_2v, groups):
    """Construction accepts the active container and rejects a foreign one."""
    from pydantic import ValidationError

    from impulso.fitted import FittedVAR
    from impulso.volatility import Constant

    posterior, _ = groups
    fitted = FittedVAR(
        idata=make_idata(posterior=posterior),
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    assert isinstance(fitted.idata, InferenceDataLike)

    with pytest.raises(ValidationError):
        FittedVAR(
            idata="not-a-container",
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )


_RUNTIME_CHECKS_SCRIPT = """
import numpy as np
import pandas as pd
import xarray as xr

import impulso

impulso.enable_runtime_checks()

from beartype.roar import BeartypeCallHintViolation
from impulso._arviz_compat import InferenceDataLike, make_idata
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.volatility import Constant

rng = np.random.default_rng(0)
posterior = xr.Dataset({
    "B": xr.DataArray(rng.standard_normal((2, 20, 2, 2)) * 0.2, dims=["chain", "draw", "var", "coeff"]),
    "intercept": xr.DataArray(np.zeros((2, 20, 2)), dims=["chain", "draw", "var"]),
    "L": xr.DataArray(
        np.broadcast_to(np.eye(2), (2, 20, 2, 2)).copy(), dims=["chain", "draw", "var1", "var2"]
    ),
})
data = VARData(
    endog=rng.standard_normal((60, 2)),
    endog_names=["y1", "y2"],
    index=pd.date_range("2000-01-01", periods=60, freq="MS"),
)

fitted = FittedVAR(
    idata=make_idata(posterior=posterior),
    n_lags=1,
    data=data,
    var_names=["y1", "y2"],
    volatility=Constant(),
)
assert isinstance(fitted.idata, InferenceDataLike)

# The seam must not weaken validation: a foreign container is still rejected.
try:
    FittedVAR(
        idata=posterior,
        n_lags=1,
        data=data,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
except Exception as exc:
    assert "idata" in str(exc), exc
else:
    raise AssertionError("a bare Dataset was accepted as the idata container")

# A full result path must survive beartype wrapping on both stacks.
result = fitted.forecast(steps=3, include_shock_uncertainty=False)
assert result.median().shape == (3, 2)
assert result.hdi(0.89).lower.shape == (3, 2)
print("OK")
"""


def test_runtime_checks_accept_the_active_container():
    """`enable_runtime_checks()` beartype-wraps the API; the alias must satisfy it.

    Driven in a fresh interpreter because `enable_runtime_checks()` mutates
    the library's classes in place and the wrapping would otherwise leak into
    the rest of the suite.
    """
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _RUNTIME_CHECKS_SCRIPT], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_public_lazy_imports_still_resolve():
    """The seam must not break `impulso.__getattr__` lazy exports."""
    for name in ("FittedVAR", "IdentifiedVAR", "ForecastResult", "FittedSV", "GrangerCausalityResult"):
        assert getattr(impulso, name).__name__ == name
