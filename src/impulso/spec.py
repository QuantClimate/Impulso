"""VAR model specification."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
from pydantic import Field, field_validator, model_validator

from impulso._arviz_compat import InferenceDataLike
from impulso._base import ImpulsoBaseModel
from impulso._design import build_lag_design_matrix
from impulso._posterior import COEFFICIENTS, EXOG_COEFFICIENTS, INTERCEPT
from impulso.data import VARData, _format_names
from impulso.observation import Gaussian, StudentT
from impulso.priors import MinnesotaPrior
from impulso.protocols import ErrorDistribution, Prior, PyMCVolatilityProcess, Sampler
from impulso.sv.spec import StochasticVolatility
from impulso.volatility import Constant

if TYPE_CHECKING:
    import pytensor.tensor as pt

    from impulso.fitted import FittedVAR

_PRIOR_REGISTRY: dict[str, type] = {
    "minnesota": MinnesotaPrior,
}

_VOLATILITY_REGISTRY: dict[str, type] = {
    "constant": Constant,
    "sv": StochasticVolatility,
}

_ERROR_DIST_REGISTRY: dict[str, type] = {
    "gaussian": Gaussian,
    "student_t": StudentT,
}

# A column whose spread is below this fraction of its own largest absolute value is
# numerically constant even though it passed VARData's exactly-constant check. Using
# its raw standard deviation would inflate the prior towards infinity, so the floor
# substitutes a scale derived from the column's level instead.
_EXOG_SD_FLOOR_FRACTION: float = 1e-3


def _validate_sigma_is_usable(
    sigma: np.ndarray,
    endog_names: Sequence[str],
    *,
    source: Literal["ar1_residual_sd", "endog_scales"] = "ar1_residual_sd",
) -> None:
    """Reject a per-variable scale that would break every prior dividing by it.

    `sigma` (`ar1_residual_sd(endog)`, or the caller's `endog_scales`) is resolved
    once in `VAR.build_in_model` and shared by `Prior.build_priors` — whose
    Minnesota cross-lag entries scale by `sigma[i] / sigma[j]` (docs/adr/0015) —
    and by `_exog_prior_sigma` below. A zero entry collapses that column's own row
    of the coefficient prior toward zero, sends every other row's coefficient on
    its lag to `inf`, and turns the own-lag entry into `0.0 / 0.0 = nan`.

    `VARData` already rejects endogenous columns that are exactly constant over the
    whole sample, which is the common way a column ends up here with `sigma == 0`.
    This check exists for the rarer column that varies (so `VARData` accepts it) but
    is nonetheless perfectly predictable from its own first lag — a short, noiseless
    sample or an exact linear relationship — so `ar1_residual_sd` still returns
    exactly `0.0` for it. It does *not* fire on a column that is merely
    near-constant: dividing by a tiny but nonzero `sigma` gives a large, finite
    prior standard deviation (which can reach `1e12` or more — see
    `MinnesotaPrior.build_priors`), not `inf` or `nan`, and that column is a real,
    varying measurement whose coefficient is still identified.

    `source` only changes the wording of the error: `sigma` reaches this function
    either computed from the data (`ar1_residual_sd`) or supplied directly by the
    caller (`endog_scales`), and a message blaming `ar1_residual_sd` for a bad
    `endog_scales` is simply wrong — that function never ran.

    Args:
        sigma: Per-variable scale, shape `(n_vars,)` — `ar1_residual_sd(endog)`,
            or the caller's `endog_scales`.
        endog_names: Names for each endogenous variable, used only to name the
            offending columns in the error message.
        source: Which one produced `sigma`. `"ar1_residual_sd"` (the default)
            blames the data-derived residual scale; `"endog_scales"` blames the
            caller-supplied array instead.

    Raises:
        ValueError: If any entry of `sigma` is zero, negative, or non-finite.
    """
    bad = np.flatnonzero(~np.isfinite(sigma) | (sigma <= 0.0))
    if not bad.size:
        return
    labels = [endog_names[i] for i in bad]
    if source == "endog_scales":
        raise ValueError(
            f"endog_scales has a zero, negative, or non-finite entry for these columns: "
            f"{_format_names(labels)}. This scale is shared by the Minnesota cross-lag prior "
            "(docs/adr/0015) and the exogenous-coefficient prior, both of which divide by it. Supply a "
            "strictly positive, finite scale for every variable, or omit `endog_scales` so it is "
            "derived from the data instead."
        )
    raise ValueError(
        f"endog columns have a zero, negative, or non-finite scale: {_format_names(labels)}. "
        "`ar1_residual_sd` came out non-positive (or non-finite) for these columns, and this scale "
        "is shared by the Minnesota cross-lag prior (docs/adr/0015) and the exogenous-coefficient "
        "prior, both of which divide by it. VARData already rejects columns that are constant over "
        "the whole sample; this column varies but is nonetheless perfectly predictable from its own "
        "first lag (e.g. a very short, noiseless sample), so its residual scale is exactly zero. Add "
        "noise, drop the column, or otherwise make its scale identified."
    )


def _resolve_sigma(
    endog: np.ndarray,
    endog_scales: np.ndarray | Sequence[float] | None,
    endog_names: Sequence[str],
    n_vars: int,
) -> np.ndarray:
    """Resolve and validate `VAR.build_in_model`'s shared `sigma`.

    `endog_scales=None` computes `sigma` from `endog` via `ar1_residual_sd`.
    Otherwise the caller's array is coerced with `np.asarray(..., dtype=float)`
    — a plain list is accepted, not only an ndarray — and its shape checked
    against `n_vars` before validation, so a wrong-length `endog_scales` raises
    a clear `ValueError` here instead of an opaque `TypeError`/`IndexError`
    from numpy code further down. Either way, `_validate_sigma_is_usable` gets
    told which path produced `sigma`, so its error names the actual source.

    Args:
        endog: Endogenous data, shape `(T, n_vars)`.
        endog_scales: Caller-supplied scale, or `None` to derive it from `endog`.
        endog_names: Names for each endogenous column, used to name offending
            columns in a validation error.
        n_vars: Expected length of `sigma` — `endog.shape[1]`.

    Returns:
        `sigma`, shape `(n_vars,)`.

    Raises:
        ValueError: If `endog_scales` does not have shape `(n_vars,)`.
        ValueError: If any entry of the resolved `sigma` is zero, negative or
            non-finite.
    """
    # Lazy: `_conjugate` imports scipy at module level, and `spec` is on the
    # package import path.
    from impulso._conjugate import ar1_residual_sd

    if endog_scales is None:
        sigma = ar1_residual_sd(endog)
        _validate_sigma_is_usable(sigma, endog_names, source="ar1_residual_sd")
        return sigma

    sigma = np.asarray(endog_scales, dtype=float)
    if sigma.shape != (n_vars,):
        raise ValueError(f"endog_scales must have shape ({n_vars},) to match n_vars={n_vars}, got shape {sigma.shape}")
    _validate_sigma_is_usable(sigma, endog_names, source="endog_scales")
    return sigma


def _check_plain_2d_tensor(endog: "pt.TensorVariable") -> None:
    """Reject a symbolic `endog` that is a dimmed xtensor or not 2-D.

    Raises:
        TypeError: If `endog` is a dimmed `XTensorVariable`.
        ValueError: If `endog` is not 2-D.
    """
    # `isinstance(endog, Variable)` also admits a dimmed xtensor, which then
    # fails deep inside PyTensor's slicing. Checked by class name so this
    # works on PyTensor versions without `pytensor.xtensor`.
    if any(cls.__name__ == "XTensorVariable" for cls in type(endog).__mro__):
        raise TypeError(
            "endog is a dimmed XTensorVariable (e.g. a `pymc.dims.Data` container); build_in_model "
            "needs a plain 2-D tensor. Pass its `.values` instead."
        )
    if endog.ndim != 2:
        raise ValueError(f"endog must be 2-D (T, n_vars), got a {endog.ndim}-D tensor")


def _symbolic_endog_n_vars(
    endog: "pt.TensorVariable",
    endog_scales: np.ndarray | Sequence[float] | None,
    endog_names: Sequence[str],
) -> int:
    """Validate a symbolic `endog` for `VAR.build_in_model` and return `n_vars`.

    A symbolic `endog` (e.g. a `pm.Data` container the caller owns) cannot go
    through the numpy-only steps: `ar1_residual_sd`, the OLS pre-fit
    residuals, or an observed RV. Its column count may not be static, so
    `n_vars` comes from `endog_names`, checked against the static count when
    there is one.

    Raises:
        TypeError: If `endog` is a dimmed `XTensorVariable` (e.g. a
            `pymc.dims.Data` container itself) rather than a plain tensor.
        ValueError: If `endog` is not 2-D, if `endog_scales` is `None`, or if
            the static column count differs from `len(endog_names)`.
    """
    _check_plain_2d_tensor(endog)
    if endog_scales is None:
        raise ValueError(
            "endog_scales is required when endog is symbolic: the default scale comes from "
            "`ar1_residual_sd`, which needs concrete data. Pass the per-variable scales the "
            "Minnesota cross-lag prior and the exogenous prior should use, e.g. "
            "`impulso.ar1_residual_sd(values)` on the data behind the tensor."
        )
    n_vars = len(endog_names)
    static_n_vars = endog.type.shape[1]
    if static_n_vars is not None and static_n_vars != n_vars:
        raise ValueError(f"endog has {static_n_vars} columns but endog_names has {n_vars} names")
    return n_vars


def _check_latent_endog_shape(
    endog: "np.ndarray | pt.TensorVariable",
    exog: np.ndarray | None,
    observed_names: Sequence[str],
) -> None:
    """Check the observed `endog` and `exog` shapes on the latent-series path.

    Raises:
        TypeError: If `endog` is a dimmed `XTensorVariable`.
        ValueError: If `endog` is not 2-D, if its column count is not
            `len(observed_names)`, or if `exog`'s row count differs from its.
    """
    from pytensor.graph.basic import Variable

    if isinstance(endog, Variable):
        _check_plain_2d_tensor(endog)
        n_rows, n_cols = endog.type.shape
    elif endog.ndim != 2:
        raise ValueError(f"endog must be 2-D (T, n_observed), got a {endog.ndim}-D array")
    else:
        n_rows, n_cols = endog.shape
    # Without latent series `build_lag_design_matrix` trims both blocks
    # together and the mean catches a mismatch; here the exogenous block
    # enters the latent drive and the observed mean separately, so check it.
    if exog is not None and n_rows is not None and exog.shape[0] != n_rows:
        raise ValueError(f"exog has {exog.shape[0]} rows but endog has {n_rows}; both must cover the same periods.")
    n_obs = len(observed_names)
    if n_cols is not None and n_cols != n_obs:
        raise ValueError(
            f"endog has {n_cols} columns but endog_names lists {n_obs} observed series after the latent ones "
            f"({_format_names(list(observed_names))}). Pass only the observed columns: latent series "
            "are generated inside the model, not passed in."
        )


def _reject_unsupported_for_embedded_path(
    *,
    lags: int | str,
    volatility: PyMCVolatilityProcess,
    error_dist: ErrorDistribution,
    symbolic: bool,
    n_latent: int,
) -> None:
    """Reject spec options the embedded path cannot support, before any variable is registered.

    `build_in_model`'s plain numpy path — concrete `endog`, no latent series
    — can run OLS on the data to pick a lag order and to seed stochastic-
    volatility priors, and its likelihood can be any `ErrorDistribution`. The
    embedded path drops each of those: a symbolic `endog` has no
    concrete values to run OLS on at graph-build time, and a latent series
    has none at all — it is generated inside the model. This
    runs first in `build_in_model`, before `_latent_n_vars`/
    `_symbolic_endog_n_vars` or any `pm.Normal`/`add_coords` call, so a spec
    these options would misconfigure never leaves a partially-built model
    behind.

    Args:
        lags: `self.lags` — the spec's lag order, an int or a selection
            criterion string (`"aic"`, `"bic"`, `"hq"`).
        volatility: `self.resolved_volatility`.
        error_dist: `self.resolved_error_dist`.
        symbolic: Whether `endog` is a PyTensor variable.
        n_latent: Number of latent series (`0` for none).

    Raises:
        ValueError: If `lags` is a selection criterion string and `endog` is
            symbolic or there are latent series (lag selection runs OLS on
            data).
        ValueError: If `volatility` is not `Constant` and `endog` is
            symbolic or there are latent series (stochastic volatility seeds
            its priors from OLS residuals).
        ValueError: If there are latent series and `error_dist` is not
            Gaussian (the conditional of a multivariate Student-t is not a
            Student-t with the same `nu`, so the non-centred conditional
            likelihood has no simple closed form).
    """
    if not (symbolic or n_latent):
        return
    if symbolic and n_latent:
        reason = "endog is symbolic and latent series are present"
    elif symbolic:
        reason = "endog is symbolic"
    else:
        reason = "latent series are present"

    if isinstance(lags, str):
        raise ValueError(  # noqa: TRY004
            f"lags={lags!r} selects the lag order by running OLS on concrete data, which is unavailable "
            f"because {reason}. Pass an integer n_lags instead (resolve the selection criterion yourself "
            "first, e.g. via select_lag_order)."
        )
    if not isinstance(volatility, Constant):
        raise ValueError(  # noqa: TRY004
            f"volatility={type(volatility).__name__} seeds its priors from OLS residuals of concrete data, "
            f"which is unavailable because {reason}. Use volatility='constant' (the default) instead."
        )
    if n_latent and not isinstance(error_dist, Gaussian):
        raise ValueError(
            f"latent series require Gaussian errors, got {type(error_dist).__name__}: the conditional of a "
            "multivariate Student-t is not a Student-t with the same `nu`, so the non-centred conditional "
            "likelihood has no simple closed form."
        )


def _latent_n_vars(
    endog: "np.ndarray | pt.TensorVariable",
    exog: np.ndarray | None,
    endog_names: Sequence[str],
    endog_scales: np.ndarray | Sequence[float] | None,
    latent_names: Sequence[str],
) -> int:
    """Validate `VAR.build_in_model`'s latent-series inputs and return `n_vars`.

    With latent series, `endog_names` is the full VAR order and `endog` holds
    only the observed columns, which follow the latent ones. `n_vars` is
    therefore `len(endog_names)`, not `endog`'s column count.

    The error-distribution check (a latent series needs Gaussian errors) is
    `_reject_unsupported_for_embedded_path`, which runs before this function.

    Raises:
        TypeError: If `endog` is a dimmed `XTensorVariable`.
        ValueError: If `endog_names` does not start with exactly
            `latent_names`, if no observed series is left, if `endog`'s
            column count is not the number of observed series, if `exog`'s
            row count differs from `endog`'s, or if `endog_scales` is
            missing or does not cover the latent series.
    """
    n_latent = len(latent_names)
    n_vars = len(endog_names)
    if len(set(latent_names)) != n_latent:
        raise ValueError(f"latent_names lists a name more than once: {_format_names(list(latent_names))}.")
    if list(endog_names[:n_latent]) != list(latent_names):
        raise ValueError(
            f"latent_names ({_format_names(list(latent_names))}) must come first in endog_names, in the same "
            f"order; got endog_names {_format_names(list(endog_names))}. The latent series lead the Cholesky "
            "ordering, which is what makes the observed block's likelihood conditional on the latent "
            "innovations exact."
        )
    n_obs = n_vars - n_latent
    if n_obs < 1:
        raise ValueError("build_in_model needs at least one observed series; every name in endog_names is latent.")
    _check_latent_endog_shape(endog, exog, endog_names[n_latent:])
    if endog_scales is None:
        raise ValueError(
            "endog_scales is required when latent_names is given: a latent series has no data to compute "
            "its scale from. Pass one scale per name in endog_names, latent series included."
        )
    if np.shape(endog_scales) == (n_obs,):
        raise ValueError(
            f"endog_scales has {n_obs} entries, one per observed series, but it must cover the latent "
            f"series too ({_format_names(list(latent_names))}): pass one scale per name in endog_names."
        )
    return n_vars


def _latent_init_sigma(latent_init_sigma: float | Sequence[float], n_latent: int) -> np.ndarray:
    """Coerce `latent_init_sigma` to a positive array of shape `(n_latent,)`."""
    sigma = np.asarray(latent_init_sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(n_latent, float(sigma))
    if sigma.shape != (n_latent,):
        raise ValueError(f"latent_init_sigma must be a scalar or have {n_latent} entries, got shape {sigma.shape}")
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError(f"latent_init_sigma must be positive and finite, got {sigma.tolist()}")
    return sigma


def _latent_b_initval(b_mu: np.ndarray, n_latent: int) -> np.ndarray:
    """Initial value for `B`: latent rows inside the stationary region, observed rows at the prior mean.

    A latent equation's own first lag starts at its prior mean when that is
    inside the stationary region (`|mean| < 1`) and at 0.5 otherwise; every
    other coefficient in the latent rows starts at 0. PyMC takes an initial
    value for the whole of `B`, so the observed rows get their prior mean,
    which is PyMC's default start for a `Normal` anyway.
    """
    b_mu = np.asarray(b_mu, dtype=float)
    idx = np.arange(n_latent)
    own_lag_mean = b_mu[idx, idx]
    initval = b_mu.copy()
    initval[:n_latent] = 0.0
    initval[idx, idx] = np.where(np.abs(own_lag_mean) < 1.0, own_lag_mean, 0.5)
    return initval


def _set_latent_b_initval(model: Any, B: Any, b_mu: np.ndarray, n_latent: int) -> None:
    """Start `B` at `_latent_b_initval`, registering an initial value only when that is not the prior mean.

    The prior mean is PyMC's default start for a `Normal`. PyMC cannot copy a
    model with a non-default initial value (`Model.copy()`, `pm.do`,
    `pm.observe`), so a redundant one would only make the host model
    unclonable.
    """
    initval = _latent_b_initval(b_mu, n_latent)
    if not np.array_equal(initval, b_mu):
        model.set_initval(B, initval)


def _latent_companion(B: Any, n_latent: int, n_vars: int, n_lags: int) -> "pt.TensorVariable":
    """Companion matrix of the latent-on-latent block of `B`, shape `(n_latent * n_lags, n_latent * n_lags)`.

    `B` is `(n_vars, n_vars * n_lags)` with lag-major columns, so lag `l`'s
    latent-on-latent block `A_l[lat, lat]` is rows `:n_latent`, columns
    `(l - 1) * n_vars` to `(l - 1) * n_vars + n_latent`. The top block row
    stacks `A_1 ... A_p`; below it an identity shifts the lags down.
    """
    import pytensor.tensor as pt

    B = pt.as_tensor_variable(B)
    top = pt.concatenate([B[:n_latent, lag * n_vars : lag * n_vars + n_latent] for lag in range(n_lags)], axis=1)
    if n_lags == 1:
        return top
    size = n_latent * n_lags
    shift = pt.eye(size - n_latent, size)
    return pt.concatenate([top, shift], axis=0)


def _register_coefficients(
    model: Any, b_mu: np.ndarray, b_sigma: np.ndarray, n_latent: int, n_lags: int
) -> "pt.TensorVariable":
    """Register `B`, the VAR lag coefficients with `dims=("var", "coeff")`, and keep its latent block stationary.

    Without latent series `B` is one `Normal` with the prior's means and
    standard deviations. With latent series an explosive latent block makes
    the generated path explode and freezes the chain, so the block is
    constrained to its stationary region, in one of two ways (ADR-0017):

    - One latent series with `n_lags == 1`: the block is the own lag
      `B[0, 0]` alone, and the region is `|B[0, 0]| < 1`. `B` is a
      `Deterministic` scattered from a free block, `B_free`, a `Normal` over
      every other entry with that entry's prior mean and standard deviation,
      and a constrained block, `latent_own_lag`, the own lag's `Normal`
      truncated to (-1, 1). PyMC samples the own lag through an interval
      transform, which starts it at 0 and keeps it inside the region, so no
      initial value is set.
    - Otherwise: `B` is one `Normal`, started inside the region by
      `_set_latent_b_initval` and kept there by the `latent_stationarity`
      Potential.

    Args:
        model: The active `pymc.Model`.
        b_mu: Prior means, shape `(n_vars, n_vars * n_lags)`.
        b_sigma: Prior standard deviations, the same shape.
        n_latent: Number of latent series (`0` for none).
        n_lags: Lag order.

    Returns:
        `B`: the `Deterministic` for one latent series with `n_lags == 1`, the
        `Normal` otherwise.
    """
    import pymc as pm
    import pytensor.tensor as pt

    if n_latent == 1 and n_lags == 1:
        b_mu = np.asarray(b_mu, dtype=float)
        b_sigma = np.asarray(b_sigma, dtype=float)
        # Flat, row-major positions in `B`; position 0 is the latent own lag `B[0, 0]`.
        constrained = np.array([0])
        free = np.setdiff1d(np.arange(b_mu.size), constrained)
        B_free = pm.Normal("B_free", mu=b_mu.ravel()[free], sigma=b_sigma.ravel()[free])
        own_lag = pm.TruncatedNormal("latent_own_lag", mu=b_mu[0, 0], sigma=b_sigma[0, 0], lower=-1.0, upper=1.0)
        flat = pt.zeros(b_mu.size)[free].set(B_free)[constrained].set(own_lag[None])
        return pm.Deterministic(COEFFICIENTS, flat.reshape(b_mu.shape), dims=("var", "coeff"))
    B = pm.Normal(COEFFICIENTS, mu=b_mu, sigma=b_sigma, dims=("var", "coeff"))
    if n_latent:
        _set_latent_b_initval(model, B, b_mu, n_latent)
        _register_latent_stationarity(B, n_latent, np.shape(b_mu)[0], n_lags)
    return B


def _register_latent_stationarity(B: Any, n_latent: int, n_vars: int, n_lags: int) -> None:
    """Register the `latent_stationarity` Potential: 0 inside the stationary region, `-inf` outside.

    The region is a spectral radius below 1 for the latent block's companion
    matrix. The observed series are data, so only this block decides whether
    the generated path explodes.
    """
    import inspect

    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.builders import OpFromGraph

    # The Potential is piecewise constant in `B`, so its gradient is zero.
    # PyTensor cannot differentiate the complex `eig`, and PyMC's graph
    # rewrites strip `disconnected_grad`, so the radius is wrapped in an
    # `OpFromGraph` whose gradient is overridden with zeros.
    # Newer PyTensor renamed `lop_overrides` to `pullback`.
    override = "pullback" if "pullback" in inspect.signature(OpFromGraph.__init__).parameters else "lop_overrides"
    companion = pt.matrix("companion")
    spectral_radius = OpFromGraph(
        [companion],
        [pt.max(pt.abs(pt.linalg.eig(companion)[0]))],
        name="spectral_radius",
        **{override: lambda inputs, outputs, output_grads: [pt.zeros_like(inputs[0])]},
    )
    # `eig` raises on a non-finite matrix, which would abort the sampler
    # instead of recording a divergence. Swap such a matrix for an
    # explosive one (radius 2) so the Potential is `-inf`.
    matrix = _latent_companion(B, n_latent, n_vars, n_lags)
    size = n_latent * n_lags
    matrix = pt.switch(pt.all(~(pt.isnan(matrix) | pt.isinf(matrix))), matrix, 2.0 * pt.eye(size))
    radius = spectral_radius(matrix)
    pm.Potential("latent_stationarity", pt.switch(pt.lt(radius, 1.0), np.float64(0.0), np.float64(-np.inf)))


def _scan(fn: Any, **kwargs: Any) -> "pt.TensorVariable":
    """`pytensor.scan` returning outputs only, on PyTensor versions with and without `return_updates`."""
    import inspect

    import pytensor

    if "return_updates" in inspect.signature(pytensor.scan).parameters:
        return pytensor.scan(fn, return_updates=False, **kwargs)
    outputs, _ = pytensor.scan(fn, **kwargs)
    return outputs


def _latent_path(
    obs: "np.ndarray | pt.TensorVariable",
    x_exog: np.ndarray | None,
    n_lags: int,
    n_vars: int,
    n_latent: int,
    intercept_term: "pt.TensorVariable",
    B: "pt.TensorVariable",
    B_exog: "pt.TensorVariable | None",
    L: "pt.TensorVariable",
    init_sigma: np.ndarray,
) -> "tuple[pt.TensorVariable, pt.TensorVariable]":
    """Register and generate the latent series non-centred.

    Registers `latent_init`, a `Normal(0, init_sigma)` prior on each latent
    series' first `n_lags` values, shape `(n_lags, n_latent)`, and
    `latent_innovations`, standard-normal `z` of shape `(T - n_lags,
    n_latent)`. For `t >= n_lags` the latent block follows its VAR equations,

        latent_t = c_lat + sum_l A_l[lat, :] full_{t-l} + B_exog[lat] x_t + L[lat, lat] z_t,

    where `full` stacks the latent path with the observed columns. The terms
    that do not depend on the latent path are computed in one vectorised
    step; `scan` carries only the latent own- and cross-lag recursion.

    Returns:
        Tuple `(path, z)`: the full latent path, shape `(T, n_latent)` and
        including the initial values, registered as the `Deterministic`
        `latent`; and the innovations `z`.
    """
    import pymc as pm
    import pytensor.tensor as pt

    obs = pt.as_tensor_variable(obs)
    static_T = obs.type.shape[0]
    T = static_T if static_T is not None else obs.shape[0]

    init = pm.Normal("latent_init", mu=0.0, sigma=init_sigma, shape=(n_lags, n_latent))
    z = pm.Normal("latent_innovations", mu=0.0, sigma=1.0, shape=(T - n_lags, n_latent))

    # `B` is lag-major over the full VAR order: column `l * n_vars + j` is
    # variable `j`'s lag `l + 1`.
    latent_cols = [lag * n_vars + j for lag in range(n_lags) for j in range(n_latent)]
    obs_cols = [lag * n_vars + j for lag in range(n_lags) for j in range(n_latent, n_vars)]
    B_lat = B[:n_latent]
    _, obs_lags, _ = build_lag_design_matrix(obs, n_lags)
    drive = intercept_term[:n_latent] + pt.dot(obs_lags, B_lat[:, obs_cols].T) + pt.dot(z, L[:n_latent, :n_latent].T)
    if x_exog is not None and B_exog is not None:
        drive = drive + pt.dot(x_exog, B_exog[:n_latent].T)
    # (n_lags, n_latent, n_latent): A_lat[l] is the latent-on-latent block of lag l + 1.
    A_lat = B_lat[:, latent_cols].reshape((n_latent, n_lags, n_latent)).dimshuffle(1, 0, 2)
    # Pin the static shapes scan sees. Scan checks that a rebuilt node's
    # inputs broadcast like the originals, and a rewrite that substitutes the
    # value variables (nutpie's compile does) can make a length-1 axis static
    # that was unknown when the scan was built, which then fails that check.
    drive = pt.specify_shape(drive, (None, n_latent))
    A_lat = pt.specify_shape(A_lat, (n_lags, n_latent, n_latent))

    def step(drive_t, *args):
        # `args` is the last `n_lags` latent values, oldest first (scan's
        # tap order), then `A_lat`.
        *history, A = args
        new = drive_t
        for lag, value in enumerate(reversed(history)):
            new = new + pt.dot(A[lag], value)
        return new

    # With a single tap scan takes the state itself (a vector), not a
    # one-row history.
    initial = init[0] if n_lags == 1 else init
    history = _scan(
        step,
        sequences=[drive],
        outputs_info=[{"initial": initial, "taps": list(range(-n_lags, 0))}],
        non_sequences=[A_lat],
    )
    path = pm.Deterministic("latent", pt.concatenate([init, history], axis=0))
    return path, z


def _ols_residuals(Y: np.ndarray, X_lag: np.ndarray, X_exog: np.ndarray | None) -> np.ndarray:
    """OLS residuals of `Y` on an intercept, `X_lag` and (optional) `X_exog`.

    Seeds the volatility process's per-variable priors in
    `VAR.build_in_model`. Numpy-only: it needs concrete data.
    """
    if X_exog is not None:
        X_full = np.hstack([np.ones((Y.shape[0], 1)), X_lag, X_exog])
    else:
        X_full = np.hstack([np.ones((Y.shape[0], 1)), X_lag])
    B_ols, *_ = np.linalg.lstsq(X_full, Y, rcond=None)
    return Y - X_full @ B_ols


def _time_coord(model: Any, n_rows: int | None) -> dict[str, object]:
    """The `"time"` coord `VAR.build_in_model` must add, if any.

    Returns `{}` when there is nothing to add, `{"time": range}` otherwise.

    Args:
        model: The active `pymc.Model`.
        n_rows: Number of likelihood rows, `T - n_lags`, or `None` for a
            symbolic `endog` whose static shape does not know `T`.

    Raises:
        ValueError: If `model` already has a `"time"` coord whose length is
            not `n_rows`.
    """
    if n_rows is None:
        # Symbolic `endog` of unknown length: its likelihood is a
        # `pm.Potential`, which carries no dims, so nothing needs a
        # "time" coord and there is no length to check one against.
        return {}
    if "time" in model.coords:
        # `add_coords` only rejects a duplicate coord whose *values* differ,
        # so a length mismatch could slip through and surface much later as
        # a shape error, e.g. inside `sample_prior_predictive`.
        existing_length = int(model.dim_lengths["time"].eval())
        if existing_length != n_rows:
            raise ValueError(
                f"the active model already has a 'time' coordinate of length "
                f"{existing_length}, but this call's likelihood has {n_rows} rows "
                "(T - n_lags). Coordinates are not prefixed by a nested "
                "pm.Model(name=...), so two VARs embedded in the same model share a "
                "single 'time' coordinate and must agree on its length. Give both VARs "
                "the same number of likelihood rows, or build them in separate "
                "pm.Model() instances."
            )
        return {}
    # PyMC requires a named dim used on an *observed* multivariate RV
    # to already exist (unlike a free RV's `dims`, which it will
    # auto-register).
    return {"time": list(range(n_rows))}


def _intercept_mask(endog_names: Sequence[str], intercept_equations: Sequence[str] | None) -> np.ndarray:
    """Boolean mask over `endog_names`: which equations get an intercept.

    `None` means every equation. Otherwise every name must appear in
    `endog_names`, at most once; the mask follows `endog_names`' order
    regardless of the order `intercept_equations` lists them in.
    """
    if intercept_equations is None:
        return np.ones(len(endog_names), dtype=bool)
    unknown = [name for name in intercept_equations if name not in endog_names]
    if unknown:
        raise ValueError(
            f"intercept_equations names {_format_names(unknown)}, which are not endogenous variables; "
            f"expected a subset of endog_names ({_format_names(endog_names)})."
        )
    duplicates = sorted({name for name in intercept_equations if list(intercept_equations).count(name) > 1})
    if duplicates:
        raise ValueError(f"intercept_equations lists {_format_names(duplicates)} more than once.")
    return np.array([name in intercept_equations for name in endog_names], dtype=bool)


def _register_intercept(intercept_mask: np.ndarray) -> "tuple[pt.TensorVariable | None, pt.TensorVariable]":
    """Register `VAR.build_in_model`'s intercept.

    Returns:
        Tuple `(intercept, intercept_term)`: the free variable (`None` when
        every equation is excluded) and the length-`n_vars` vector added to
        the mean, with literal zeros for excluded equations.
    """
    import pymc as pm
    import pytensor.tensor as pt

    n_vars = intercept_mask.size
    if intercept_mask.all():
        intercept = pm.Normal(INTERCEPT, mu=0, sigma=1, dims="var")
        return intercept, intercept
    if intercept_mask.any():
        intercept = pm.Normal(INTERCEPT, mu=0, sigma=1, dims="var_intercept")
        return intercept, pt.zeros(n_vars)[np.flatnonzero(intercept_mask)].set(intercept)
    return None, pt.zeros(n_vars)


def _register_likelihood(
    error_dist: ErrorDistribution,
    mu: "pt.TensorVariable",
    L: "pt.TensorVariable",
    Y: "np.ndarray | pt.TensorVariable",
    *,
    potential: bool,
    n_latent: int,
    z: "pt.TensorVariable | None",
) -> "pt.TensorVariable":
    """Register `VAR.build_in_model`'s observation likelihood, named `"obs"`.

    A numpy `endog` without latent series gets an observed RV. Otherwise the
    likelihood is `error_dist.logp` wrapped in a `pm.Potential`. With latent
    series, only the observed block enters, conditional on the latent
    innovations `z`: the latent series lead the Cholesky ordering, so
    `resid_obs - L[obs, lat] z ~ MvN(0, L[obs, obs] L[obs, obs]')` exactly.
    """
    import pymc as pm

    if n_latent:
        mu_obs = mu[:, n_latent:] + pm.math.dot(z, L[n_latent:, :n_latent].T)
        return pm.Potential("obs", error_dist.logp(mu=mu_obs, chol=L[n_latent:, n_latent:], value=Y[:, n_latent:]))
    if potential:
        return pm.Potential("obs", error_dist.logp(mu=mu, chol=L, value=Y))
    return error_dist.build_likelihood("obs", mu=mu, chol=L, observed=Y, dims=("time", "var"))


def _exog_prior_sigma(
    sigma: np.ndarray,
    x_exog: np.ndarray,
    scale: float | Sequence[float],
    exog_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Prior standard deviations for the exogenous coefficients `B_exog`.

    The coefficient on an exogenous regressor is not a unit-free quantity: it
    converts the regressor's units into the dependent variable's. A prior fixed
    in coefficient space therefore encodes a different belief for every dataset
    — crushing coefficients on small-scale regressors and leaving coefficients
    on large-scale ones effectively unrestricted. This scales the prior so the
    belief lives in *contribution* space instead:

        sd[i, j] = scale_i * sigma_i / s_j

    where `sigma_i` is the AR(1) residual standard deviation of endogenous
    variable `i` (the same scale `MinnesotaPrior.build_priors` uses for its lag
    coefficients — see docs/adr/0015) and `s_j` is the sample standard
    deviation of exogenous column `j`. One prior standard deviation of
    `B_exog[i, j]` then moves variable `i` by `scale_i` of its own residual
    standard deviation when regressor `j` moves by one of its own. `scale_i`
    is `scale` itself when `scale` is a scalar and its `i`-th entry when it is
    a sequence. The default `scale` is deliberately loose (see
    `VAR.exog_prior_scale`).

    Args:
        sigma: Per-endogenous-variable AR(1) residual standard deviation,
            shape `(n_vars,)` — `ar1_residual_sd(endog)`, or the caller's
            `endog_scales`. `VAR.build_in_model` resolves this once and passes
            the same array here and to `Prior.build_priors`, so the
            coefficient and exogenous priors are expressed in the same units.
        x_exog: Exogenous regressor block of shape `(T_eff, n_exog)`, already
            trimmed to the rows the likelihood sees.
        scale: Multiplier in units of "residual standard deviations of the
            dependent variable per standard deviation of the regressor". A
            scalar applies to every equation; a sequence gives one entry per
            equation, in the order of `sigma`.
        exog_names: Optional column names, used only to make the
            constant-column error message readable.

    Returns:
        Array of shape `(n_vars, n_exog)` of prior standard deviations.

    Raises:
        ValueError: If `scale` is a sequence whose length is not `sigma`'s.
        ValueError: If a column of `x_exog` is exactly constant. `VARData`
            rejects columns that are constant over the whole sample, but
            trimming the first `n_lags` rows can flatten a column that did
            vary — a dummy that only switches inside the initial conditions,
            say. What the likelihood then sees is collinear with the
            intercept, so the coefficient is not identified; the floor below
            would happily hand it a wide prior and hide that.
    """
    scale_arr = np.asarray(scale, dtype=float)
    if scale_arr.ndim and scale_arr.shape != sigma.shape:
        raise ValueError(
            f"exog_prior_scale must be a scalar or have {sigma.size} entries to match n_vars, got {scale_arr.size}"
        )

    s = x_exog.std(axis=0, ddof=1)
    # Checked before the floor is applied: the floor exists to tame columns with
    # tiny-but-real variation, not to manufacture a scale for columns with none.
    degenerate = np.flatnonzero(s <= 0.0)
    if degenerate.size:
        labels = [exog_names[j] if exog_names is not None else f"column {j}" for j in degenerate]
        raise ValueError(
            f"exog columns are constant over the estimation sample: {_format_names(labels)}. "
            "The first n_lags rows are consumed as initial conditions, and what remains of these columns "
            "does not vary, so their coefficients are collinear with the intercept and not identified. "
            "Drop the columns, or reduce `lags` so the rows that do vary enter the estimation sample."
        )
    peak = np.abs(x_exog).max(axis=0)
    s_eff = np.maximum(s, _EXOG_SD_FLOOR_FRACTION * peak)
    return scale_arr.reshape(-1, 1) * np.outer(sigma, 1.0 / s_eff)


@dataclass(frozen=True)
class VARModelHandles:
    """PyMC variables `VAR.build_in_model` registers into the active model.

    Handed back so a caller embedding a VAR inside a larger PyMC model (or
    inspecting the graph `fit`/`prior_predictive` build) can reach the
    pieces directly, without re-deriving PyMC's own name-mangling inside a
    nested `pm.Model(name=...)`.

    Attributes:
        intercept: Per-equation intercept. `dims=("var",)` when every
            equation has one (the default); `dims=("var_intercept",)`,
            covering only the included equations, when `build_in_model` was
            given a strict subset via `intercept_equations`; `None` when
            `intercept_equations` excluded every equation.
        B: VAR lag coefficients, `dims=("var", "coeff")`. A `Normal`, except
            for one latent series with `n_lags=1`, where it is a
            `Deterministic` assembled from `B_free` and `latent_own_lag` (see
            "Latent stationarity" in `VAR.build_in_model`).
        B_exog: Exogenous coefficients, `dims=("var", "exog")`, or `None`
            when no exogenous block was registered.
        L: Lower-triangular Cholesky factor of the structural-shock scale
            matrix — `(n_vars, n_vars)` for constant volatility, `(T,
            n_vars, n_vars)` for stochastic volatility.
        obs: The registered observation likelihood. For a numpy `endog`,
            `error_dist.build_likelihood`'s return value, an observed RV.
            For a symbolic `endog`, the `pm.Potential` wrapping
            `error_dist.logp`: not a random variable, so it has no dims,
            is invisible to both `sample_prior_predictive` and
            `sample_posterior_predictive`, and cannot be predicted. Named
            `"obs"` either way. With latent series, the `pm.Potential` of the
            observed block's log-likelihood conditional on the latent
            innovations.
        latent: The generated latent path, shape `(T, n_latent)` including
            the `n_lags` initial values, registered as the `Deterministic`
            `"latent"`; `None` without latent series. Its columns follow
            `latent_names`. The latent variables carry no dims and the names
            live on `latent_names` instead, because Impulso's coordinates are
            not prefixed by a nested model and a latent coordinate would
            collide between VARs embedded in the same model.
        latent_names: Names of the latent series, in the path's column
            order; empty without latent series.
    """

    intercept: "pt.TensorVariable | None"
    B: "pt.TensorVariable"
    B_exog: "pt.TensorVariable | None"
    L: "pt.TensorVariable"
    obs: "pt.TensorVariable"
    latent: "pt.TensorVariable | None" = None
    latent_names: tuple[str, ...] = ()


class VAR(ImpulsoBaseModel):
    """Immutable VAR model specification.

    `VAR` specifies the *reduced-form* model — lag order, coefficient prior,
    volatility process, and observation error distribution. Nothing here says
    which shock is which: structural meaning is layered on afterwards, by
    applying an identification scheme to the `FittedVAR` that `fit` returns.

    Attributes:
        lags: Fixed lag order (int >= 1) or selection criterion string.
        max_lags: Upper bound for automatic selection. Only valid with string lags.
        prior: Prior shorthand string or Prior protocol instance.
        volatility: Volatility shorthand string or PyMCVolatilityProcess protocol instance.
        exog_prior_scale: Tightness of the prior on the exogenous coefficients
            `B_exog`, read in contribution space: one prior standard deviation
            moves an endogenous variable by this many of its own AR(1) residual
            standard deviations when the regressor moves by one of its own. The
            default of 100 is deliberately loose — deterministic and exogenous
            terms are conventionally left near-uninformative (the conjugate
            engine uses `Vc = 10e6` on the intercept), and the prior's job here
            is to stop the scale of the regressor from silently setting the
            answer, not to shrink. Lower it to shrink `B_exog` towards zero.
            A scalar applies to every equation. A tuple gives one scale per
            equation, in `endog_names` order, for a caller that needs one
            equation tight (such as an embedded latent series, which has no
            data to pull a loose prior back) and the rest loose; its length is
            checked against the number of endogenous variables when the graph
            is built. Every entry must be positive and finite. Governs the
            exogenous block only; `prior` governs the lag coefficients.
            `VAR.fit`, `VAR.prior_predictive` and `VAR.build_in_model` build
            the same graph, so it applies to all three.
        error_dist: Observation error distribution — shorthand string
            (`"gaussian"`, the default, or `"student_t"`) or an
            `ErrorDistribution` protocol instance. The string form takes the
            adapter's defaults, so `error_dist="student_t"` *infers* the
            degrees of freedom; pass `StudentT(nu=5.0)` to fix them. Heavy-
            tailed errors are rejected in combination with time-varying
            volatility.
    """

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    # Concrete adapters are listed ahead of the bare protocol so a dict (from
    # model_dump) is validated against a concrete schema instead of falling through
    # to the protocol arm, which only accepts an already-built instance. No explicit
    # `Field(discriminator=...)`: each adapter's own `name: Literal[...]` already
    # disambiguates a dict under pydantic's default union validation, and a
    # discriminator here trips a pydantic 2.0 bug when the tagged union's variant
    # (StochasticVolatility) itself has a discriminated field (dynamics/prior).
    prior: Literal["minnesota"] | MinnesotaPrior | Prior = "minnesota"
    volatility: Literal["constant", "sv"] | Constant | StochasticVolatility | PyMCVolatilityProcess = "constant"
    exog_prior_scale: float | tuple[float, ...] = 100.0
    error_dist: Literal["gaussian", "student_t"] | Gaussian | StudentT | ErrorDistribution = "gaussian"

    @field_validator("exog_prior_scale")
    @classmethod
    def _exog_prior_scale_is_positive(cls, value: float | tuple[float, ...]) -> float | tuple[float, ...]:
        values = value if isinstance(value, tuple) else (value,)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(f"exog_prior_scale must be positive and finite, got {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        if self.max_lags is not None and isinstance(self.lags, int):
            raise ValueError("max_lags is only valid when lags is a selection criterion ('aic', 'bic', 'hq')")
        if isinstance(self.lags, int) and self.lags < 1:
            raise ValueError(f"lags must be >= 1, got {self.lags}")
        if self.resolved_error_dist.is_heavy_tailed and self.resolved_volatility.is_time_varying:
            raise ValueError(
                "Heavy-tailed observation errors are not yet supported with "
                "time-varying volatility: the degrees of freedom and the "
                "log-volatility innovation variance both absorb outliers, so "
                "the two are only weakly identified jointly and NUTS mixes "
                "poorly. Use volatility='constant' with error_dist='student_t', "
                "or stochastic volatility with Gaussian errors."
            )
        return self

    @property
    def resolved_prior(self) -> Prior:
        """Resolve string prior shorthand to a Prior instance."""
        if isinstance(self.prior, str):
            return _PRIOR_REGISTRY[self.prior]()
        return self.prior

    @property
    def resolved_volatility(self) -> PyMCVolatilityProcess:
        """Resolve string volatility shorthand to a PyMCVolatilityProcess instance."""
        if isinstance(self.volatility, str):
            return _VOLATILITY_REGISTRY[self.volatility]()
        # `Constant`/`StochasticVolatility` are frozen, so their `name`/`is_time_varying`
        # fields cannot satisfy the protocol's plain (writable) attributes under ty's
        # invariance check, even though nothing ever writes to them.
        return cast(PyMCVolatilityProcess, self.volatility)

    @property
    def resolved_error_dist(self) -> ErrorDistribution:
        """Resolve string error-distribution shorthand to an ErrorDistribution instance."""
        if isinstance(self.error_dist, str):
            return _ERROR_DIST_REGISTRY[self.error_dist]()
        # See resolved_volatility: same frozen-field/protocol-invariance mismatch.
        return cast(ErrorDistribution, self.error_dist)

    @staticmethod
    def _default_sampler() -> Sampler:
        """Default sampler for VAR: cores=1 (macOS PyMC segfault), target_accept=0.8."""
        from impulso.samplers import NUTSSampler

        return NUTSSampler(cores=1, chains=4)

    def fit(
        self,
        data: VARData,
        sampler: Sampler | None = None,
    ) -> "FittedVAR":
        """Estimate the Bayesian VAR model.

        Args:
            data: VARData instance.
            sampler: Sampler protocol instance. Defaults to `_default_sampler()`
                (`cores=1`, `chains=4`, `target_accept=0.8`). Pass an explicit
                `NUTSSampler(cores=n)` to opt into parallel chains.

        Returns:
            FittedVAR with posterior draws.
        """
        from impulso.fitted import FittedVAR

        if sampler is None:
            sampler = self._default_sampler()

        model, n_lags = self._build_pymc_model(data)

        # Sample
        idata = sampler.sample(model)

        return FittedVAR.from_posterior(
            idata,
            data,
            n_lags,
            volatility=self.resolved_volatility,
            error_dist=self.resolved_error_dist,
            pymc_model=model,
        )

    def prior_predictive(
        self,
        data: VARData,
        *,
        draws: int = 500,
        random_seed: int | np.random.Generator | None = None,
    ) -> InferenceDataLike:
        """Simulate data from the prior, before seeing the likelihood.

        Builds the same PyMC graph `fit` builds and calls
        `pymc.sample_prior_predictive` on it, so the prior that gets
        simulated is exactly the prior that gets sampled — no hand-rolled
        second implementation to drift out of sync.

        The simulated `obs` paths are **one-step-ahead given the observed
        lags**: for each prior draw, `y_t = c + B x_t^obs (+ B_exog z_t) +
        L_t eps_t` where `x_t^obs` stacks the *observed* lags of `data`.
        The design matrices are baked into the graph, so this is the prior
        predictive of the estimation-sample conditional means, not a
        simulated path iterated from initial conditions. That is what
        `arviz.plot_ppc(..., group="prior")` expects and what makes the
        prior comparable to the data on the same time axis.

        Note:
            Under `volatility="sv"` the per-variable log-volatility priors
            are seeded from the OLS residuals of `data` (see
            `StochasticVolatility.build_pymc_latent`), so the "prior" is
            mildly data-informed in its scale. The constant-volatility
            default is not.

        Note:
            PyMC returns a single chain, so the `obs` variable has shape
            `(1, draws, T - n_lags, n_vars)`.

        Args:
            data: VARData instance. Anchors the prior simulation on the real
                lags (and, if present, the real exogenous regressors), and
                fixes the lag order when `lags` is a selection criterion.
            draws: Number of prior draws.
            random_seed: Seed or Generator passed straight through to
                `pymc.sample_prior_predictive`.

        Returns:
            InferenceData-schema container with `prior` (every latent), `prior_predictive`
            (the simulated `obs`, dims `(chain, draw, time, var)`) and
            `observed_data` (the realised `obs`) groups.
        """
        import pymc as pm

        model, _ = self._build_pymc_model(data)
        with model:
            return pm.sample_prior_predictive(draws=draws, random_seed=random_seed)

    def build_in_model(
        self,
        endog: "np.ndarray | pt.TensorVariable",
        exog: np.ndarray | None,
        n_lags: int,
        endog_names: Sequence[str],
        exog_names: Sequence[str] | None = None,
        endog_scales: np.ndarray | Sequence[float] | None = None,
        intercept_equations: Sequence[str] | None = None,
        latent_names: Sequence[str] = (),
        latent_init_sigma: float | Sequence[float] = 1.0,
    ) -> VARModelHandles:
        """Register this VAR specification into the active PyMC model.

        The public counterpart of `_build_pymc_model`: where that wrapper
        opens a fresh model and converts a `VARData` into arrays, this
        method takes the arrays directly and registers the intercept, lag
        coefficients, (optional) exogenous coefficients, volatility latents
        and observation likelihood into whichever `pymc.Model` is active on
        entry (`pymc.modelcontext(None)`). `_build_pymc_model` routes
        through this method too, so `fit` and `prior_predictive` share the
        same code path with a caller embedding a VAR inside a larger PyMC
        model — a marketing-mix model with a VAR-shaped baseline, say.

        Symbolic `endog`: `endog` may be a PyTensor variable instead of a
        numpy array, e.g. a `pm.Data` container the caller registered. The
        observed block's likelihood is then `error_dist.logp` wrapped in a
        `pm.Potential` named `"obs"` (a symbolic value cannot be an RV's
        `observed`), with the same density the numpy path's observed RV
        contributes. A Potential is not a random variable: it is invisible
        to both `sample_prior_predictive` and `sample_posterior_predictive`,
        so the symbolic path's observations cannot be predicted. Pass a
        plain tensor: for a `pymc.dims.Data` container, its `.values`. The numpy-only steps are skipped: `endog_scales` is
        required, since `ar1_residual_sd` needs concrete data, and the
        volatility process gets `data=None` instead of OLS pre-fit
        residuals, which only `Constant` volatility accepts, as it ignores
        them. Callers resolve string lag-selection criteria (e.g.
        via `select_lag_order`) before calling this method — it always
        takes a concrete integer `n_lags`.

        Latent series: `latent_names` declares endogenous series that have no
        data. They come first in `endog_names`, and `endog` holds only the
        observed columns that follow them. Their paths are generated inside
        the model, non-centred: `latent_init` puts a `Normal(0,
        latent_init_sigma)` prior on each latent series' first `n_lags`
        values, `latent_innovations` holds standard-normal innovations `z`
        of shape `(T - n_lags, n_latent)`, and for `t >= n_lags` a `scan`
        runs the latent equations of the VAR, `latent_t = c + sum_l
        A_l[lat, :] full_{t-l} + B_exog[lat] x_t + L[lat, lat] z_t`, where
        `full` stacks the latent path with the observed columns. The path,
        shape `(T, n_latent)` including the initial values, is registered as
        the `Deterministic` `"latent"` and returned as `handles.latent`, its
        columns ordered like `handles.latent_names`. Because the latent series
        lead the Cholesky ordering, the observed block's likelihood
        conditional on `z` is exact: `resid_obs - L[obs, lat] z ~ MvN(0,
        L[obs, obs] L[obs, obs]')`, evaluated with `error_dist.logp` on that
        sub-block and registered as a `pm.Potential` named `"obs"`. This is
        the non-centred design of `prototype/REPORT.md`: passing a free latent
        column as symbolic `endog` instead gives a funnel in the latent
        innovation scale. Latent series need Gaussian errors and an
        `endog_scales` entry, since they have no data to compute a scale
        from. `latent_init`, `latent_innovations` and `latent` carry no dims
        (their time axis has no coordinate), so no new coordinate is
        registered for them. `latent_init_sigma` sets only the start of the
        path; the VAR is the latent series' only prior after that.

        Latent stationarity: an explosive draw of the latent
        equations' coefficients makes the generated path explode over the
        sample and can freeze a chain. So the latent-on-latent block (the
        latent rows' coefficients on latent lags, over all `n_lags`) is
        constrained to its stationary region, where the spectral radius of
        its companion matrix is below 1. The observed series are data, not
        generated, so this block alone decides whether the path explodes.
        Either way the posterior is the coefficients' prior truncated to that
        region. How the constraint is enforced depends on the case
        (ADR-0017).

        One latent series with `n_lags=1`: the block is the latent own lag
        `B[0, 0]` alone, and the region is `|B[0, 0]| < 1`. The own lag is
        registered as `latent_own_lag`, a `Normal` with its prior mean and
        standard deviation truncated to (-1, 1); every other coefficient is
        in `B_free`, a flat `Normal` with the prior means and standard
        deviations of those entries; and `B` is a `Deterministic` assembled
        from the two, with the same name and dims as in every other case.
        PyMC samples the own lag through an interval transform, so the log
        density is smooth up to the boundary and NUTS has no wall to diverge
        against, whatever the own-lag prior mean. The own lag starts at 0,
        the interval's midpoint, and the free block at its prior mean, so no
        initial value is set and the host model can always be cloned
        (`model.copy()`, `pm.do`, `pm.observe`); PyMC's start jitter acts in
        the transformed space and stays inside the region. Prior predictive
        draws of the own lag come from the truncated prior, so
        prior-predictive latent paths are stationary. There is no `eig` in
        the graph, so the JAX backend can differentiate the model. The
        truncation's normalising constant is part of the model's log
        density.

        Every other latent case (`n_lags > 1`, or several latent series):
        `B` stays a single `Normal`, and the `pm.Potential`
        `"latent_stationarity"` is 0 inside the region and `-inf` outside.
        With a latent own-lag prior mean near 1 the posterior can press
        against that boundary and give divergences. The prior sets that
        mean: `MinnesotaPrior(own_lag_mean=...)` takes one entry per series,
        so a caller modelling a stationary deviation passes 0 for the latent
        series and 1 for the random-walk observed ones. The model's initial
        point puts every latent equation inside the stationary region: own
        first lag at its prior mean when that is inside the region,
        otherwise 0.5, and every other coefficient in the latent rows 0.
        PyMC's default start for a `Normal` is its mean, so when every
        latent own-lag prior mean is inside the region and the prior puts
        mean 0 on the rest of the latent rows, as `MinnesotaPrior` does, no
        initial value is set. Otherwise this method sets one for `B` as a
        whole, so the observed rows start at their prior mean, which is
        PyMC's default start for them anyway. PyMC cannot clone a model with
        a non-default initial value: `model.copy()`, `pm.do` and
        `pm.observe` raise `NotImplementedError`. A latent own-lag prior mean
        outside the stationary region, the default of 1 included, therefore
        leaves the host model unclonable. PyMC's `jitter+adapt_diag` start
        moves this by up to +-1; a jittered start outside the region has
        `-inf` log density, and PyMC redraws it (`jitter_max_retries`). A
        Potential is not a random variable, so `pm.sample_prior_predictive`
        ignores `"latent_stationarity"`: prior-predictive latent paths are
        drawn from the untruncated prior and can still explode.

        Nesting: open a `pm.Model(name=prefix)` before calling this method
        and every free random variable, `Deterministic` and the likelihood
        it registers come out named `prefix::...` — ordinary PyMC nested-
        model behaviour (see the "Nested `pm.Model(name=prefix)`" section
        of `prototype/REPORT.md`). Coordinates are the one exception: PyMC
        does not prefix coords, so `add_coords` below always lands on the
        *root* model, shared by every nested submodel. Embed at most one
        VAR's variable labelling per model — two VARs with different
        `endog_names`/`exog_names` embedded in the same model will collide
        on `var`/`coeff`/`exog` (identical labels are shared silently;
        different labels raise `ValueError`). `"time"` is a coordinate too,
        so it is subject to the same sharing: two VARs embedded in the same
        model must agree on its *length* (see "Time coordinate" below) —
        checked explicitly, because `add_coords` alone only rejects a
        duplicate coordinate whose *values* differ, not one whose length
        happens to differ while its (unlabelled) content still matches.

        Time coordinate: the likelihood is registered with `dims=("time",
        "var")`, and PyMC requires an *observed* multivariate RV's named
        dims to already be coordinates on the model — unlike a free RV, it
        will not silently auto-register them. If the active model does not
        already carry a `"time"` coordinate, this method adds a plain
        positional one (`range(T_eff)`). `_build_pymc_model` pre-registers
        `"time"` from `data.index` before calling this method, so `fit` and
        `prior_predictive` keep real dates; a caller invoking this method
        directly gets the positional fallback unless it registers `"time"`
        itself first. If the active model *already* carries a `"time"`
        coordinate — this VAR's own wrapper, or a second VAR embedded in
        the same model — and its length does not match this call's number
        of likelihood rows (`T - n_lags`), this method raises `ValueError`
        rather than silently reusing the wrong length; equal length is
        fine regardless of the actual values. A symbolic `endog` is handled
        the same way when its static shape knows `T`. When it does not, as
        for `pm.Data` or `pytensor.shared`, whose value can be swapped for
        another length, no `"time"` coordinate is registered or checked:
        the `pm.Potential` likelihood carries no dims.

        Args:
            endog: Endogenous data, shape `(T, n_vars)`: a numpy array, or a
                2-D PyTensor variable (see "Symbolic `endog`" above). With
                latent series, only the observed columns, shape `(T, n_vars
                - n_latent)`.
            exog: Optional exogenous regressors, shape `(T, n_exog)`. `None`
                if the model has no exogenous block.
            n_lags: Lag order. Always a concrete integer — resolving a
                string selection criterion is the caller's job.
            endog_names: Names for each endogenous variable, length
                `n_vars`, in VAR order: latent series first, then the
                observed columns of `endog`. Labels the
                `var`/`var1`/`var2`/`coeff` coordinates.
            exog_names: Names for each exogenous column, length `n_exog`.
                Required when `exog` is given; labels the `exog` coordinate.
            endog_scales: Per-variable scale `sigma`, shape `(n_vars,)` — an
                array or any array-like (e.g. a plain list) accepted by
                `np.asarray(..., dtype=float)`. `None` (the default) computes
                it from `endog` with `ar1_residual_sd`; required when `endog`
                is symbolic or there are latent series, and then covering
                every name in `endog_names`. The same array feeds
                both the prior's Minnesota cross-lag scaling `sigma_i /
                sigma_j` (`Prior.build_priors`, docs/adr/0015) and the
                exogenous prior (`_exog_prior_sigma`, docs/adr/0012), so it
                matters even when `exog` is `None`.
            intercept_equations: Names of the endogenous equations that get
                an intercept, a subset of `endog_names` in any order. `None`
                (the default) means every equation — the same graph as
                before this argument existed. An excluded equation has no
                intercept term at all (its `mu` adds zero), so its series is
                modelled as a zero-mean deviation around a level owned
                elsewhere in the model. Naming every equation, in any order,
                is the same as `None`: the intercept keeps `dims="var"`. A
                strict subset gets its own `"var_intercept"` coordinate,
                ordered like `endog_names` (not like this argument), and
                like every Impulso coordinate it is not prefixed by a nested
                model. An empty sequence registers no intercept variable,
                and the returned handles' `intercept` is `None`.
                Latent series may be excluded, e.g. to model a latent
                series as a zero-mean deviation.
            latent_names: Names of the latent endogenous series (see "Latent
                series" above), which must be the first entries of
                `endog_names`, in the same order. Empty (the default) means
                no latent series and the graph described above for observed
                data only.
            latent_init_sigma: Standard deviation of the zero-mean Normal
                prior on each latent series' first `n_lags` values: a scalar
                shared by every latent series, or one entry per latent
                series. Ignored without latent series.

        Returns:
            `VARModelHandles` wrapping the intercept, coefficient,
            volatility and likelihood variables this call registered.

        Raises:
            ValueError: If the active model already carries a `"time"`
                coordinate whose length does not match this call's number
                of likelihood rows (`T - n_lags`) — see "Time coordinate"
                above.
            ValueError: If any entry of the scale — computed or supplied via
                `endog_scales` — is zero, negative or non-finite.
            ValueError: If `endog_scales` does not have shape `(n_vars,)`.
            TypeError: If `endog` is a dimmed `XTensorVariable` rather than a
                plain tensor — pass its `.values`.
            ValueError: If `endog` is symbolic and `endog_scales` is `None`,
                if it is not 2-D, or if its static column count differs
                from `len(endog_names)`.
            ValueError: If `intercept_equations` names an equation not in
                `endog_names`, or names one more than once.
            ValueError: With latent series: if `endog_names`
                does not start with exactly `latent_names`, if no observed
                series is left, if `endog`'s column count is not the number
                of observed series, if `exog` has a different number of
                rows from `endog`, if `endog_scales` is missing or has
                entries for the observed series only, or if
                `latent_init_sigma` has the wrong length or a non-positive
                entry.
            ValueError: On the embedded path — `endog` symbolic or latent
                series present: if `self.lags` is a selection
                criterion string (lag selection runs OLS on data), or if
                `self.volatility` is not `"constant"`/`Constant` (stochastic
                volatility seeds its priors from OLS residuals). With latent
                series specifically, also if `self.error_dist` is not
                Gaussian (a multivariate Student-t's conditional is not a
                Student-t with the same `nu`). Raised before anything is
                registered into the model.
        """
        import pymc as pm
        import pytensor.tensor as pt
        from pytensor.graph.basic import Variable

        model = pm.modelcontext(None)

        symbolic = isinstance(endog, Variable)
        error_dist = self.resolved_error_dist
        latent_names = tuple(latent_names)
        n_latent = len(latent_names)
        _reject_unsupported_for_embedded_path(
            lags=self.lags,
            volatility=self.resolved_volatility,
            error_dist=error_dist,
            symbolic=symbolic,
            n_latent=n_latent,
        )
        if n_latent:
            n_vars = _latent_n_vars(endog, exog, endog_names, endog_scales, latent_names)
            init_sigma = _latent_init_sigma(latent_init_sigma, n_latent)
        else:
            n_vars = _symbolic_endog_n_vars(endog, endog_scales, endog_names) if symbolic else endog.shape[1]
        # With latent series the observed block is conditioned on
        # generated values, so its likelihood is a Potential, as for a
        # symbolic `endog`, and the numpy-only steps are skipped.
        potential = symbolic or n_latent > 0

        # `sigma` is the per-variable scale — computed once here (or taken
        # from the caller) and reused for both the Minnesota lag-coefficient
        # prior (cross-lag sigma_i/sigma_j scaling, docs/adr/0015) and the
        # exogenous-coefficient prior below, so a zero/non-finite entry would
        # blow up both.
        sigma = _resolve_sigma(endog, endog_scales, endog_names, n_vars)
        intercept_mask = _intercept_mask(endog_names, intercept_equations)
        prior_params = self.resolved_prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=sigma)

        # With latent series the design matrix needs the generated path, so
        # it is built further down, once the coefficients exist; only the
        # exogenous block is needed before then.
        if n_latent:
            X_exog = exog[n_lags:] if exog is not None else None
        else:
            Y, X_lag, X_exog = build_lag_design_matrix(endog, n_lags, exog)

        # Number of likelihood rows, `T - n_lags`. A symbolic `endog` only
        # knows it when its static shape does: `pm.Data` and
        # `pytensor.shared` leave it `None`, since their value can be
        # swapped for one of a different length.
        if symbolic:
            static_T = endog.type.shape[0]
            n_rows = None if static_T is None else static_T - n_lags
        else:
            n_rows = endog.shape[0] - n_lags

        # OLS pre-fit residuals seed the volatility process's per-variable
        # priors. Numpy-only: they need concrete data for every column, so a
        # symbolic `endog` or a latent series passes `None`.
        # Constant-volatility adapters ignore this; only stochastic ones use it.
        resid = None if potential else _ols_residuals(Y, X_lag, X_exog)

        # Coordinates make the posterior self-describing: `B` comes back labelled
        # by variable and by "L<lag>.<variable>" coefficient instead of positional
        # `B_dim_0` / `B_dim_1`. Variable names come from `impulso._posterior` —
        # the schema ConjugateVAR constructs against too, so both estimators
        # agree. `coeff` is lag-major to mirror the X_lag hstack above. Not
        # prefixed by a nested `pm.Model(name=...)` — see the docstring above.
        coords: dict[str, object] = {
            "var": list(endog_names),
            "var1": list(endog_names),
            "var2": list(endog_names),
            "coeff": [f"L{lag}.{name}" for lag in range(1, n_lags + 1) for name in endog_names],
        }
        if exog_names is not None:
            coords["exog"] = list(exog_names)
        intercepted = [name for name, keep in zip(endog_names, intercept_mask, strict=True) if keep]
        if 0 < len(intercepted) < n_vars:
            coords["var_intercept"] = intercepted
        coords.update(_time_coord(model, n_rows))
        model.add_coords(coords)

        # Intercept. Every equation gets one by default (`dims="var"`, the
        # graph from before `intercept_equations` existed). A strict subset
        # gets a shorter free variable on its own coord, scattered into a
        # length-`n_vars` vector with literal zeros for the excluded
        # equations; excluding every equation drops the term entirely.
        intercept, intercept_term = _register_intercept(intercept_mask)

        # VAR coefficients with Minnesota prior. With latent series this also keeps the latent block
        # stationary (ADR-0016, ADR-0017).
        B = _register_coefficients(model, prior_params["B_mu"], prior_params["B_sigma"], n_latent, n_lags)

        # Exogenous coefficients. The prior scales with the data so that it
        # encodes the same belief regardless of the units the regressors
        # happen to be measured in (ADR-0012).
        if X_exog is not None:
            B_exog = pm.Normal(
                EXOG_COEFFICIENTS,
                mu=0,
                sigma=_exog_prior_sigma(sigma, X_exog, self.exog_prior_scale, exog_names),
                dims=("var", "exog"),
            )
        else:
            B_exog = None

        # Volatility process: registers latent vars, returns L (Cholesky factor of Σ_t).
        # For constant volatility, L is (n_vars, n_vars) and time-invariant.
        # For stochastic volatility, L is (T, n_vars, n_vars) — per-t.
        volatility = self.resolved_volatility
        # `T` is ignored by `Constant`, the one adapter the symbolic path
        # supports; an unknown static length falls back to the symbolic one.
        L = volatility.build_pymc_latent(
            n_vars=n_vars, T=endog.shape[0] - n_lags if n_rows is None else n_rows, data=resid
        )
        # Sigma deterministic is only registered for time-invariant L —
        # for SV, materialising (T, n, n) per draw is wasteful; users can
        # reconstruct per-t Σ via `volatility.cholesky_at(posterior, t)`.
        if L.ndim == 2:
            pm.Deterministic("Sigma", pm.math.dot(L, L.T), dims=("var1", "var2"))

        # Under Student-t errors L L' is the *scale* matrix (ADR-0007). With a symbolic `endog` or latent
        # series the likelihood is a `pm.Potential`, invisible to predictive sampling; latent series are
        # non-centred, so the observed rows are conditioned on their innovations `z` (ADR-0016).
        latent = z = None
        if n_latent:
            latent, z = _latent_path(endog, X_exog, n_lags, n_vars, n_latent, intercept_term, B, B_exog, L, init_sigma)
            full = pt.concatenate([latent, pt.as_tensor_variable(endog)], axis=1)
            Y, X_lag, _ = build_lag_design_matrix(full, n_lags)
        mu = intercept_term + pm.math.dot(X_lag, B.T)
        if B_exog is not None:
            mu = mu + pm.math.dot(X_exog, B_exog.T)

        obs = _register_likelihood(error_dist, mu, L, Y, potential=potential, n_latent=n_latent, z=z)

        return VARModelHandles(
            intercept=intercept, B=B, B_exog=B_exog, L=L, obs=obs, latent=latent, latent_names=latent_names
        )

    def _build_pymc_model(self, data: VARData) -> tuple[Any, int]:
        """Build the PyMC model graph for this specification.

        Resolves the lag order (running `select_lag_order` when `lags` is a
        criterion string), opens a fresh `pymc.Model`, and delegates to
        `build_in_model` to register the intercept, coefficient, exogenous,
        volatility and likelihood nodes. The design matrices are baked into
        the graph as constants, so the returned model is tied to `data`.

        Shared by `fit` (which samples the graph) and `prior_predictive`
        (which draws from it without conditioning on the observations).

        Args:
            data: VARData instance.

        Returns:
            Tuple of the built `pymc.Model` and the resolved lag order. The
            model is typed `Any` so that importing `impulso.spec` does not
            pull in PyMC — the same reason `FittedVAR.pymc_model` is.
        """
        import pymc as pm

        from impulso._lag_selection import select_lag_order

        # Resolve lags
        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        # "time" is registered here, from `data.index`, rather than inside
        # `build_in_model` — that method has no date index to draw one
        # from, only arrays. Registering it before `build_in_model` runs
        # means the likelihood's `dims=("time", "var")` binds to real dates.
        with pm.Model(coords={"time": data.index[n_lags:]}) as model:
            self.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=n_lags,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )

        return model, n_lags
