"""Schema gate for `FittedVAR.from_posterior`.

`VAR.fit` and `ConjugateVAR.fit` each build their own posterior — a PyMC
graph and a hand-built `xarray.Dataset` respectively (ADR-0004) — but both
answer to the same coefficient-block contract (`impulso._posterior`), the
same volatility-seam contract (`Constant.cholesky_at` / `ConjugateVolatility.
cholesky_at` reading `L`, `StochasticVolatility.cholesky_at` reading `h` and
`R_chol`), and, under Student-t errors, the same `nu` deterministic
(`impulso.observation._nu_draws`). Both callers get that schema for free
from the code that builds their posterior. `FittedVAR.from_posterior` is a
*public* seam for a posterior estimated elsewhere — by a future Impulso
estimator or by an embedding caller — so unlike those two callers
it cannot trust the shape it receives. This module is that trust boundary.

Every check names the offending variable and the layout expected, so a
caller who gets a `ValueError` can fix the posterior without reading this
file.

One deliberate asymmetry: `B`, `intercept` and `B_exog` are checked by dim
*name* (`var`, `coeff`, `exog`) because `VAR.fit` registers them with
explicit PyMC `dims=`, so both estimators agree on the names. The
volatility-seam variables (`L`, and `h` / `R_chol` under stochastic
volatility) are checked by *shape* only, because `Constant.build_pymc_latent`
and `StochasticVolatility.build_pymc_latent` both register them via a bare
`pm.Deterministic(name, ...)` with no `dims=`, so PyMC assigns synthetic
per-fit names (`L_dim_0`, `h_dim_0`, ...) that `ConjugateVAR.fit`'s explicit
`var1`/`var2` on `L` never match. `Constant.cholesky_at` and
`StochasticVolatility.cholesky_at` themselves read these positionally, for
the same reason.

Which shape is required depends on which volatility adapter produced the
posterior, so `validate_posterior_schema` takes the resolved adapter and
dispatches on its type: `Constant` and every `ConjugateVolatility` break
(`ConjugateVAR.fit`'s two possible outputs) both read a base `L`; a
`StochasticVolatility` posterior carries no `L` at all and reads `h` /
`R_chol` instead (see `StochasticVolatility.cholesky_at` in
`impulso/sv/spec.py`). An adapter this module does not recognise raises a
clear error rather than a misleading "missing L".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from impulso._posterior import CHOLESKY_FACTOR, COEFFICIENTS, DEGREES_OF_FREEDOM, EXOG_COEFFICIENTS, INTERCEPT
from impulso.conjugate_volatility import ConjugateVolatility
from impulso.sv.spec import StochasticVolatility
from impulso.volatility import Constant

if TYPE_CHECKING:
    import xarray as xr

    from impulso.data import VARData
    from impulso.protocols import ErrorDistribution, VolatilityProcess

_LOG_VOLATILITY = "h"
_MIXING_CHOLESKY = "R_chol"


def _require_variable(
    posterior: xr.Dataset,
    name: str,
    *,
    expected_dims: tuple[str, ...],
    sizes: dict[str, int],
    layout: str,
) -> xr.DataArray:
    """Fetch `posterior[name]`, checked against `expected_dims` and `sizes`.

    Args:
        posterior: The `posterior` group.
        name: Variable to fetch.
        expected_dims: The dim names `posterior[name]` must carry, as a set
            (order is not significant — both estimators may transpose).
        sizes: Mapping of dim name to the size it must have.
        layout: Human-readable description of the expected layout, echoed
            in every error this raises.

    Returns:
        `posterior[name]`.

    Raises:
        ValueError: If `name` is absent, its dims are not exactly
            `expected_dims`, or a dim in `sizes` has the wrong size.
    """
    if name not in posterior:
        raise ValueError(f"FittedVAR.from_posterior: posterior is missing required variable {name!r}. {layout}")
    da = posterior[name]
    if set(da.dims) != set(expected_dims):
        raise ValueError(
            f"FittedVAR.from_posterior: {name!r} has dims {da.dims}, expected {expected_dims} (any order). {layout}"
        )
    for dim, size in sizes.items():
        actual = da.sizes[dim]
        if actual != size:
            raise ValueError(
                f"FittedVAR.from_posterior: {name!r} dim {dim!r} has size {actual}, expected {size}. {layout}"
            )
    return da


def _require_shape_only(posterior: xr.Dataset, name: str, expected_trailing: list[int], layout: str) -> None:
    """Fetch `posterior[name]`, checked by shape alone: no dim-name check.

    For a variable PyMC registers via a bare `pm.Deterministic(name, ...)`
    with no `dims=` — `L`, `h`, `R_chol` — the trailing dim *names* are
    synthetic and per-fit (`{name}_dim_0`, ...), so only shape survives
    across fits: exactly 4 dims including `chain` and `draw`, with the
    other two matching `expected_trailing` **in order**. Order matters
    whenever the two trailing axes are not both `n_vars` (e.g. `h`'s
    `(T, n_vars)`), since the adapter that reads the variable back
    (`Constant.cholesky_at`, `StochasticVolatility.cholesky_at`) indexes it
    positionally.

    Args:
        posterior: The `posterior` group.
        name: Variable to fetch.
        expected_trailing: The two non-`chain`/`draw` axis sizes, in the
            order the adapter reading `name` expects them.
        layout: Human-readable description of the expected layout, echoed
            in every error this raises.

    Raises:
        ValueError: If `name` is absent, does not have exactly 4 dims
            including `chain` and `draw`, or its trailing shape does not
            equal `expected_trailing`.
    """
    if name not in posterior:
        raise ValueError(f"FittedVAR.from_posterior: posterior is missing required variable {name!r}. {layout}")
    da = posterior[name]
    if len(da.dims) != 4 or "chain" not in da.dims or "draw" not in da.dims:
        raise ValueError(f"FittedVAR.from_posterior: {name!r} has dims {da.dims}. {layout}")
    trailing = [dim for dim in da.dims if dim not in ("chain", "draw")]
    sizes = [da.sizes[dim] for dim in trailing]
    if sizes != expected_trailing:
        raise ValueError(f"FittedVAR.from_posterior: {name!r} has trailing shape {sizes}. {layout}")


def _explicit_string_labels(da: xr.DataArray, dim: str) -> list[str] | None:
    """Labels of `dim` on `da`, or `None` when no explicit string coordinate is set.

    `ConjugateVAR.fit` builds its posterior `DataArray`s with dims only, no
    coordinate labels, so `dim` falls back to a positional default index
    with no entry in `.coords` at all. `VAR.fit` registers every dim through
    `pm.Model(coords=...)`, so its labels are real strings. Only the latter
    is checked — a hand-built posterior is free to leave labels off and be
    trusted positionally.
    """
    if dim not in da.coords:
        return None
    labels = da.coords[dim].values
    if labels.dtype.kind not in "OU":  # not string-like: a default integer index
        return None
    return [str(label) for label in labels]


def _validate_labels(da: xr.DataArray, dim: str, expected: list[str], variable_name: str) -> None:
    """Check `dim` labels on `da` against `expected`, when labels are explicit."""
    labels = _explicit_string_labels(da, dim)
    if labels is not None and labels != expected:
        raise ValueError(
            f"FittedVAR.from_posterior: {variable_name!r} {dim!r} labels {labels} do not match the expected {expected}."
        )


def _validate_coefficients(posterior: xr.Dataset, var_names: list[str], n_lags: int) -> xr.DataArray:
    """Validate `B`: dims, shape, and (when explicit) lag-major `coeff` / `var` labels."""
    n_vars = len(var_names)
    n_coeff = n_vars * n_lags
    layout = (
        f"'{COEFFICIENTS}' must have dims (chain, draw, var, coeff) with var={n_vars} and "
        f"coeff={n_coeff} (= n_vars * n_lags), lag-major: the lag-1 block of every variable "
        "first, then lag-2, etc."
    )
    b_da = _require_variable(
        posterior,
        COEFFICIENTS,
        expected_dims=("chain", "draw", "var", "coeff"),
        sizes={"var": n_vars, "coeff": n_coeff},
        layout=layout,
    )
    expected_coeff_labels = [f"L{lag}.{name}" for lag in range(1, n_lags + 1) for name in var_names]
    _validate_labels(b_da, "coeff", expected_coeff_labels, COEFFICIENTS)
    _validate_labels(b_da, "var", var_names, COEFFICIENTS)
    return b_da


def _validate_intercept(posterior: xr.Dataset, var_names: list[str]) -> None:
    """Validate `intercept`: dims, shape, and (when explicit) `var` labels."""
    n_vars = len(var_names)
    intercept_da = _require_variable(
        posterior,
        INTERCEPT,
        expected_dims=("chain", "draw", "var"),
        sizes={"var": n_vars},
        layout=f"'{INTERCEPT}' must have dims (chain, draw, var) with var={n_vars}.",
    )
    _validate_labels(intercept_da, "var", var_names, INTERCEPT)


def _validate_constant_cholesky(posterior: xr.Dataset, n_vars: int) -> None:
    """Validate `L`: present, 4-dimensional, with a (chain, draw, n_vars, n_vars) shape.

    Used for `Constant` and every `ConjugateVolatility` break — both read
    the same base Cholesky factor `L` (`ConjugateVolatility.cholesky_at`
    scales it by a deterministic `s_t`, but `L` itself is the same shape
    either way). Checked by shape rather than dim name — see the module
    docstring for why `VAR.fit`'s `L` and `ConjugateVAR.fit`'s `L` do not
    share dim names.
    """
    layout = (
        f"'{CHOLESKY_FACTOR}' must have 4 dims including 'chain' and 'draw', with the remaining two "
        f"axes both of size {n_vars} — the lower-triangular Cholesky factor of Sigma that "
        "`Constant.cholesky_at` reads directly from the posterior."
    )
    _require_shape_only(posterior, CHOLESKY_FACTOR, [n_vars, n_vars], layout)


def _validate_stochastic_volatility(posterior: xr.Dataset, n_vars: int, expected_t: int) -> None:
    """Validate `h` and `R_chol`: the posterior contract `StochasticVolatility.cholesky_at` reads.

    `StochasticVolatility` carries no `L` at all — `Constant.cholesky_at`'s
    `L` is reconstructed on the fly as `diag(exp(h/2)) @ R_chol`
    (`StochasticVolatility._clark_reconstruct`), so the posterior must carry
    both `h` (the per-variable log-volatility path) and `R_chol` (the
    unit-diagonal mixing factor) instead.

    Args:
        posterior: The `posterior` group.
        n_vars: Number of endogenous variables.
        expected_t: In-sample length after lag trimming
            (`data.endog.shape[0] - n_lags`), the size `h`'s time axis must
            have.
    """
    h_layout = (
        f"'{_LOG_VOLATILITY}' must have 4 dims including 'chain' and 'draw', with the remaining "
        f"two axes of size {expected_t} (in-sample T = len(data.endog) - n_lags) then {n_vars} "
        "(n_vars), in that order — the per-variable log-volatility path "
        "`StochasticVolatility.cholesky_at` reads as `h[:, :, t, :]`."
    )
    _require_shape_only(posterior, _LOG_VOLATILITY, [expected_t, n_vars], h_layout)

    r_layout = (
        f"'{_MIXING_CHOLESKY}' must have 4 dims including 'chain' and 'draw', with the remaining "
        f"two axes both of size {n_vars} — the unit-diagonal lower-triangular mixing factor "
        "`StochasticVolatility.cholesky_at` reads directly from the posterior."
    )
    _require_shape_only(posterior, _MIXING_CHOLESKY, [n_vars, n_vars], r_layout)


def _validate_volatility(posterior: xr.Dataset, data: VARData, n_lags: int, volatility: VolatilityProcess) -> None:
    """Validate the posterior's volatility-seam variables, dispatched by adapter type.

    `L` and `h`/`R_chol` are different, non-overlapping contracts (a
    `StochasticVolatility` posterior carries no `L` at all), so which one is
    required cannot be inferred from the posterior alone — it depends on
    which adapter produced it, the resolved `volatility` `from_posterior`
    was given (or defaulted to `Constant()`).

    Args:
        posterior: The `posterior` group.
        data: The `VARData` the posterior is claimed to have been fitted on;
            `data.endog.shape[0] - n_lags` gives the in-sample length a
            `StochasticVolatility` posterior's `h` must span.
        n_lags: Lag order, used with `data` to size `h`'s time axis.
        volatility: The resolved volatility adapter.

    Raises:
        ValueError: If the adapter-appropriate variables are missing or
            mis-shaped.
        TypeError: If `volatility` is not a `Constant`, `ConjugateVolatility`,
            or `StochasticVolatility` instance.
    """
    n_vars = len(data.endog_names)
    if isinstance(volatility, (Constant, ConjugateVolatility)):
        _validate_constant_cholesky(posterior, n_vars)
    elif isinstance(volatility, StochasticVolatility):
        expected_t = data.endog.shape[0] - n_lags
        _validate_stochastic_volatility(posterior, n_vars, expected_t)
    else:
        raise TypeError(
            f"FittedVAR.from_posterior: unrecognised volatility adapter {type(volatility).__name__!r}. "
            f"Schema validation knows how to check a Constant or ConjugateVolatility posterior "
            f"(a Cholesky factor {CHOLESKY_FACTOR!r}) and a StochasticVolatility posterior "
            f"({_LOG_VOLATILITY!r} and {_MIXING_CHOLESKY!r}); pass one of those as `volatility`, or "
            "extend impulso._posterior_validation to recognise a new adapter."
        )


def _validate_exog_coefficients(posterior: xr.Dataset, data: VARData) -> None:
    """Validate `B_exog` when `data` carries exogenous regressors; required in that case only."""
    if data.exog is None:
        return
    n_vars = len(data.endog_names)
    n_exog = len(data.exog_names) if data.exog_names is not None else data.exog.shape[1]
    layout = (
        f"'{EXOG_COEFFICIENTS}' must have dims (chain, draw, var, exog) with var={n_vars} and "
        f"exog={n_exog}; required because `data` carries exogenous regressors."
    )
    exog_da = _require_variable(
        posterior,
        EXOG_COEFFICIENTS,
        expected_dims=("chain", "draw", "var", "exog"),
        sizes={"var": n_vars, "exog": n_exog},
        layout=layout,
    )
    _validate_labels(exog_da, "var", data.endog_names, EXOG_COEFFICIENTS)
    if data.exog_names is not None:
        _validate_labels(exog_da, "exog", data.exog_names, EXOG_COEFFICIENTS)


def _validate_degrees_of_freedom(posterior: xr.Dataset, error_dist: ErrorDistribution) -> None:
    """Validate `nu` when `error_dist` is heavy-tailed; required in that case only."""
    if not error_dist.is_heavy_tailed:
        return
    _require_variable(
        posterior,
        DEGREES_OF_FREEDOM,
        expected_dims=("chain", "draw"),
        sizes={},
        layout=(
            f"'{DEGREES_OF_FREEDOM}' must have dims (chain, draw); required because error_dist "
            f"{type(error_dist).__name__} is heavy-tailed."
        ),
    )


def validate_posterior_schema(
    posterior: xr.Dataset,
    data: VARData,
    n_lags: int,
    volatility: VolatilityProcess,
    error_dist: ErrorDistribution,
) -> None:
    """Validate a hand-built posterior against the schema both estimators produce.

    Args:
        posterior: The `posterior` group, as an `xarray.Dataset`.
        data: The `VARData` the posterior is claimed to have been fitted on;
            supplies `n_vars` (`len(data.endog_names)`) and, when present,
            the exogenous block.
        n_lags: Lag order the posterior's `coeff` dimension is checked
            against.
        volatility: Resolved volatility adapter; determines whether the
            posterior must carry `L` (`Constant` / `ConjugateVolatility`) or
            `h` and `R_chol` (`StochasticVolatility`).
        error_dist: Resolved error-distribution adapter; when
            `error_dist.is_heavy_tailed`, `nu` is required.

    Raises:
        ValueError: If a required variable is missing, has the wrong dims,
            the wrong size along a dim, or — when the posterior carries
            explicit string labels — lag-major / variable-name labels that
            disagree with the layout `VAR.fit` and `ConjugateVAR.fit` both
            produce. The message names the offending variable and the
            expected layout.
        TypeError: If `volatility` is an adapter this module does not
            recognise (not a `Constant`, `ConjugateVolatility`, or
            `StochasticVolatility`).
    """
    _validate_coefficients(posterior, data.endog_names, n_lags)
    _validate_intercept(posterior, data.endog_names)
    _validate_volatility(posterior, data, n_lags, volatility)
    _validate_exog_coefficients(posterior, data)
    _validate_degrees_of_freedom(posterior, error_dist)
