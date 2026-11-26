"""THE posterior schema — the coefficient-block contract shared by both estimators.

`spec.py`'s PyMC graph and `conjugate.py`'s hand-built Dataset are separate
likelihood implementations by design (ADR-0011); what they share is only the
schema below. Each registers its reduced-form coefficient blocks under these
names and layouts, and every consumer — residual reconstruction, forecasting,
the scenario engines, identification schemes — reads them back through this
module's accessors instead of re-knowing the key strings by convention.

Variable layouts (all leading with the `(chain, draw)` sample dims):

* ``B`` — `(chain, draw, n, n*p)`: VAR lag coefficients with **lag-major**
  coefficient columns (the lag-1 block first, mirroring the `X_lag` hstack
  both estimators build their design matrix from).
* ``intercept`` — `(chain, draw, n)`: per-equation intercepts.
* ``B_exog`` — `(chain, draw, n, n_exog)`: contemporaneous exogenous
  coefficients; present only when the estimator consumed an exogenous block.

Not owned here: the layouts of the volatility seam's ``L`` (owned by
`volatility.py` / `conjugate_volatility.py`) and the error-distribution
seam's ``nu`` (owned by `observation.py`). Their names are spelled here,
with the latent paths' ``latent``, so code that reads them by name shares
one spelling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

    from impulso._arviz_compat import InferenceDataLike

COEFFICIENTS = "B"
"""VAR lag-coefficient draws, `(chain, draw, n, n*p)`, lag-major columns."""

INTERCEPT = "intercept"
"""Per-equation intercept draws, `(chain, draw, n)`."""

EXOG_COEFFICIENTS = "B_exog"
"""Exogenous-coefficient draws, `(chain, draw, n, n_exog)`; present only when consumed."""

CHOLESKY_FACTOR = "L"
"""Name of the constant-volatility Cholesky factor of the innovation scale matrix; its layout is `volatility.py`'s."""

DEGREES_OF_FREEDOM = "nu"
"""Name of the Student-t errors' degrees of freedom; its layout is `observation.py`'s."""

LATENT_PATHS = "latent"
"""Name of the generated latent-series paths, `(chain, draw, T, n_latent)`; present only with latent series."""


def posterior_dataset(idata: InferenceDataLike) -> xr.Dataset:
    """The `posterior` group as an `xarray.Dataset`.

    On ArviZ 1 the raw group is a `DataTree` node rather than a Dataset,
    which would violate the `xr.Dataset` contract every volatility,
    error-distribution, and identification implementation is written
    against. A fresh Dataset is built on each call, so callers that use the
    posterior more than once — or hand it to a memoising helper such as
    `_PosteriorCache` — must bind the result to a local first and pass that
    local through.

    Args:
        idata: The container holding the fit's groups.

    Returns:
        The posterior group as an `xarray.Dataset`.

    Raises:
        KeyError: If the container has no posterior group.
    """
    from impulso._arviz_compat import get_group_dataset

    return get_group_dataset(idata, "posterior")


def coefficient_draws(posterior: xr.Dataset) -> np.ndarray:
    """`B` draws, `(chain, draw, n, n*p)` with lag-major coefficient columns."""
    return posterior[COEFFICIENTS].values


def intercept_draws(posterior: xr.Dataset) -> np.ndarray:
    """`intercept` draws, `(chain, draw, n)`."""
    return posterior[INTERCEPT].values


def exog_coefficient_draws(posterior: xr.Dataset) -> np.ndarray | None:
    """`B_exog` draws, `(chain, draw, n, n_exog)`, or `None` when absent."""
    if EXOG_COEFFICIENTS not in posterior:
        return None
    return posterior[EXOG_COEFFICIENTS].values


def has_exog_block(posterior: xr.Dataset) -> bool:
    """Whether the estimator consumed an exogenous block (`B_exog` present)."""
    return EXOG_COEFFICIENTS in posterior
