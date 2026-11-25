"""A VAR compiled from its PyMC model into a block a NumPyro model can call.

`VAR.compile_for_numpyro` builds the VAR's PyMC model once and returns a
`CompiledNumPyroBlock`. The block holds two JAX functions compiled from that
model: the joint log-density, and the map to the quantities it records. No
NumPyro code is written for the VAR itself (ADR-0016 keeps one model, defined
in PyMC).

JAX and NumPyro are imported only inside the methods that need them, so
importing this module needs neither.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

UNCONSTRAINED_SITE_PREFIX = "unconstrained/"
"""Prefix of the NumPyro site holding each PyMC value variable, on PyMC's unconstrained scale."""

LOG_DENSITY_SITE = "VAR_log_density"
"""Name of the `numpyro.factor` site that adds the VAR's joint log-density."""


def add_with_zero_gradient(
    log_density: Callable[..., Any], zero_gradient_term: Callable[..., Any]
) -> Callable[..., Any]:
    """Add `zero_gradient_term` to `log_density`, so that JAX passes the term no gradient.

    Both arguments are JAX functions that take the same inputs and return a
    one-element sequence, as `pymc.sampling.jax.get_jaxified_graph` compiles
    them. The returned function takes those inputs and returns a one-element
    sequence holding the sum. `zero_gradient_term` sees `jax.lax.stop_gradient`
    copies of the inputs, so JAX never differentiates it: its value counts and
    its gradient is zero. That is the gradient of a term that is constant
    wherever it is finite, such as the `latent_stationarity` Potential.
    """

    def total(*inputs: Any) -> list[Any]:
        import jax

        held = [jax.lax.stop_gradient(value) for value in inputs]
        return [log_density(*inputs)[0] + zero_gradient_term(*held)[0]]

    return total


@dataclass(frozen=True)
class VARModelValues:
    """Values, as JAX arrays, of the VAR quantities a `CompiledNumPyroBlock` call records.

    The NumPyro counterpart of `VARModelHandles`: where those are PyMC
    variables, these are the values the block computed from the NumPyro
    model's current parameter values.

    Attributes:
        intercept: Per-equation intercept, shape `(n_vars,)`.
        B: VAR lag coefficients, shape `(n_vars, n_vars * n_lags)`, lag-major
            columns.
        L: Lower-triangular Cholesky factor of the structural-shock scale
            matrix, shape `(n_vars, n_vars)`.
        latent: The generated latent paths, shape `(T, n_latent)` including
            the initial values; `None` without latent series. Its columns
            follow `latent_names`.
        latent_names: Names of the latent series, in the column order of
            `latent`; empty without latent series.
    """

    intercept: Any
    B: Any
    L: Any
    latent: Any = None
    latent_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledNumPyroBlock:
    """A VAR compiled once from its PyMC model, called inside a NumPyro model.

    Built by `VAR.compile_for_numpyro`; see that method for what a call adds to
    the NumPyro model and for the limitations.

    Attributes:
        numpyro_site_names: NumPyro sample-site name of each PyMC value
            variable, keyed by the value variable's name. Each site holds that
            variable on PyMC's unconstrained scale.
        site_shapes: Shape of each site's value, keyed by NumPyro site name.
        recorded_names: Names under which a call records the VAR quantities
            with `numpyro.deterministic`, in the order of
            `recorded_quantities`'s outputs.
        log_density: JAX function from the value of every site (in
            `numpyro_site_names` order) and `endog` to a one-element sequence
            holding the VAR's joint log-density, PyMC's transform corrections
            included. JAX passes the `latent_stationarity` Potential, when
            there is one, no gradient (see `add_with_zero_gradient`).
        recorded_quantities: JAX function from the same inputs to the
            recorded quantities, in `recorded_names` order.
        initial_point: PyMC's initial point of the VAR's parameters, keyed by
            NumPyro site name, on PyMC's unconstrained scale. Pass it to
            NumPyro's `init_to_value(values=...)` to start the sampler where
            PyMC would.
        latent_names: Names of the latent series, in the column order of the
            recorded `latent` paths; empty without latent series.
        n_endog_rows: Number of rows of the `endog` every call must receive,
            or `None` when a call accepts an `endog` of any length.
    """

    numpyro_site_names: Mapping[str, str]
    site_shapes: Mapping[str, tuple[int, ...]]
    recorded_names: tuple[str, ...]
    log_density: Callable[..., Any]
    recorded_quantities: Callable[..., Any]
    initial_point: Mapping[str, np.ndarray]
    latent_names: tuple[str, ...] = ()
    n_endog_rows: int | None = None

    def __call__(self, endog: Any) -> VARModelValues:
        """Add the VAR to the NumPyro model being run, with `endog` as its data.

        Args:
            endog: Endogenous series, shape `(T, n_vars)`: observed data, or an
                array the NumPyro model generated.

        Returns:
            The recorded quantities' values at the NumPyro model's current
            parameter values.

        Raises:
            ValueError: If the block was compiled with `n_endog_rows` and
                `endog` has another number of rows.
        """
        import numpyro
        import numpyro.distributions as dist

        n_rows = np.shape(endog)[0]
        if self.n_endog_rows is not None and n_rows != self.n_endog_rows:
            raise ValueError(
                f"endog has {n_rows} rows, but the block was compiled for {self.n_endog_rows} "
                f"(n_endog_rows): compile a block with n_endog_rows={n_rows} for this endog."
            )
        values = [
            numpyro.sample(site, dist.ImproperUniform(dist.constraints.real, (), self.site_shapes[site]))
            for site in self.numpyro_site_names.values()
        ]
        numpyro.factor(LOG_DENSITY_SITE, self.log_density(*values, endog)[0])
        recorded = dict(zip(self.recorded_names, self.recorded_quantities(*values, endog), strict=True))
        for name, value in recorded.items():
            numpyro.deterministic(name, value)
        return VARModelValues(**recorded, latent_names=self.latent_names)
