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

UNCONSTRAINED_SITE_PREFIX = "unconstrained/"
"""Prefix of the NumPyro site holding each PyMC value variable, on PyMC's unconstrained scale."""

LOG_DENSITY_SITE = "VAR_log_density"
"""Name of the `numpyro.factor` site that adds the VAR's joint log-density."""


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
    """

    intercept: Any
    B: Any
    L: Any


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
            included.
        recorded_quantities: JAX function from the same inputs to the
            recorded quantities, in `recorded_names` order.
    """

    numpyro_site_names: Mapping[str, str]
    site_shapes: Mapping[str, tuple[int, ...]]
    recorded_names: tuple[str, ...]
    log_density: Callable[..., Any]
    recorded_quantities: Callable[..., Any]

    def __call__(self, endog: Any) -> VARModelValues:
        """Add the VAR to the NumPyro model being run, with `endog` as its data.

        Args:
            endog: Endogenous series, shape `(T, n_vars)`: observed data, or an
                array the NumPyro model generated.

        Returns:
            The recorded quantities' values at the NumPyro model's current
            parameter values.
        """
        import numpyro
        import numpyro.distributions as dist

        values = [
            numpyro.sample(site, dist.ImproperUniform(dist.constraints.real, (), self.site_shapes[site]))
            for site in self.numpyro_site_names.values()
        ]
        numpyro.factor(LOG_DENSITY_SITE, self.log_density(*values, endog)[0])
        recorded = dict(zip(self.recorded_names, self.recorded_quantities(*values, endog), strict=True))
        for name, value in recorded.items():
            numpyro.deterministic(name, value)
        return VARModelValues(**recorded)
