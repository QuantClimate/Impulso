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
from typing import TYPE_CHECKING, Any

from impulso._posterior import COEFFICIENTS, INTERCEPT

if TYPE_CHECKING:
    from impulso._arviz_compat import InferenceDataLike

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
        dimension_names: Dimension names of each recorded quantity, keyed by
            its recorded name, without the sample dimensions `chain` and
            `draw`. They are the PyMC model's own (`B` on `("var", "coeff")`);
            a quantity the PyMC model gives no dimension names gets PyMC's
            default ones, `{name}_dim_0`, `{name}_dim_1`, and so on.
        coordinate_labels: Labels along each dimension in
            `dimension_names` that the PyMC model labels, keyed by dimension
            name: the variable names on `"var"`, the lag-major `L{lag}.{name}`
            labels on `"coeff"`, and so on. Together with `dimension_names`
            they are the `coords` and `dims` of the PyMC model.
    """

    numpyro_site_names: Mapping[str, str]
    site_shapes: Mapping[str, tuple[int, ...]]
    recorded_names: tuple[str, ...]
    log_density: Callable[..., Any]
    recorded_quantities: Callable[..., Any]
    dimension_names: Mapping[str, tuple[str, ...]]
    coordinate_labels: Mapping[str, tuple[str, ...]]

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

    def posterior_from_samples(self, samples: Mapping[str, Any]) -> "InferenceDataLike":
        """Turn NumPyro samples of a model that calls this block into a posterior for `FittedVAR.from_posterior`.

        NumPyro samples carry no dimension names, and the posterior schema
        checks `B`, `intercept` and `B_exog` by dimension name. This method
        names the sample axes `chain` and `draw` and gives each quantity the
        block records its `dimension_names` and `coordinate_labels`. The
        whole sequence, from compiling to impulse responses:

        ```python
        spec = VAR(lags=2)
        block = spec.compile_for_numpyro(
            exog=None, n_lags=2, endog_names=data.endog_names, endog_scales=impulso.ar1_residual_sd(data.endog)
        )

        def model(endog):
            values = block(endog)  # values.B, values.intercept, values.L
            ...  # the rest of the NumPyro model

        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model), num_warmup=500, num_samples=500, num_chains=2, chain_method="sequential"
        )
        mcmc.run(jax.random.key(0), data.endog)

        posterior = block.posterior_from_samples(mcmc.get_samples(group_by_chain=True))
        fitted = FittedVAR.from_posterior(posterior, data, n_lags=2, error_dist=spec.resolved_error_dist)
        irf = fitted.set_identification_strategy(Cholesky(ordering=data.endog_names)).impulse_response()
        ```

        The posterior holds:

        - each quantity the block records, under its posterior-schema name.
          The intercept has an entry for every equation, on `var`: an
          equation that `intercept_equations` excluded gets zero, as in the
          PyMC model's mean;
        - every other variable in `samples`, such as the NumPyro model's own
          sample and deterministic sites, under its own name, with PyMC's
          default dimension names `{name}_dim_0`, `{name}_dim_1`, and so on.
          `FittedVAR.from_posterior` reads none of them.

        It leaves out the block's own sample sites (`numpyro_site_names`):
        they hold the VAR's parameters on PyMC's unconstrained scale, which
        the posterior schema does not read, and their names contain `/`,
        which ArviZ 1 refuses in a variable name.

        `FittedVAR.from_posterior` needs a `VARData` (ADR-0016): pass the
        series given to the block as `endog`, with the `endog_names`, `exog`
        and `exog_names` the block was compiled with. Under Student-t errors,
        also pass `error_dist=spec.resolved_error_dist`; without it
        `FittedVAR.from_posterior` treats the errors as Gaussian.

        Args:
            samples: NumPyro samples keyed by site name, each with a chain
                axis and a draw axis first:
                `mcmc.get_samples(group_by_chain=True)`.

        Returns:
            A container with one group, `posterior`: an `arviz.InferenceData`
            under ArviZ 0 and an `xarray.DataTree` under ArviZ 1, the
            container `FittedVAR.from_posterior` takes on each.

        Raises:
            ValueError: If `samples` lacks a quantity the block records (as
                when the block was called inside a `numpyro.handlers.scope`),
                or a recorded quantity lacks its chain or draw axis (as from
                `mcmc.get_samples()` without `group_by_chain=True`).
        """
        import numpy as np
        import xarray as xr

        from impulso._arviz_compat import make_idata

        for name, dims in self.dimension_names.items():
            if name not in samples:
                raise ValueError(
                    f"The samples hold no {name!r}, which every call of this block records. Pass "
                    "`mcmc.get_samples(group_by_chain=True)` of a NumPyro model that calls this block outside any "
                    "`numpyro.handlers.scope`."
                )
            if np.ndim(samples[name]) != 2 + len(dims):
                raise ValueError(
                    f"The samples of {name!r} have shape {np.shape(samples[name])}, but they need "
                    f"{2 + len(dims)} axes: {', '.join(('chain', 'draw', *dims))}. Pass "
                    "`mcmc.get_samples(group_by_chain=True)`, which keeps the chains apart."
                )
        block_sites = set(self.numpyro_site_names.values())
        variables = {}
        for name, value in samples.items():
            if name in block_sites:
                continue
            array = np.asarray(value)
            dims = self.dimension_names.get(name, tuple(f"{name}_dim_{i}" for i in range(array.ndim - 2)))
            variables[name] = (("chain", "draw", *dims), array)
        variables[INTERCEPT] = (("chain", "draw", "var"), self._intercept_of_every_equation(samples))
        used_dims = {dim for dims, _ in variables.values() for dim in dims}
        coords = {dim: list(labels) for dim, labels in self.coordinate_labels.items() if dim in used_dims}
        return make_idata(posterior=xr.Dataset(variables, coords=coords))

    def _intercept_of_every_equation(self, samples: Mapping[str, Any]) -> Any:
        """The intercept samples of every equation, shape `(chain, draw, n_vars)`, as the posterior schema needs.

        An equation that `intercept_equations` excluded has no recorded
        intercept, and its intercept is zero, as in the PyMC model's mean.
        """
        import numpy as np

        variable_names = self.coordinate_labels["var"]
        recorded_dims = self.dimension_names.get(INTERCEPT)
        if recorded_dims == ("var",):
            return np.asarray(samples[INTERCEPT])
        full = np.zeros((*np.shape(samples[COEFFICIENTS])[:2], len(variable_names)))
        if recorded_dims is not None:
            (equations_dim,) = recorded_dims
            positions = [variable_names.index(name) for name in self.coordinate_labels[equations_dim]]
            full[..., positions] = np.asarray(samples[INTERCEPT])
        return full
