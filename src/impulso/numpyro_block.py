"""A VAR compiled from its PyMC model into a block a NumPyro model can call.

`VAR.compile_for_numpyro` builds the VAR's PyMC model once and returns a
`CompiledNumPyroBlock`. The block holds two JAX functions compiled from that
model: the joint log-density, and the map to the recorded quantities. No
NumPyro code is written for the VAR itself (ADR-0016 keeps one model, defined
in PyMC).

JAX and NumPyro are imported only inside the methods that need them, so
importing this module needs neither.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from impulso._posterior import (
    CHOLESKY_FACTOR,
    COEFFICIENTS,
    DEGREES_OF_FREEDOM,
    EXOG_COEFFICIENTS,
    INTERCEPT,
    LATENT_PATHS,
)
from impulso._zero_gradient import has_zero_gradient

if TYPE_CHECKING:
    from impulso._arviz_compat import InferenceDataLike
    from impulso.spec import VARModelHandles

UNCONSTRAINED_SITE_PREFIX = "unconstrained/"
"""Prefix of the NumPyro site holding each PyMC value variable, on PyMC's unconstrained scale."""

LOG_DENSITY_SITE = "VAR_log_density"
"""Name of the `numpyro.factor` site that adds the VAR's joint log-density."""


def add_with_zero_gradient(
    log_density: Callable[..., Any], zero_gradient_term: Callable[..., Any]
) -> Callable[..., Any]:
    """Add `zero_gradient_term` to `log_density`, so that JAX passes the term no gradient.

    `zero_gradient_term` sees `jax.lax.stop_gradient` copies of the inputs, so
    JAX never differentiates it: its value counts and its gradient is zero.
    That is the gradient of a term that is constant wherever it is finite,
    such as the `latent_stationarity` Potential.

    Args:
        log_density: JAX function, as `pymc.sampling.jax.get_jaxified_graph`
            compiles it, from some inputs to a one-element sequence holding a
            log-density.
        zero_gradient_term: JAX function of the same form, from the same
            inputs, holding the term to add.

    Returns:
        JAX function from those inputs to a one-element sequence holding the
        sum of the two.
    """

    def total(*inputs: Any) -> list[Any]:
        import jax

        inputs_without_gradient = [jax.lax.stop_gradient(value) for value in inputs]
        return [log_density(*inputs)[0] + zero_gradient_term(*inputs_without_gradient)[0]]

    return total


def _default_dimension_names(name: str, n_dims: int) -> tuple[str, ...]:
    """PyMC's default dimension names for a variable that the model gives none.

    Args:
        name: Name of the variable.
        n_dims: Number of its dimensions, without the sample dimensions
            `chain` and `draw`.

    Returns:
        `{name}_dim_0`, `{name}_dim_1`, and so on, one per dimension.
    """
    return tuple(f"{name}_dim_{i}" for i in range(n_dims))


@dataclass(frozen=True, kw_only=True)
class VARModelValues:
    """Values, as JAX arrays, of the recorded quantities: the quantities a `CompiledNumPyroBlock` call records.

    The NumPyro counterpart of `VARModelHandles`: where those are PyMC
    variables, these are the values the block computed from the NumPyro
    model's current parameter values. A quantity the VAR does not have is
    `None` and is not recorded.

    Attributes:
        intercept: Per-equation intercept. Shape `(n_vars,)` when every
            equation has one (the default); shape `(n_included,)`, covering
            only the included equations in `endog_names` order, when
            `intercept_equations` named a strict subset; `None` when
            `intercept_equations` excluded every equation.
        B: VAR lag coefficients, shape `(n_vars, n_vars * n_lags)`, lag-major
            columns.
        B_exog: Exogenous coefficients, shape `(n_vars, n_exog)`; `None`
            without exogenous regressors.
        L: Lower-triangular Cholesky factor of the structural-shock scale
            matrix, shape `(n_vars, n_vars)`. Under Student-t errors this is
            the factor of the scale matrix, not of the covariance (ADR-0007).
        nu: Degrees of freedom of the Student-t errors, a scalar, whether
            inferred or fixed; `None` under Gaussian errors.
        latent: The generated latent paths, shape `(T, n_latent)` including
            the initial values; `None` without latent series. Its columns
            follow `latent_names`.
        latent_names: Names of the latent series, in the column order of
            `latent`; empty without latent series.
    """

    intercept: Any = None
    B: Any
    B_exog: Any = None
    L: Any
    nu: Any = None
    latent: Any = None
    latent_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledNumPyroBlock:
    """A VAR compiled once from its PyMC model, called inside a NumPyro model.

    Built by `VAR.compile_for_numpyro`, through `from_pymc_model`; see
    `VAR.compile_for_numpyro` for what a call adds to the NumPyro model and
    for the limitations.

    Attributes:
        numpyro_site_names: NumPyro sample-site name of each PyMC value
            variable, keyed by the value variable's name. A PyMC value
            variable is the unconstrained value of one free parameter of the
            PyMC model, and its site holds it on that scale.
        site_shapes: Shape of each site's value, keyed by NumPyro site name.
        recorded_names: Names under which a call records the recorded
            quantities with `numpyro.deterministic`, in the order of
            `compute_recorded_quantities`'s outputs.
        log_density: JAX function from the value of every site (in
            `numpyro_site_names` order) and `endog` to a one-element sequence
            holding the VAR's joint log-density, PyMC's transform corrections
            included. JAX passes every term that the PyMC model gives a
            zero-gradient override, such as the `latent_stationarity`
            Potential, no gradient (see `add_with_zero_gradient`).
        compute_recorded_quantities: JAX function from the same inputs to the
            recorded quantities, in `recorded_names` order.
        dimension_names: Dimension names of each recorded quantity, keyed by
            its recorded name, without the sample dimensions `chain` and
            `draw`. They are the PyMC model's own (`B` on `("var", "coeff")`;
            an intercept for a strict subset of equations on
            `("var_intercept",)`); a quantity the PyMC model gives no
            dimension names gets PyMC's default ones, `{name}_dim_0`,
            `{name}_dim_1`, and so on, and the scalar `nu` gets none.
        coordinate_labels: Labels along each dimension in
            `dimension_names` that the PyMC model labels, keyed by dimension
            name: the variable names on `"var"`, the lag-major `L{lag}.{name}`
            labels on `"coeff"`, the exogenous names on `"exog"`, and the
            equations that have an intercept on `"var_intercept"`. Together
            with `dimension_names` they are the PyMC model's `coords` and
            `dims` for the recorded quantities. `posterior_from_samples` uses
            both.
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
    compute_recorded_quantities: Callable[..., Any]
    dimension_names: Mapping[str, tuple[str, ...]]
    coordinate_labels: Mapping[str, tuple[str, ...]]
    initial_point: Mapping[str, np.ndarray]
    latent_names: tuple[str, ...] = ()
    n_endog_rows: int | None = None

    @classmethod
    def from_pymc_model(
        cls,
        model: Any,
        endog: Any,
        handles: "VARModelHandles",
        *,
        degrees_of_freedom: Any = None,
        n_endog_rows: int | None = None,
    ) -> "CompiledNumPyroBlock":
        """Compile a VAR's PyMC model, built by `VAR.build_in_model` with a symbolic `endog`, into a block.

        `VAR.compile_for_numpyro` builds the model and calls this constructor.

        Args:
            model: The `pymc.Model` that holds the VAR `build_in_model`
                registered, and nothing else.
            endog: The symbolic `endog` that `build_in_model` received. Both
                compiled functions take it after the value of every site.
            handles: The handles `build_in_model` returned.
            degrees_of_freedom: The PyMC variable `nu`, the degrees of freedom
                of the Student-t errors, under heavy-tailed errors; `None`
                under Gaussian errors.
            n_endog_rows: Number of rows of the `endog` every call of the
                block must receive, or `None` when a call accepts an `endog`
                of any length.

        Returns:
            The compiled block.
        """
        from pymc.sampling.jax import get_jaxified_graph

        inputs = [*model.value_vars, endog]
        # The log-density sums the priors, the Potentials (the likelihood
        # `obs` among them) and the transforms' correction terms. PyMC gives a
        # term declared with `declare_zero_gradient`, such as the
        # `latent_stationarity` Potential, a zero gradient through a gradient
        # override. PyTensor's JAX conversion drops the override, so the
        # declared terms are compiled on their own and added with a zero
        # gradient.
        terms = [*model.basic_RVs, *model.potentials]
        zero_gradient_terms = [term for term in terms if has_zero_gradient(term)]
        differentiated_terms = [term for term in terms if not has_zero_gradient(term)]
        log_density = get_jaxified_graph(inputs=inputs, outputs=[model.logp(vars=differentiated_terms)])
        if zero_gradient_terms:
            log_density = add_with_zero_gradient(
                log_density, get_jaxified_graph(inputs=inputs, outputs=[model.logp(vars=zero_gradient_terms)])
            )
        quantities = {
            INTERCEPT: handles.intercept,
            COEFFICIENTS: handles.B,
            EXOG_COEFFICIENTS: handles.B_exog,
            CHOLESKY_FACTOR: handles.L,
            DEGREES_OF_FREEDOM: degrees_of_freedom,
            LATENT_PATHS: handles.latent,
        }
        # A quantity the model does not have (no intercept, no exogenous
        # regressors, Gaussian errors, no latent series) is not recorded,
        # and the call returns `None` for it.
        recorded = {name: quantity for name, quantity in quantities.items() if quantity is not None}
        site_names = {value.name: f"{UNCONSTRAINED_SITE_PREFIX}{value.name}" for value in model.value_vars}
        # On the unconstrained scale, keyed by value-variable name. It holds
        # the stationary start `B` gets with latent series (ADR-0016).
        initial_point = model.initial_point()
        # PyMC's own dimension names, or its defaults for a variable it gives
        # none: `L_dim_0` and `L_dim_1` for `L`, and none for the scalar `nu`.
        dimension_names = {
            name: tuple(model.named_vars_to_dims.get(name) or _default_dimension_names(name, variable.ndim))
            for name, variable in recorded.items()
        }
        labelled = {dim for dims in dimension_names.values() for dim in dims if model.coords.get(dim) is not None}
        return cls(
            numpyro_site_names=site_names,
            # PyMC gives some value variables no static shape (`B`'s among
            # them), so the shapes come from evaluating the initial point.
            site_shapes={site_names[name]: np.shape(value) for name, value in initial_point.items()},
            recorded_names=tuple(recorded),
            log_density=log_density,
            # Rewritten in terms of the value variables, undoing their
            # transforms, so the function reads its inputs instead of drawing.
            compute_recorded_quantities=get_jaxified_graph(
                inputs=inputs, outputs=model.replace_rvs_by_values(list(recorded.values()))
            ),
            dimension_names=dimension_names,
            coordinate_labels={dim: tuple(model.coords[dim]) for dim in labelled},
            initial_point={site_names[name]: value for name, value in initial_point.items()},
            latent_names=handles.latent_names,
            n_endog_rows=n_endog_rows,
        )

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
        recorded = dict(zip(self.recorded_names, self.compute_recorded_quantities(*values, endog), strict=True))
        for name, value in recorded.items():
            numpyro.deterministic(name, value)
        return VARModelValues(**recorded, latent_names=self.latent_names)

    def posterior_from_samples(self, samples: Mapping[str, Any]) -> "InferenceDataLike":
        """Turn NumPyro samples of a model that calls this block into a posterior for `FittedVAR.from_posterior`.

        NumPyro samples carry no dimension names, and the posterior schema
        checks `B`, `intercept` and `B_exog` by dimension name. This method
        names the sample axes `chain` and `draw` and gives each recorded
        quantity its `dimension_names` and `coordinate_labels`. The example in
        `VAR.compile_for_numpyro` shows the whole sequence, from compiling to
        impulse responses.

        The posterior holds:

        - each recorded quantity, under its posterior-schema name.
          The intercept has an entry for every equation, on `var`: an
          equation that `intercept_equations` excluded gets zero, as in the
          PyMC model's mean;
        - every other variable in `samples`, such as the NumPyro model's own
          sample and deterministic sites, under its own name with each `/`
          replaced by `.`, and with PyMC's default dimension names built from
          that name: `{name}_dim_0`, `{name}_dim_1`, and so on. ArviZ 1
          refuses `/` in a variable name, and
          `numpyro.handlers.scope(prefix="user")` names a site `user/scale`.
          The posterior holds that site as `user.scale` under ArviZ 0 as
          well, so the names are the same under both.
          `FittedVAR.from_posterior` reads none of these variables.

        It leaves out the block's own sites: the sample sites in
        `numpyro_site_names`, which hold the VAR's parameters on PyMC's
        unconstrained scale and whose names contain `/`, which ArviZ 1
        refuses in a variable name; and the log-density site.

        Two arguments of `FittedVAR.from_posterior` are the caller's to get
        right:

        - `data`, a `VARData` (ADR-0016): the series given to the block as
          `endog`, with the `endog_names`, `exog` and `exog_names` the block
          was compiled with. With latent series it also needs a plug-in
          column for each latent series, ahead of the observed ones, such as
          the posterior mean of `latent`.
        - `error_dist=spec.resolved_error_dist`, as in the example in
          `VAR.compile_for_numpyro`. Under Gaussian errors it makes no
          difference. Under Student-t errors, leaving it out makes
          `FittedVAR.from_posterior` accept the posterior as Gaussian, so
          forecasts draw Gaussian innovations.

        Args:
            samples: NumPyro samples keyed by site name, each with a chain
                axis and a draw axis first:
                `mcmc.get_samples(group_by_chain=True)`.

        Returns:
            A container with one group, `posterior`: an `arviz.InferenceData`
            under ArviZ 0 and an `xarray.DataTree` under ArviZ 1, the
            container `FittedVAR.from_posterior` takes on each.

        Raises:
            ValueError: If `samples` lacks a recorded quantity (as when the
                block was called inside a `numpyro.handlers.scope`), or a
                recorded quantity lacks its chain or draw axis (as from
                `mcmc.get_samples()` without `group_by_chain=True`).
            ValueError: If two variables in `samples` other than the block's
                own sites would get the same name in the posterior once each
                `/` is replaced by `.`, such as `user/scale` and `user.scale`.
        """
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
        block_sites = {*self.numpyro_site_names.values(), LOG_DENSITY_SITE}
        variables = {}
        site_of_posterior_name: dict[str, str] = {}
        for name, value in samples.items():
            if name in block_sites:
                continue
            posterior_name = name.replace("/", ".")
            if posterior_name in site_of_posterior_name:
                first, second = sorted((site_of_posterior_name[posterior_name], name))
                raise ValueError(
                    f"The samples hold {first!r} and {second!r}, which would both be {posterior_name!r} in the "
                    "posterior, where each `/` in a site's name becomes `.`. Rename one of the two sites."
                )
            site_of_posterior_name[posterior_name] = name
            array = np.asarray(value)
            dims = self.dimension_names.get(name, _default_dimension_names(posterior_name, array.ndim - 2))
            variables[posterior_name] = (("chain", "draw", *dims), array)
        variables[INTERCEPT] = (("chain", "draw", "var"), self._intercept_of_every_equation(samples))
        used_dims = {dim for dims, _ in variables.values() for dim in dims}
        coords = {dim: list(labels) for dim, labels in self.coordinate_labels.items() if dim in used_dims}
        return make_idata(posterior=xr.Dataset(variables, coords=coords))

    def _intercept_of_every_equation(self, samples: Mapping[str, Any]) -> Any:
        """The intercept samples of every equation, as the posterior schema needs.

        An equation that `intercept_equations` excluded has no recorded
        intercept, and its intercept is zero, as in the PyMC model's mean.

        Args:
            samples: NumPyro samples keyed by site name, as
                `posterior_from_samples` takes them, with `B` and, unless
                `intercept_equations` excluded every equation, `intercept`.

        Returns:
            The intercept samples, shape `(chain, draw, n_vars)`, with the
            equations in `endog_names` order.
        """
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
