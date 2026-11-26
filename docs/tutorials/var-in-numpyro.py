# ---
# jupyter:
#   jupytext:
#     default_lexer: python
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # A VAR Inside a NumPyro Model

# %% tags=["remove-cell"]
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# %% tags=["remove-cell"]
import os

# Smoke-render flag: set IMPULSO_DOCS_CI=1 to shrink MCMC for fast CI builds.
ci = os.environ.get("IMPULSO_DOCS_CI") == "1"

# %% [markdown]
# Impulso defines its VAR once, in PyMC. This tutorial is for a reader who builds models in
# [NumPyro](https://num.pyro.ai/) and wants a Bayesian VAR as one part of a larger NumPyro
# model. `VAR.compile_for_numpyro` builds the VAR's PyMC model once, through the same
# `VAR.build_in_model` that embeds a VAR in a PyMC model (the **embedded path**), and compiles
# that model's log-density into JAX functions. The result is a **compiled NumPyro block**: an
# object your NumPyro model function calls with the VAR's endogenous series. No NumPyro code
# is written for the VAR itself, so the priors and the likelihood are the ones `VAR.fit` uses,
# and the route runs one way, from PyMC to NumPyro.
#
# The tutorial embeds a VAR in a NumPyro model, samples the model with NumPyro's NUTS, turns
# the NumPyro samples into a `FittedVAR` through `FittedVAR.from_posterior`, and plots
# impulse responses.
#
# The compile step needs JAX and NumPyro, which the core install leaves out. Install the
# `numpyro` extra:
#
# ```
# pip install "impulso[numpyro]"
# ```
#
# or, with uv:
#
# ```
# uv add "impulso[numpyro]"
# ```
#
# Calling `VAR.compile_for_numpyro` without them raises an `ImportError` that names the
# missing packages.

# %%
import time

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_value
from qc_core import plotting

from impulso import VAR, VARData, ar1_residual_sd
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.priors import MinnesotaPrior

plotting.use_ledger_style()

# %% [markdown]
# ## Simulate two series with a level shift
#
# The NumPyro model in this tutorial explains two quarterly series, output growth and
# inflation, both in percent. Each series is the sum of two parts:
#
# - a **shift-free series** that follows a VAR(1) with an intercept, and
# - a **level shift** of unknown size that starts at a known quarter, the first quarter of
#   2005, and lasts to the end of the sample.
#
# $$
# y_t = \delta \, s_t + v_t, \qquad v_t = c + A v_{t-1} + \varepsilon_t,
# \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma),
# $$
#
# where $s_t$ is 0 before the break quarter and 1 from it onwards, and $\delta$ holds the two
# level shifts in percentage points. The NumPyro model infers $\delta$, and the VAR describes
# the shift-free series $v_t = y_t - \delta \, s_t$. The VAR therefore receives an array the
# NumPyro model computes from its own parameter, not observed data.
#
# We simulate the data so the true values are known. Before the break, mean output growth is
# 2.5 % and mean inflation is 2.0 %. At the break, output growth falls by 1.0 percentage point
# and inflation rises by 1.5 percentage points.
#
# | Equation | Own lag | Cross lag | Innovation standard deviation |
# |----------|---------|-----------|-------------------------------|
# | **Output growth** | 0.5 | −0.2 on last quarter's inflation | 0.6 percentage points |
# | **Inflation** | 0.6 | 0.3 on last quarter's output growth | 0.3 percentage points |
#
# The two innovations have correlation 0.3.

# %%
rng = np.random.default_rng(7)
n_quarters = 200
names = ["output_growth", "inflation"]
index = pd.date_range("1975-01-01", periods=n_quarters, freq="QS")
after_break = (index >= "2005-01-01").astype(float)  # s_t

A_true = np.array([
    [0.5, -0.2],  # output growth on its own lag and on lagged inflation
    [0.3, 0.6],  # inflation on lagged output growth and on its own lag
])
sd_true = np.array([0.6, 0.3])
corr_true = np.array([[1.0, 0.3], [0.3, 1.0]])
Sigma_true = np.outer(sd_true, sd_true) * corr_true
mean_before_true = np.array([2.5, 2.0])
shift_true = np.array([-1.0, 1.5])

c_true = (np.eye(2) - A_true) @ mean_before_true
chol_true = np.linalg.cholesky(Sigma_true)
v = mean_before_true.copy()
shift_free_true = np.empty((n_quarters, 2))
for t in range(-100, n_quarters):  # the first 100 quarters are burn-in
    v = c_true + A_true @ v + chol_true @ rng.standard_normal(2)
    if t >= 0:
        shift_free_true[t] = v
observed = shift_free_true + after_break[:, None] * shift_true

# %% mystnb={"image": {"alt": "Simulated quarterly output growth and inflation from 1975 to 2024; output growth steps down and inflation steps up at the first quarter of 2005."}}
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, name, column in zip(axes, ["Output growth (%)", "Inflation (%)"], observed.T, strict=True):
    ax.plot(index, column, color=plotting.COLORS.body, linewidth=0.9)
    ax.axvline(pd.Timestamp("2005-01-01"), color=plotting.COLORS.oxblood, linewidth=1.2, linestyle="--")
    ax.set_ylabel(name)
plotting.serif_title("Simulated series; the dashed line marks the level shift", axes[0])
fig.tight_layout()

# %% [markdown]
# ## Compile the VAR once
#
# `VAR.compile_for_numpyro` takes the arguments of `VAR.build_in_model` except `endog`, the
# endogenous series, which your NumPyro model passes on each call. Because the series is not
# known when the VAR is compiled, three things differ from `VAR.fit`:
#
# - `lags` must be an integer, because lag selection needs the data.
# - `endog_scales` is required. It is the per-series scale that the Minnesota prior uses to
#   set the prior standard deviation of each cross-lag coefficient. `VAR.fit` computes it
#   from the data with `ar1_residual_sd`, the residual standard deviation of an AR(1) fit to
#   each series; here we compute it ourselves from the observed series.
# - `exog`, the exogenous regressors, is fixed here, because the prior on their coefficients
#   scales with each regressor's sample standard deviation. This VAR has none.
#
# The method builds the PyMC model and compiles it to JAX once. Your NumPyro model can then
# call the block as often as NumPyro traces it, without rebuilding anything.
#
# The `VAR` specification is the one `VAR.fit` takes, prior included. The Minnesota prior's
# default `tightness` of 0.1 would pull the cross-lag coefficients of this simulation most of
# the way to zero (see [The Minnesota Prior, From Scratch](minnesota-prior.py)), so we loosen
# it to 0.5.

# %%
spec = VAR(lags=1, prior=MinnesotaPrior(tightness=0.5))
block = spec.compile_for_numpyro(exog=None, n_lags=1, endog_names=names, endog_scales=ar1_residual_sd(observed))
block.numpyro_site_names

# %% [markdown]
# The block declares one NumPyro sample site for each free variable of the PyMC model. Each
# site holds its variable on PyMC's unconstrained scale: `sigma_sd_log__`, for example, is the
# logarithm of the innovation standard deviations. The sites have no NumPyro prior of their
# own; the compiled log-density supplies the VAR's priors, its likelihood and PyMC's
# corrections for the change of scale.
#
# ## Write the NumPyro model
#
# The NumPyro model owns the level shift. It draws $\delta$, subtracts $\delta \, s_t$ from the
# observed series and passes the shift-free series to the block. Each call of the block:
#
# 1. declares the VAR's sample sites,
# 2. adds the VAR's log-density at the shift-free series to the model with `numpyro.factor`,
#    and
# 3. records the lag coefficients `B`, the `intercept` and the Cholesky factor `L` of the
#    innovation covariance with `numpyro.deterministic`, under the names `VAR.fit` gives them
#    (the **posterior schema**).
#
# It returns those values, so the rest of a NumPyro model can use them.

# %%
def model(observed, after_break):
    shift = numpyro.sample("shift", dist.Normal(0.0, 2.0).expand([2]).to_event(1))
    shift_free = observed - after_break[:, None] * shift
    block(shift_free)


# %% [markdown]
# ## Sample with NumPyro's NUTS
#
# NumPyro sets initial values on the sampler, not in the model. `block.initial_point` holds
# the initial point PyMC would use, keyed by site name, and `init_to_value` starts the
# sampler there. The `shift` site is not in it, so `init_to_value` falls back to NumPyro's
# default for it: a uniform draw between −2 and 2.

# %%
if ci:
    num_warmup, num_samples, num_chains = 200, 50, 2
else:
    num_warmup, num_samples, num_chains = 1000, 1000, 4

kernel = NUTS(model, init_strategy=init_to_value(values=block.initial_point))
mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    chain_method="sequential",
    progress_bar=False,
)
start = time.perf_counter()
mcmc.run(jax.random.key(0), observed, after_break)
print(f"Sampling took {time.perf_counter() - start:.0f} seconds.")
print(f"Divergent transitions: {int(mcmc.get_extra_fields()['diverging'].sum())}")

# %% [markdown]
# ## Hand the posterior to `FittedVAR`
#
# NumPyro's samples carry no dimension names, and `FittedVAR.from_posterior` checks `B` and
# `intercept` by dimension name. `block.posterior_from_samples` names the axes `chain` and
# `draw`, gives each quantity the block records the dimension names and labels of the PyMC
# model, and keeps your own sites, here `shift`, under their own names. Pass it the samples
# grouped by chain.

# %%
samples = mcmc.get_samples(group_by_chain=True)
posterior = block.posterior_from_samples(samples)
az.summary(posterior, var_names=["shift", "B", "intercept"])

# %% [markdown]
# The posterior of `shift` should sit near the true level shifts of −1.0 and 1.5 percentage
# points, and `B` near the true coefficients in `A_true`. In smoke-render builds of these docs
# the sampler runs only a few draws, so the numbers above are not representative there.
#
# `FittedVAR.from_posterior` also needs a `VARData`: the series the block received, with the
# `endog_names` the block was compiled with. Here the block received a different shift-free
# series at every draw, because the series depends on `shift`. We pass the shift-free series
# at the posterior mean of `shift`. Impulse responses do not read the series' values;
# forecasts and historical decompositions do, and they then describe the shift-free series,
# not the observed series.
#
# Pass `error_dist=spec.resolved_error_dist` as well. It makes no difference for this VAR's
# Gaussian errors, but under Student-t errors leaving it out makes `FittedVAR.from_posterior`
# treat the posterior as Gaussian.

# %%
shift_mean = np.asarray(samples["shift"]).mean(axis=(0, 1))
data = VARData(endog=observed - after_break[:, None] * shift_mean, endog_names=names, index=index)
fitted = FittedVAR.from_posterior(posterior, data, n_lags=1, error_dist=spec.resolved_error_dist)
fitted

# %% [markdown]
# ## Impulse responses
#
# From here every tool of `FittedVAR` works as it does after `VAR.fit`. We identify the
# structural shocks with a Cholesky ordering, output growth first, and plot the response of
# each series to a one-standard-deviation shock in each series over 12 quarters. The shaded
# band is the 89 % highest-density interval. The dashed line is the true impulse response
# of the simulated VAR, $A^h P$ at horizon $h$, where $P$ is the Cholesky factor of the true
# innovation covariance $\Sigma$.

# %% mystnb={"image": {"alt": "Two-by-two grid of impulse responses over 12 quarters for output growth and inflation, with posterior median, 89% band and the true response as a dashed line."}}
horizon = 12
irf = fitted.set_identification_strategy(Cholesky(ordering=names)).impulse_response(horizon=horizon)
fig = irf.plot()

true_irf = np.stack([np.linalg.matrix_power(A_true, h) @ chol_true for h in range(horizon + 1)])
for i in range(2):  # response
    for j in range(2):  # shock
        fig.axes[2 * i + j].plot(
            range(horizon + 1), true_irf[:, i, j], color=plotting.COLORS.ink, linestyle="--", linewidth=1.0
        )

# %% [markdown]
# ## Latent endogenous series
#
# A **latent endogenous series** is an endogenous series with no data: the VAR generates it
# from its own equation. The compiled NumPyro block supports latent series as
# `VAR.build_in_model` does, through `latent_names` and `latent_init_sigma`. Four things change
# from this tutorial:
#
# - The latent series come first in `endog_names`, and the series each call receives holds
#   only the observed columns that follow them.
# - Pass `n_endog_rows`, the number of rows of the series each call receives. JAX needs the
#   length of the generated latent paths when it compiles them, and every call must then pass
#   that many rows.
# - Start the sampler with `init_to_value(values=block.initial_point)`. When a latent series'
#   own-lag prior mean lies outside the stationary region, PyMC's initial point moves the lag
#   coefficients inside it. For a latent series that is a stationary deviation, set its
#   own-lag prior mean to 0 with `MinnesotaPrior(own_lag_mean=...)`.
# - The `VARData` for `FittedVAR.from_posterior` needs a column for each latent series, ahead
#   of the observed columns, such as the posterior mean of the recorded `latent` paths.
#
# The block keeps the latent block of the lag coefficients stationary as PyMC does. With one
# latent series and one lag, the latent series' own lag has a Normal prior truncated to
# (−1, 1). In every other case the `latent_stationarity` Potential makes the log-density minus
# infinity outside the stationary region. The Potential uses a non-symmetric
# eigendecomposition, which JAX computes on a CPU and on a GPU (on a GPU it calls LAPACK on the
# host CPU by default), and does not compute on a TPU.
#
# ## What the compiled NumPyro block cannot do
#
# - **Prior predictive sampling of the VAR.** The VAR's sample sites have no NumPyro
#   distribution to draw from, so `numpyro.infer.Predictive` cannot draw the VAR's parameters
#   from the prior.
# - **`numpyro.handlers.do` on the VAR's parameters.** `do` ignores recorded quantities such as
#   `B`, and it cannot remove a parameter's prior from the compiled log-density.
# - **Exogenous regressors that change between calls.** `exog` is fixed at compile time.
# - **The refusals of the embedded path.** As with `VAR.build_in_model` on a symbolic series,
#   the VAR needs an integer `lags`, constant volatility, and Gaussian errors when it has latent
#   series.
#
# ## Summary
#
# 1. **Compile** the VAR once with `VAR.compile_for_numpyro`, outside the NumPyro model.
# 2. **Call** the compiled NumPyro block inside the NumPyro model with the endogenous series,
#    observed or computed by the model.
# 3. **Sample** with NumPyro's NUTS, starting at `block.initial_point`.
# 4. **Convert** the samples with `block.posterior_from_samples` and wrap them with
#    `FittedVAR.from_posterior`, passing a `VARData` and `error_dist=spec.resolved_error_dist`.
# 5. **Analyse** with the usual tools: impulse responses, forecasts, forecast error variance
#    decompositions and historical decompositions.
#
# ADR-0018 records why the VAR reaches NumPyro by compilation from PyMC rather than through a
# second implementation written in NumPyro.
