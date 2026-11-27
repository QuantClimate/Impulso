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
# # A Factor-Augmented VAR for Monetary Policy

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
# A VAR in a few series reports the effect of a monetary policy shock on those few series.
# Two problems follow:
#
# - It says nothing about the series it leaves out, so a VAR in industrial production cannot
#   say how payroll employment responds to the monetary policy shock.
# - Its monetary policy shock is the part of the federal funds rate that the series in the
#   VAR cannot predict. If the Federal Reserve reacts to information outside the VAR, that
#   reaction becomes part of the monetary policy shock. The best-known evidence of this is the
#   **price puzzle**: prices rise after a monetary policy shock that raises the federal funds
#   rate. [Identification in structural VARs](monetary-policy.py) shows it in a
#   three-variable VAR.
#
# A **factor-augmented VAR** {cite:p}`bernankeBoivinEliasz2005` summarises many indicators
# by a few unobserved common factors and puts the factors in the VAR next to the federal funds
# rate. Each indicator then has a response to the monetary policy shock, computed from its
# loading on the factors.
#
# This tutorial estimates a small factor-augmented VAR in one step, in one NumPyro model. The
# NumPyro model builds one factor from four indicators. An Impulso VAR, compiled with
# `VAR.compile_for_numpyro`, holds the dynamics of the factor, consumer price inflation and the
# federal funds rate. [A VAR Inside a NumPyro Model](var-in-numpyro.py) explains the compiled
# NumPyro block; this tutorial uses it without repeating that explanation. The example samples
# reliably, and it does not remove the price puzzle; a later section gives a possible reason.

# %%
import time
from typing import NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from IPython.display import Markdown
from numpyro.infer import MCMC, NUTS, init_to_value
from qc_core import plotting

from impulso import VAR, VARData, ar1_residual_sd
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.priors import MinnesotaPrior

plotting.use_ledger_style()

# %% [markdown]
# Four helpers format the results: `numbered_table` turns a table into a numbered table that
# the text can cite; `posterior_quantiles` and `median_and_band_text` summarise draws by
# their posterior median and **89 % band**, the interval between the 5.5 % and 94.5 %
# posterior quantiles; `plot_median_and_band` draws both.

# %%
quantile_levels = [0.055, 0.5, 0.945]


def numbered_table(frame, name, caption):
    """A MyST table with a number, a caption and a name for cross-references, from a DataFrame."""
    header = "| " + " | ".join([frame.index.name or "", *map(str, frame.columns)]) + " |"
    rule = "|---" * (len(frame.columns) + 1) + "|"
    rows = ["| " + " | ".join(map(str, [index, *values])) + " |" for index, values in frame.iterrows()]
    return Markdown("\n".join([f"```{{table}} {caption}", f":name: {name}", "", header, rule, *rows, "```"]))


def posterior_quantiles(draws):
    """The 5.5 %, 50 % and 94.5 % posterior quantiles of `draws` over its first two axes, chain and draw."""
    return np.quantile(draws, quantile_levels, axis=(0, 1))


def median_and_band_text(draws):
    """Posterior median and 89 % band of `draws`, one text entry per entry of its last axis."""
    low, median, high = posterior_quantiles(draws)
    return [f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for lo, m, hi in zip(low, median, high, strict=True)]


def plot_median_and_band(ax, x, draws, label=None):
    """Plot the posterior median of `draws` against `x`, its 89 % band and a line at 0."""
    low, median, high = posterior_quantiles(draws)
    ax.fill_between(x, low, high, color=plotting.COLORS.oxblood, alpha=0.25, linewidth=0)
    ax.plot(x, median, color=plotting.COLORS.oxblood, linewidth=1.2, label=label)
    ax.axhline(0.0, color=plotting.COLORS.hairline, linewidth=0.8)


# %% [markdown]
# ## Data
#
# The series come from FRED-MD {cite:p}`mccrackenNg2016`, the monthly database for
# macroeconomic research that the Federal Reserve Bank of St. Louis publishes at
# <https://www.stlouisfed.org/research/economists/mccracken/fred-databases>. We use the
# vintage of October 2026 (file `2026-10-md.csv`).
#
# The four **indicators** are payroll employment, industrial production, real personal income
# less transfers, and real manufacturing and trade sales. They are the four components of the
# composite index of coincident indicators, which the Bureau of Economic Analysis of the US
# Department of Commerce published until 1995 and the Conference Board has published since
# {cite:p}`conferenceBoard2001`. {cite:t}`stockWatson1989` modelled the same four series with
# one common factor, except that their model used employee-hours in nonagricultural
# establishments in place of payroll employment. The VAR adds consumer price inflation and
# the effective federal funds rate. Every series is published by the Bureau of Labor
# Statistics, the Bureau of Economic Analysis, the Federal Reserve Board or the Federal Reserve
# Bank of St. Louis; the `source` column of {numref}`tbl-series` gives each series' source.
#
# Each indicator and the consumer price index enter as $100 (\log x_t - \log x_{t-1})$,
# FRED-MD's transformation code 5. This is the change in the logarithm from the previous
# month, times 100, which approximates the **percent change from the previous month**; we use
# that name for it throughout. **Consumer price inflation** is the percent change from the
# previous month of the consumer price index. FRED-MD gives the consumer price index code 6,
# the change in consumer price inflation; we use code 5, because consumer price inflation is
# stationary over the sample and its change is mostly noise. The federal funds rate enters in
# percent, untransformed, as it does in Bernanke, Boivin and Eliasz's VAR; FRED-MD's code 2
# would take its change from the previous month. The `unit` column of {numref}`tbl-series`
# gives each series' unit after the transformation.
#
# The sample runs from January 1984 to December 2007: after the disinflation that the Federal
# Reserve under Paul Volcker completed in the early 1980s, and before the federal funds rate
# reached its lower bound near 0 in December 2008, after which the rate could no longer be cut
# and stopped measuring the stance of monetary policy. `scripts/fetch_fred_md.py` downloads
# the vintage, transforms the series and writes the two CSV files read below.

# %% mystnb={"markdown_format": "myst"}
panel = pd.read_csv("data/fred_md_1984_2007.csv", index_col="date", parse_dates=True)
series_metadata = pd.read_csv("data/fred_md_1984_2007_series.csv", index_col="name")
numbered_table(
    series_metadata[["fred_md_mnemonic", "description", "source", "fred_md_transformation_code", "transformation_code", "unit"]],
    "tbl-series",
    "The six series, from FRED-MD vintage 2026-10.",
)

# %% [markdown]
# We standardise each indicator: we subtract its sample mean and divide by its sample
# standard deviation, so that the loadings are on a common scale. We keep the standard
# deviations to undo the standardisation for the impulse responses. We **demean** consumer
# price inflation and the federal funds rate: we subtract each one's sample mean. Payroll
# employment comes first among the indicators, because it is the normalising indicator of the
# model below.

# %%
indicator_titles = {
    "payroll_employment": "Payroll employment",
    "industrial_production": "Industrial production",
    "real_personal_income_less_transfers": "Real personal income less transfers",
    "real_manufacturing_trade_sales": "Real manufacturing and trade sales",
}
indicator_names = list(indicator_titles)
indicators = panel[indicator_names]
indicator_sd = indicators.std()
standardised = ((indicators - indicators.mean()) / indicator_sd).to_numpy()

price_and_rate_names = ["consumer_price_inflation", "federal_funds_rate"]
price_and_rate_mean = panel[price_and_rate_names].mean()
demeaned_price_and_rate = (panel[price_and_rate_names] - price_and_rate_mean).to_numpy()
n_months, n_indicators = standardised.shape
print(f"{n_months} months, {n_indicators} indicators")
print(
    f"Mean consumer price inflation: {price_and_rate_mean['consumer_price_inflation']:.2f}"
    " (percent change from the previous month)"
)
print(f"Mean federal funds rate: {price_and_rate_mean['federal_funds_rate']:.2f} percent")

# %% [markdown]
# ## The model
#
# With $X_{it}$ the standardised indicator $i$ in month $t$ and $F_t$ the factor, the
# **observation equation** is
#
# $$
# X_{it} = \lambda_i F_t + e_{it}, \qquad
# e_{it} \sim \mathcal{N}(0, \sigma_i^2) \text{ independent across } i \text{ and } t,
# $$ (eq-factor-augmented-observation)
#
# where $\lambda_i$ is indicator $i$'s **loading** on the factor, $e_{it}$ its
# **idiosyncratic error** and $\sigma_i$ its **idiosyncratic standard deviation**.
# {cite:t}`bernankeBoivinEliasz2005` also let each indicator load on the federal funds rate in
# the same month. We leave that term out. Bernanke, Boivin and Eliasz class real-activity
# series such as these four as slow-moving, which they define as "largely predetermined as of
# the current period", so we assume that the four indicators do not respond to the federal
# funds rate within the month. The Cholesky ordering below makes the same assumption.
#
# The **transition equation** is a VAR with 13 lags, the number Bernanke, Boivin and Eliasz
# use, in the factor, consumer price inflation $\pi_t$ and the federal funds rate $R_t$:
#
# $$
# y_t = \sum_{\ell=1}^{13} A_\ell \, y_{t-\ell} + u_t, \qquad
# y_t = \begin{bmatrix} F_t \\ \pi_t \\ R_t \end{bmatrix}, \qquad
# u_t \sim \mathcal{N}(0, \Sigma).
# $$ (eq-factor-augmented-transition)
#
# It has no intercepts, because every series has mean 0: each indicator by standardisation,
# and therefore the factor, which the indicators measure with idiosyncratic errors of mean 0;
# consumer price inflation and the federal funds rate by demeaning.
#
# **Normalisation.** The factor has no data of its own, so its scale and sign are not
# identified: doubling $F_t$ and halving every $\lambda_i$ gives the same indicators.
# Following {cite:t}`bernankeBoivinEliasz2005`, who fix loadings for the same reason, we fix
# the loading of one **normalising indicator**, payroll employment, to $\lambda_1 = 1$. Every
# choice of normalising indicator gives the same likelihood; only the priors, which are set on
# the normalised scale, differ. We choose payroll employment because it determines the factor:
# {numref}`tbl-measurement` below shows that its idiosyncratic standard deviation is far
# smaller than the other three indicators'. The factor is then on the scale of standardised
# payroll employment: one unit of the factor is one standard deviation of payroll employment's
# percent change from the previous month, and a rise in the factor is a rise in payroll
# employment.
#
# **How the NumPyro model builds the factor.** The compiled block can generate a latent series
# itself, through `latent_names`: it samples one standardised VAR innovation per month and
# computes the series from them with the VAR's equation. A series generated from standardised
# innovations samples well when the data inform it weakly, and badly when the data inform it
# strongly {cite:p}`papaspiliopoulosRobertsSkold2007`. Payroll employment informs this factor
# strongly, as {numref}`tbl-measurement` shows. The NumPyro model therefore builds the factor itself, from standardised payroll
# employment and payroll employment's standardised idiosyncratic error
# ({eq}`eq-factor-augmented-change-of-variables` below), and passes it to the block as an
# endogenous series, next to consumer price inflation and the federal funds rate; the `endog`
# a block receives may be an array that the NumPyro model generated. The block then has no
# latent series. With latent series and 2 or more lags, the block adds the
# `latent_stationarity` Potential, a term that makes the log density minus infinity outside
# the stationary region and has no gradient. Passing the factor as an endogenous series
# avoids that term.
#
# **Sampling the factor's values directly.** We first sampled the factor's value in each month
# directly: each $F_t$ was a NumPyro sample site with no distribution of its own, which gives
# it an improper flat base measure, so that the factor's prior was the VAR density the block
# adds, which conditions on the first 13 months. The posterior puts mass near 0 on payroll employment's idiosyncratic standard
# deviation $\sigma_1$, because the factor can be almost equal to payroll employment. Given
# $\sigma_1$, the posterior spread of each $F_t$ around $X_{1t}$ is at most about $\sigma_1$,
# so the posterior of the 288 values of the factor is wide where $\sigma_1$ is large and narrow
# where it is small. We ran three trials, each with 4 chains of 1,000 kept draws and random key
# 0: industrial production's loading fixed at 1 with NumPyro's default random start; payroll
# employment's loading fixed at 1 with the default random start; and payroll employment's
# loading fixed at 1 with the other loadings starting at 0 and the factor starting at
# standardised payroll employment. In each trial the four chains' draws of $\sigma_1$ had
# clearly different medians. **R-hat** measures such disagreement: it compares the spread of
# the draws within each chain with the spread between chains, it is 1 when the chains agree,
# lower is better, and values above 1.01 mean the chains disagree. The largest R-hat was far
# above 1.01 in all three trials.
#
# **A change of variables.** Payroll employment's observation equation says
# $X_{1t} = F_t + \sigma_1 \eta_t$, where $\eta_t = e_{1t} / \sigma_1 \sim \mathcal{N}(0, 1)$
# is **payroll employment's standardised idiosyncratic error**. The model samples $\eta_t$ and
# sets
#
# $$
# F_t = X_{1t} - \sigma_1 \eta_t .
# $$ (eq-factor-augmented-change-of-variables)
#
# The flat base measure of $F_t$ times the density of $X_{1t}$ given $F_t$ is
# $\sigma_1^{-1} \varphi(\eta_t)$, with $\varphi$ the standard normal density, and the
# Jacobian of {eq}`eq-factor-augmented-change-of-variables` is $\sigma_1$, so the two factors
# of $\sigma_1$ cancel. Sampling $\eta_t \sim \mathcal{N}(0, 1)$ therefore replaces the flat
# base measure of $F_t$ and payroll employment's observation equation. The block adds the VAR
# density of the factor in both forms, so the posterior is unchanged. In the new variables, as
# $\sigma_1 \to 0$, the factor approaches payroll employment and the posterior of each
# $\eta_t$ approaches its $\mathcal{N}(0, 1)$ prior, instead of narrowing. The change of
# variables does not make every region of the posterior easy to sample: the diagnostics below
# report any divergent transitions and where they occurred.
#
# **Priors.** Each loading other than payroll employment's is $\mathcal{N}(0, 1)$, and each
# $\sigma_i$ is half-normal with scale 1: weakly informative on the standardised scale, where
# an indicator's idiosyncratic standard deviation cannot exceed its total standard deviation
# of 1 by much. The VAR's coefficients get the Minnesota prior with Impulso's default
# tightness. The coefficient on lag $\ell$ of series $j$ in the equation of series $i$ has
# prior standard deviation $0.1 / \ell$ when $j = i$, and $0.05 \, s_i / (\ell \, s_j)$
# otherwise, where $s_i$ is the scale of series $i$ set in the next section. The coefficients
# with $j \neq i$ are the **cross-lag coefficients**. The prior mean is 0, except on a series'
# own first lag. The factor, on the scale of standardised payroll employment, and consumer
# price inflation are stationary, so their own-lag prior mean is 0, as
# {cite:t}`banburaGiannoneReichlin2010` set for stationary series. The federal funds rate is a
# slow-moving level and keeps the random-walk prior mean of 1.
#
# **Identification.** The monetary policy shock is identified with a Cholesky ordering of the
# VAR's innovations $u_t$: the factor first, consumer price inflation second and the federal
# funds rate last. The Federal Reserve then reacts within the month to the factor and to
# consumer price inflation, and neither responds to the monetary policy shock until the next
# month.
#
# ## Compile the VAR
#
# Each call of the block receives all three series, so `endog_names` lists them in VAR order
# and `latent_names` is not given. `intercept_equations=()` gives no equation an intercept.
# `endog_scales` sets the scales $s_i$ of the Minnesota prior: for consumer price inflation and
# the federal funds rate, the standard deviation of the residuals of a regression of the series
# on a constant and its own previous value (`ar1_residual_sd`). The factor has no data, so its
# scale is the same quantity for standardised payroll employment, the normalising indicator.

# %%
n_lags = 13
var_names = ["factor", *price_and_rate_names]
endog_scales = ar1_residual_sd(np.column_stack([standardised[:, 0], demeaned_price_and_rate]))
spec = VAR(lags=n_lags, prior=MinnesotaPrior(own_lag_mean=(0.0, 0.0, 1.0)))
block = spec.compile_for_numpyro(
    exog=None,
    n_lags=n_lags,
    endog_names=var_names,
    endog_scales=endog_scales,
    intercept_equations=(),
)

# %% [markdown]
# ## Write the NumPyro model
#
# The NumPyro model samples the three free loadings, the four idiosyncratic standard
# deviations and payroll employment's standardised idiosyncratic errors. It builds the factor
# with {eq}`eq-factor-augmented-change-of-variables` and passes the factor, demeaned consumer
# price inflation and the demeaned federal funds rate to the block, which adds the VAR's prior
# and density. The observation equation of the other three indicators completes the model;
# payroll employment's observation equation is already part of the change of variables. The
# recorded `factor_loadings` puts payroll employment's fixed loading of 1 in front of the three
# sampled loadings.


# %%
def model(standardised, demeaned_price_and_rate):
    other_factor_loadings = numpyro.sample(
        "other_factor_loadings", dist.Normal(0.0, 1.0).expand([n_indicators - 1]).to_event(1)
    )
    idiosyncratic_sd = numpyro.sample("idiosyncratic_sd", dist.HalfNormal(1.0).expand([n_indicators]).to_event(1))
    payroll_employment_standardised_error = numpyro.sample(
        "payroll_employment_standardised_error", dist.Normal(0.0, 1.0).expand([n_months]).to_event(1)
    )
    factor = numpyro.deterministic(
        "factor", standardised[:, 0] - idiosyncratic_sd[0] * payroll_employment_standardised_error
    )
    block(jnp.column_stack([factor, demeaned_price_and_rate]))

    numpyro.deterministic("factor_loadings", jnp.concatenate([jnp.ones(1), other_factor_loadings]))
    mean = factor[:, None] * other_factor_loadings
    numpyro.sample("other_indicators", dist.Normal(mean, idiosyncratic_sd[1:]).to_event(2), obs=standardised[:, 1:])


# %% [markdown]
# ## Sample with NumPyro's NUTS
#
# NUTS runs with NumPyro's default settings: a diagonal mass matrix and a target acceptance
# probability of 0.8. The VAR's sites start at `block.initial_point`, and the other sites at
# NumPyro's default random start, uniform between −2 and 2 on the unconstrained scale. The
# random key, the argument of `jax.random.key`, sets the random start and the draws. We run 4
# chains, each with 1,000 warm-up draws and 4,000 kept draws. Payroll employment's
# idiosyncratic standard deviation has the smallest bulk effective sample size, defined in the
# next section. In trials with 2,000 kept draws per chain, that smallest bulk effective sample
# size was barely above the target of 400; 4,000 kept draws per chain give it a margin.

# %%
if ci:
    num_warmup, num_samples, num_chains = 200, 100, 2
else:
    num_warmup, num_samples, num_chains = 1000, 4000, 4

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
mcmc.run(jax.random.key(0), standardised, demeaned_price_and_rate)
print(f"Sampling took {time.perf_counter() - start:.0f} seconds.")

# %% [markdown]
# ## Diagnostics
#
# Three checks answer whether the draws describe the posterior; [Model Checks and
# Validation](model-checking.py) explains them in detail.
#
# - **Divergent transitions** are transitions at which NUTS's numerical integrator broke down,
#   evidence of posterior geometry the sampler cannot follow. Lower is better; 0 is the target.
# - **R-hat**, defined above. Lower is better; at most 1.01 is the target.
# - **Bulk effective sample size** is the number of independent draws that would estimate
#   the posterior mean as precisely as the correlated draws do. Higher is better; at least 400
#   is the usual target.
#
# R-hat and the effective sample size are computed for every quantity in the posterior: the
# VAR's coefficients, the Cholesky factor of its innovation covariance, the loadings, the
# idiosyncratic standard deviations, payroll employment's standardised idiosyncratic errors
# and the factor in each of the 288 months. Quantities fixed by construction, such as payroll
# employment's loading and the zeros above the diagonal of the Cholesky factor, have none and
# are skipped. The cell prints whether the largest R-hat is at most 1.01 and whether the
# smallest bulk effective sample size is at least 400. If there are divergent transitions, it
# also prints the range of percentiles, among the posterior draws, of payroll employment's
# idiosyncratic standard deviation and of the standard deviation of the factor's VAR
# innovation at which they occurred.

# %%
samples = mcmc.get_samples(group_by_chain=True)
posterior = block.posterior_from_samples(samples)
diagnostics = az.summary(posterior, kind="diagnostics", round_to="none")
divergent = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
largest_r_hat = diagnostics["r_hat"].max()
smallest_ess_bulk = diagnostics["ess_bulk"].min()

print(f"Divergent transitions: {int(divergent.sum())} of {divergent.size}")
print(f"Largest R-hat: {largest_r_hat:.4f} ({diagnostics['r_hat'].idxmax()})")
print(f"Largest R-hat at most 1.01: {'yes' if largest_r_hat <= 1.01 else 'no'}")
print(f"Smallest bulk effective sample size: {smallest_ess_bulk:.0f} ({diagnostics['ess_bulk'].idxmin()})")
print(f"Smallest bulk effective sample size at least 400: {'yes' if smallest_ess_bulk >= 400 else 'no'}")

where_divergent = {
    "payroll employment's idiosyncratic standard deviation": np.asarray(samples["idiosyncratic_sd"])[..., 0],
    "the standard deviation of the factor's VAR innovation": np.asarray(samples["L"])[..., 0, 0],
}
if divergent.any():
    for name, draws in where_divergent.items():
        percentiles = [100 * np.mean(draws < value) for value in draws[divergent]]
        print(f"Percentiles of {name} at the divergent transitions: {min(percentiles):.1f} to {max(percentiles):.1f}")

# %% [markdown]
# In smoke-render builds of these docs the sampler runs few draws, so the numbers above are not
# representative there.
#
# **Validation.** We ran this notebook's model and sampling code with random keys 0 to 7,
# changing nothing else, in two environments: Python 3.11 with PyMC 5.28 and JAX 0.10, and Python 3.13 with PyMC 6.2
# and JAX 0.12. In every run the largest R-hat was at most 1.01 and the smallest bulk effective
# sample size was well above 400, and in every run the smallest belonged to payroll
# employment's idiosyncratic standard deviation. Most runs had no divergent transitions and the
# others had few. In the run with the most, every divergent transition occurred among the
# largest values of payroll employment's idiosyncratic standard deviation and the smallest
# values of the standard deviation of the factor's VAR innovation, where the factor differs
# most from payroll employment. The run on this page uses random key 0, but the machine and
# the package versions that built this page may differ from those of the validation runs.
#
# ## The estimated factor
#
# {numref}`tbl-measurement` gives the posterior median and the 89 % band of each loading, each
# idiosyncratic standard deviation and each share $1 - \sigma_i^2$. A loading is in standard
# deviations of indicator $i$'s percent change per standard deviation of payroll employment's
# percent change, both from the previous month; an idiosyncratic standard deviation is in
# standard deviations of indicator $i$'s percent change. Each standardised indicator has
# variance 1, so $1 - \sigma_i^2$ is roughly the share of its variance that the factor
# explains; it is below 0 in draws where $\sigma_i$ exceeds 1.

# %% mystnb={"markdown_format": "myst"}
idiosyncratic_sd_draws = np.asarray(samples["idiosyncratic_sd"])
measurement = pd.DataFrame(
    {
        "loading": median_and_band_text(np.asarray(samples["factor_loadings"])),
        "idiosyncratic standard deviation": median_and_band_text(idiosyncratic_sd_draws),
        "share of variance the factor explains": median_and_band_text(1.0 - idiosyncratic_sd_draws**2),
    },
    index=pd.Index(list(indicator_titles.values()), name="indicator"),
)
numbered_table(
    measurement,
    "tbl-measurement",
    "Posterior median and 89 % band of each indicator's loading, idiosyncratic standard deviation and share of "
    "variance the factor explains.",
)

# %% [markdown]
# Payroll employment's idiosyncratic standard deviation is the smallest of the four, and the
# lower end of its 89 % band is close to 0: the posterior puts mass near 0, as the model
# section described. For real personal income less transfers and real manufacturing and trade
# sales, the factor explains a small share of the variance: most of their percent change from
# the previous month is idiosyncratic.
#
# {numref}`fig-factor` plots the posterior median of $F_t$ with its 89 % band next to
# standardised payroll employment. The printout gives the correlation of the posterior-median
# factor with each standardised indicator.

# %% mystnb={"figure": {"name": "fig-factor", "caption": "Posterior median and 89 % band of the factor, and standardised payroll employment, January 1984 to December 2007, in standard deviations of payroll employment's percent change from the previous month."}, "image": {"alt": "Posterior median and 89% band of the factor from 1984 to 2007, plotted over standardised payroll employment; the factor is close to payroll employment and falls below 0 in the 1990-91 and 2001 recessions."}}
factor_draws = np.asarray(samples["factor"])  # (chain, draw, month)
factor_median = np.median(factor_draws, axis=(0, 1))
for i, title in enumerate(indicator_titles.values()):
    correlation = np.corrcoef(factor_median, standardised[:, i])[0, 1]
    print(f"Correlation with standardised {title.lower()}: {correlation:.3f}")

fig, ax = plt.subplots()
ax.plot(panel.index, standardised[:, 0], color=plotting.COLORS.muted, linewidth=0.7, label="Standardised payroll employment")
plot_median_and_band(ax, panel.index, factor_draws, label="Factor, posterior median and 89 % band")
ax.set_ylabel("Standard deviations")
plotting.serif_title("The factor is close to standardised payroll employment", ax)
_ = plotting.legend_below(ax, per_row=1)

# %% [markdown]
# The printed correlation with payroll employment is close to 1, and the correlations with the
# other three indicators are much lower: the factor is close to payroll employment because
# payroll employment's idiosyncratic standard deviation is small. The factor falls below 0 in
# the recessions of 1990–91 and 2001, when payroll employment fell.
#
# ## Pass the posterior to `FittedVAR`
#
# `FittedVAR.from_posterior` needs a `VARData` with the series the block received. The block
# received a different factor in each draw, so the factor's column holds a plug-in series: the
# posterior mean of the factor in each month. The docstring of
# {py:meth}`CompiledNumPyroBlock.posterior_from_samples <impulso.numpyro_block.CompiledNumPyroBlock.posterior_from_samples>`
# asks for the same plug-in for a latent series, which has no data either. Impulse responses
# do not read these values; forecasts and historical decompositions do. We pass
# `error_dist=spec.resolved_error_dist` for the reason that docstring gives.

# %%
data = VARData(
    endog=np.column_stack([factor_draws.mean(axis=(0, 1)), demeaned_price_and_rate]),
    endog_names=var_names,
    index=panel.index,
)
fitted = FittedVAR.from_posterior(posterior, data, n_lags=n_lags, error_dist=spec.resolved_error_dist)
identified = fitted.set_identification_strategy(Cholesky(ordering=var_names))

# %% [markdown]
# ## Impulse responses
#
# `identified.impulse_response` gives, for each posterior draw, the response of the factor,
# consumer price inflation and the federal funds rate to the monetary policy shock. We scale
# each draw's responses so that the federal funds rate rises by 25 basis points (0.25
# percentage points) in the month of the monetary policy shock. The response of indicator $i$
# follows from {eq}`eq-factor-augmented-observation`: $\lambda_i$ times the factor's response.
# We compute it draw by draw, from the loadings of the same draw as the VAR coefficients: the
# impulse responses keep the posterior's `chain` and `draw` axes in the order of `samples`.
#
# To read the responses in each series' own units, we multiply an indicator's response by its
# standard deviation, which gives the response of its percent change from the previous month,
# and sum over the months, which gives the response of its level in percent. Summing the
# response of consumer price inflation gives the response of the consumer price index's level
# in percent. Each response is a `Response`: a title, a unit and the draws, with shape
# (chain, draw, month).


# %%
class Response(NamedTuple):
    """Draws of one series' response to the monetary policy shock, with the series' title and unit."""

    title: str
    unit: str
    draws: np.ndarray


horizon = 48
months = np.arange(horizon + 1)
irf = identified.impulse_response(horizon=horizon)
to_monetary_policy_shock = irf.idata["posterior_predictive"]["irf"].sel(shock="federal_funds_rate")
to_monetary_policy_shock = (
    to_monetary_policy_shock * 0.25 / to_monetary_policy_shock.sel(response="federal_funds_rate", horizon=0)
)


def response_draws(name):
    """Draws of the response of VAR series `name`, shape (chain, draw, month)."""
    return to_monetary_policy_shock.sel(response=name).transpose("chain", "draw", "horizon").to_numpy()


factor_response = response_draws("factor")
factor_loadings = np.asarray(samples["factor_loadings"])  # (chain, draw, indicator)
assert factor_loadings.shape[:2] == factor_response.shape[:2]

rate_and_price_responses = [
    Response("Federal funds rate", "percentage points", response_draws("federal_funds_rate")),
    Response("Consumer price index", "percent", response_draws("consumer_price_inflation").cumsum(axis=-1)),
]
indicator_responses = [
    Response(title, "percent", (factor_loadings[:, :, i, None] * factor_response * indicator_sd[name]).cumsum(axis=-1))
    for i, (name, title) in enumerate(indicator_titles.items())
]


def plot_responses(responses, n_rows, n_cols, claim):
    """Plot each response's posterior median and 89 % band in its own panel, under the figure's claim."""
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False)
    for ax, response in zip(axes.ravel(), responses, strict=True):
        plot_median_and_band(ax, months, response.draws)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.25 * (top - bottom))  # room above the data for the panel's title
        ax.text(0.03, 0.95, response.title, transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_ylabel(response.unit)
    for ax in axes[-1]:
        ax.set_xlabel("Months after the monetary policy shock")
    plotting.serif_title(claim, axes[0, 0])


# %% [markdown]
# {numref}`fig-rate-and-price` plots the responses of the federal funds rate and of the
# consumer price index, and {numref}`fig-indicators` those of the four indicators.

# %% mystnb={"figure": {"name": "fig-rate-and-price", "caption": "Responses over 48 months to a monetary policy shock that raises the federal funds rate by 25 basis points: posterior median and 89 % band of the federal funds rate, in percentage points, and of the consumer price index's level, in percent."}, "image": {"alt": "Two panels over 48 months: the federal funds rate rises from 0.25 for several months and then falls back; the consumer price index's level rises, with a band above 0."}}
plot_responses(rate_and_price_responses, 1, 2, "Prices rise after a rise in the federal funds rate")

# %% mystnb={"figure": {"name": "fig-indicators", "caption": "Responses of the four indicators' levels, in percent, over 48 months to the same monetary policy shock: posterior median and 89 % band."}, "image": {"alt": "Two-by-two grid over 48 months: payroll employment, industrial production, real personal income less transfers and real manufacturing and trade sales, each with a band that includes 0 throughout."}}
plot_responses(indicator_responses, 2, 2, "The indicators show no clear response")

# %% [markdown]
# {numref}`tbl-responses` gives the posterior median and the 89 % band of each response 12, 24
# and 48 months after the monetary policy shock, and the number of months, from 0 to 48, in
# which the 89 % band excludes 0. It includes the factor's own response, in standard
# deviations of payroll employment's percent change from the previous month.

# %% mystnb={"markdown_format": "myst"}
report_months = [12, 24, 48]
factor_row = Response("Factor", "standard deviations of payroll employment's percent change", factor_response)
rows = {}
for response in [*rate_and_price_responses, factor_row, *indicator_responses]:
    low, _, high = posterior_quantiles(response.draws)
    months_excluding_0 = int(np.sum((low > 0) | (high < 0)))
    rows[f"{response.title} ({response.unit})"] = [
        *median_and_band_text(response.draws[..., report_months]),
        months_excluding_0,
    ]
responses_table = pd.DataFrame.from_dict(
    rows,
    orient="index",
    columns=[*(f"{m} months" for m in report_months), "months of 49 with a band excluding 0"],
)
responses_table.index.name = "response"
numbered_table(
    responses_table,
    "tbl-responses",
    "Posterior median and 89 % band of each response to a 25 basis point rise in the federal funds rate.",
)

# %% [markdown]
# Standard monetary theory predicts that a monetary policy shock that raises the federal funds
# rate lowers output, employment, income and sales, and lowers prices or leaves them flat.
# {numref}`tbl-responses` shows what this run estimates:
#
# - **The federal funds rate rises further and then falls back.** Its median at 12 months is
#   above the 0.25 percentage points of the first month, and its median at 48 months is below
#   it.
# - **The consumer price index rises: a price puzzle.** Its 89 % band lies above 0 at 12, 24
#   and 48 months.
# - **The factor shows no clear response.** Its 89 % band includes 0 in every month, so its
#   count of months with a band excluding 0 is 0. The indicators' responses are the factor's
#   response times their loadings, cumulated over the months; their counts are 0 too.
#
# Each of these statements held in every validation run.
#
# ## A possible reason prices rise, and what the prior contributes
#
# - **A possible reason: the factor contains only real-activity information.** The Federal
#   Reserve raises the federal funds rate when it expects consumer price inflation to rise. A
#   VAR whose series do not predict that rise includes the Federal Reserve's response to it in
#   the monetary policy shock, and the monetary policy shock then appears to raise prices:
#   {cite:t}`sims1992` traced the price puzzle to this. {cite:t}`bernankeBoivinEliasz2005`
#   summarised 120 series, prices and financial series among them, by three factors to shrink
#   the price puzzle. Four real-activity indicators do not contain that information.
# - **The prior determines much of the spread of the cross-lag coefficients.** The Cholesky
#   ordering sets the responses of the factor and of consumer price inflation in the month of
#   the monetary policy shock to 0, so all their later responses come from the cross-lag
#   coefficients: if those were all 0, neither series would respond at all.
#   {numref}`tbl-prior-ratio` gives, for each equation and each lagged series, the median and
#   the smallest over the 13 lags of the ratio of a coefficient's posterior standard deviation
#   to its prior standard deviation. A ratio close to 1 means that the data changed the
#   coefficient's spread little from the prior's.

# %% mystnb={"markdown_format": "myst"}
prior_sd = spec.resolved_prior.build_priors(len(var_names), n_lags, sigma=endog_scales)["B_sigma"]
posterior_sd = np.asarray(samples["B"]).std(axis=(0, 1))  # (equation, lag and series)
ratio = (posterior_sd / prior_sd).reshape(len(var_names), n_lags, len(var_names))  # (equation, lag, series)
var_titles = ["Factor", "Consumer price inflation", "Federal funds rate"]
prior_ratio = pd.DataFrame(
    [[f"{np.median(ratio[i, :, j]):.2f} ({ratio[i, :, j].min():.2f})" for j in range(len(var_names))] for i in range(len(var_names))],
    index=pd.Index([f"{title} equation" for title in var_titles], name="equation"),
    columns=[f"lags of {title.lower()}" for title in var_titles],
)
numbered_table(
    prior_ratio,
    "tbl-prior-ratio",
    "Posterior standard deviation over prior standard deviation of the VAR's coefficients: median over the 13 lags, "
    "with the smallest in parentheses.",
)

# %% [markdown]
# In {numref}`tbl-prior-ratio` the median ratios, those of the cross-lag coefficients off the
# diagonal among them, are close to 1, and the smallest ratios, in parentheses, are lower. For
# most cross-lag coefficients, then, the data change the spread little from that of their
# Minnesota prior, which has mean 0. The responses of the factor and of consumer price
# inflation, computed from these coefficients, therefore depend on the prior's tightness as
# well as on the data. This tutorial keeps the tightness at Impulso's default.
#
# These are limits of the example, not of the factor-augmented VAR. We keep the priors as
# stated rather than change them to change the result.
#
# ## Summary
#
# 1. **Prepare** the indicators as standardised percent changes from the previous month, and
#    demean the other series that enter the VAR.
# 2. **Compile** the VAR with the factor as an endogenous series, no latent series, and
#    `intercept_equations=()`.
# 3. **Write** the NumPyro model: build the factor from standardised payroll employment and
#    payroll employment's standardised idiosyncratic error, pass it to the block with the other
#    series, and write the observation equation of the other indicators.
# 4. **Sample** with NumPyro's NUTS, starting the VAR's sites at `block.initial_point`, and
#    check divergent transitions, R-hat and bulk effective sample size.
# 5. **Pass** the posterior to `FittedVAR.from_posterior` through
#    `block.posterior_from_samples`, with the posterior-mean factor as the factor's column.
# 6. **Combine** the factor's impulse responses with the loadings, draw by draw, to get the
#    response of every indicator.
#
# ## References
#
# The works cited above are collected on the [project bibliography](../references.md) page.
