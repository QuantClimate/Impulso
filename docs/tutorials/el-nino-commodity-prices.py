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
# # El Niño and Food Commodity Prices

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
# How much, and for how long, does an El Niño raise world food commodity prices? {cite:t}`brunner1998`
# answered this question with vector autoregressions (VARs) estimated by least squares on quarterly data
# from 1963 to 1997. A version of the paper was published as {cite:t}`brunner2002`; this tutorial
# follows the 1998 discussion paper, and its page numbers refer to it. He measured El Niño in two ways, with a
# sea-surface-temperature anomaly and with the Southern Oscillation Index, a difference in sea-level
# pressure across the Pacific. He writes that the two measures are "highly but not perfectly
# correlated" and "offer a somewhat different view" of the 1997 El Niño (p. 4), and he estimated a
# separate VAR for each.
#
# This tutorial treats El Niño as one quantity that no single index measures exactly: the **El Niño
# factor**, an unobserved series that four climate indices measure with error. One model estimates the
# El Niño factor and its effect on food commodity prices together:
#
# - a **measurement equation**, written in [NumPyro](https://num.pyro.ai/), relates the four climate
#   indices to the El Niño factor;
# - a **VAR**, specified once in Impulso and compiled for NumPyro with `VAR.compile_for_numpyro`, holds
#   the dynamics of the El Niño factor and of real food commodity price growth. The NumPyro model
#   samples the El Niño factor's value in every quarter and passes these values to the compiled VAR as
#   an endogenous series;
# - **one posterior** covers both, so the impulse responses' intervals include the uncertainty about
#   the El Niño factor in every quarter.
#
# The tutorial assumes you have read [A VAR Inside a NumPyro Model](var-in-numpyro.py), which explains
# the compiled NumPyro block. It needs the `numpyro` extra:
#
# ```
# pip install "impulso[numpyro]"
# ```

# %%
import time
from functools import partial
from typing import NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
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
# ## The data
#
# The four climate indices are monthly series from the NOAA Climate Prediction Center, described in
# {numref}`tbl-el-nino-climate-indices`.
#
# :::{table} The four climate indices.
# :name: tbl-el-nino-climate-indices
#
# | Climate index | What it measures | Unit |
# |---------------|------------------|------|
# | **Niño 3** | Sea-surface-temperature anomaly over 5°N–5°S, 150°W–90°W, the eastern equatorial Pacific, from the 1991–2020 monthly means of the Extended Reconstructed Sea Surface Temperature, version 5 {cite:p}`huang2017` | °C |
# | **Niño 4** | The same anomaly over 5°N–5°S, 160°E–150°W, the central equatorial Pacific | °C |
# | **Southern Oscillation Index** | Standardised sea-level pressure at Tahiti minus standardised sea-level pressure at Darwin, divided by the standard deviation of that difference | standard deviations |
# | **Equatorial Southern Oscillation Index** | Standardised sea-level-pressure anomaly over the eastern equatorial Pacific (5°N–5°S, 130°W–80°W) minus that over Indonesia (5°N–5°S, 90°E–140°E), normalised | standard deviations |
# :::
#
# Brunner describes El Niño as a large-scale warming of the surface of the eastern equatorial Pacific,
# and the Southern Oscillation as the shift of low sea-level pressure to the east of the Pacific and
# high pressure to the west that comes with it (pp. 3–4). Niño 3 and Niño 4 are the two
# **temperature indices**, and the Southern Oscillation Index and the equatorial Southern Oscillation
# Index are the two **pressure indices**. So during an El Niño, the temperature indices rise and the
# pressure indices fall.
#
# Brunner's sea-surface-temperature measure was the anomaly of the Niño 1+2 region (0–10°S, 90°W–80°W),
# off the coasts of Ecuador and Peru (p. 4). The data file holds it, from the same NOAA file as Niño 3
# and Niño 4, but the model leaves it out, for the reason that the correlations printed below show. The
# model also leaves out Niño 3.4 (5°N–5°S, 170°W–120°W), whose region overlaps both Niño 3 and Niño 4, so
# that no two temperature indices average over the same area of ocean.
#
# The **food price index** is the World Bank's Food price index, a weighted index of food commodity
# prices in nominal US dollars (2010 = 100). We deflate it by the **consumer price index**, the US
# consumer price index for all urban consumers, all items, seasonally adjusted (1982–84 = 100), from
# the Bureau of Labor Statistics. Brunner's real food commodity price inflation was the inflation rate
# of the IMF's food price index minus average G-7 consumer price inflation (pp. 6–7); this tutorial
# uses neither of his series.
#
# The script `scripts/fetch_el_nino_commodity_prices.py` in the Impulso repository downloads every
# series and averages each to quarters: the mean of the quarter's three months. Its output ships with
# these docs as `data/el_nino_commodity_prices.csv`. Sources and licences are in
# [Data sources and licences](#data-sources-and-licences) at the end.

# %%
panel = pd.read_csv("data/el_nino_commodity_prices.csv", index_col="quarter_start", parse_dates=True)
panel.loc["1961-07-01":"1997-10-01"].head()

# %% [markdown]
# **Real food commodity price growth** is 100 times the quarterly change in the logarithm of the food
# price index divided by the consumer price index, in percent per quarter:
#
# $$
# g_t = 100 \, \bigl[\log(p_t / c_t) - \log(p_{t-1} / c_{t-1})\bigr],
# $$ (el-nino-growth)
#
# where $p_t$ is the food price index and $c_t$ the consumer price index in quarter $t$. One hundred
# times a change in a logarithm is a log change, which approximates the percent change; this tutorial
# calls it a percent throughout.
#
# The model uses the quarters from 1961Q3 to 1997Q4. With 6 lags, the VAR's equations then explain the
# quarters from 1963Q1 to 1997Q4, which is how we read Brunner's "quarterly data from 1963 through
# 1997" (p. 8). The cell below prints both counts.
#
# We standardise each climate index over 1961Q3 to 1997Q4 (subtract its mean, divide by its standard
# deviation), so the four share one scale, and we subtract the mean over the same quarters from real
# food commodity price growth. Every series then has mean 0, which is why the VAR below has no
# intercepts.

# %%
INDEX_NAMES = ["nino3", "nino4", "southern_oscillation_index", "equatorial_southern_oscillation_index"]
INDEX_LABELS = ["Niño 3", "Niño 4", "Southern Oscillation Index", "Equatorial Southern Oscillation Index"]
FACTOR_NAME = "el_nino_factor"
GROWTH_NAME = "real_food_commodity_price_growth"
VAR_NAMES = [FACTOR_NAME, GROWTH_NAME]
N_LAGS = 6

panel[GROWTH_NAME] = 100 * np.log(panel["food_price_index"] / panel["consumer_price_index"]).diff()
sample = panel.loc["1961-07-01":"1997-10-01"]
quarters = sample.index
n_quarters = len(sample)

climate_indices = sample[INDEX_NAMES]
standardised_indices = ((climate_indices - climate_indices.mean()) / climate_indices.std()).to_numpy()
growth = sample[GROWTH_NAME].to_numpy()
demeaned_growth = growth - growth.mean()

print(f"Quarters the model uses: {quarters[0].to_period('Q')} to {quarters[-1].to_period('Q')}, {n_quarters} quarters")
print(f"Quarters the VAR's equations explain: {quarters[N_LAGS].to_period('Q')} to "
      f"{quarters[-1].to_period('Q')}, {n_quarters - N_LAGS} quarters")
print(f"Mean real food commodity price growth: {growth.mean():.2f} % per quarter")

# %% [markdown]
# {numref}`fig-el-nino-climate-indices` plots the four standardised climate indices, and
# {numref}`fig-el-nino-food-growth` plots real food commodity price growth.

# %% mystnb={"figure": {"caption": "The four climate indices, each standardised over 1961Q3–1997Q4. Niño 3 and Niño 4 rise and both pressure indices fall in El Niño episodes such as 1972, 1982–83 and 1997.", "name": "fig-el-nino-climate-indices"}, "image": {"alt": "Four standardised climate indices from 1961 to 1997: Niño 3 and Niño 4 rise and both Southern Oscillation indices fall during El Niño episodes such as 1972, 1982-83 and 1997."}}
fig, ax = plt.subplots()
for column, label, color in zip(standardised_indices.T, INDEX_LABELS, plotting.COLORS.cycle, strict=False):
    ax.plot(quarters, column, color=color, linewidth=0.9, label=label)
ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
ax.set_ylabel("Standard deviations")
plotting.serif_title("The temperature indices rise when the pressure indices fall", ax)
_ = plotting.legend_below(ax, per_row=2)

# %% [markdown]
# The correlations of the climate indices over 1961Q3 to 1997Q4, printed below, include Niño 1+2. Among
# the four climate indices in the model they have the signs that Brunner's description of El Niño
# implies: positive between the two temperature indices and between the two pressure indices, negative
# between a temperature index and a pressure index. The cell also prints whether Niño 3 correlates more
# strongly than Niño 1+2, in absolute value, with each of the other three climate indices in the model.

# %%
index_correlations = sample[["nino12", *INDEX_NAMES]].corr()
others_in_model = INDEX_NAMES[1:]
nino3_correlates_more_strongly = bool(
    (index_correlations.loc["nino3", others_in_model].abs() > index_correlations.loc["nino12", others_in_model].abs()).all()
)
print("Niño 3 correlates more strongly than Niño 1+2 with each of Niño 4 and the two pressure indices: "
      f"{nino3_correlates_more_strongly}")
correlation_labels = ["Niño 1+2", *INDEX_LABELS]
index_correlations.set_axis(correlation_labels, axis=0).set_axis(correlation_labels, axis=1).round(2)

# %% [markdown]
# The model below has one El Niño factor and assumes that the measurement errors of different climate
# indices are independent. Under those assumptions, the correlation of two standardised climate indices
# is the product of their loadings, each in standard deviations of the index per standard deviation of
# the El Niño factor, and the square of such a loading is the share of the index's variance that the El
# Niño factor explains. Niño 1+2 correlates more weakly than Niño 3 with each of Niño 4 and the two
# pressure indices, as the check above prints, so by these correlations its loading is smaller than Niño
# 3's and the El Niño factor explains less of its variance. Niño 1+2 and Niño 3 are both temperature
# indices of the eastern equatorial Pacific, so the model keeps Niño 3, the one more closely related to
# the other three indices, and leaves Niño 1+2 out.

# %% mystnb={"figure": {"caption": "Real food commodity price growth, percent per quarter, 1961Q3–1997Q4.", "name": "fig-el-nino-food-growth"}, "image": {"alt": "Quarterly real food commodity price growth from 1961 to 1997, with the largest swings between 1972 and 1977."}}
fig, ax = plt.subplots()
ax.plot(quarters, growth, color=plotting.COLORS.body, linewidth=0.9)
ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
ax.set_ylabel("% per quarter")
_ = plotting.serif_title("Real food commodity price growth varied most in the 1970s", ax)

# %%
decade = pd.Series(growth, index=quarters).groupby((quarters.year // 10) * 10)
decade.std().round(1).rename("standard deviation of real food commodity price growth (% per quarter)")

# %% [markdown]
# ## The model
#
# Write $f_t$ for the El Niño factor in quarter $t$ and $x_{kt}$ for standardised climate index $k$,
# where $k$ = 1, 2, 3 and 4 are Niño 3, Niño 4, the Southern Oscillation Index and the equatorial
# Southern Oscillation Index. The **measurement equation** says each climate index is a multiple of the
# El Niño factor plus an error:
#
# $$
# x_{kt} = \lambda_k \, f_t + e_{kt}, \qquad e_{kt} \sim \mathcal{N}(0, \psi_k^2).
# $$ (el-nino-measurement)
#
# The **loading** $\lambda_k$ is how far climate index $k$ moves when the El Niño factor rises by 1. The
# **measurement error** $e_{kt}$ is the part of climate index $k$ that the El Niño factor does not
# explain: errors in the measurement, and variation specific to the index's own region. The
# measurement errors are independent across climate indices and quarters, with standard deviation
# $\psi_k$, the **measurement-error standard deviation**.
#
# The **VAR** holds $z_t = (f_t, g_t)$, the El Niño factor and demeaned real food commodity price
# growth:
#
# $$
# z_t = \sum_{l=1}^{6} A_l \, z_{t-l} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma),
# $$ (el-nino-var)
#
# with 6 lags, as in Brunner, and no intercepts.
#
# ### Units, sign and scale of the El Niño factor
#
# Equation {eq}`el-nino-measurement` cannot tell $f_t$ from $-f_t$ with every loading's sign reversed,
# nor $f_t$ from $2 f_t$ with every loading halved. We fix the loading of Niño 3 at 1:
#
# - **Sign.** A positive El Niño factor then means warm water in the Niño 3 region, the eastern
#   equatorial Pacific, which is how Brunner describes an El Niño. His own sea-surface-temperature
#   measure, Niño 1+2, is also an eastern-Pacific anomaly.
# - **Scale.** The El Niño factor is measured in standard deviations of Niño 3 over 1961Q3 to 1997Q4.
#
# Each other loading is then in standard deviations of its climate index per standard deviation of
# Niño 3, and has a Normal(0, 1) prior. Each measurement-error standard deviation is in standard
# deviations of its climate index, and has a half-Normal(1) prior.
#
# ### How the El Niño factor is sampled
#
# `VAR.compile_for_numpyro` can generate a latent endogenous series itself, through `latent_names`: an
# endogenous series with no data, which the block builds from standardised innovations
# $\eta_t \sim \mathcal{N}(0, 1)$ with the VAR's equation, $f_t = \dots + L_{11} \eta_t$, where $L_{11}$
# is the standard deviation of the series' innovation. This is the **non-centred form**: the sampler's
# coordinates are the innovations $\eta_t$, not the values of $f_t$.
#
# This tutorial uses the **centred form** instead: the sampler's coordinates are the values of the El
# Niño factor, $f_1, \dots, f_{146}$. The NumPyro model declares them as one sample site,
# `el_nino_factor`, whose base measure is flat over the real line (NumPyro's `ImproperUniform`), so the
# site adds nothing to the log-density. It passes them to the compiled VAR as its first endogenous
# series, an ordinary column of data from the block's point of view, and compiles the block with no
# `latent_names`, so the block has no latent series. The block adds the priors on the lag coefficients
# `B` and on the Cholesky factor `L` of the innovation covariance $\Sigma$, and the VAR's joint density of
# both series in quarters 7 to 146 given their values in the first 6 quarters. The El Niño factor's
# values enter that joint density, and the measurement equation adds the density of the four climate
# indices given them. The first 6 values of the El Niño factor enter the VAR's joint density only as
# lagged values.
#
# Why the centred form? {cite:t}`betancourtGirolami2013` show that which form samples better depends on
# how strongly the data inform each value: "when the data are weak", the non-centred form "yield[s]
# simpler posterior geometries" (Fig. 4). Here each quarter's value of the El Niño factor is measured by
# four climate indices, and the cell after sampling prints how precisely they determine it.
#
# With latent series, the block keeps the latent series' lag coefficients in their stationary region:
# for one latent series with one lag, through a prior on its own lag truncated to (−1, 1), and in every
# other case through the `latent_stationarity` Potential, which makes the log-density minus infinity
# outside that region (see the docstring of `VAR.compile_for_numpyro`). Here the block has no latent
# series, so it adds no stationarity term at all. The 6-lag VAR is sampled without one, and the notebook
# counts the posterior draws that are not stationary after sampling.
#
# ### What is simplified relative to Brunner
#
# - **Two series, not four.** Brunner's VARs also hold average G-7 consumer price inflation and G-7
#   GDP growth. This VAR holds only the El Niño factor and real food commodity price growth.
# - **El Niño is not strictly exogenous.** Brunner's El Niño equation has only its own lags (p. 8).
#   Here the El Niño equation also has lags of real food commodity price growth, and the Minnesota
#   prior below shrinks their coefficients towards 0, which imposes his restriction only softly.
# - **No seasonal constants.** Brunner's constants vary by season; this VAR has no constants.
# - **A prior.** The VAR's coefficients have a Minnesota prior, and the model is estimated by Bayesian
#   inference, not by least squares.
#
# ## The priors, declared before estimation
#
# Every prior in {numref}`tbl-el-nino-priors`, and the model itself, was fixed before the model was first
# estimated and has not changed since. Two sampler settings were changed after the first runs: the
# number of kept draws and where the chains start. [Sample](#sample) says what was changed and why.
#
# :::{table} Priors and settings.
# :name: tbl-el-nino-priors
#
# | Part | Setting |
# |------|---------|
# | VAR lags | 6 (Brunner) |
# | Minnesota prior | `tightness=0.2`, `cross_shrinkage=0.5`, `decay="harmonic"` |
# | Own-lag prior means | 0.7 for the El Niño factor; 0 for real food commodity price growth |
# | Innovation covariance $\Sigma$ | Impulso's defaults: half-Cauchy(2.5) on the diagonal of its Cholesky factor, Normal(0, 0.5) on the off-diagonal scaled by its row's diagonal |
# | Loading of Niño 3 | fixed at 1 |
# | Other loadings | Normal(0, 1) |
# | Measurement-error standard deviations | half-Normal(1) |
# | El Niño factor | a sample site with a flat base measure, so no prior of its own; its values enter the VAR's joint density of both series, which the block adds |
# :::
#
# The tightness of 0.2 is the starting point that [The Minnesota Prior, From Scratch](minnesota-prior.py)
# suggests for a small system with a long sample. Brunner found his real commodity price inflation
# series stationary (p. 8), and real food commodity price growth is a growth rate of the same kind, so
# its own-lag prior mean is 0. The own-lag prior mean of the El Niño factor is the first-order
# autoregression coefficient of quarterly Niño 3 over the quarters before 1961Q3, 1950Q1 to 1961Q2,
# rounded to one decimal. The model does not use those quarters, so the prior and the likelihood use
# different data.

# %%
before_1961q3 = panel.loc["1950-01-01":"1961-04-01", "nino3"].to_numpy()
regressors = np.column_stack([np.ones(len(before_1961q3) - 1), before_1961q3[:-1]])
first_order_autoregression_coefficient = np.linalg.lstsq(regressors, before_1961q3[1:], rcond=None)[0][1]
print(f"First-order autoregression coefficient of quarterly Niño 3, 1950Q1-1961Q2 "
      f"({len(before_1961q3)} quarters): {first_order_autoregression_coefficient:.3f}")

OWN_LAG_MEANS = (0.7, 0.0)  # El Niño factor, real food commodity price growth
TIGHTNESS = 0.2

# %% [markdown]
# The Minnesota prior scales each cross-lag coefficient's prior standard deviation by the ratio of the
# two series' scales, `endog_scales`. The El Niño factor has no data before it is sampled, so it gets
# the scale of standardised Niño 3, the climate index that fixes its scale. Each scale is the residual
# standard deviation of a first-order autoregression fitted to the series, `ar1_residual_sd`.
#
# The impulse responses after the shock quarter are computed from the lag coefficients. Of these, the
# coefficients on the lagged El Niño factor in the equation of real food commodity price growth link
# the two series. Their prior standard deviation on lag $l$ is
# $\text{tightness} \times 0.5 \times (\sigma_g / \sigma_f) / l$, where $\sigma_g$ and $\sigma_f$ are
# the two scales, so it is proportional to the tightness. The cell prints it, in percent per quarter
# per standard deviation of Niño 3.

# %%
endog_scales = ar1_residual_sd(np.column_stack([standardised_indices[:, 0], demeaned_growth]))
cross_lag_prior_standard_deviation = TIGHTNESS * 0.5 * endog_scales[1] / endog_scales[0] / np.arange(1, N_LAGS + 1)
print(f"Scales: El Niño factor {endog_scales[0]:.2f} standard deviations of Niño 3, "
      f"real food commodity price growth {endog_scales[1]:.2f} % per quarter")
print("Prior standard deviation of the coefficient on the lagged El Niño factor in the equation of")
print(f"real food commodity price growth, lags 1 to {N_LAGS}:", np.round(cross_lag_prior_standard_deviation, 2))

# %% [markdown]
# ## Compile the VAR
#
# The block is compiled once, outside the NumPyro model. `intercept_equations=()` gives no equation an
# intercept. `compile_var` is used again, unchanged, in the section on sensitivity to the tightness.

# %%
def compile_var(tightness, endog_scales):
    """Specify the VAR with the given Minnesota tightness and compile it for NumPyro."""
    prior = MinnesotaPrior(tightness=tightness, cross_shrinkage=0.5, decay="harmonic", own_lag_mean=OWN_LAG_MEANS)
    spec = VAR(lags=N_LAGS, prior=prior)
    block = spec.compile_for_numpyro(
        exog=None, n_lags=N_LAGS, endog_names=VAR_NAMES, endog_scales=endog_scales, intercept_equations=()
    )
    return spec, block


spec, block = compile_var(TIGHTNESS, endog_scales)

# %% [markdown]
# ## The NumPyro model
#
# The model function declares the three free loadings, the four measurement-error standard deviations
# and the El Niño factor in every quarter. It passes the El Niño factor and demeaned real food commodity
# price growth to the block, which adds the priors on `B` and `L` and the VAR's joint density of both
# series, and then adds the log-density of the measurement equation {eq}`el-nino-measurement` at the four
# standardised climate indices.

# %%
def model(standardised_indices, demeaned_growth, block):
    n_quarters, n_indices = standardised_indices.shape
    with numpyro.plate("free_indices", n_indices - 1):
        free_loadings = numpyro.sample("free_loadings", dist.Normal(0.0, 1.0))
    error_standard_deviation = numpyro.sample(
        "measurement_error_standard_deviation", dist.HalfNormal(1.0).expand([n_indices]).to_event(1)
    )
    factor = numpyro.sample(FACTOR_NAME, dist.ImproperUniform(dist.constraints.real, (), (n_quarters,)))
    block(jnp.column_stack([factor, demeaned_growth]))  # the priors on B and L, and the VAR's joint density
    loadings = jnp.concatenate([jnp.ones(1), free_loadings])  # the loading of Niño 3 is fixed at 1
    numpyro.sample(
        "climate_indices",
        dist.Normal(factor[:, None] * loadings, error_standard_deviation).to_event(2),
        obs=standardised_indices,
    )


# %% [markdown]
# ## Sample
#
# We run NumPyro's NUTS with its defaults (target acceptance probability 0.8, diagonal mass matrix):
# 4 chains, run one after another, each with 1,000 warm-up draws and 2,000 kept draws.
#
# **Where the chains start.** With the loading of Niño 3 fixed at 1, the posterior can have a second
# mode, the **mode with reversed loading signs**: on it, the other three loadings have the opposite
# signs to the ones Brunner's description implies, and the El Niño factor is negatively correlated with
# Niño 3. Every chain starts the El Niño factor at standardised Niño 3, the climate index whose loading
# fixes the sign, and the three free loadings at their prior mean of 0. The VAR's parameters start at
# the block's `initial_point`, PyMC's initial point. The measurement-error standard deviations start at
# NumPyro's default, a uniform draw between −2 and 2 on the logarithmic scale, so between 0.14 and 7.4.
# A start changes where the chains begin, not the posterior.
#
# **What was changed after the first runs.** The first runs started the El Niño factor at 0 and kept
# 1,000 draws per chain, with every other setting as above. At tightness 0.1, one of those runs put one
# chain on the mode with reversed loading signs, which holds far less posterior probability than the
# main mode, and another run had a largest R-hat above 1.01. We then made the two changes described
# above, starting the El Niño factor at standardised Niño 3 and keeping 2,000 draws per chain, and reran
# every tightness with them. No prior and no part of the model changed.
#
# `sample_posterior`, `convergence` and `per_chain_signs` are used again, unchanged, in the section on
# sensitivity to the tightness.

# %%
KEY = 0  # the random key of every run in this tutorial
if ci:
    sampler_settings = {"num_warmup": 200, "num_samples": 50, "num_chains": 2}
else:
    sampler_settings = {"num_warmup": 1000, "num_samples": 2000, "num_chains": 4}


def sample_posterior(block, standardised_indices, demeaned_growth, sampler_settings, key):
    """Sample the model with NumPyro's NUTS; return the MCMC object and the wall time in seconds."""
    start = dict(block.initial_point)
    start["free_loadings"] = np.zeros(standardised_indices.shape[1] - 1)
    start[FACTOR_NAME] = standardised_indices[:, 0]  # standardised Niño 3
    kernel = NUTS(partial(model, block=block), init_strategy=init_to_value(values=start))
    mcmc = MCMC(kernel, **sampler_settings, chain_method="sequential", progress_bar=False)
    started = time.perf_counter()
    mcmc.run(jax.random.key(key), standardised_indices, demeaned_growth)
    return mcmc, time.perf_counter() - started


class Convergence(NamedTuple):
    """Convergence diagnostics of one run, over every sampled quantity."""

    sampling_seconds: float
    divergent_transitions: int
    transitions: int
    largest_r_hat: float
    quantity_with_largest_r_hat: str
    smallest_bulk_effective_sample_size: float
    smallest_tail_effective_sample_size: float


def convergence(mcmc, seconds):
    """Divergent transitions, largest R-hat and smallest effective sample sizes over every sampled quantity."""
    samples = mcmc.get_samples(group_by_chain=True)
    rows, cols = np.tril_indices(len(VAR_NAMES))
    draws = {
        "B": np.asarray(samples["B"]),
        "L": np.asarray(samples["L"])[:, :, rows, cols],  # the upper triangle of L is fixed at 0
        "free_loadings": np.asarray(samples["free_loadings"]),
        "measurement_error_standard_deviation": np.asarray(samples["measurement_error_standard_deviation"]),
        FACTOR_NAME: np.asarray(samples[FACTOR_NAME]),
    }
    summary = az.summary(draws, kind="diagnostics", round_to="none")
    divergent = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
    return Convergence(
        sampling_seconds=seconds,
        divergent_transitions=int(divergent.sum()),
        transitions=divergent.size,
        largest_r_hat=float(summary["r_hat"].max()),
        quantity_with_largest_r_hat=summary["r_hat"].idxmax(),
        smallest_bulk_effective_sample_size=float(summary["ess_bulk"].min()),
        smallest_tail_effective_sample_size=float(summary["ess_tail"].min()),
    )


mcmc, seconds = sample_posterior(block, standardised_indices, demeaned_growth, sampler_settings, KEY)
diagnostics = convergence(mcmc, seconds)
print(f"Sampling time: {diagnostics.sampling_seconds:.0f} seconds")
print(f"Divergent transitions: {diagnostics.divergent_transitions} of {diagnostics.transitions}")
print(f"Largest R-hat: {diagnostics.largest_r_hat:.4f} ({diagnostics.quantity_with_largest_r_hat})")
print(f"Largest R-hat below 1.01: {diagnostics.largest_r_hat < 1.01}")
print(f"Smallest bulk effective sample size: {diagnostics.smallest_bulk_effective_sample_size:.0f}")
print(f"Smallest bulk effective sample size at least 400: {diagnostics.smallest_bulk_effective_sample_size >= 400}")
print(f"Smallest tail effective sample size: {diagnostics.smallest_tail_effective_sample_size:.0f}")

# %% [markdown]
# The diagnostics above cover every sampled quantity: the VAR's lag coefficients `B`, the Cholesky
# factor `L` of the innovation covariance, the free loadings, the measurement-error standard deviations
# and the El Niño factor in all 146 quarters. A divergent transition is one in which NUTS's numerical
# integration failed; fewer is better, and any divergent transition deserves a look. R-hat compares the
# variance between chains with the variance within them; lower is better, and {cite:t}`vehtari2021`
# recommend using the draws only if it is below 1.01. The bulk and tail effective sample sizes are the
# number of independent draws that the correlated draws are worth, in the centre and in the tails of
# each posterior; higher is better, and {cite:t}`vehtari2021` recommend at least 100 per chain, 400 for
# these 4 chains.
#
# The statements on this page about results were checked by rerunning the sampling code with other
# random keys. In smoke-render builds of these docs the sampler runs only a few draws, so the numbers on
# this page are not representative there.
#
# The table below checks each chain for the mode with reversed loading signs. A chain on that mode would
# show a negative correlation between its mean El Niño factor and Niño 3, a negative loading of Niño 4
# or a positive loading of a pressure index; the last column marks such a chain.

# %%
class ChainSigns(NamedTuple):
    """Per-chain signs of one run: a table to display, and which chains are on the mode with reversed loading signs."""

    table: pd.DataFrame
    on_reversed_mode: np.ndarray


def per_chain_signs(samples, standardised_indices):
    """Each chain's correlation of its mean El Niño factor with standardised Niño 3, and its mean free loadings."""
    factor_draws = np.asarray(samples[FACTOR_NAME])  # (chain, draw, quarter)
    mean_loadings = np.asarray(samples["free_loadings"]).mean(axis=1)  # (chain, free loading): Niño 4, pressure indices
    correlation = np.array(
        [np.corrcoef(chain_factor.mean(axis=0), standardised_indices[:, 0])[0, 1] for chain_factor in factor_draws]
    )
    on_reversed_mode = (correlation < 0) | (mean_loadings[:, 0] < 0) | (mean_loadings[:, 1:] > 0).any(axis=1)
    table = pd.DataFrame(
        {
            "correlation of the chain's mean El Niño factor with Niño 3": correlation,
            **{f"mean loading of {label}": mean_loadings[:, i] for i, label in enumerate(INDEX_LABELS[1:])},
            "on the mode with reversed loading signs": on_reversed_mode,
        },
        index=pd.Index(range(len(correlation)), name="chain"),
    )
    return ChainSigns(table, on_reversed_mode)


samples = mcmc.get_samples(group_by_chain=True)
chain_signs = per_chain_signs(samples, standardised_indices)
print(f"Chains on the mode with reversed loading signs: {int(chain_signs.on_reversed_mode.sum())} of "
      f"{len(chain_signs.on_reversed_mode)}")
chain_signs.table.round(2)

# %% [markdown]
# ## The El Niño factor
#
# {numref}`fig-el-nino-factor` shows the posterior median of the El Niño factor with its **89 % interval**,
# the interval between the posterior's 5.5 % and 94.5 % quantiles, against standardised Niño 3. Every
# 89 % interval on this page is computed by `posterior_quantiles`, over every draw of every chain.

# %%
QUANTILE_LEVELS = [0.055, 0.5, 0.945]  # the lower end of the 89 % interval, the median, the upper end


def posterior_quantiles(draws):
    """The 5.5 %, 50 % and 94.5 % posterior quantiles of `draws` over its first two axes, chain and draw."""
    return np.quantile(draws, QUANTILE_LEVELS, axis=(0, 1))


def median_and_interval(draws, digits=2):
    """Posterior median with its 89 % interval, as text, for draws of one quantity with axes (chain, draw)."""
    low, mid, high = posterior_quantiles(draws)
    return f"{mid:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


factor_draws = np.asarray(samples[FACTOR_NAME])  # (chain, draw, quarter)
factor_quantiles = posterior_quantiles(factor_draws)

# %% mystnb={"figure": {"caption": "Posterior median and 89 % interval of the El Niño factor, in standard deviations of Niño 3, with standardised Niño 3 as a dashed line.", "name": "fig-el-nino-factor"}, "image": {"alt": "Posterior median and 89% interval of the El Niño factor from 1961 to 1997, with standardised Niño 3 as a dashed line that stays close to the median."}}
fig, ax = plt.subplots()
ax.fill_between(quarters, factor_quantiles[0], factor_quantiles[2], color=plotting.COLORS.oxblood, alpha=0.3,
                linewidth=0, label="El Niño factor, 89 % interval")
ax.plot(quarters, factor_quantiles[1], color=plotting.COLORS.oxblood, linewidth=1.2, label="El Niño factor, median")
ax.plot(quarters, standardised_indices[:, 0], color=plotting.COLORS.ink, linewidth=0.7, linestyle="--",
        label="Niño 3, standardised")
ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
ax.set_ylabel("Standard deviations of Niño 3")
plotting.serif_title("The El Niño factor follows Niño 3 closely", ax)
_ = plotting.legend_below(ax)

# %%
median_factor_correlation = np.corrcoef(factor_quantiles[1], standardised_indices[:, 0])[0, 1]
print(f"Correlation of the posterior median El Niño factor with standardised Niño 3: {median_factor_correlation:.3f}")

# %% [markdown]
# The table below gives, for each climate index, its loading, its measurement-error standard deviation,
# and the share of its variance over 1961Q3 to 1997Q4 that the El Niño factor explains,
# $\lambda_k^2 \operatorname{var}(f) / (\lambda_k^2 \operatorname{var}(f) + \psi_k^2)$, computed draw by
# draw. Each entry is the posterior median with its 89 % interval in brackets. Loadings are in standard
# deviations of the climate index per standard deviation of Niño 3, and measurement-error standard
# deviations in standard deviations of the climate index. A share near 1 means the El Niño factor
# explains that climate index almost entirely.

# %%
loading_draws = np.asarray(samples["free_loadings"])  # (chain, draw, free loading)
all_loadings = np.concatenate([np.ones((*loading_draws.shape[:2], 1)), loading_draws], axis=-1)
error_standard_deviation_draws = np.asarray(samples["measurement_error_standard_deviation"])
factor_variance = factor_draws.var(axis=-1)[..., None]
explained = all_loadings**2 * factor_variance / (all_loadings**2 * factor_variance + error_standard_deviation_draws**2)
pd.DataFrame(
    {
        "loading": ["1 (fixed)"] + [median_and_interval(loading_draws[..., i]) for i in range(3)],
        "measurement-error standard deviation": [
            median_and_interval(error_standard_deviation_draws[..., i]) for i in range(4)
        ],
        "share of variance the El Niño factor explains": [median_and_interval(explained[..., i]) for i in range(4)],
    },
    index=INDEX_LABELS,
)

# %% [markdown]
# How precisely do the four climate indices determine the El Niño factor? The first number below is the
# posterior standard deviation of the El Niño factor in one quarter (the median over the 146 quarters).
# The second is the standard deviation of the El Niño factor's innovation $\varepsilon_{1t}$ in the VAR,
# $L_{11}$. Both are in standard deviations of Niño 3. A ratio below 1 means the climate indices
# determine each quarter's value of the El Niño factor more precisely than the size of one quarter's
# innovation.

# %%
factor_standard_deviation = np.median(factor_draws.std(axis=(0, 1)))
innovation_standard_deviation = np.median(np.asarray(samples["L"])[..., 0, 0])
print(f"Posterior standard deviation of the El Niño factor in one quarter: {factor_standard_deviation:.2f}")
print(f"Standard deviation of the El Niño factor's innovation, posterior median: {innovation_standard_deviation:.2f}")
print(f"Ratio: {factor_standard_deviation / innovation_standard_deviation:.2f}")

# %% [markdown]
# A VAR is stationary when every eigenvalue of its companion matrix has modulus below 1. The block adds
# no stationarity term here, so the cell counts the posterior draws that are not stationary.

# %%
n_vars = len(VAR_NAMES)
B_draws = np.asarray(samples["B"]).reshape(-1, n_vars, n_vars * N_LAGS)
companion = np.zeros((len(B_draws), n_vars * N_LAGS, n_vars * N_LAGS))
companion[:, :n_vars] = B_draws
companion[:, n_vars:, :-n_vars] = np.eye(n_vars * (N_LAGS - 1))
largest_modulus = np.abs(np.linalg.eigvals(companion)).max(axis=-1)
print(f"Draws that are not stationary: {int((largest_modulus >= 1).sum())} of {len(largest_modulus)}")
print(f"Largest eigenvalue modulus, posterior median: {np.median(largest_modulus):.3f}")

# %% [markdown]
# ## Hand the posterior to `FittedVAR`
#
# `block.posterior_from_samples` turns the samples into a posterior, and `FittedVAR.from_posterior`
# wraps it. `FittedVAR.from_posterior` also needs a `VARData` holding the series the block received.
# The block received a different El Niño factor at every draw, so `identify` passes its posterior mean.
#
# Under Cholesky identification, impulse responses and forecast-error variance shares are computed only
# from the lag coefficients `B` and the Cholesky factor `L`, not from this column. Each draw of `B` and
# `L` was drawn jointly with a draw of the El Niño factor, so their intervals include the uncertainty
# about the El Niño factor. Results that use the VAR's residuals, such as a historical decomposition,
# would use the posterior-mean column and would leave that uncertainty out.

# %%
def identify(spec, block, samples, demeaned_growth, quarters):
    """Wrap the samples in a `FittedVAR` and identify it with a Cholesky ordering, El Niño factor first."""
    posterior = block.posterior_from_samples(samples)
    factor_mean = np.asarray(samples[FACTOR_NAME]).mean(axis=(0, 1))
    data = VARData(endog=np.column_stack([factor_mean, demeaned_growth]), endog_names=VAR_NAMES, index=quarters)
    fitted = FittedVAR.from_posterior(posterior, data, n_lags=N_LAGS, error_dist=spec.resolved_error_dist)
    return fitted.set_identification_strategy(Cholesky(ordering=VAR_NAMES))


identified = identify(spec, block, samples, demeaned_growth, quarters)

# %% [markdown]
# ## Impulse responses to an El Niño shock
#
# `identify` orders the El Niño factor first, as Brunner orders his El Niño measure (p. 8): an El Niño
# shock can move real food commodity price growth in the quarter it happens, and a shock to real food
# commodity price growth cannot move the El Niño factor in that quarter. The **El Niño shock** is the El
# Niño factor's innovation $\varepsilon_{1t}$, and we compute the responses to a one-standard-deviation
# El Niño shock over 16 quarters:
#
# - the **El Niño factor**, in standard deviations of Niño 3;
# - **real food commodity price growth, annualised**: 4 times the response in percent per quarter, in
#   percentage points, to compare with Brunner, whose inflation rates we take to be annualised (see
#   [Comparison with Brunner](#comparison-with-brunner));
# - the **real food commodity price level**, in percent: in each draw, the sum of that draw's responses
#   of real food commodity price growth, in percent per quarter, up to each quarter. Its 89 % intervals
#   are computed from these summed draws; they are not the sums of the intervals of the growth
#   responses.
#
# The figures mark Brunner's response two quarters after the shock. In the code, Brunner's numbers are
# typed once, in the cell below, and every table, check and figure on this page takes them from there.
#
# The table below gives the posterior median and 89 % interval at selected quarters after the shock.
# {numref}`fig-el-nino-growth-response` and {numref}`fig-el-nino-level-response` plot every quarter.

# %%
HORIZON = 16

# Brunner (1998). The response of real commodity price inflation, all primary commodities, to a
# one-standard-deviation El Niño shock two quarters after it, in percentage points (p. 9):
BRUNNER_ALL_COMMODITIES_RESPONSE = (3.5, 4.0)
# The number of quarters after the shock for which the shock raises his El Niño measures (p. 9):
BRUNNER_QUARTERS_RAISED = 4
# The percentage of the 4- and 16-quarter-ahead forecast error variance of real food commodity price
# inflation attributable to El Niño, with each of his El Niño measures (Tables 2a and 2b):
BRUNNER_FOOD_SHARES = pd.DataFrame(
    {4: [14.3, 14.7], 16: [19.6, 23.9]},
    index=pd.Index(["sea-surface temperature", "Southern Oscillation Index"], name="Brunner's El Niño measure"),
)


class Responses(NamedTuple):
    """Draws of the responses to a one-standard-deviation El Niño shock, each with axes (chain, draw, quarter)."""

    factor: np.ndarray  # standard deviations of Niño 3
    growth_annualised: np.ndarray  # percentage points
    price_level: np.ndarray  # %


def el_nino_responses(identified):
    """The responses of the El Niño factor, of annualised growth and of the price level to an El Niño shock."""
    irf = identified.impulse_response(horizon=HORIZON).idata.posterior_predictive["irf"].sel(shock=FACTOR_NAME)
    quarterly_growth = irf.sel(response=GROWTH_NAME).values  # % per quarter
    return Responses(
        factor=irf.sel(response=FACTOR_NAME).values,
        growth_annualised=4 * quarterly_growth,
        price_level=np.cumsum(quarterly_growth, axis=-1),  # summed draw by draw
    )


def plot_band(ax, draws, color, label):
    """Plot the posterior median and 89 % interval of a response over the quarters after the shock."""
    low, mid, high = posterior_quantiles(draws)
    ax.fill_between(range(HORIZON + 1), low, high, color=color, alpha=0.3, linewidth=0, label=f"{label}, 89 % interval")
    ax.plot(range(HORIZON + 1), mid, color=color, linewidth=1.4, label=f"{label}, median")


def plot_brunner_response(ax):
    """Mark Brunner's response two quarters after the shock, for all primary commodities, as a bar."""
    low, high = BRUNNER_ALL_COMMODITIES_RESPONSE
    ax.vlines(2, low, high, color=plotting.COLORS.ink, linewidth=4, label="Brunner (1998), all primary commodities")


responses = el_nino_responses(identified)
TABLE_QUARTERS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16]
pd.DataFrame(
    {
        "El Niño factor (standard deviations of Niño 3)": [
            median_and_interval(responses.factor[..., h]) for h in TABLE_QUARTERS
        ],
        "real food commodity price growth, annualised (percentage points)": [
            median_and_interval(responses.growth_annualised[..., h], 1) for h in TABLE_QUARTERS
        ],
        "real food commodity price level (%)": [
            median_and_interval(responses.price_level[..., h], 1) for h in TABLE_QUARTERS
        ],
    },
    index=pd.Index(TABLE_QUARTERS, name="quarters after the shock"),
)

# %% mystnb={"figure": {"caption": "Response of real food commodity price growth, annualised, to a one-standard-deviation El Niño shock: posterior median and 89 % interval. The black bar marks Brunner's response two quarters after the shock for all primary commodities; he reports no response for food.", "name": "fig-el-nino-growth-response"}, "image": {"alt": "Annualised real food commodity price growth response over 16 quarters after an El Niño shock: the median is below zero in the shock quarter and above zero in quarters 1 to 4, with Brunner's all-commodity response marked as a bar two quarters after the shock."}}
fig, ax = plt.subplots()
plot_band(ax, responses.growth_annualised, plotting.COLORS.oxblood, "real food commodity price growth")
plot_brunner_response(ax)
ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
ax.set_xlabel("Quarters after the shock")
ax.set_ylabel("Percentage points, annualised")
plotting.serif_title("Real food commodity price growth falls in the shock quarter, then rises", ax)
_ = plotting.legend_below(ax)

# %% mystnb={"figure": {"caption": "Response of the real food commodity price level to a one-standard-deviation El Niño shock, in percent: posterior median and 89 % interval of the summed responses of real food commodity price growth.", "name": "fig-el-nino-level-response"}, "image": {"alt": "Real food commodity price level response over 16 quarters after an El Niño shock: the median is below zero in the shock quarter and above zero from quarter 2 to quarter 16."}}
fig, ax = plt.subplots()
plot_band(ax, responses.price_level, plotting.COLORS.oxblood, "real food commodity price level")
ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
ax.set_xlabel("Quarters after the shock")
ax.set_ylabel("%")
plotting.serif_title("The real food commodity price level falls in the shock quarter, then rises", ax)
_ = plotting.legend_below(ax)

# %% [markdown]
# Under the Cholesky ordering, the response of real food commodity price growth in the shock quarter is
# $L_{21}$, the second row and first column of the Cholesky factor of $\Sigma$: the covariance of the
# two innovations divided by the standard deviation of the El Niño factor's innovation. Its sign is the
# sign of the correlation of the two innovations, printed below. A negative correlation means that in
# quarters when the El Niño factor was higher than the VAR predicted from the past, real food commodity
# price growth tended to be lower than the VAR predicted.

# %%
L_draws = np.asarray(samples["L"])
innovation_covariance = L_draws @ np.swapaxes(L_draws, -1, -2)
innovation_correlation = innovation_covariance[..., 1, 0] / np.sqrt(
    innovation_covariance[..., 0, 0] * innovation_covariance[..., 1, 1]
)
print(f"Correlation of the two innovations: {median_and_interval(innovation_correlation)}")

# %% [markdown]
# ## Forecast-error variance shares
#
# The **forecast-error variance share** of El Niño shocks at $k$ quarters ahead is the share of the
# variance of the $k$-quarter-ahead forecast error of real food commodity price growth that El Niño
# shocks explain, in percent. `IdentifiedVAR.fevd` computes it in each posterior draw from that draw's
# lag coefficients `B` and Cholesky factor `L`, so it is the share that the VAR implies for its forecast
# errors. The table below gives it for $k$ = 4 and 16 quarters, the horizons of Brunner's Tables 2a and
# 2b.

# %%
def el_nino_forecast_error_variance_shares(identified):
    """Draws of El Niño shocks' forecast-error variance share for real food commodity price growth, in %."""
    fevd = identified.fevd(horizon=HORIZON).idata.posterior_predictive["fevd"]
    shares = 100 * fevd.sel(response=GROWTH_NAME, shock=FACTOR_NAME)
    return {k: shares.sel(horizon=k - 1).values for k in (4, 16)}  # horizon h holds the (h + 1)-quarter-ahead error


forecast_error_variance_shares = el_nino_forecast_error_variance_shares(identified)
pd.Series(
    {f"{k} quarters ahead": median_and_interval(draws, 1) for k, draws in forecast_error_variance_shares.items()},
    name="forecast-error variance share of El Niño shocks (%)",
)

# %% [markdown]
# ## Comparison with Brunner
#
# Brunner reports that a one-standard-deviation El Niño shock "raises commodity price inflation (in real
# terms) an estimated 3-1/2 to 4 percentage points two quarters after the initial surprise", and that
# "commodity prices fall by a similar amount in the second and third years after the initial surprise"
# (p. 9). Both statements are about the IMF index of all primary commodities: his Figures 4a
# (sea-surface temperature) and 4b (Southern Oscillation Index) are the only impulse responses he shows,
# and both are for that index. For food he reports no impulse response. He writes: "A similar pattern
# is evident for other measures of commodity prices (not shown). Not surprisingly, prices for food
# commodities and agricultural raw materials are affected the most by El Niño" (pp. 9–10). So this tutorial compares a response of
# food with a response of all primary commodities, and a match is weak evidence of agreement. Our
# inference from his remark, not a number he reports, is that his response of food would be larger than
# his response of all primary commodities.
#
# He also reports that an El Niño shock raises his El Niño measures "for four subsequent quarters after
# the initial surprise" (p. 9).
#
# His Tables 2a and 2b do report food: the "Percentage of K-step-ahead Forecast Error Variance
# Attributable to El Niño" for real food commodity price inflation, 4 and 16 quarters ahead. He computed
# them differently from this tutorial's forecast-error variance shares. Each of his VARs "was used to
# calculate k-step-ahead dynamic forecasts … for each time period in the data sample", the "resulting
# forecast errors were then decomposed into the portion attributable to El Niño … and the portion
# attributable to all other factors", and "the variances of these forecast errors were also decomposed in
# a similar manner" (pp. 10–11). His percentages therefore describe the forecast errors his VARs made in
# his sample. This tutorial's describe the forecast errors that the VAR implies, from each draw's
# coefficients and innovation covariance. The comparison below sets these two different statistics side
# by side.
#
# Brunner does not say whether his inflation rates are annualised. His Figure 3 plots G-7 consumer
# price inflation on an axis that runs to 30 %, which fits annualised quarterly rates and not rates per
# quarter, so we compare annualised rates.
#
# The cell below sets Brunner's numbers beside this run's, and prints, for each comparison, whether
# this run's 89 % interval contains Brunner's value or lies on one side of 0. "Years two and three"
# are quarters 5 to 12 after the shock, so the change in the real food commodity price level over them
# is its response in quarter 12 minus its response in quarter 4, computed draw by draw. For the El Niño
# factor, the cell prints the median and the lower end of the 89 % interval of its response in each of
# the four quarters after the shock.

# %%
growth_responses = responses.growth_annualised
level_change_years_two_and_three = responses.price_level[..., 12] - responses.price_level[..., 4]
lowest_quarter = 5 + int(np.argmin(np.median(growth_responses[..., 5:13], axis=(0, 1))))
brunner_low, brunner_high = BRUNNER_ALL_COMMODITIES_RESPONSE


def brunner_shares_text(k):
    """Brunner's percentages for food, k quarters ahead, with the El Niño measure of each."""
    shares = ", ".join(f"{share} ({measure})" for measure, share in BRUNNER_FOOD_SHARES[k].items())
    return f"{shares}; his VARs' forecast errors in his sample"


comparison = pd.DataFrame(
    {
        "Brunner (1998)": [
            f"+{brunner_low} to +{brunner_high} (all primary commodities; no number for food)",
            "not discussed",
            "prices 'fall by a similar amount' (p. 9)",
            "prices 'fall by a similar amount' (p. 9)",
            brunner_shares_text(4),
            brunner_shares_text(16),
        ],
        "this run (food)": [
            median_and_interval(growth_responses[..., 2], 1),
            median_and_interval(growth_responses[..., 0], 1),
            f"{median_and_interval(growth_responses[..., lowest_quarter], 1)} in quarter {lowest_quarter}",
            median_and_interval(level_change_years_two_and_three, 1),
            median_and_interval(forecast_error_variance_shares[4], 1),
            median_and_interval(forecast_error_variance_shares[16], 1),
        ],
    },
    index=[
        "real food commodity price growth 2 quarters after the shock, annualised (percentage points)",
        "real food commodity price growth in the shock quarter, annualised (percentage points)",
        "lowest median real food commodity price growth in quarters 5 to 12, annualised (percentage points)",
        "change in the real food commodity price level over quarters 5 to 12 (%)",
        "forecast-error variance share, 4 quarters ahead (%)",
        "forecast-error variance share, 16 quarters ahead (%)",
    ],
)
low, _, high = posterior_quantiles(growth_responses[..., 2])
print(f"Brunner's {brunner_low} to {brunner_high} percentage points lie inside this run's 89 % interval of real "
      f"food commodity price growth 2 quarters after the shock: {bool(low <= brunner_low and high >= brunner_high)}")
print("89 % interval of real food commodity price growth in the shock quarter lies below 0: "
      f"{bool(posterior_quantiles(growth_responses[..., 0])[2] < 0)}")
print("89 % interval of the change in the real food commodity price level over quarters 5 to 12 lies below 0: "
      f"{bool(posterior_quantiles(level_change_years_two_and_three)[2] < 0)}")
factor_low, factor_median, _ = posterior_quantiles(responses.factor[..., 1 : BRUNNER_QUARTERS_RAISED + 1])
print(f"El Niño factor's response in quarters 1 to {BRUNNER_QUARTERS_RAISED} after the shock, median: "
      f"{np.round(factor_median, 2)}; lower end of the 89 % interval: {np.round(factor_low, 2)}")
for k in (4, 16):
    low, _, high = posterior_quantiles(forecast_error_variance_shares[k])
    for measure, share in BRUNNER_FOOD_SHARES[k].items():
        print(f"Brunner's {share} % ({measure}, {k} quarters ahead) lies inside this run's 89 % interval of the "
              f"forecast-error variance share: {bool(low <= share <= high)}")
comparison

# %% [markdown]
# Three of the checks above bear on Brunner's description of the response. Brunner's range lies inside
# the 89 % interval of the response of food two quarters after the shock; for the reason given above,
# this is weak evidence of agreement. The 89 % interval of the response in the shock quarter lies below
# 0. Brunner's text does not discuss the response in the shock quarter, so this tutorial cannot say
# whether his VARs show a fall in that quarter, and it does not explain the fall. The 89 % interval of
# the change in the real food commodity price level over years two and three contains 0, so the
# posterior does not place the fall that Brunner describes for all primary commodities clearly below 0.
#
# The median response of the El Niño factor is above 0 in each of the four quarters after the shock, as
# Brunner reports for his El Niño measures, and the lower end of its 89 % interval falls close to 0 by
# the fourth quarter.
#
# The two models differ in their data and their specification: the World Bank food price index
# deflated by the US consumer price index against the IMF food price index net of G-7 consumer price
# inflation; an El Niño factor measured by four climate indices against one index at a time; two series
# against four; no constants against seasonal constants; and a Minnesota prior against least squares.
# This tutorial tests only one of these differences, the tightness of the prior, in the next section.
#
# ## Sensitivity to the prior's tightness
#
# Before estimation we declared a set of tightness values for this check: 0.1, 0.2 and 0.5. A larger
# `tightness` gives larger prior standard deviations, so 0.1 is the most restrictive prior of the three
# and 0.5 the loosest. The cell below fits the model at 0.1 and 0.5 with `compile_var`,
# `sample_posterior`, `convergence`, `per_chain_signs` and `identify`, so with the same start, sampler
# settings and key as above, and compares the three, including each fit's chains on the mode with
# reversed loading signs.

# %%
class Fit(NamedTuple):
    """One fit of the model: diagnostics, chains on the mode with reversed loading signs, and results."""

    diagnostics: Convergence
    chains_on_reversed_mode: int
    responses: Responses
    forecast_error_variance_shares: dict


fits = {
    TIGHTNESS: Fit(diagnostics, int(chain_signs.on_reversed_mode.sum()), responses, forecast_error_variance_shares)
}
for tightness in (0.1, 0.5):
    other_spec, other_block = compile_var(tightness, endog_scales)
    other_mcmc, other_seconds = sample_posterior(
        other_block, standardised_indices, demeaned_growth, sampler_settings, KEY
    )
    other_samples = other_mcmc.get_samples(group_by_chain=True)
    other_identified = identify(other_spec, other_block, other_samples, demeaned_growth, quarters)
    fits[tightness] = Fit(
        diagnostics=convergence(other_mcmc, other_seconds),
        chains_on_reversed_mode=int(per_chain_signs(other_samples, standardised_indices).on_reversed_mode.sum()),
        responses=el_nino_responses(other_identified),
        forecast_error_variance_shares=el_nino_forecast_error_variance_shares(other_identified),
    )

SENSITIVITY_QUARTERS = [(0, "in the shock quarter"), (2, "2 quarters after the shock"), (10, "10 quarters after the shock")]


def sensitivity_column(fit):
    """The rows of the sensitivity table for one fit, in the order of `sensitivity_rows`."""
    return [
        fit.diagnostics.divergent_transitions,
        round(fit.diagnostics.largest_r_hat, 4),
        fit.diagnostics.largest_r_hat < 1.01,
        round(fit.diagnostics.smallest_bulk_effective_sample_size),
        fit.chains_on_reversed_mode,
        round(fit.diagnostics.sampling_seconds),
        *[median_and_interval(fit.responses.growth_annualised[..., h], 1) for h, _ in SENSITIVITY_QUARTERS],
        median_and_interval(fit.responses.price_level[..., 8], 1),
        median_and_interval(fit.forecast_error_variance_shares[16], 1),
    ]


sensitivity_rows = [
    "divergent transitions",
    "largest R-hat",
    "largest R-hat below 1.01",
    "smallest bulk effective sample size",
    "chains on the mode with reversed loading signs",
    "sampling time (seconds)",
    *[f"real food commodity price growth {when}, annualised (percentage points)" for _, when in SENSITIVITY_QUARTERS],
    "real food commodity price level 8 quarters after the shock (%)",
    "forecast-error variance share, 16 quarters ahead (%)",
]
medians_two_quarters_after = [np.median(fits[t].responses.growth_annualised[..., 2]) for t in sorted(fits)]
print("Median real food commodity price growth 2 quarters after the shock rises with the tightness: "
      f"{bool(np.all(np.diff(medians_two_quarters_after) > 0))}")
pd.DataFrame({t: sensitivity_column(fits[t]) for t in sorted(fits)}, index=sensitivity_rows).rename_axis(
    columns="tightness"
)

# %% [markdown]
# {numref}`fig-el-nino-tightness` plots the response of real food commodity price growth at each
# tightness, with the 89 % interval at tightness 0.2.

# %% mystnb={"figure": {"caption": "Response of real food commodity price growth, annualised, to a one-standard-deviation El Niño shock, at Minnesota tightness 0.1, 0.2 and 0.5: posterior medians, with the 89 % interval at tightness 0.2.", "name": "fig-el-nino-tightness"}, "image": {"alt": "Annualised real food commodity price growth response over 16 quarters at Minnesota tightness 0.1, 0.2 and 0.5; at every tightness the median is below zero in the shock quarter and above zero two quarters after it, and the median two quarters after the shock is larger at a looser prior."}}
fig, ax = plt.subplots()
for tightness, color in zip(sorted(fits), plotting.COLORS.cycle[1:], strict=False):
    draws = fits[tightness].responses.growth_annualised
    if tightness == TIGHTNESS:
        plot_band(ax, draws, plotting.COLORS.oxblood, f"tightness {tightness}")
    else:
        ax.plot(range(HORIZON + 1), np.median(draws, axis=(0, 1)), color=color, linewidth=1.2, linestyle="--",
                label=f"tightness {tightness}, median")
plot_brunner_response(ax)
ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
ax.set_xlabel("Quarters after the shock")
ax.set_ylabel("Percentage points, annualised")
plotting.serif_title("A looser prior gives a larger response two quarters after the shock", ax)
_ = plotting.legend_right(ax)

# %% [markdown]
# The prior standard deviations of the coefficients on the lagged El Niño factor in the equation of real
# food commodity price growth are proportional to the tightness, as printed in
# [The priors, declared before estimation](#the-priors-declared-before-estimation), and the median
# response two quarters after the shock rises with the tightness, as printed above.
#
# We keep tightness 0.2 for the results above because it was declared before estimation, not because of
# its result. Choosing the tightness by how close its response comes to Brunner's would make the
# comparison with Brunner circular.
#
# ## Limitations
#
# - **Gaussian errors and constant volatility.** The standard deviation of real food commodity price
#   growth differs by decade (printed in [The data](#the-data)), and the VAR assumes one innovation
#   covariance for the whole sample, with Gaussian innovations.
# - **Two series.** Brunner identifies El Niño shocks in VARs that also hold G-7 inflation and growth;
#   this VAR holds two series.
# - **A posterior-mean El Niño factor in residual-based results.** Historical decompositions and other
#   results that use the VAR's residuals would use the posterior mean of the El Niño factor and leave
#   out its uncertainty.
# - **Standardisation over the whole sample.** The climate indices are standardised with their means and
#   standard deviations over 1961Q3 to 1997Q4, which suits an analysis of past data like this one but
#   not a forecast made in real time.
# - **One sample.** The responses describe 1963 to 1997, Brunner's sample. The data file runs to
#   2024Q4, so the same model can be fitted on a longer sample by changing the dates of `sample`.
#
# ## Summary
#
# 1. **Measure** an unobserved series with several indices: the measurement equation is ordinary NumPyro
#    code, and fixing one loading at 1 sets the series' sign and scale.
# 2. **Sample the series in the centred form**: declare its values as a NumPyro sample site with a flat
#    base measure and pass them, as the first endogenous series, to a VAR compiled with
#    `VAR.compile_for_numpyro` and no `latent_names`. The block then adds the priors on `B` and `L` and
#    the VAR's joint density of that series and the others.
# 3. **Start the chains** at the climate index whose loading fixes the sign, and check every chain of
#    every fit for the mode with reversed loading signs.
# 4. **Use one posterior**: impulse responses and forecast-error variance shares under Cholesky
#    identification include the uncertainty about the El Niño factor, because `B` and `L` were drawn
#    jointly with it.
# 5. **Declare the prior before estimation**, and report the sensitivity to it rather than tuning it.
#
# ## Data sources and licences
#
# - **Niño 1+2, Niño 3 and Niño 4**: NOAA Climate Prediction Center, Niño-region
#   sea-surface-temperature anomalies from the Extended Reconstructed Sea Surface Temperature, version 5
#   {cite:p}`huang2017`, 1991–2020 base period, file
#   [`ersst5.nino.mth.91-20.ascii`](https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii).
#   **Southern Oscillation Index**: NOAA Climate Prediction Center, file
#   [`soi`](https://www.cpc.ncep.noaa.gov/data/indices/soi), standardised table. **Equatorial Southern
#   Oscillation Index**: NOAA Climate Prediction Center, file
#   [`reqsoi.for`](https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for). These five series are NOAA
#   National Weather Service material, which is in the public domain and not subject to copyright
#   protection ([National Weather Service disclaimer](https://www.weather.gov/disclaimer)). We averaged
#   them to quarters and standardised them.
# - **Food price index**: World Bank, Commodity Price Data (the Pink Sheet), monthly Food price index,
#   from [World Bank Commodity Markets](https://www.worldbank.org/en/research/commodity-markets),
#   licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). We
#   changed the series: we averaged it to quarters, deflated it by the consumer price index, and took
#   100 times the quarterly change in its logarithm.
# - **Consumer price index**: US Bureau of Labor Statistics, consumer price index for all urban
#   consumers, all items in U.S. city average, seasonally adjusted (series CUSR0000SA0), which is in the
#   public domain. Source: Bureau of Labor Statistics.
#
# The works cited above are collected on the [project bibliography](../references.md) page.
