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
# # Does El Niño Still Raise Food Prices?

# %% tags=["remove-cell"]
# A notebook displays a cell's last expression (B018), imports after this hidden cell (E402) and uses
# typographic en dashes in its text (RUF001, RUF003).
# ruff: noqa: B018, E402, RUF001, RUF003
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
# In April 1998, as one of the strongest El Niños on record was ending, the Federal Reserve
# economist Allan Brunner asked whether El Niño moves world commodity prices or whether the idea
# was hot air {cite:p}`brunner1998`. His answer was warm water. In quarterly data from 1963 to
# 1997, a typical El Niño surprise raised real commodity price inflation by 3.5 to 4 percentage
# points two quarters later, and El Niño accounted for about a fifth of the movements in
# commodity price inflation.
#
# The years since look different. Real food prices fell during the two strongest El Niños
# that followed, in 2015–16 and 2023–24. The three large food price surges of the period, in
# 2007–08, 2010–11 and 2020–22, each came while the Pacific was colder than normal, the
# opposite phase, called La Niña.
#
# This case study asks whether Brunner's result still holds. It does not. But El Niño has not
# stopped moving food prices: the effect has changed shape. We rebuild his analysis as one
# statistical model, fit it to his years and to the years since, and then change one thing at a
# time. Four findings come out.
#
# 1. **The rise is there in Brunner's years.** In 1963–1997 real food prices are about 1.7 %
#    higher a year after a typical El Niño surprise. Two quarters after the surprise, price
#    growth is about 4 percentage points higher at an annual rate, as he reported.
# 2. **It is gone in the years since.** In 1999–2024 the same model, unchanged, puts real food
#    prices about 1.2 % *lower* a year after the surprise. The probability that the response
#    fell between the two periods is about 0.95.
# 3. **La Niña is not El Niño in reverse.** When the model lets the two phases differ, an El
#    Niño surprise raises prices by about 3.4 % before 1998 and a La Niña surprise does nothing.
#    After 1998 the roles are swapped: La Niña raises prices by about 2.7 % and El Niño does
#    nothing.
# 4. **Today the effect shows between crops.** After 1998 an El Niño surprise makes palm oil
#    and rice about 3.8 % dearer relative to soybeans, maize and wheat within a year, and the
#    probability that this relative price rises is 0.99. La Niña does the opposite.
#
# Read together, the four say that El Niño and La Niña still move food prices. They now push
# crops grown on opposite sides of the Pacific in opposite directions, and an index that
# averages those crops shows little. Brunner's number describes 1963–1997. It is not a forecast
# for today.
#
# The model treats El Niño as a series that nobody observes directly. Four published climate
# indices each measure it with error, and the model infers the hidden series from all four.
# That series and the growth of real food prices then follow a **vector autoregression** (VAR),
# a model that predicts each series from the recent past of both. The VAR is Impulso's. The
# measurement of El Niño is a few lines of [NumPyro](https://num.pyro.ai/) written around it.
# [A VAR Inside a NumPyro Model](var-in-numpyro.py) explains how the two fit together; this
# page uses that mechanism and keeps to the question.
#
# The code needs JAX and NumPyro, which the core install leaves out:
# `pip install "impulso[numpyro]"`.

# %%
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
# ## El Niño, La Niña and the price of food
#
# **El Niño** is a warming of the surface of the tropical Pacific Ocean that returns every few
# years and lasts about a year. **La Niña** is the cold opposite phase. Both move rainfall
# around the globe. El Niño tends to bring drought to Southeast Asia, India and Australia, and
# La Niña to parts of the Americas. Harvests depend on that rain, so a link from the Pacific to
# the price of food is plausible. Brunner's paper, published as {cite:t}`brunner2002`, put a
# number on it.
#
# No single number measures El Niño. Forecasters watch the temperature of the sea surface in
# several regions of the Pacific and the difference in air pressure across it. Brunner used one
# temperature series and one pressure series, and estimated every model twice. We use four
# monthly indices from the NOAA Climate Prediction Center, described in
# {numref}`tbl-el-nino-climate-indices`. During an El Niño the two **temperature indices** rise
# and the two **pressure indices** fall.
#
# :::{table} The four climate indices.
# :name: tbl-el-nino-climate-indices
#
# | Climate index | What it measures | Unit |
# |---------------|------------------|------|
# | **Niño 3** | How far the sea surface of the eastern equatorial Pacific (5°N–5°S, 150°W–90°W) is from its usual temperature | °C |
# | **Niño 4** | The same for the central equatorial Pacific (5°N–5°S, 160°E–150°W) | °C |
# | **Southern Oscillation Index** | Air pressure at Tahiti minus air pressure at Darwin, Australia | standard deviations |
# | **Equatorial Southern Oscillation Index** | Air pressure over the eastern equatorial Pacific minus air pressure over Indonesia | standard deviations |
# :::
#
# The price of food is the World Bank's **food price index**: the world prices, in US dollars,
# of cereals, vegetable oils and other foods traded in bulk. These are the prices a grain
# trader pays, not the prices in a shop. We divide the index by the US consumer price index to
# remove general inflation and call the result the **real food price**. Its **growth** is its
# percentage change from one quarter to the next, computed as 100 times the change in its
# logarithm. The data file also holds the World Bank's prices of five crops, which the last
# fit uses. All series are quarterly averages; the script
# `scripts/fetch_el_nino_commodity_prices.py` in the Impulso repository downloads them, and
# [Data sources and licences](#data-sources-and-licences) lists where they come from.

# %%
INDEX_NAMES = ["nino3", "nino4", "southern_oscillation_index", "equatorial_southern_oscillation_index"]
INDEX_LABELS = ["Niño 3", "Niño 4", "Southern Oscillation Index", "Equatorial Southern Oscillation Index"]
FACTOR_NAME = "el_nino_factor"
PRICE_NAME = "price_growth"
VAR_NAMES = [FACTOR_NAME, PRICE_NAME]
N_LAGS = 6
SAMPLES = {"1963–1997": ("1961-07-01", "1997-10-01"), "1999–2024": ("1997-07-01", "2024-10-01")}

panel = pd.read_csv("data/el_nino_commodity_prices.csv", index_col="quarter_start", parse_dates=True)
real_food_price = 100 * np.log(panel["food_price_index"] / panel["consumer_price_index"])
panel.loc["1961-07-01":, [*INDEX_NAMES, "food_price_index", "consumer_price_index"]].head()

# %% [markdown]
# {numref}`fig-el-nino-data` shows the data. In the top panel the two pressure indices are
# turned upside down, and the four lines then move almost as one: they are four measurements of
# the same thing. The bottom panel is the real food price. Red bands mark quarters in which the
# average of the four lines is more than one standard deviation above normal, an El Niño. Blue
# bands mark quarters in which it is more than one below, a La Niña. The dashed line is the end
# of Brunner's sample.

# %% mystnb={"figure": {"caption": "Top: the four climate indices from 1961Q3 to 2024Q4, each rescaled to mean 0 and standard deviation 1, with the two pressure indices reversed. Bottom: the real food price, on a logarithmic scale. Red bands are El Niño quarters and blue bands are La Niña quarters: those in which the average of the four lines is more than one standard deviation from normal. The dashed line marks the end of Brunner's sample.", "name": "fig-el-nino-data"}, "image": {"alt": "Two stacked time series panels from 1961 to 2024. In the top panel four climate indices move together, with peaks in 1972, 1982-83, 1997-98, 2015-16 and 2023-24. In the bottom panel the real food price spikes in 1973-74, falls until about 2000, and rises in 2007-08, 2010-11 and 2020-22 during blue La Niña bands, while it falls during the red El Niño bands of 2015-16 and 2023-24."}} tags=["remove-input"]
indices = panel.loc["1961-07-01":, INDEX_NAMES]
aligned = (indices - indices.mean()) / indices.std() * [1, 1, -1, -1]  # pressure indices reversed
state = aligned.mean(axis=1)
real_index = (panel["food_price_index"] / panel["consumer_price_index"]).loc["1961-07-01":]
real_index = 100 * real_index / real_index.mean()

fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
for column, label, color in zip(INDEX_NAMES, INDEX_LABELS, plotting.COLORS.cycle, strict=False):
    axes[0].plot(aligned.index, aligned[column], color=color, linewidth=0.8, label=label)
axes[0].axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
axes[0].set_ylabel("Standard deviations")
axes[1].plot(real_index.index, real_index, color=plotting.COLORS.ink, linewidth=1.1)
axes[1].set_yscale("log")
axes[1].set_yticks([50, 100, 200], labels=["50", "100", "200"])
axes[1].minorticks_off()
axes[1].set_ylabel("Real food price (average = 100)")
for ax in axes:
    for quarters, color in [(state.index[state > 1], plotting.COLORS.cycle[0]), (state.index[state < -1], plotting.COLORS.cycle[1])]:
        for quarter in quarters:
            ax.axvspan(quarter, quarter + pd.offsets.QuarterBegin(startingMonth=1), color=color, alpha=0.18, linewidth=0)
    ax.axvline(pd.Timestamp("1998-01-01"), color=plotting.COLORS.muted, linewidth=0.8, linestyle=(0, (4, 3)))
plotting.serif_title("Four measurements of El Niño, and the real price of food", axes[0])
labels = [*INDEX_LABELS[:2], *(f"{label} (reversed)" for label in INDEX_LABELS[2:])]
_ = plotting.legend_below(axes[1], handles=axes[0].get_lines()[:4], labels=labels, per_row=2, pad=0.2)

# %% [markdown]
# To the left of the dashed line the eye finds Brunner's result. The great price spike of
# 1973–74 starts with the El Niño of 1972–73, and prices firm after the El Niños of 1982–83 and
# 1986–87. To the right it finds something close to the reverse. Prices fall through the red
# bands of 2015–16 and 2023–24, and the three surges sit in or beside blue ones.
#
# A picture like this proves little. Each of those surges had large causes that had nothing to
# do with the Pacific, from the cost of energy to war. And El Niño is partly predictable: a
# warm quarter usually follows a warm quarter, so markets see much of it coming. To do better
# we need a model that separates the *surprise* in El Niño from what was already expected, and
# then follows food prices in the quarters after the surprise.
#
# ## One El Niño, four measurements
#
# The model has two parts.
#
# **Measuring El Niño.** Write $f_t$ for the state of El Niño in quarter $t$. We call it the
# **El Niño factor**. Nobody observes it. What we observe are the four climate indices, each
# rescaled to mean 0 and standard deviation 1 over the sample. The model says that index $k$ in
# quarter $t$, $x_{kt}$, is a multiple of the factor plus an error:
#
# $$
# x_{kt} = \lambda_k \, f_t + e_{kt}, \qquad e_{kt} \sim \mathcal{N}(0, \psi_k^2).
# $$ (el-nino-measurement)
#
# The **loading** $\lambda_k$ is how far index $k$ moves when the factor rises by 1, and
# $\psi_k$ is the size of its **measurement error**. An index with a small error has more say
# in where the factor is. We fix the loading of Niño 3 at 1. A positive factor then means warm
# water in the eastern Pacific, and the factor is measured in standard deviations of Niño 3,
# about 0.9 °C.
#
# **Linking El Niño to prices.** Let $g_t$ be the growth of the real food price in quarter $t$,
# less its average over the sample. The factor and price growth, $z_t = (f_t, g_t)$, follow a
# VAR: each is a weighted sum of the last six quarters of both, plus a surprise.
#
# $$
# z_t = \sum_{l=1}^{6} A_l \, z_{t-l} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma).
# $$
#
# The matrices $A_l$ hold the weights on the values $l$ quarters back, and $\Sigma$ is the
# covariance of the two surprises $\varepsilon_t$. Six quarters is Brunner's choice. A surprise
# is the part of a quarter that the previous six did not predict. An **El Niño surprise** is the
# surprise in the factor, and every result below is the response to one of typical size, one
# standard deviation. A **La Niña surprise** is the same with the sign reversed.
#
# One assumption is needed to read the surprises. When El Niño and food prices both surprise in
# the same quarter, which moved which? We assume that El Niño can move food prices within the
# quarter and that food prices cannot move the Pacific. Brunner assumes the same.
#
# The model is Bayesian. Each unknown starts with a **prior**, a distribution that says which
# values are plausible before the data are seen. The fit returns a **posterior**, the
# distribution after they are seen. For each result we report the posterior median and an
# **89 % interval**, the range between the posterior's 5.5 % and 94.5 % quantiles. The prior
# that matters most is on the 24 weights in $A_1, \ldots, A_6$. It pulls them towards a plain
# benchmark in which each series depends mostly on its own recent past. A sample of 35 years
# holds only a handful of strong El Niños, and without that pull the model would read their
# accidents as effects. The priors were fixed before the model was first estimated on
# 1963–1997.
#
# :::{dropdown} The priors, and how the model differs from Brunner's
#
# | Part | Prior |
# |------|-------|
# | VAR weights $A_l$ | Minnesota prior with `tightness=0.2`, `cross_shrinkage=0.5`, harmonic decay; own-lag prior mean 0.7 for the El Niño factor and 0 for price growth |
# | Covariance of the surprises $\Sigma$ | Impulso's defaults |
# | Loading of Niño 3 | fixed at 1 |
# | Other loadings | Normal(0, 1) |
# | Measurement-error standard deviations | half-Normal(1) |
# | El Niño factor | no prior of its own; its values enter the VAR's density |
#
# [The Minnesota Prior, From Scratch](minnesota-prior.py) explains the first row. Both series
# have mean 0, so the VAR has no intercepts.
#
# The model is simpler than Brunner's in three ways. His VARs also hold consumer price
# inflation and GDP growth for the G-7 countries; this one holds two series. His El Niño
# equation has only El Niño's own past in it; here the prior shrinks the weights on past food
# prices towards 0 without removing them. And his VARs are estimated by least squares, with no
# prior. The data differ too: his prices are an IMF index deflated by G-7 consumer prices, and
# ours are the World Bank index deflated by US consumer prices.
# :::
#
# ## The model in code
#
# Four short pieces build the model. The first prepares a sample and specifies the VAR.
# `prepare` rescales the climate indices over a sample and returns them with the growth of a
# price series, less its mean. `compile_var` declares the VAR with its prior and calls
# `VAR.compile_for_numpyro`, which builds the VAR's probability model once and returns a
# **compiled block**: an object that a NumPyro model can call with the two series.

# %%
TIGHTNESS, CROSS_SHRINKAGE = 0.2, 0.5


class Sample(NamedTuple):
    """One sample, ready for the model."""

    quarters: pd.DatetimeIndex
    indices: np.ndarray  # the four climate indices, standardised over the sample: (quarter, index)
    growth: np.ndarray  # growth of the price series in % per quarter, less its mean over the sample


def prepare(sample_name, price):
    """Standardise the climate indices over a sample and take the demeaned growth of a price series."""
    first, last = SAMPLES[sample_name]
    indices = panel.loc[first:last, INDEX_NAMES]
    growth = price.diff().loc[first:last]
    standardised = (indices - indices.mean()) / indices.std()
    return Sample(indices.index, standardised.to_numpy(), (growth - growth.mean()).to_numpy())


def compile_var(sample):
    """Specify the VAR and compile it for NumPyro. Also return the two series' scales."""
    prior = MinnesotaPrior(
        tightness=TIGHTNESS, cross_shrinkage=CROSS_SHRINKAGE, decay="harmonic", own_lag_mean=(0.7, 0.0)
    )
    spec = VAR(lags=N_LAGS, prior=prior)
    # The prior needs a scale for each series. The factor has no data yet, so it gets that of Niño 3.
    endog_scales = ar1_residual_sd(np.column_stack([sample.indices[:, 0], sample.growth]))
    block = spec.compile_for_numpyro(
        exog=None, n_lags=N_LAGS, endog_names=VAR_NAMES, endog_scales=endog_scales, intercept_equations=()
    )
    return spec, block, endog_scales


# %% [markdown]
# The second piece is the NumPyro model, and it is the measurement equation plus one call. It
# declares the three free loadings, the four measurement errors and the value of the El Niño
# factor in every quarter. It passes the factor and price growth to the block, which adds the
# VAR's priors and the VAR's probability of both series. Then it adds the probability of the
# four climate indices given the factor, equation {eq}`el-nino-measurement`. To the VAR the
# factor is an ordinary column of data. It happens to be a column that the sampler redraws at
# every step, so the uncertainty about where El Niño really was carries through to every
# result.


# %%
def model(indices, growth, block):
    n_quarters, n_indices = indices.shape
    with numpyro.plate("free_indices", n_indices - 1):
        free_loadings = numpyro.sample("free_loadings", dist.Normal(0.0, 1.0))
    error_standard_deviation = numpyro.sample(
        "measurement_error_standard_deviation", dist.HalfNormal(1.0).expand([n_indices]).to_event(1)
    )
    factor = numpyro.sample(FACTOR_NAME, dist.ImproperUniform(dist.constraints.real, (), (n_quarters,)))
    block(jnp.column_stack([factor, growth]))  # the VAR's priors and its density of both series
    loadings = jnp.concatenate([jnp.ones(1), free_loadings])  # the loading of Niño 3 is fixed at 1
    numpyro.sample(
        "climate_indices",
        dist.Normal(jnp.outer(factor, loadings), error_standard_deviation).to_event(2),
        obs=indices,
    )


# %% [markdown]
# The third piece draws from the posterior with NumPyro's NUTS sampler: 4 chains, each with
# 2,000 warm-up draws and 4,000 kept draws. A fit takes well under a minute on a laptop.
# `diagnostics` computes the standard checks on a run. [Checks](#checks), near the end,
# explains them and reports them for every fit on this page.
#
# :::{dropdown} Where the sampler starts
# NumPyro sets starting values on the sampler, not in the model. `block.initial_point` holds
# the starting values PyMC would use for the VAR, and `init_to_value` starts the sampler there.
# We add two of our own. With the loading of Niño 3 fixed at 1, the posterior can have a
# second, far less probable mode on which the other three loadings have the wrong signs and the
# factor runs against Niño 3. So every chain starts the factor at standardised Niño 3 and the
# free loadings at 0. A starting value changes where the chains begin, not the posterior.
# :::

# %%
if ci:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 200, 50, 2
else:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 2000, 4000, 4


def sample_posterior(model, block, sample, extra_start=None):
    """Sample a model with NumPyro's NUTS, starting the VAR's parameters at PyMC's initial point."""
    start = dict(block.initial_point)
    start["free_loadings"] = np.zeros(len(INDEX_NAMES) - 1)
    start[FACTOR_NAME] = sample.indices[:, 0]  # standardised Niño 3
    start |= extra_start or {}
    kernel = NUTS(partial(model, block=block), init_strategy=init_to_value(values=start))
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.key(0), sample.indices, sample.growth)
    return mcmc


def diagnostics(mcmc, block):
    """Convergence diagnostics of one run, over every sampled quantity."""
    samples = mcmc.get_samples(group_by_chain=True)
    # ArviZ does not accept the "/" in the names of the VAR's sample sites.
    sampled = {
        name.replace("/", "."): np.asarray(draws) for name, draws in samples.items() if name not in block.recorded_names
    }
    summary = az.summary(sampled, kind="diagnostics", round_to="none")
    loadings = np.asarray(samples["free_loadings"]).mean(axis=1)  # (chain, [Niño 4, the two pressure indices])
    reversed_signs = (loadings[:, 0] < 0) | (loadings[:, 1:] > 0).any(axis=1)
    return {
        "divergent transitions": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
        "largest R-hat": round(float(np.asarray(summary["r_hat"]).max()), 3),
        "smallest bulk effective sample size": round(float(np.asarray(summary["ess_bulk"]).min())),
        "chains with reversed loading signs": int(reversed_signs.sum()),
    }


# %% [markdown]
# The fourth piece hands the draws back to Impulso. `block.posterior_from_samples` and
# `FittedVAR.from_posterior` turn NumPyro's draws into an ordinary `FittedVAR`, and
# `Cholesky(ordering=...)` states the assumption about timing: the El Niño factor comes first,
# so its surprise can move food prices in the same quarter and not the other way round. `fit`
# chains the four pieces for any sample and any price series.


# %%
class Fit(NamedTuple):
    """One fit: its sample, its draws grouped by chain, the identified VAR and its diagnostics."""

    sample: Sample
    samples: dict
    identified: object
    diagnostics: dict


def identify(spec, block, samples, sample):
    """Wrap the draws in a `FittedVAR` and order the El Niño factor first."""
    posterior = block.posterior_from_samples(samples)
    # `VARData` wants one column per series. The factor differs at every draw, so pass its posterior mean;
    # the responses below are computed from the VAR's weights and covariance and do not read this column.
    factor_mean = np.asarray(samples[FACTOR_NAME]).mean(axis=(0, 1))
    data = VARData(endog=np.column_stack([factor_mean, sample.growth]), endog_names=VAR_NAMES, index=sample.quarters)
    fitted = FittedVAR.from_posterior(posterior, data, n_lags=N_LAGS, error_dist=spec.resolved_error_dist)
    return fitted.set_identification_strategy(Cholesky(ordering=VAR_NAMES))


def fit(sample_name, price):
    """Compile the VAR for a sample and a price series, sample the model and identify the VAR."""
    sample = prepare(sample_name, price)
    spec, block, _ = compile_var(sample)
    mcmc = sample_posterior(model, block, sample)
    samples = mcmc.get_samples(group_by_chain=True)
    return Fit(sample, samples, identify(spec, block, samples, sample), diagnostics(mcmc, block))


# %% [markdown]
# From here every tool of a `FittedVAR` works as it does after `VAR.fit`. We need two.
# `IdentifiedVAR.impulse_response` gives the path of a series in the quarters after a surprise,
# and `IdentifiedVAR.fevd` gives the share of a series' unpredictable movements that each kind
# of surprise accounts for. The helpers below take the posterior draws of both and turn them
# into the quantities reported on this page:
#
# - **price growth at an annual rate**: 4 times the response of quarterly growth, to compare
#   with Brunner;
# - the **price level**: in each draw, the growth responses added up to a quarter. This is how
#   much higher or lower the price is by then;
# - the **share of movements explained**: the share of the unpredictable movements in price
#   growth over the next 16 quarters that El Niño surprises account for.

# %%
HORIZON = 16
QUANTILES = [0.055, 0.5, 0.945]  # the lower end of the 89 % interval, the median, the upper end


def response_to_el_nino(identified, series=PRICE_NAME):
    """Draws of a series' response to an El Niño surprise, with axes (chain, draw, quarter)."""
    irf = identified.impulse_response(horizon=HORIZON).idata.posterior_predictive["irf"]
    return irf.sel(shock=FACTOR_NAME, response=series).values


def share_explained(identified):
    """Draws of the share of 16-quarter-ahead movements in price growth due to El Niño surprises, in %."""
    fevd = identified.fevd(horizon=HORIZON).idata.posterior_predictive["fevd"]
    return 100 * fevd.sel(response=PRICE_NAME, shock=FACTOR_NAME, horizon=15).values


def price_level(growth):
    """The response of the price level, in %: the growth responses summed draw by draw."""
    return np.cumsum(growth, axis=-1)


def median_and_interval(draws, digits=1):
    """Posterior median with its 89 % interval, as text."""
    low, mid, high = np.quantile(draws, QUANTILES)
    return f"{mid:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def level_rows(growth):
    """Rows of the tables below for one response of growth: the price level after one and two years."""
    level = price_level(growth)
    return {
        "price level 4 quarters after the surprise (%)": median_and_interval(level[..., 4]),
        "price level 8 quarters after the surprise (%)": median_and_interval(level[..., 8]),
        "probability that the price level is higher after 4 quarters": f"{(level[..., 4] > 0).mean():.2f}",
    }


def plot_band(ax, draws, color, label):
    """Plot the posterior median and 89 % interval of a response over the quarters after the surprise."""
    low, mid, high = np.quantile(draws, QUANTILES, axis=(0, 1))
    ax.fill_between(range(HORIZON + 1), low, high, color=color, alpha=0.25, linewidth=0)
    ax.plot(range(HORIZON + 1), mid, color=color, linewidth=1.4, label=label)


# %% [markdown]
# ## 1963–1997: the rise is there
#
# We start where Brunner did. The table sets the model's answers for his sample beside his.
# Each entry in the first column is a posterior median with its 89 % interval.

# %%
food = {"1963–1997": fit("1963–1997", real_food_price)}
growth = response_to_el_nino(food["1963–1997"].identified)
pd.DataFrame(
    {
        "this model, food": [
            median_and_interval(4 * growth[..., 0]),
            median_and_interval(4 * growth[..., 2]),
            median_and_interval(price_level(growth)[..., 4]),
            median_and_interval(share_explained(food["1963–1997"].identified)),
        ],
        "Brunner (1998)": [
            "not discussed",
            "3.5 to 4.0 (all primary commodities)",
            "not reported",
            "19.6 and 23.9 (food; his two El Niño measures)",
        ],
    },
    index=[
        "price growth in the quarter of the surprise, annual rate (percentage points)",
        "price growth 2 quarters after the surprise, annual rate (percentage points)",
        "price level 4 quarters after the surprise (%)",
        "share of movements in price growth explained (%)",
    ],
)

# %% [markdown]
# The model finds what Brunner found. Two quarters after a typical El Niño surprise, real food
# prices are growing about 4 percentage points faster at an annual rate. That is inside the
# range of 3.5 to 4 he reports for all primary commodities; he shows no response for food
# alone. A year after the surprise the real food price is about 1.7 % higher. The interval
# reaches just below zero, and the probability that the price is higher at all is about 0.9:
# likely, not certain.
#
# How big is a typical surprise? The cell below converts it to degrees.

# %%
typical_surprise = np.median(response_to_el_nino(food["1963–1997"].identified, FACTOR_NAME)[..., 0])
nino3_sd = panel.loc[slice(*SAMPLES["1963–1997"]), "nino3"].std()
print(f"A typical El Niño surprise lifts the factor by {typical_surprise:.2f} standard deviations of Niño 3,")
print(f"which is about {typical_surprise * nino3_sd:.1f} °C in the eastern equatorial Pacific.")

# %% [markdown]
# A typical surprise is a small event, less than half a degree. At its peak in late 1997 the
# eastern Pacific was more than 3 °C warmer than usual, so a great El Niño is several large
# surprises in a row, and its effect on prices is correspondingly larger.
#
# Two things differ from Brunner. Prices dip in the quarter of the surprise before they rise.
# His text does not discuss that quarter, and his figure for the pressure measure shows a dip
# there too. And El Niño surprises account for about an eighth of the movements in food price
# growth, against his fifth. He estimated his models by least squares, which tends to overstate
# such shares, because weights that are only noise still add to them. Our prior shrinks those
# weights.
#
# ## 1999–2024: the rise is gone
#
# Nothing in the model is specific to a period. So we fit it again, with the same priors and
# the same code, on the 26 years that came after Brunner's sample.

# %%
food["1999–2024"] = fit("1999–2024", real_food_price)
pd.DataFrame({
    name: {
        "price growth 2 quarters after the surprise, annual rate (percentage points)": median_and_interval(
            4 * response_to_el_nino(f.identified)[..., 2]
        ),
        **level_rows(response_to_el_nino(f.identified)),
        "share of movements in price growth explained (%)": median_and_interval(share_explained(f.identified)),
    }
    for name, f in food.items()
})

# %% mystnb={"figure": {"caption": "Response of the real food price to a typical El Niño surprise in each sample: posterior median and 89 % interval.", "name": "fig-el-nino-two-samples"}, "image": {"alt": "Real food price response over 16 quarters after an El Niño surprise: in 1963-1997 the median dips at first and then rises to about 2.5 percent, in 1999-2024 it stays at or below zero."}}
fig, ax = plt.subplots()
for (name, f), color in zip(food.items(), plotting.COLORS.cycle, strict=False):
    plot_band(ax, price_level(response_to_el_nino(f.identified)), color, name)
ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
ax.set_xlabel("Quarters after the surprise")
ax.set_ylabel("%")
plotting.serif_title("Food prices rise in Brunner's sample, not after it", ax)
_ = plotting.legend_below(ax, pad=0.22)  # below the x-axis label

# %% [markdown]
# After 1998 the rise is gone ({numref}`fig-el-nino-two-samples`). A year after an El Niño
# surprise the real food price is about 1.2 % lower, not higher. El Niño surprises account for
# about 3 % of the movements in price growth, down from an eighth.
#
# This is not a precise zero. The later interval is as wide as the earlier one, and it includes
# both no effect and a fall of 3 %. What the later data speak against is the earlier answer.
# The cell below compares the two posteriors draw by draw.

# %%
early, late = (price_level(response_to_el_nino(f.identified))[..., 4].ravel() for f in food.values())
print(f"Probability that the response after 4 quarters is lower in 1999–2024 than in 1963–1997: {(late < early).mean():.2f}")

# %% [markdown]
# The Pacific did not change. Niño 3 varies as much in the later sample as in the earlier one,
# with a standard deviation of 0.9 °C in both, and the later sample holds two of the five
# strongest El Niños in the data, those of 2015–16 and 2023–24. The signal is as strong as it
# was. Food prices stopped answering it in the old way.
#
# ## La Niña is not El Niño in reverse
#
# The model so far makes an assumption that is easy to miss. It is linear in the El Niño
# factor, so a fall in the factor moves food prices by exactly minus what an equal rise does.
# La Niña is El Niño's mirror image by construction. If El Niño raises prices, La Niña must
# lower them by the same amount. A fall in prices after El Niño and a rise in prices after La
# Niña are then one and the same finding, and the model cannot say which of the two the later
# sample shows.
#
# Harvests give a reason to doubt the mirror. A crop can fail because the season was too dry or
# because it was too wet, and each phase brings bad weather to some growing region.
# {cite:t}`iizumi2014` find that the global yields of maize, rice, wheat and soybeans all tend
# to be below normal in La Niña years, and that El Niño raises the yield of soybeans and lowers
# or leaves unchanged the other three. Both phases can hurt.
#
# So we let the two phases differ. The **separated model** writes price growth as
#
# $$
# g_t = \tilde g_t + \mu + \sum_{l=0}^{6} \kappa_l \, \bigl(|f_{t-l}| - c\bigr),
# $$
#
# where $\tilde g_t$ and the factor $f_t$ follow the VAR as before. The sum is new. We call it
# the **common part**: it depends on $|f_t|$, how far the Pacific is from normal, and not on
# the direction, so El Niño and La Niña move it the same way. The weights $\kappa_l$ are
# estimated, $\mu$ is an intercept, and $c$ is the average of $|f_t|$, approximated by the
# average size of standardised Niño 3. With every $\kappa_l$ at 0 the separated model is the
# mirror-image model again, so the data decide.
#
# The response to an El Niño surprise is now the VAR's response plus the common part. The
# response to a La Niña surprise is minus the VAR's response plus the common part.
#
# In code the change is small, because the compiled block does not care where its input comes
# from. `separated_model` reuses `model` unchanged and hands it a wrapper around the block. The
# wrapper subtracts the common part from price growth before the VAR sees it.
#
# :::{dropdown} The prior on the common part, and how its response is computed
# Each $\kappa_l$ has a Normal prior with mean 0 and the standard deviation that the Minnesota
# prior gives the VAR's weight on the factor at the same lag in the equation of price growth
# (lag 0 takes that of lag 1). The common part is therefore shrunk as much as the linear part.
# $\mu$ has a Normal(0, 1) prior. Before the first quarter of the sample, $|f_t|$ is set to $c$.
#
# A typical surprise moves the factor along the path $\psi_h$ that the VAR gives: upwards for
# El Niño, downwards for La Niña. Starting from neutral conditions, with the factor at 0 in
# every earlier quarter and no later surprises, the common part of the response of price growth
# in quarter $h$ is $\sum_l \kappa_l \, |\psi_{h-l}|$. `common_part` computes it from the draws
# of $\kappa_l$ and of the factor's response.
# :::


# %%
def separated_model(indices, growth, block, common_part_sd, centre):
    common_coefficients = numpyro.sample("common_coefficients", dist.Normal(0.0, common_part_sd).to_event(1))
    common_intercept = numpyro.sample("common_intercept", dist.Normal(0.0, 1.0))

    def block_on_the_linear_part(endog):
        factor, growth = endog[:, 0], endog[:, 1]
        size = jnp.concatenate([jnp.zeros(N_LAGS), jnp.abs(factor) - centre])
        lagged_size = jnp.stack([size[N_LAGS - lag : len(size) - lag] for lag in range(N_LAGS + 1)], axis=1)
        return block(jnp.column_stack([factor, growth - common_intercept - lagged_size @ common_coefficients]))

    model(indices, growth, block_on_the_linear_part)


def fit_separated(sample_name, price):
    """As `fit`, for the separated model."""
    sample = prepare(sample_name, price)
    spec, block, endog_scales = compile_var(sample)
    # The Minnesota prior's standard deviations of the weights on the lagged factor in the growth equation.
    lag_sd = TIGHTNESS * CROSS_SHRINKAGE * endog_scales[1] / endog_scales[0] / np.arange(1, N_LAGS + 1)
    separated = partial(
        separated_model,
        common_part_sd=np.concatenate([lag_sd[:1], lag_sd]),
        centre=np.abs(sample.indices[:, 0]).mean(),
    )
    start = {"common_coefficients": np.zeros(N_LAGS + 1), "common_intercept": np.zeros(())}
    mcmc = sample_posterior(separated, block, sample, extra_start=start)
    samples = mcmc.get_samples(group_by_chain=True)
    return Fit(sample, samples, identify(spec, block, samples, sample), diagnostics(mcmc, block))


def common_part(f):
    """Draws of the part of the growth response common to El Niño and La Niña surprises."""
    path = np.abs(response_to_el_nino(f.identified, FACTOR_NAME))
    coefficients = np.asarray(f.samples["common_coefficients"])
    common = np.zeros_like(path)
    for lag in range(N_LAGS + 1):
        common[..., lag:] += coefficients[..., lag, None] * path[..., : HORIZON + 1 - lag]
    return common


separated = {name: fit_separated(name, real_food_price) for name in SAMPLES}
phases = {}
for name, f in separated.items():
    linear, common = response_to_el_nino(f.identified), common_part(f)
    phases[name] = {"El Niño surprise": linear + common, "La Niña surprise": common - linear, "common part": common}
pd.concat({
    name: pd.DataFrame({surprise: level_rows(growth) for surprise, growth in responses.items()}).T
    for name, responses in phases.items()
})

# %% mystnb={"figure": {"caption": "Response of the real food price to a typical El Niño surprise and to a typical La Niña surprise in the separated model, in each sample: posterior medians and 89 % intervals.", "name": "fig-el-nino-phases"}, "image": {"alt": "Two panels of real food price responses over 16 quarters. In 1963-1997 the response to an El Niño surprise rises to about 4 percent and the response to a La Niña surprise stays near zero; in 1999-2024 the response to a La Niña surprise rises to about 2.5 percent and the response to an El Niño surprise stays near zero."}}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (name, responses) in zip(axes, phases.items(), strict=True):
    for surprise, color in zip(["El Niño surprise", "La Niña surprise"], plotting.COLORS.cycle, strict=False):
        plot_band(ax, price_level(responses[surprise]), color, surprise)
    ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
    ax.set_xlabel("Quarters after the surprise")
    ax.set_title(name)
axes[0].set_ylabel("%")
_ = axes[0].legend(frameon=False)

# %% [markdown]
# Separating the phases turns one change of sign into two one-sided effects
# ({numref}`fig-el-nino-phases`). Before 1998 an El Niño surprise raises the real food price by
# about 3.4 % within a year, twice what the mirror-image model found, and a La Niña surprise
# does nothing. After 1998 it is the other way round. A La Niña surprise raises the price by
# about 2.7 %, with a probability of about 0.9 that the price is higher, and an El Niño
# surprise does nothing.
#
# The mirror-image model was reporting half the gap between the two responses. Before 1998 that
# is half of 3.4 less 0.1, about 1.7. After 1998 it is half of 0.2 less 2.7, about −1.2. Its
# sign flipped because the phase that raises prices changed, not because El Niño began to make
# food cheap.
#
# The evidence for separating the phases is moderate. The common part is probably above zero in
# both samples, with a probability of 0.80 to 0.85, and its interval includes zero. The pattern
# is suggestive. It is not established.
#
# ## What the food index hides
#
# Why would the phase that raises prices change? A food index is an average, and it averages
# crops that the two phases treat differently. Geography suggests one comparison in particular.
# Palm oil and rice come mainly from South and Southeast Asia, where El Niño brings drought.
# Soybeans and maize are exported largely from the United States, Brazil and Argentina, where
# La Niña brings drought to large growing regions, and the wheat price in the data is a US one.
# If the weather matters for prices, El Niño should make the first group dearer relative to the
# second, and La Niña should do the opposite.
#
# The comparison has two advantages. Here the two phases *should* be mirror images, so the
# first model applies as it stands. And a price of one crop relative to another removes
# whatever moves all food prices together, such as the cost of energy or a boom in world
# demand. Those are the things that made {numref}`fig-el-nino-data` hard to read.
#
# The **relative crop price** is the average price of palm oil and rice divided by the average
# price of soybeans, maize and wheat, with averages taken in logarithms. We chose the two
# groups from geography before looking at their prices. The model, the priors and the sampler
# are the ones above. Only the series passed to `fit` changes.

# %%
ASIAN_CROPS = ["palm_oil_price", "rice_price"]
AMERICAN_CROPS = ["soybean_price", "maize_price", "wheat_price"]
relative_crop_price = 100 * (np.log(panel[ASIAN_CROPS]).mean(axis=1) - np.log(panel[AMERICAN_CROPS]).mean(axis=1))
crops = {name: fit(name, relative_crop_price) for name in SAMPLES}
pd.DataFrame({name: level_rows(response_to_el_nino(f.identified)) for name, f in crops.items()})

# %% mystnb={"figure": {"caption": "Response of the price of palm oil and rice relative to soybeans, maize and wheat to a typical El Niño surprise in each sample: posterior median and 89 % interval.", "name": "fig-el-nino-crops"}, "image": {"alt": "Relative crop price response over 16 quarters after an El Niño surprise: in 1999-2024 the median rises to about 4 percent within a year and its interval stays above zero, in 1963-1997 the median rises to about 1 to 2 percent and its interval includes zero."}}
fig, ax = plt.subplots()
for (name, f), color in zip(crops.items(), plotting.COLORS.cycle, strict=False):
    plot_band(ax, price_level(response_to_el_nino(f.identified)), color, name)
ax.axhline(0, color=plotting.COLORS.hairline, linewidth=0.6)
ax.set_xlabel("Quarters after the surprise")
ax.set_ylabel("%")
plotting.serif_title("After 1998 El Niño shifts prices between crops", ax)
_ = plotting.legend_below(ax, pad=0.22)  # below the x-axis label

# %% [markdown]
# This is the clearest result on the page ({numref}`fig-el-nino-crops`). After 1998 an El Niño
# surprise makes palm oil and rice about 3.8 % dearer relative to soybeans, maize and wheat
# within a year, and the probability of a rise is 0.99. A La Niña surprise does the reverse.
# Before 1998 the response is a third as large and its interval includes zero. In Brunner's
# years El Niño lifted the index, and the two groups show no clear move against each other.
# Since then the index has not moved and the groups have.
#
# The index's weights connect this to the last section. Soybeans and their oil and meal, maize
# and wheat make up about 45 % of the World Bank's food index. Palm oil, rice and coconut oil
# make up about 22 %. A La Niña that makes the first group dear therefore lifts the index more
# than an El Niño that makes the second group dear, which is the direction the separated model
# found after 1998. We did not test that link.
#
# ## What we learned
#
# Brunner's result holds in his sample and fails outside it. In 1963–1997 an El Niño surprise
# was followed by higher food prices, with no clear shift between crops. In 1999–2024 the same
# surprise is followed
# by a shift between crops, in the direction that the geography of drought predicts, and by
# little change in their average. El Niño still matters for the price of food. It no longer
# matters in the way Brunner measured.
#
# For anyone who watches El Niño forecasts for news about food prices, the reading of the last
# 26 years is this. An El Niño alert is not a reason to expect dearer food overall. It is a
# reason to expect rice and palm oil to gain on maize, wheat and soybeans, and a La Niña alert
# is a reason to expect the reverse. What either does to the average is small and uncertain: in
# the separated model a rise of a few percent after La Niña and nothing after El Niño, with
# intervals that include zero.
#
# We are not the first to see the change. {cite:t}`borrallo2024`, at the Banco de España, use a
# different method and find that the effect of El Niño on food commodity prices is strong in
# 1972–1992 and "completely fades away" in 1997–2023. They also find that La Niña lowers prices
# in the earlier period and raises them in the later one, and write: "We do not have an
# explanation for that." Neither do we. The four fits show what changed and not why.
#
# Three cautions apply.
#
# - **Few events.** Each sample holds a handful of strong El Niños and La Niñas, and every
#   estimate here rests on them.
# - **Follow-up, not a planned test.** The later sample, the separated model and the crop
#   comparison were added after the results for 1963–1997 were known.
# - **A simple model of the noise.** The VAR assumes that surprises are as large, on average,
#   in one decade as in another, and real food prices were far more volatile in the 1970s than
#   before or since.
#
# Waiting for more data will not settle the question soon, because a strong event comes along
# only every few years. Two other designs could. One follows the chain link by link, from El
# Niño to harvests and from harvests to prices, with production data for many countries. The
# other measures how crop futures react when El Niño forecasts are revised, which happens every
# month and would turn a handful of events into hundreds.
#
# ## Checks
#
# **Do the four indices measure one thing?** The table gives each index's loading and
# measurement error in the two fits of the first model to the food index.

# %%
def measurement(f):
    """Posterior medians and 89 % intervals of the loadings and measurement errors of one fit."""
    loadings = np.asarray(f.samples["free_loadings"])
    errors = np.asarray(f.samples["measurement_error_standard_deviation"])
    return pd.DataFrame(
        {
            "loading": ["1 (fixed)", *[median_and_interval(loadings[..., k], 2) for k in range(len(INDEX_NAMES) - 1)]],
            "measurement-error standard deviation": [
                median_and_interval(errors[..., k], 2) for k in range(len(INDEX_NAMES))
            ],
        },
        index=INDEX_LABELS,
    )


pd.concat({name: measurement(f) for name, f in food.items()}, axis=1)

# %% [markdown]
# Niño 4 loads positively and the two pressure indices negatively, as they should. Every
# measurement error is well below 1, the standard deviation of each index, so most of the
# movement in each index is the factor.
#
# **Did the sampler converge?** The table reports four checks for each of the six fits, over
# every sampled quantity, the factor in each quarter included.
#
# - **Divergent transitions** are steps at which the sampler's numerical integration failed.
#   Any deserves a look.
# - The **largest R-hat** compares the variance between chains with the variance within them.
#   {cite:t}`vehtari2021` recommend using the draws only if it is below 1.01.
# - The **smallest bulk effective sample size** is the number of independent draws that the
#   correlated draws are worth. They recommend at least 100 per chain.
# - **Chains with reversed loading signs** counts chains that ended on the second mode
#   described under "Where the sampler starts".

# %%
pd.DataFrame({
    **{f"food index, {name}": f.diagnostics for name, f in food.items()},
    **{f"food index, phases separated, {name}": f.diagnostics for name, f in separated.items()},
    **{f"relative crop price, {name}": f.diagnostics for name, f in crops.items()},
}).T

# %% [markdown]
# The fits on the later sample are harder for the sampler. There the Equatorial Southern
# Oscillation Index tracks the factor closely: its measurement error is less than half what it
# is in Brunner's sample, and its posterior reaches towards 0. Near 0 the factor is pinned to
# that one index and the posterior narrows sharply, which NUTS explores slowly. The long runs
# are for this, and a fit on the later sample can still report a few divergent transitions out
# of 16,000 draws. They do not move the answers. When we reran all six fits with another random
# key, and again with a stricter sampler setting (`target_accept_prob=0.95`), no median or
# interval end of a price response moved by more than 0.1.
#
# In smoke-render builds of these docs, which include the previews of pull requests, the
# sampler runs only a few draws. The tables and figures of such a build do not match the
# numbers in the text, which come from a full run.
#
# ## The Impulso pieces
#
# For readers who want to build something similar, the page used six steps.
#
# 1. **Compile** the VAR once with `VAR.compile_for_numpyro`, outside the NumPyro model.
# 2. **Call** the compiled block inside the NumPyro model with the VAR's series. A series can
#    be observed, sampled by the model, as the El Niño factor is, or computed by it, as price
#    growth less the common part is.
# 3. **Sample** with NumPyro's NUTS, starting at `block.initial_point`.
# 4. **Convert** the draws with `block.posterior_from_samples` and wrap them with
#    `FittedVAR.from_posterior`.
# 5. **Analyse** with the usual tools, here `impulse_response` and `fevd`.
# 6. **Reuse** the pieces. One model ran on two samples and two price series, and a second
#    NumPyro model wrapped the same block.
#
# [A VAR Inside a NumPyro Model](var-in-numpyro.py) covers each step and what the compiled
# block cannot do.
#
# ## Data sources and licences
#
# - **Niño 3 and Niño 4**: NOAA Climate Prediction Center, Niño-region sea-surface-temperature
#   anomalies from the Extended Reconstructed Sea Surface Temperature, version 5
#   {cite:p}`huang2017`, 1991–2020 base period, file
#   [`ersst5.nino.mth.91-20.ascii`](https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii).
#   **Southern Oscillation Index**: NOAA Climate Prediction Center, file
#   [`soi`](https://www.cpc.ncep.noaa.gov/data/indices/soi), standardised table. **Equatorial
#   Southern Oscillation Index**: NOAA Climate Prediction Center, file
#   [`reqsoi.for`](https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for). These series are NOAA
#   National Weather Service material, which is in the public domain and not subject to
#   copyright protection
#   ([National Weather Service disclaimer](https://www.weather.gov/disclaimer)). We averaged them
#   to quarters and standardised them.
# - **Food price index and crop prices**: World Bank, Commodity Price Data (the Pink Sheet),
#   monthly Food price index and the prices of palm oil, rice (Thai 5%), soybeans, maize and
#   wheat (US hard red winter), from
#   [World Bank Commodity Markets](https://www.worldbank.org/en/research/commodity-markets),
#   licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
#   We changed the series: we averaged them to quarters, deflated the food price index by the
#   consumer price index, formed a relative price from the crop prices, and took 100 times the
#   quarterly change in each logarithm. The index weights quoted above are from the same file,
#   sheet "Index Weights".
# - **Consumer price index**: US Bureau of Labor Statistics, consumer price index for all urban
#   consumers, all items in U.S. city average, seasonally adjusted (series CUSR0000SA0), which
#   is in the public domain. Source: Bureau of Labor Statistics.
#
# The works cited above are collected on the [project bibliography](../references.md) page.
