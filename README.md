# Impulso

[![Release](https://img.shields.io/github/v/release/QuantClimate/Impulso)](https://github.com/QuantClimate/Impulso/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/QuantClimate/Impulso/main.yml?branch=main)](https://github.com/QuantClimate/Impulso/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/QuantClimate/Impulso/branch/main/graph/badge.svg)](https://codecov.io/gh/QuantClimate/Impulso)
[![Commit activity](https://img.shields.io/github/commit-activity/m/QuantClimate/Impulso)](https://github.com/QuantClimate/Impulso/commits/main)
[![License](https://img.shields.io/github/license/QuantClimate/Impulso)](https://github.com/QuantClimate/Impulso/blob/main/LICENSE)

Bayesian Vector Autoregression (VAR) in Python.

<section class="consulting-cta">
    <p>We currently have some <strong>availability for consulting</strong> on how Bayesian modelling, vector autoregressions, and impulso can be integrated into your team's macroeconomic and financial forecasting work. If this sounds relevant, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use <a href="https://github.com/QuantClimate/Impulso/issues">GitHub Issues</a> and the documentation.</p>
</section>

## Overview

**impulso** provides a modern, Pythonic interface for Bayesian Vector Autoregression modeling. Built on PyMC, it enables full posterior inference for VAR models with informative priors, structural identification, impulse response analysis, and forecast error variance decomposition.

### Core Pipeline

The library follows an immutable, type-safe pipeline:

```
VARData → VAR.fit() → FittedVAR → .set_identification_strategy() → IdentifiedVAR
```

- **`VARData`**: Validated time series data (endogenous/exogenous variables + DatetimeIndex)
- **`VAR`**: Model specification (lags, priors, exogenous variables)
- **`FittedVAR`**: Reduced-form posterior estimates with forecasting capabilities
- **`IdentifiedVAR`**: Structural VAR with impulse responses, FEVD, and historical decomposition

### Key Features

- **Full Bayesian inference** via PyMC (NUTS sampling, automatic diagnostics)
- **Minnesota priors** for regularization in high-dimensional VARs, with cross-lag scaling by relative variable scale (ADR-0015)
- **Per-variable innovation priors**: Override the innovation scale prior for individual variables
- **Flexible identification schemes**: Recursive (Cholesky), sign restrictions
- **Forecasting**: Point forecasts, credible intervals, and scenario analysis
- **Impulse response functions** (IRFs) with uncertainty quantification
- **Forecast error variance decomposition** (FEVD)
- **Historical decomposition** of variables into structural shocks
- **Dynamic multipliers**: Response of endogenous variables to exogenous (VARX) drivers
- **Embeddable VARs**: Build a VAR inside an existing PyMC model with `VAR.build_in_model`, with symbolic observed data or latent endogenous series generated in the model under a stationarity constraint, then wrap the posterior with `FittedVAR.from_posterior` (see ADR-0016). `VAR.compile_for_numpyro` compiles the same VAR for use inside a NumPyro model (see ADR-0018)
- **Extensible protocols**: Plug in custom priors, samplers, and identification schemes
- **Type-safe**: Frozen Pydantic models with full type hints

## Installation

```bash
pip install impulso
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install impulso
```

### Faster sampling with nutpie

For significantly faster NUTS sampling, install with the optional [nutpie](https://github.com/pymc-devs/nutpie) backend:

```bash
pip install "impulso[nutpie]"
```

Or with uv:

```bash
uv add impulso --extra nutpie
```

When nutpie is installed, it is used automatically as the default sampler. You can also select the backend explicitly:

```python
from impulso.samplers import NUTSSampler

sampler = NUTSSampler(nuts_sampler="nutpie")   # or "pymc"
fitted = var.fit(data, sampler=sampler)
```

### A VAR inside a NumPyro model

`VAR.compile_for_numpyro` compiles a VAR's PyMC model to JAX so that a [NumPyro](https://num.pyro.ai/) model can call it. It needs JAX and NumPyro, in the optional `numpyro` extra:

```bash
pip install "impulso[numpyro]"
```

Or with uv:

```bash
uv add impulso --extra numpyro
```

The tutorial [A VAR Inside a NumPyro Model](https://impulso.quantclimate.com/tutorials/var-in-numpyro.html) walks through compiling, sampling with NumPyro's NUTS and handing the posterior to `FittedVAR`.

## Quick Start

```python
import pandas as pd
from impulso import VARData, VAR

# Load your time series data
df = pd.read_csv("data.csv", index_col="date", parse_dates=True)

# Create validated VAR data
data = VARData.from_df(df, endog_vars=["gdp", "inflation", "interest_rate"])

# Specify and fit a VAR(4) model with Minnesota prior
var = VAR(lags=4, prior="minnesota")
fitted = var.fit(data)

# Generate forecasts
forecast = fitted.forecast(steps=12)
forecast.plot()

# Structural identification and impulse responses
identified = fitted.set_identification_strategy("cholesky")
irf = identified.impulse_response(steps=20)
irf.plot()

# Forecast error variance decomposition
fevd = identified.fevd(steps=20)
fevd.plot()
```

## Documentation

Full documentation, tutorials, and API reference: [https://impulso.quantclimate.com](https://impulso.quantclimate.com)

## Development

See [AGENTS.md](AGENTS.md) for development setup, testing, and contribution guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use impulso in your research, please cite:

```bibtex
@software{impulso,
  author = {Pinder, Thomas},
  title = {impulso: Bayesian Vector Autoregression in Python},
  year = {2026},
  url = {https://github.com/QuantClimate/Impulso}
}
```
