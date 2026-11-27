# Tutorials

These tutorials walk you through Impulso's core workflow: fitting a Bayesian VAR, producing probabilistic forecasts, and running structural analysis. They assume familiarity with regression and autoregressive models but explain VAR-specific concepts as they arise.


| Tutorial | What you'll learn |
|----------|-------------------|
| [Fitting Your First Bayesian VAR](quickstart.py) | Data loading, lag selection, model fitting, posterior inspection |
| [The Minnesota Prior, From Scratch](minnesota-prior.py) | Why VARs need shrinkage, the prior's maths, prior predictive checks, tuning `tightness` |
| [Probabilistic Forecasts](forecasting.py) | Multi-step forecasts, credible intervals, fan charts |
| [Structural Shocks and Their Effects](structural-analysis.py) | Cholesky identification, impulse responses, FEVD, historical decomposition |
| [Model Checks and Validation](model-checking.py) | Stationarity pretests, prior predictive checks, MCMC diagnostics in ArviZ, posterior predictive checks |
| [Monetary Policy Analysis](monetary-policy.py) | Policy reaction functions, scenario analysis |
| [Stochastic Volatility](stochastic-volatility.py) | Time-varying residual volatility via univariate SV: fit, interpret, and forecast |
| [Oil Supply News with an External Instrument](proxy-svar.py) | Proxy-SVAR identification, external instruments, Känzig (2021) replication |
| [The Conjugate VAR](conjugate-var.py) | Closed-form NIW estimation, data-selected shrinkage, conjugate-vs-NUTS comparison |
| [Estimating a VAR after March 2020](post-march-2020.py) | Conjugate NIW VAR, COVID volatility break, conditional forecasts, Lenza & Primiceri (2022) replication |
| [Counterfactuals & Scenario Analysis](scenario-analysis.py) | Historical counterfactuals, conditional forecasts, structural scenarios, plausibility statistics |
| [A VAR Inside a NumPyro Model](var-in-numpyro.py) | Compiling a VAR for NumPyro, sampling with NumPyro's NUTS, handing the posterior to `FittedVAR` |
| [El Niño and Food Commodity Prices](el-nino-commodity-prices.py) | An unobserved El Niño factor measured by four climate indices, sampled in a NumPyro model and passed to a compiled VAR as an endogenous series, comparison with Brunner (1998) |

Start with the **Quickstart** if you're new to Impulso, then read **The Minnesota Prior** to understand the shrinkage the Quickstart switched on by default. The Forecasting and Structural Analysis tutorials build on concepts introduced there.

```{toctree}
:hidden:
:maxdepth: 1

quickstart
minnesota-prior
forecasting
structural-analysis
monetary-policy
model-checking
stochastic-volatility
proxy-svar
conjugate-var
post-march-2020
scenario-analysis
var-in-numpyro
el-nino-commodity-prices
```
