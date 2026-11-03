# Impulso

Bayesian Vector Autoregression (VAR) with structural identification and stochastic volatility extensions. This file defines the load-bearing terms used in the codebase, docs, and public API. Add new terms here as they're sharpened in design discussions; don't drift into synonyms.

Definitions only. Formulas, field names, exact value sets and failure rules live in the docstring or ADR that owns the behaviour — this file says what a thing *is*, not how it works.

## Language

**VAR (Vector Autoregression)**:
A multivariate linear time-series model where each variable is regressed on its own lags and the lags of every other variable. The class `VAR` is the *reduced-form* specification; structural meaning is layered on afterwards through identification.
_Avoid_: "VAR model" (redundant), "autoregression" (the singular form is misleading for multivariate models).

**Endogenous / exogenous**:
The two blocks of a dataset. Endogenous variables are modelled jointly and each carries a structural shock; exogenous regressors enter contemporaneously and carry none. The split is a modelling assumption the data cannot check, and it decides which questions are askable — identification lives only on the endogenous block.
_Avoid_: "dependent" / "independent" variables (regression vocabulary that hides the jointness); "controls" for the exogenous block.

**VARData**:
The validated entry point of the pipeline: an endogenous block, an optional exogenous block, their names, and a time index. Frozen and read-only, so a fitted model can never be re-pointed at data that changed underneath it.
_Avoid_: "dataset" / "dataframe" — a DataFrame is what `from_df` consumes, not what the pipeline holds.

**Reduced-form / Structural**:
The reduced-form VAR fits dynamics without economic interpretation; the structural VAR adds an identification scheme that maps reduced-form residuals to economically-meaningful shocks. In code, `FittedVAR` is reduced-form; `IdentifiedVAR` is structural.
_Avoid_: "raw" / "interpretable" — they obscure the technical meaning.

**Posterior schema**:
The reduced-form parameter block every estimator must produce and every consumer may assume — lag coefficients, intercept, and exogenous coefficients. It is what lets two unrelated estimators feed identical downstream code. The volatility and error-distribution seams name their own parameters and sit deliberately outside it.
_Avoid_: `B` for anything but the lag coefficients — the structural shock matrix is `P`, and `A_1..A_p` are the per-lag matrices `B` splits into.

**Identification scheme**:
A rule for recovering structural shocks from reduced-form covariance. Pure in its output — it consumes a Cholesky factor and produces a structural shock matrix — with one declared side effect, its identification diagnostics. It does not own time iteration.
_Avoid_: "identification strategy" (used colloquially; the Protocol is named "scheme"). `B` for the structural shock matrix — that name belongs to the reduced-form coefficient block.

**Identification diagnostics (`last_diagnostics`)**:
The scalars a scheme reports about its own work — acceptance rates, instrument strength, screen conditioning. A single-call scratchpad rather than accumulated state, and an optional capability of the seam: a scheme with nothing to report simply lacks it.
_Avoid_: treating it as cumulative or reentrant state; reading the private backing attribute rather than the property.

**Long-run multiplier (C(1))**:
The total effect of a one-off reduced-form innovation on the *level* of each variable — the sum of the moving-average coefficients. Defined only for a stable VAR, and numerical near-singularity is a distinct failure from outright divergence.

**Cumulative MA impact matrix (Θ(1))**:
The structural counterpart: the total effect of each structural shock on each variable's level. Where most schemes constrain the impact matrix Θ(0), `LongRunRestriction` constrains Θ(1).
_Avoid_: "Blanchard-Quah identification" as an API name — the scheme is named for what it restricts, not for its first users (the prose citation is fine). "Permanent shock" for anything but the shock restricted nowhere.

**Zero-and-sign restrictions (ARW construction)**:
Identification combining exact zeros on the impact matrix with sign restrictions on impulse responses, via the recursive orthogonalisation of Arias, Rubio-Ramírez & Waggoner (2018). The zeros hold by construction; only the signs are accept/reject.
_Avoid_: "penalty-function zeros" — that is a different (Uhlig-style, loss-minimising) construction that only approximates the zeros; these are exact.

**Volatility process**:
The seam that owns how the structural-shock covariance Σ_t is built, evolved and queried. It owns its downstream computation too — forecast paths, time-`t` query, per-variable volatility paths — so the pipeline never branches on adapter type.
_Avoid_: "variance model", "covariance specification" — these describe the *output*, not the process.

**Constant volatility**:
The default volatility process: one Σ shared across all time points, carried as its Cholesky factor rather than re-decomposed on demand.

**Stochastic volatility**:
Time-varying volatility where Σ_t evolves stochastically. The shipped flavour is Clark-style — per-variable log-volatility with a constant correlation factor; Primiceri-style time-varying correlations are a future adapter the seam is shaped to admit. The class serves a dual role: a standalone univariate model, and a volatility process plugged into a VAR.

**Log-volatility dynamics (`SVDynamics`)**:
The seam *inside* stochastic volatility that owns the law of the latent log-volatility path — how it is built and how it is simulated forward. One rung below the volatility process: that owns Σ_t, this owns the path driving it.
_Avoid_: "SV prior" for the dynamics (the prior is a separate seam); "the SV model" when the dynamics adapter is what's meant.

**Observation error distribution**:
The seam that owns *which law* the observation error follows. Sibling of the volatility process, which owns how *big* the error is; the separation is strict in both directions — this seam never touches Σ, and the volatility process never touches tail shape.
_Avoid_: "likelihood" as the name of the seam (the likelihood is what the adapter *builds*, and the word also gets used for the whole model's `logp`); "error model", "noise distribution".

**Degrees of freedom (ν)**:
The tail-weight parameter of Student-t errors, either fixed or inferred. Bounded strictly above 2, and not configurable: below it the t has infinite variance and every variance-shaped consumer stops being defined.
_Avoid_: "df" as a field name (collides with the DataFrame idiom); "tail parameter".

**Scale matrix (Ω)**:
What a Cholesky factor squares to under a heavy-tailed error distribution. It equals the innovation covariance under Gaussian errors but not under Student-t, where the covariance is a tail-dependent multiple of it. Identification factorises the scale matrix, so impulse responses are in scale units while FEVD, historical decompositions and zero-edit counterfactuals are exactly invariant to the distinction (ADR-0007).
_Avoid_: calling `sigma()` "the covariance" without qualification — true only under Gaussian errors.

**L_t**:
Lower-triangular Cholesky factor of the time-`t` structural-shock covariance, and the primary output of the volatility process. Constant in `t` under constant volatility, varying under stochastic volatility. Identification operates on it directly — no redundant decomposition.

**Impulse response function (IRF)**:
The dynamic response of each variable to a unit structural shock over a horizon. Under stochastic volatility an IRF depends on when the shock lands, which is what `at` selects.

**Dynamic multiplier**:
The response of each endogenous variable to a unit impulse in an *exogenous* regressor. Shares the moving-average coefficients with the IRF but needs no identification scheme — exogenous regressors are exogenous by assumption — so it lives on the reduced-form fit. The cumulative variant is the response to a permanent *step* rather than a one-off impulse.
_Avoid_: "exogenous IRF" — IRF is reserved for responses to identified structural shocks.

**Historical decomposition (HD)**:
The attribution of each in-sample observation to a deterministic baseline plus the *propagated* contribution of each structural shock. "Propagated" is load-bearing: a shock's contribution carries forward through the lag dynamics beyond its impact period.
_Avoid_: presenting the contemporaneous split of the one-step forecast error as HD — that is a residual decomposition, and was the (incorrect) behaviour prior to the scenario-analysis stack (2026-07).

**Historical counterfactual**:
The in-sample "what if": back out the realised structural shocks, overwrite a chosen shock's path over a window, and re-propagate the system. Shocks are edited, never re-drawn, so the posterior spread reflects parameter and identification uncertainty only.
_Avoid_: bare "counterfactual" for forecast-side operations (those are conditional forecasts or structural scenarios); "policy counterfactual" — rule replacement is a different, out-of-scope object.

**Predictive check (prior / posterior)**:
Data simulated from the model on the *estimation window*, to be compared against the data itself. Both variants are one step ahead given the observed lags, so they sit on the data's own time axis.
_Avoid_: "forecast" for either — a forecast leaves the sample and iterates its own predictions; a predictive check never does.

**Conditional forecast**:
A forecast constrained so chosen endogenous variables follow pinned future paths, with *all* structural shocks free to adjust (Waggoner–Zha 1999). Identification-free: the observable-space answer does not depend on the scheme, so it lives on the reduced-form fit — the same placement logic as the dynamic multiplier.
_Avoid_: "scenario" for an all-shocks-adjust conditional forecast — scenarios restrict *who* adjusts.

**Structural scenario**:
A conditional forecast with structural attribution (Antolín-Díaz, Petrella & Rubio-Ramírez 2021): pinned variable paths must be absorbed by a named set of adjusting shocks, and/or future shock paths are prescribed directly.
_Avoid_: "conditional forecast" when an adjusting set is named — the restriction is the point.

**Condition vocabulary (`ShockPath`, `VariablePath`)**:
The frozen spec objects expressing scenario content: a structural shock's path, and a pinned future path for an endogenous variable. Each scenario method accepts only the condition types legal for it and rejects the rest by type.
_Avoid_: "Conditions object" / "Scenario object" for these primitives — a bundling `Scenario` container may arrive later for connector round-trips; the primitives are paths.

**Adjusting shocks**:
The subset of structural shocks permitted to absorb a structural scenario's conditions. Existence is a per-draw rank condition: enough adjusting-shock dimensions to span the conditions asked of them.
_Avoid_: "driving" / "offsetting" shocks in API surface (fine in prose, where ADPRR and Leeper–Zha use them).

**Plausibility statistic (q)**:
How far a scenario reaches beyond what the model considers ordinary — the per-draw distance of its binding restrictions from their unconditional law, reported with a calibrated companion on a bounded scale. The Leeper–Zha "modest interventions" check in the ADPRR lineage: a large `q` means the scenario demands incredible shocks and the model's answer should not be trusted.
_Avoid_: calling it a Kullback–Leibler divergence under hard conditions — the conditional law is singular there and that KL is infinite; the KL form applies only to future *soft* conditioning. "Modesty statistic" stays prose-only.

**at**:
The time-index parameter on time-varying queries: a specific index, the most recent slice, or the whole time axis. Ignored under constant volatility, where there is nothing to select.

**Estimation paradigm (`VAR` vs `ConjugateVAR`)**:
Two ways to estimate the reduced-form VAR: independent-Normal coefficient priors sampled by NUTS, or a conjugate Normal-Inverse-Wishart prior with closed-form posteriors and a Metropolis step on hyperparameters (Giannone et al. 2015). Both return a `FittedVAR`, so identification and forecasting are identical downstream.
_Reader-facing shorthand_: "the conjugate VAR" (`ConjugateVAR`) vs "the NUTS VAR" (`VAR`) — the contrast axis is the mode of inference (closed-form vs MCMC).
_Avoid_: "the Bayesian VAR" as if there were one estimator; name the paradigm.

**NIW prior (`NIWPrior`)**:
The conjugate Normal-Inverse-Wishart Minnesota prior consumed by `ConjugateVAR`, encoded via dummy observations. Distinct from `MinnesotaPrior`, which parameterises the independent-Normal prior for the NUTS path.
_Avoid_: conflating the two — different priors for different estimators.

**Minnesota tightness (λ)**:
The scalar controlling how hard all coefficients shrink toward the random-walk prior mean. Fixed on the NUTS path; the conjugate prior can instead treat it as a hyperparameter selected against the marginal likelihood, which makes selection a property of the prior's configuration rather than of the estimator.
_Avoid_: bare "shrinkage" (ambiguous with cross-variable shrinkage).

**Contribution space (`exog_prior_scale`)**:
The units a prior on exogenous coefficients is stated in. Such a coefficient converts a regressor's units into a variable's, so a prior fixed in *coefficient* space is not a fixed belief — rescaling a regressor silently changes it. Stating the belief as a fraction of the variable's own residual scale is invariant to both (ADR-0012).
_Avoid_: reading `exog_prior_scale` as a tightness — it runs the opposite direction from **Minnesota tightness (λ)**, and lowering it is what shrinks.

**Embedded path (`build_in_model`)**:
Registering a VAR into a PyMC model someone else owns, rather than the fresh model `fit` opens. There is one model-building path: `fit` and `prior_predictive` wrap it, so a standalone fit and an embedded VAR describe the same density. "Embedded" in the narrow sense, where the numpy-only steps and some spec options drop out, means a symbolic `endog` or any **latent endogenous series** (ADR-0016).
_Avoid_: "custom VAR" or "VAR inside the MMM" for the mechanism: the host model is incidental. Avoid "prefix" as an Impulso option: namespacing comes from the caller's nested `pm.Model(name=...)`, and coordinates are not namespaced at all.

**Latent endogenous series**:
An endogenous variable with no data, generated inside the model from its own VAR equation driven by standard-normal innovations (non-centred). Latent series lead the VAR ordering and the Cholesky ordering, so the observed block's likelihood is exact once it is conditioned on the latent innovations. A latent series is a full member of the VAR, not an exogenous input: it has lags, a shock, and coefficients in every equation.
_Avoid_: "missing data" or "unobserved column": nothing is imputed, and a latent series is never passed in. Avoid "state" (state-space vocabulary for a model Impulso does not fit).

**Latent block / stationarity constraint**:
The latent-on-latent part of the lag coefficients: the latent equations' coefficients on latent lags, across all lags. The observed series are data, so this block alone decides whether a generated path explodes. The stationarity constraint restricts it to the region where its companion matrix has spectral radius below 1, and the posterior is the prior truncated to that region. For one latent series with one lag the block is the latent own lag alone and the region is the interval (−1, 1), so the constraint is a truncated own-lag prior, which prior predictive sampling respects. In every other case it is a Potential, which prior predictive sampling ignores (ADR-0017).
_Avoid_: "stationary VAR" for the constraint: the observed equations are unconstrained. Avoid "stationarity pretest", which is the unrelated frequentist test run before specification.

**Model evidence (`ModelEvidence`)**:
The conjugate estimator's closed-form log marginal likelihood of the observed data. Because it accounts for any volatility rescaling it is a density over the observations themselves, so a break model and a homoscedastic model fitted to the same data are directly comparable. The NUTS path has no closed form and carries none.
_Avoid_: "log ML" in the API surface (spell out marginal likelihood); "model probability" for a raw Bayes factor — the probability requires prior model weights.

**Deterministic volatility break (`ConjugateVolatility`)**:
A volatility process whose per-period scale follows a deterministic, hyperparameter-driven path with a known break date. Used only by the conjugate estimator, where the scale enters as a rescaling of the data. `PandemicBreak` is the concrete case, reproducing Lenza & Primiceri (2020).
_Avoid_: "stochastic volatility" — the break is deterministic given its hyperparameters.

**Stationarity pretest**:
A classical frequentist test run *before* a VAR is specified, to decide levels versus differences. ADF and KPSS have opposite nulls, so both are reported. The pretests report and never decide: no mechanical "difference it" rule is applied, because unit-root tests have low power against persistent alternatives and are sensitive to deterministic terms and breaks.
_Avoid_: "stationarity check" / "unit-root check" — "pretest" carries the sequencing (it precedes specification) and the pretesting-bias caveat.

**Integration order (`d`, `d_max`)**:
The number of differences a series needs before ADF rejects a unit root, per variable, plus the system-wide maximum a Toda-Yamamoto procedure consumes. Variables whose tests conflict, or that are still integrated where the search stopped, are reported as inconclusive rather than given a silent verdict — so the maximum is a floor rather than a finding whenever that list is non-empty.
_Avoid_: "order of differencing" for `d_max` — it is the maximum across the system, not any one series' `d`.

**Cointegration rank**:
The number of independent long-run relationships among integrated series, from the Johansen procedure. Both sequential tests are reported, because trace and max-eigen can disagree. A non-zero rank means differencing every series discards the long-run relationship; the recommended response is a VAR in levels, since a vector error-correction model is deliberately out of scope.
_Avoid_: "number of cointegrating vectors" in API surface (fine in prose); "cointegration test" without saying which statistic.

**Granger causality**:
Conditional predictive precedence: the past of one variable improves the prediction of another beyond that other's own past, *within the fitted system of variables*. Ordered and directional — the two orderings are separate queries with unrelated answers. Reduced-form, so no identification scheme is involved.
_Avoid_: "X causes Y" for a Granger result, and "causal effect" — the finding is about information sets, not interventions; an omitted common driver manufactures it. Reserve "effect" for identified structural objects (IRFs, counterfactuals).

**ROPE (region of practical equivalence)**:
The magnitude below which the analyst declares a relationship practically negligible, and the only input from which the Granger surface will make a probability statement. There is deliberately no default: the threshold is the analyst's judgement, and it travels with the number that came from it.
_Avoid_: "probability of no causality" / "probability the coefficient is zero" — under continuous coefficient priors that probability is zero before and after the data, and an edge-inclusion probability needs a spike-and-slab prior Impulso does not fit (ADR-0010). Also avoid bare "threshold" — the ROPE is on the magnitude, not on a p-value.

**Toda-Yamamoto augmentation**:
Fitting a VAR in levels with extra lags and testing only the original ones, which restores standard Granger inference on possibly-integrated series without differencing (Toda & Yamamoto 1995). The augmented lags are never reported, and the procedure refuses to run on inconclusive integration diagnostics — an under-augmented test is invalid rather than imprecise.
_Avoid_: "extra lags" without saying they are untested; "corrected for non-stationarity" — nothing is corrected, the asymptotics are restored.

## Relationships

- A **VAR** carries one **prior**, one **volatility process**, and one **observation error distribution**.
- A **FittedVAR** plus an **identification scheme** produces an **IdentifiedVAR**.
- An **identification scheme** consumes an `L_t` (queried from the volatility process) and produces a structural shock matrix `P`.
- An **identification scheme** may additionally consume the posterior lag coefficients: `LongRunRestriction` needs them for the **long-run multiplier**, as `SignRestriction(restriction_horizon > 0)`, `ZeroSignRestriction(restriction_horizon > 0)` and `ProxySVAR` already do. `L_t` alone is the minimum, not the maximum, of what a scheme may ask for.
- An **IdentifiedVAR** computes **IRFs**, FEVDs, and historical decompositions by asking the volatility process for `L_t` at the requested `at`, then applying the identification scheme.
- A **FittedVAR** fitted with exogenous regressors computes **dynamic multipliers** on its own; no identification scheme is involved, because the driver is already exogenous.
- A **stochastic volatility** can plug into a **VAR** as its volatility process *or* be fitted standalone on a univariate series.
- A **stochastic volatility** carries one **log-volatility dynamics** adapter and one SV prior; whether the dynamics pins its own level decides if the model adds an outer per-variable level term.
- A **FittedVAR** computes **conditional forecasts** on its own — all shocks adjust, so no identification scheme is involved (the dynamic-multiplier placement logic). Under the hood it is the degenerate case of the structural scenario's three-way partition solve: no prescribed shocks, every shock adjusting, with the volatility process's `L` standing in for the structural matrix.
- An **IdentifiedVAR** computes **historical counterfactuals** and **structural scenarios** through the four-layer scenario engine (back out → constrain → solve → propagate); the solve layer is one function shared with the conditional forecast, and the propagate layer is one lag recursion shared with `forecast()`, the counterfactual paths and the **historical decomposition**'s baseline.
- The **condition vocabulary** is consumed by all three scenario methods; each method accepts only the condition types legal for it and rejects the rest by type.
- A **VAR** simulates its own **prior predictive** from the graph it would fit; a **FittedVAR** replicates the estimation sample as a **posterior predictive**, computed in NumPy from the posterior and the volatility seam so the conjugate estimator gets it for free (ADR-0011).
- A **VAR** is estimated by NUTS; a **ConjugateVAR** is estimated analytically with a Metropolis step on hyperparameters. Both produce a **FittedVAR**.
- A **stationarity pretest** consumes `VARData` (endogenous block only), a DataFrame, or a Series, and produces a result object — never a modified dataset and never a specification. It sits *beside* the pipeline, not in it: nothing downstream of `VAR.fit()` reads its output.
- **Integration order** feeds **cointegration rank**: the Johansen test is only meaningful for series that are individually integrated, and it is conditioned on a lag order that `select_lag_order` supplies.
- A **FittedVAR** answers **Granger causality** queries on its own — the coefficients are reduced-form and carry no time dimension, so neither an identification scheme nor an `at` is involved. The query never refits; it selects which of the already-fitted lags are tested.
- **Toda-Yamamoto augmentation** consumes an **integration order**, fits the augmented lag order with a **ConjugateVAR**, and produces the same result object the `FittedVAR` query does — which is why the manual route is equivalent, and is the documented escape hatch for exogenous regressors, NUTS, and stochastic volatility.
- A **VAR** registers itself into a host model through the **embedded path**. The host owns sampling. `FittedVAR.from_posterior` turns a posterior into a **FittedVAR** after checking the **posterior schema**, by exact unprefixed names against a `VARData`. An embedding caller must first return its posterior to that schema: strip the nested-model prefix, and supply a `VARData`, with a plug-in column (e.g. the posterior-mean path) standing in for each latent series (ADR-0016).
- A **latent endogenous series** exists only on the **embedded path**. Its **latent block** carries the **stationarity constraint**, and latent series require Gaussian errors and constant volatility.
- A **ConjugateVAR** carries an **NIW prior** and optionally a **deterministic volatility break**; a **VAR** carries a **MinnesotaPrior** and a PyMC volatility process. Each estimator's fields accept only its compatible components, enforced by types and validators rather than a builder.

## Example dialogue

> **User:** "Fit a 4-variable VAR with stochastic volatility, AR(1) log-vol."
> **Library:** `VAR(lags=4, volatility=StochasticVolatility(dynamics="ar1")).fit(VARData(...))`. The `volatility` parameter accepts a string shorthand (`"constant"`, `"sv"`) or any `VolatilityProcess` instance.
>
> **User:** "Is my prior sane before I spend an hour on NUTS?"
> **Library:** `VAR(lags=4).prior_predictive(data, draws=500)` then `az.plot_ppc(idata, group="prior")`. After fitting, `fitted.posterior_predictive()` returns the same-shaped in-sample replicates for the other end of the check.
>
> **User:** "Show me the IRF for shocks hitting in 2008Q3."
> **Library:** `identified.impulse_response(horizon=20, at=t_2008Q3)`. The pipeline queries `volatility.cholesky_at(t_2008Q3)` for `L`, the identification scheme rotates it into `P`, and the IRF is computed from `A_1..A_p` and `P`.
>
> **User:** "Just a univariate SV fit."
> **Library:** `StochasticVolatility(dynamics="ar1").fit(SVData(y))`. Same class, standalone code path.
>
> **User:** "What would food prices have done if the 2022 energy shock had never happened?"
> **Library:** `identified.counterfactual(shocks=[ShockPath(shock="energy", values=0.0, start="2022-01", end="2022-12")])`. Realised shocks are edited and re-propagated; `result.difference()` is the shock's effect.
>
> **User:** "Forecast inflation if the policy rate follows this path for two years."
> **Library:** `fitted.conditional_forecast(steps=8, conditions=[VariablePath(variable="rate", values=path)])`. All shocks adjust (Waggoner–Zha); no identification scheme needed. Add `path_uncertainty="unconditional"` to restrict the mean only and keep honest bands.
>
> **User:** "Same rate path, but make the *policy* shock do the work — and how believable is that?"
> **Library:** `identified.structural_scenario(steps=8, conditions=[VariablePath(variable="rate", values=path)], adjusting=["policy"])`. Non-adjusting shocks keep their unconditional draws; `result.plausibility()` reports `q` and the calibrated `q_cal`.
>
> **User:** "2020Q2 is wrecking my estimates. Can I stop dummying it out?"
> **Library:** `VAR(lags=4, error_dist="student_t").fit(VARData(...))`. The t likelihood downweights the observation automatically; the degrees of freedom come back in the posterior as `nu`. Pass `StudentT(nu=5.0)` to fix them instead — the robust choice on short samples.

## Conventions

**Discriminator field on adapters**: adapters of the seams whose Protocol declares one — `VolatilityProcess`, `ErrorDistribution`, `SVDynamics`, plus the `ConjugateVolatility` base — declare their registry key as `name: Literal["x"] = "x"`, never `ClassVar`. `IdentificationScheme`, `Prior` and `Sampler` adapters carry no discriminator, because those Protocols do not declare one; they are selected by type or by an estimator-local registry instead. The Pydantic rationale and the class-level-access sharp edge are documented on `impulso._base`.

**Scheme-prefixed diagnostic keys**: every key a scheme writes into its **identification diagnostics** names its scheme family in a prefix, so entries surfaced onto shared `attrs` can never collide with — or mislabel themselves as — another scheme's. The existing spellings are public behaviour and do not get tidied.

**The result quartet**: every result carrying posterior draws exposes `median()`, `hdi()`, `to_dataframe()` and `plot()`, and plotting consumes only that interface. Results that are not posterior-shaped — the Granger and the stationarity / cointegration / integration-order / lag-order diagnostics — deliberately opt out and offer `summary()` instead. "Result object" therefore does not imply the quartet; ask which shape it is.

## Flagged ambiguities

- "SV" is both a noun (the model family — *stochastic volatility*) and an adjective ("an SV adapter"). The class `StochasticVolatility` is the canonical noun reference; the adjective form is fine in prose after the term has been spelled out.
- "Volatility" alone is ambiguous between *volatility process* (the seam) and *volatility paths* (the per-variable σ_i,t time series, useful for plotting). Be explicit when the distinction matters.
- "Minnesota prior" now denotes two distinct encodings: the independent-Normal `MinnesotaPrior` (NUTS path) and the conjugate `NIWPrior` (`ConjugateVAR`). Name the estimator when it matters.
- "Σ" means the *scale* matrix under Student-t errors and the covariance under Gaussian errors. `sigma()` returns the same object either way; when the number has to be a variance, say so and use `innovation_covariance()`.
- "Counterfactual" in the wider literature spans shock-path edits (Impulso's meaning), policy-rule replacement (Sims–Zha style; out of scope), and Lucas-robust constructions (McKay–Wolf; out of scope). When comparing with external work, say which one is meant.
- `B` vs `P`. `B` is the reduced-form lag-coefficient block of the **posterior schema** (and `B_exog` its exogenous sibling); `P` is the structural shock matrix an identification scheme produces. Both are "the coefficient matrix" in casual speech and the literature uses `B` for either. Say which.
