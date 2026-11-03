# One latent series at lag order 1 gets a truncated own lag, not the stationarity Potential

ADR-0016 keeps a latent block stationary with the `latent_stationarity` Potential: 0 where the companion matrix of the latent-on-latent block has spectral radius below 1, −∞ elsewhere, and a zero gradient everywhere. NUTS cannot see that wall coming and can only diverge against it. What tells a latent series apart from white noise is its persistence, so a caller sets a persistent own-lag prior mean, 0.5 to 0.8, and the posterior then sits about two prior standard deviations from the wall. The adversarial review of pymc-labs/pymc-marketing#3293 (item P1.2) measured divergence rates of up to a third of draws on null data in that setup. Impulso's default own-lag mean of 1 is worse: it starts the latent equation outside the region, which forces an initial value on `B` and leaves the host model unclonable, and ADR-0016 recorded 212–290 divergent draws of 400 under it. The `eig` inside the Potential is also what keeps the JAX backend from differentiating the model (#378).

For exactly one latent series and `n_lags = 1` the latent block is the 1×1 own lag `B[0, 0]`. Its companion matrix is that number, so the stationary region is the interval `|B[0, 0]| < 1`, and the constraint can be exact and smooth. `build_in_model` then registers three variables in place of the single `Normal` `B`:

- `latent_own_lag`: the own lag's `Normal`, with its Minnesota mean and standard deviation, truncated to (−1, 1). PyMC samples it through its default interval transform.
- `B_free`: a flat `Normal` over every other lag coefficient, with the Minnesota means and standard deviations of those entries.
- `B`: the `Deterministic` scattered from the two, under its existing name and `(var, coeff)` dims.

No Potential and no initial value are registered in this case. Every other case is unchanged. Without latent series `B` is the single `Normal`. With `n_lags > 1` or several latent series it is the single `Normal` with the Potential and the stationary initial value of ADR-0016.

**The posterior does not change.** The Potential truncated `B`'s Normal prior to `|B[0, 0]| < 1` and left every other entry alone. A Normal truncated to that interval is the same density on the own lag, up to a normalising constant that depends only on the prior's fixed mean and standard deviation, and the other entries keep their Normals. What changes is the geometry NUTS sees: in the transformed coordinate the boundary is infinitely far away, and the log density is smooth all the way to it.

`_register_coefficients` is written as a free block plus a constrained block, with the constrained block at the flat position of `B[0, 0]`. A general parametrisation of the latent block replaces the constrained block and widens its positions; it does not need a second rewrite.

## Considered options

- **A partial-autocorrelation parametrisation of the general latent block** — deferred to its own issue. Mapping unconstrained variables onto the stationary region of the latent VAR(p) is the general fix for `n_lags > 1` and several latent series, where bounding each own lag does not bound the spectral radius. For one series at lag 1 the partial autocorrelation is the own lag, so that parametrisation reduces to a bounded own lag, which the truncated Normal already is.
- **A smooth penalty on the spectral radius** — rejected. A soft barrier in place of the wall changes the posterior. The truncation is the belief. A penalty moves prior mass inside the region, which nothing in the data asked for.
- **Sampler guidance alone** — rejected in ADR-0016 already.
- **ADR-0016's bounded own-lag variable, reconsidered.** ADR-0016 rejected a bounded own lag (`tanh` of a free variable) on cost. For this case its costs do not apply or are worth paying. The truncated Normal is the Minnesota Normal on the region, so the latent row's prior does not change, where a `tanh` of a Normal would change it. It adds no parameters: `B_free` and `latent_own_lag` together have as many entries as `B`. `B` survives as a `Deterministic` under its name and dims, so the posterior schema is unchanged. And it needs no initial value, so the ignored `set_initval` on a `Deterministic` that ADR-0016 worried about never arises. ADR-0016 accepted boundary divergences as the price of the Potential; under the default own-lag mean that price was most of the draws.

## Consequences

- The own lag starts at 0, the interval's midpoint, whatever its prior mean, and the free block starts at its prior mean. The latent equation therefore starts inside the region with no initial value, `_latent_b_initval` does not apply to this case, and the host model is clonable (`Model.copy()`, `pm.do`, `pm.observe`) for any own-lag prior mean, the default of 1 included.
- PyMC's start jitter acts in the transformed space, so every jittered start is inside the region and no retry is needed.
- `sample_prior_predictive` draws the own lag from the truncated prior, so prior-predictive latent paths are stationary in this case. The Potential cases still ignore the constraint in prior predictive sampling.
- There is no `eig` in the graph, so the JAX backend differentiates the model in this case. #378 stays open for the Potential cases.
- On the setup of the slow one-latent sampling test under the default own-lag mean of 1, the Potential gave 268–300 divergent draws of 400 on three seeds. The truncated own lag gives none, with R-hat at most 1.011 on the own lag. The slow test allows at most 1% divergent draws and requires R-hat below 1.05.
- The posterior carries two extra variables, `B_free` and `latent_own_lag`. `B` is a `Deterministic`, which PyMC records in the posterior like a free variable, so the posterior schema and `FittedVAR.from_posterior`, which reads `B` by name and ignores variables it does not know, are unchanged. `handles.B` is the `Deterministic`.
- The model's log density includes the truncation's normalising constant, `−log(Φ((1 − μ)/σ) − Φ((−1 − μ)/σ))` for the own lag's prior mean μ and standard deviation σ. It is a constant, so it changes no posterior, but a comparison of the joint density with a reference has to account for it.
- ADR-0016's "Initial values" and "The `latent_stationarity` Potential" bullets now apply only to the other latent cases, and so does its rejection of a bounded own-lag variable.
