"""Vulture whitelist: public API reached only by users, docs and tests.

Regenerate with:
    uv run vulture src --min-confidence 60 --make-whitelist
Add a new entry only for a method or field that is public API. Never add dead code.
"""

_ = None

# impulso.data / impulso.sv.data
_.from_df
_.from_series

# impulso.evidence
hyperparameters  # ModelEvidence pydantic field
_.best
_.to_dataframe

# impulso.fitted.FittedVAR
_.coefficients
_.intercepts
_.innovation_covariance
_.posterior_predictive
_.conditional_forecast
_.dynamic_multiplier
_.set_identification_strategy

# impulso.identification
_.from_zero_restrictions
_.long_run_diagnostics
_.first_stage

# impulso.identified.IdentifiedVAR
_.impulse_response
_.fevd
_.historical_decomposition
_.counterfactual
_.structural_scenario

# impulso.results
_.difference
_.summary
null_hypothesis  # StationarityTestResult pydantic field
_.conclusions
_.pvalues
eigenvalues  # CointegrationTestResult pydantic field
_.rank

# impulso.spec.VAR
_.prior_predictive
_.compile_for_numpyro

# impulso.sv.fitted.FittedSV
_.log_volatility
