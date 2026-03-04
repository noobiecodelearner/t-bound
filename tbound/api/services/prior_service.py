"""
[api/services/prior_service). — prior database lookup service.

DAYANCH — implement this file.

What to do:
    Read prior/prior.csv and literature_values.csv.
    Return the best matching prior for a given (architecture_family, domain, dataset_size_regime).

    This service is called by fitting_service when a customer has < 6 runs.
    After 6 runs, the prior is irrelevant and this service is not called.

Main function to implement:
    get_prior(architecture_family, domain, dataset_size_regime, sweep_type="n_d_lr_grid")
        → returns dict with keys:
            alpha_mean, alpha_std
            beta_mean,  beta_std
            gamma_mean, gamma_std
            delta_mean, delta_std
            n_source_curves  (how many curves this prior is based on)
            source           (internal | literature | customer | global_fallback)
            confidence_in_prior  (low if n_source_curves < 3, medium if < 10, high if 10+)

Fallback chain (MUST follow this order):
    1. Check prior/prior.csv for exact match
       (architecture_family + domain + dataset_size_regime + sweep_type)
    2. Check prior/prior.csv for family + domain match (drop regime)
    3. Check prior/prior.csv for domain-only match (drop family)
    4. Check prior/literature_values.csv for architecture_family + domain match
    5. Global fallback: alpha_mean=0.30, alpha_std=0.15 (valid for any architecture)

You can use the PriorLogger.load_prior_for() method from utils/logger.py
as a starting point — it already implements the fallback chain for prior.csv.
Extend it to also check literature_values.csv.

Notes:
    - Cache the prior CSV in memory after first read — do not re-read on every request
    - If both prior.csv and literature_values.csv have entries, prefer prior.csv
      (it's based on real experiments, not just published papers)
    - The global fallback (alpha=0.30 ± 0.15) is the honest answer when you have
      nothing better — do not invent precise values you don't have
"""
