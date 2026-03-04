"""
[meta_model/prior_updater). — absorbs customer curves into GP training set.

V2 — NOT implemented at launch. Stub only.

When implemented, this module will:
    Extract scaling exponents from completed customer projects.
    Add them to the GP training dataset.
    Retrain the GP.
    Update prior.csv with improved aggregate statistics.

Planned interface:
    absorb_customer_curve(project_id, alpha, architecture_features)
        Called when a customer project reaches 10+ runs and alpha converges.
        Adds the (architecture_features, alpha) pair to GP training data.
        Triggers GP retraining.
        Updates prior.csv aggregate stats for the relevant group.

Trigger condition:
    Run count >= 10 AND the last 3 alpha estimates vary by < 0.02
    (alpha has converged — adding more runs won't change it)

Privacy:
    Only the fitted alpha is absorbed — never raw (params, accuracy) pairs.
    The customer's actual training data never leaves their project.
"""

def absorb_customer_curve(*args, **kwargs):
    raise NotImplementedError("prior_updater is a V2 feature.")
