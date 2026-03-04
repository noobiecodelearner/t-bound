# [schema). — the complete data contract for [t-bound).

Do not change column names without both team members agreeing.
This document is the single source of truth for all data structures.

---

## Core principle

The scaling law fits the customer's numbers, not ours.
Exponents (α, β, γ, δ) are scale-invariant — they transfer across customers.
Absolute values (a, b) are customer-specific — they never leave a project.

---

## runs.csv

Every training run — internal or customer — logged here.
One row per training run. Written by ExperimentLogger in utils/logger.py.

| Column | Type | Values | Notes |
|---|---|---|---|
| run_id | str | `run_abc12345` | UUID hex prefix |
| source | str | `internal` \| `customer` | Who ran the experiment |
| project_id | str | any | Customer project name or internal sweep name |
| timestamp | str | ISO 8601 UTC | When the run completed |
| domain | str | `vision` \| `nlp` \| `tabular` | Task domain |
| dataset | str | see table below | Specific dataset name |
| architecture_family | str | `cnn` \| `transformer` \| `mlp` | Broad family |
| num_classes | int | ≥ 2 | Output classes |
| dataset_size | int | > 0 | Actual training samples used this run (absolute count) |
| full_dataset_size | int | nullable | Total samples in customer's full dataset — passed once via tbound.init |
| subsampling_extrapolation | bool | False | True when dataset_size < 1% of full_dataset_size — triggers CI inflation and dashboard warning |
| dataset_fraction | float | (0, 1] | Fraction of full training pool |
| sweep_type | str | `n_d_lr_grid` \| `batch_grid` | Which sweep this run belongs to |
| params | int | > 0 | Trainable parameter count |
| learning_rate | float | > 0 | LR used |
| batch_size | int | > 0 | Batch size used |
| weight_decay | float | ≥ 0 | L2 regularization |
| optimizer | str | `adam` \| `adamw` \| `sgd` | Optimizer type |
| num_steps | int | > 0 | Gradient steps (NOT epochs) |
| val_accuracy | float | [0, 1] | **Best** val accuracy across all steps |
| train_accuracy | float | [0, 1] | Final training accuracy |
| best_step | int | ≥ 1 | Step at which best val accuracy occurred |
| train_time_seconds | float | > 0 | Wall-clock training time |
| energy_kwh | float | > 0 | Estimated energy consumed |
| compute_flops | int | > 0 | Estimated FLOPs for this run |
| generalization_gap | float | any | train_accuracy - val_accuracy |
| gen_warning | str | `LOW` \| `MEDIUM` \| `HIGH` | Overfitting risk |

### Supported datasets

| domain | dataset | classes | approx_size | architecture_family |
|---|---|---|---|---|
| vision | cifar10 | 10 | 50K | cnn |
| vision | cifar100 | 100 | 50K | cnn |
| vision | stl10 | 10 | 5K | cnn |
| nlp | yahoo | 10 | 1.4M | transformer |
| nlp | agnews | 4 | 120K | transformer |
| nlp | dbpedia | 14 | 560K | transformer |
| tabular | covertype | 7 | 581K | mlp |
| tabular | otto | 9 | 61K | mlp |
| tabular | higgs | 2 | 1M (of 11M) | mlp |

---

## fits.csv

One row per fitted exponent per dataset.
Written by FitsLogger in utils/logger.py after each sweep completes.

Each dataset produces up to 4 rows (one per exponent type):
- `alpha` — model size exponent
- `beta`  — lr exponent
- `gamma` — batch exponent
- `delta` — dataset size exponent

| Column | Type | Notes |
|---|---|---|
| fit_id | str | UUID |
| dataset | str | Which dataset this curve was fitted on |
| architecture_family | str | cnn \| transformer \| mlp |
| domain | str | vision \| nlp \| tabular |
| dataset_size_regime | str | small \| medium \| large |
| sweep_type | str | n_d_lr_grid \| batch_grid |
| exponent_type | str | alpha \| beta \| gamma \| delta |
| exponent_value | float | The fitted exponent |
| a | float | Accuracy ceiling (only for alpha fit, blank otherwise) |
| b | float | Scaling coefficient (only for alpha fit, blank otherwise) |
| c | float | lr coefficient (only for beta fit, blank otherwise) |
| d | float | batch coefficient (only for gamma fit, blank otherwise) |
| r2 | float | Coefficient of determination |
| mae | float | Mean absolute error |
| n_runs_used | int | Number of runs used for this fit |
| optimal_n | float | N* at target accuracy (only for alpha, blank otherwise) |
| optimal_lr | float | lr*(N*) (only for beta, blank otherwise) |
| optimal_batch | float | batch*(N*) (only for gamma, blank otherwise) |
| optimal_dataset_fraction | float | D* (only for delta, blank otherwise) |
| ci_lower_95 | float | Lower bound of 95% CI on optimal value |
| ci_upper_95 | float | Upper bound of 95% CI on optimal value |
| bootstrap_success | bool | Whether bootstrap CI succeeded |
| timestamp | str | ISO 8601 UTC |

---

## prior.csv

Aggregated prior database. One row per (architecture_family, domain, dataset_size_regime, sweep_type) group.
Written by scripts/build_prior.py — never by individual experiments.

**What is stored: exponents only. Never a or b.**
Exponents are scale-invariant — they transfer across customers with different
pipelines, normalizations, and val splits. a and b are customer-specific and
live on the customer's measurement scale.

| Column | Type | Notes |
|---|---|---|
| prior_id | str | UUID |
| architecture_family | str | cnn \| transformer \| mlp \| resnet \| vit \| etc. |
| domain | str | vision \| nlp \| tabular \| timeseries \| medical |
| dataset_size_regime | str | small (<20K) \| medium (20K–200K) \| large (>200K) |
| sweep_type | str | n_d_lr_grid \| batch_grid |
| n_source_curves | int | Number of fitted curves contributing to this prior |
| n_source_implementations | int | Number of distinct implementations (>1 = family-level prior) |
| alpha_mean | float | Mean model size exponent across source curves |
| alpha_std | float | Std dev of alpha — **determines cold start CI width** |
| alpha_min | float | Minimum alpha observed |
| alpha_max | float | Maximum alpha observed |
| beta_mean | float | Mean lr exponent |
| beta_std | float | Std dev of beta |
| gamma_mean | float | Mean batch exponent |
| gamma_std | float | Std dev of gamma |
| delta_mean | float | Mean dataset size exponent |
| delta_std | float | Std dev of delta |
| decomposition_alpha | float | Alpha predicted by additive model: μ + arch_offset + domain_offset |
| decomposition_residual | float | observed alpha minus decomposition_alpha (small = transfer reliable) |
| transfer | bool | True = decomposition reliable for novel arch transfer. False = use own alpha only |
| last_updated | str | ISO 8601 UTC |
| source | str | internal \| literature \| customer |

### Prior fallback chain

When a customer declares their architecture and domain, prior lookup follows:
1. Exact match: architecture_family + domain + dataset_size_regime + sweep_type
2. Family + domain: drop regime
3. Domain only: drop family
4. decomposition.json additive estimate (only if transfer=True for nearest cell, inflate alpha_std * 1.5)
5. literature_values.csv match
6. Global fallback: alpha_mean=0.30, alpha_std=0.15

### decomposition.json

Written by scripts/build_prior.py alongside prior.csv. Contains the fitted additive model:

```
alpha(arch, domain) = mu + arch_offset[arch] + domain_offset[domain]
```

Fields:
- mu: global mean alpha across all observed cells
- arch_offsets: {arch: offset} — how much each family shifts alpha up/down
- domain_offsets: {domain: offset} — how much each domain shifts alpha up/down
- predicted: {arch_domain: alpha} — predicted alpha for all (arch, domain) combinations
- residuals: {arch_domain: residual} — observed minus predicted for cells we ran
- residual_threshold: 0.03 — residual above this sets transfer=False in prior.csv

Used by prior_service.py to estimate alpha for novel architectures not in prior.csv.
Never commit this file — it is derived from runs.csv and rebuilt by build_prior.py.
3. Domain only: drop family
4. Global fallback: α=0.30 ± 0.15 (valid for any neural architecture, any domain)

---

## val_splits/*.npz

Generated once by scripts/generate_val_splits.py. Committed to repo. Never regenerated.

Each file: `{dataset}_seed42.npz`

Contents:
```
train_idx: np.ndarray  — indices into training pool (never used for val)
val_idx:   np.ndarray  — indices into val set (fixed, never changes)
seed:      int         — 42
dataset:   str         — dataset name
n_total:   int         — total training samples before split
n_val:     int         — number of val samples
```

Val split sizes:
```
cifar10:   5,000  val (10% of 50K)
cifar100:  5,000  val (10% of 50K)
stl10:       500  val (10% of 5K)
yahoo:    10,000  val
agnews:   10,000  val
dbpedia:  10,000  val
covertype: 10,000 val
otto:       5,000 val
higgs:     10,000 val (of 1M subset)
```

---

## literature_values.csv

Manually curated from peer-reviewed papers.
Source of prior data before any internal experiments are complete.

| Column | Notes |
|---|---|
| architecture_family | cnn \| transformer \| mlp etc. |
| domain | vision \| nlp \| tabular |
| sweep_type | model_size (maps to n_d_lr_grid) |
| alpha_mean | Reported or derived exponent |
| alpha_std | Reported uncertainty or estimated from range |
| source_paper | Full citation |
| notes | Architecture specifics |

Key sources:
- Rosenfeld et al. 2020 — vision CNNs
- Kaplan et al. 2020 — language model scaling
- Zhai et al. 2022 — vision transformers
- Gordon et al. 2021 — NLP transformers
- Abnar et al. 2022 — cross-architecture scaling

---

## Customer contract

### What customers must keep constant within a project
```
Val split          — same samples every run, same size
Normalization      — same mean and std every run
Num steps          — same training budget every run
Optimizer type     — same every run
Weight decay       — same every run
Loss function      — same every run
Data augmentation  — same every run
Architecture family — only scale varies across runs
```

### What customers log
```python
tbound.log(
    params=model.count_parameters(),
    val_accuracy=best_val_acc,   # BEST across all steps, not final
    num_steps=10000,
    learning_rate=0.001,
    batch_size=128,
    dataset_fraction=1.0,
)
```

### What customers receive
```python
rec.optimal_n                  # minimum params to hit target accuracy
rec.optimal_lr                 # optimal learning rate at N*
rec.optimal_batch              # optimal batch size
rec.optimal_dataset_fraction   # optimal D* (Chinchilla path only)
rec.expected_accuracy          # predicted accuracy at N*
rec.ci_lower, rec.ci_upper     # 95% confidence interval
rec.confidence                 # very_low | low | medium | high
rec.compute_saved              # fraction saved vs naive approach
rec.energy_saved_kwh           # energy savings
rec.carbon_saved_g             # CO₂ savings in grams
rec.runs_used                  # how many customer runs contributed
rec.prior_weight               # fraction of recommendation from prior vs data
```

---

## Exponent definitions

These are scale-invariant. They describe how accuracy changes, not what accuracy is.

```
α (alpha) — model size scaling exponent
  Accuracy*(N) = a - b · N^(-α)
  Typical range: 0.15–0.45 across all architectures
  Higher α → accuracy improves faster with more parameters

β (beta) — lr scaling exponent
  lr*(N) = c · N^(-β)
  Typical range: 0.05–0.25
  Optimal lr decreases as model size increases

γ (gamma) — batch scaling exponent
  batch*(N) = d · N^(γ)
  Typical range: 0.05–0.20
  Optimal batch increases slightly with model size

δ (delta) — dataset size scaling exponent
  Accuracy*(D) = a - b · D^(-δ)
  Typical range: 0.15–0.55
  Higher δ → dataset size matters more for this task

a — accuracy ceiling (customer-specific, never stored in prior)
  The accuracy the model approaches as N → ∞
  Depends on task difficulty, normalization, val split, etc.

b — scaling coefficient (customer-specific, never stored in prior)
  Controls how quickly accuracy rises from 0 toward ceiling
  Also depends on measurement pipeline
```

---

## Confidence levels

| Runs logged | Confidence | Description |
|---|---|---|
| 0 | `very_low` | Prior only. Wide CI. Rough estimate. |
| 1–2 | `low` | Prior + 1-2 data points. CI narrowing. |
| 3–5 | `medium` | Data starting to dominate. Usable recommendation. |
| 6+ | `high` | Customer data dominates. Prior nearly irrelevant. |
| 10+ | `high` | Full fit. Prior completely irrelevant. Tight CI. |

---

## Dataset size regimes

Used to select the right prior when customer's dataset size is known.

```
small:  < 20,000 samples
medium: 20,000 – 200,000 samples
large:  > 200,000 samples
```

---

## Fixed assumptions for internal experiments

These are held constant across all of Naeem's internal experiments to ensure
the prior database is internally consistent.

```
Optimizer:      Adam (vision, tabular), AdamW (nlp)
Batch size:     128 during N × D × lr grid
Fixed batch:    128 during N × D × lr grid
Num steps:      10,000 (vision), 5,000 (nlp), 8,000 (tabular)
Seed:           42 for val splits, 0 for training
Loss:           CrossEntropyLoss
Activation:     ReLU
Init:           He (MLP), Xavier (Transformer), default (CNN)
```