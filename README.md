# [t-bound)

Scaling-law experiments for predicting the minimum model size needed to hit
a target accuracy, before training it. This started as an attempt to build
a product around that idea. It's released here instead as an open dataset,
a working fitting pipeline, and a documented negative result: **the core
prediction breaks down badly in exactly the regime where it would matter
most.**

## TL;DR

We fit `Accuracy(N) = a - b·N^(-α)` (a standard scaling-law form) to 1,801
real training runs across 7 datasets, 3 domains, and 2 architectures. The
fits look excellent by every conventional metric — R² above 0.95 in most
cases. But when we used those fits to predict the smallest model that would
reach a target accuracy close to what we'd already observed, the prediction
was off by 42–71% (mean 63%) across every dataset we tested. The fits
interpolate well. They extrapolate badly near the accuracy ceiling. That's
the headline result of this repo.

## What's actually here

- **1,801 real training runs** across 7 datasets (`results/runs.csv`):
  CIFAR-10, CIFAR-100, STL-10 (vision, CNN), AG News (NLP, Transformer),
  Covertype, MNIST-as-tabular, HIGGS (tabular, MLP).
- **A working scaling-law fitting pipeline** (`scaling/`): power-law fits,
  a 2D Chinchilla-style N×D surface fit, bootstrap confidence intervals.
- **Diagnostic tooling that found the problem** (`scripts/`):
  `analyze_exponents.py` and `test_conditional_stability.py` are what
  actually surfaced the extrapolation failure below — not a hunch, a
  measured result you can reproduce from the included data.
- **A dashboard** (`dashboard/`) that reads `results/runs.csv` directly and
  lets you explore this interactively — pick a dataset, move the target
  accuracy slider, and watch the reliability warning appear as you approach
  that dataset's fitted ceiling.
- **A reference API + SDK** (`api/`, `sdk/`) — a working FastAPI backend and
  Python client for logging runs and getting recommendations over HTTP.
  This is what we built when we still thought this would be a hosted
  product. **There is no hosted service.** The SDK's default API URL
  (`https://api.tbound.ai`) does not resolve to anything live. Run the API
  locally if you want to use it; see [Running the reference API/SDK](#running-the-reference-apisdk) below.

## The finding, in more detail

For each dataset, we ran a full grid sweep over model size (6 sizes, ×3
dataset fractions, ×lr/batch grid — 216–288 runs per dataset), took the best
validation accuracy at each model size, and fit a power law. Holdout
validation asked: given a target accuracy close to what the largest model
in the grid actually achieved, does the fitted curve correctly predict the
smallest model size that reaches it?

| dataset | architecture | true N* | predicted N* | error |
|---|---|---|---|---|
| cifar10 | cnn | 2,580,394 | 755,187 | 70.7% |
| cifar100 | cnn | 4,629,476 | 2,689,857 | 41.9% |
| stl10 | cnn | 4,583,306 | 1,442,273 | 68.5% |
| covertype | mlp | 1,639,431 | 628,641 | 61.7% |
| mnist_tabular | mlp | 126,538 | 37,503 | 70.4% |
| higgs | mlp | 29,250 | 10,301 | 64.8% |

Every single one is flagged "HIGH ERROR" by our own diagnostic script. This
isn't one bad dataset — it's consistent across vision and tabular domains,
across CNN and MLP architectures, with errors all clustering in the 42–71%
range.

**Why this happens:** a power law approaching its asymptote (`a`) gets very
flat in accuracy near the top. A target accuracy that's only 1–2 points
below the best accuracy you've observed sits in that flat region, where a
tiny vertical uncertainty in the fitted ceiling translates into a huge
horizontal swing in the predicted N. We confirmed this directly with
leave-one-out cross-validation: prediction error for points in the *middle*
of the model-size range is typically under 10%; for points near the top of
the range it routinely exceeds 100%, and in a couple of cases exceeded
1000%.

We also checked whether the scaling exponent α itself is even stable. It
isn't, for most datasets. Re-fitting α using only data at different dataset
fractions (`scripts/test_conditional_stability.py`) gives meaningfully
different exponents depending on how much data you fit on:

```
cifar10:        D=0.1 → α=0.40   D=1.0 → α=0.62   (verdict: UNSTABLE)
cifar100:       D=0.1 → α=0.77   D=1.0 → α=0.40   (verdict: UNSTABLE)
covertype:      D=0.1 → α=0.35   D=1.0 → α=0.31   (verdict: CONDITIONALLY_STABLE)
higgs:          D=0.1 → α=0.64   D=1.0 → α=0.64   (range still 0.10, borderline)
```

Only covertype's exponent held up as something close to a stable property
of the architecture/domain pair. For the rest, α depends on how much
training data you used to fit it — which undermines the idea of a portable
prior that transfers across customers with different dataset sizes, at
least for the functional form we used.

We also tried a 2D N×D surface fit (`scaling.surface_fit.fit_nd_surface`,
the Chinchilla-style joint model+data scaling law) to see if conditioning
on dataset size fixed the extrapolation problem. It didn't — the fitted
ceiling parameter pushed against the optimizer's upper bound, which is a
sign of an ill-conditioned fit, not a more reliable one.

**What we think this means:** scaling-law extrapolation in the form we
used is trustworthy for "how much accuracy will I get at this model size,
somewhere in the middle of my swept range," and not trustworthy for "what's
the smallest model that gets me basically the accuracy I'm already seeing."
The second question is the one a "train less" product would actually need
to answer reliably, which is why we stopped pursuing this as a product
rather than continuing to paper over the gap.

## Reproducing the finding

No GPU, no raw datasets needed — everything below runs against the included
`results/runs.csv`.

```bash
pip install -r requirements.txt
python3 scripts/analyze_exponents.py        # exponent stability + N* holdout errors
python3 scripts/test_conditional_stability.py  # per-dataset stability verdicts
```

Both scripts are idempotent reads except where noted — `analyze_exponents.py`
only writes `results/exponent_analysis.csv`. Separately,
`scripts/backfill_dataset_features.py` and `scripts/migrate_runs_csv.py`
**modify `results/runs.csv` in place** (writing a timestamped backup first).
Copy the file before running those if you want to keep the original
untouched.

## Exploring the data interactively

```bash
pip install streamlit plotly pandas numpy scipy
streamlit run dashboard/app.py
```

Pick a dataset from the sidebar, then try the Recommendation page: set the
target accuracy slider near the top of that dataset's range and watch the
reliability warning appear. Back it off and the warning disappears. This is
the most direct way to feel the boundary of where these fits can be trusted.

## What's solid vs. unfinished

**Solid:** the data loaders, the trainer, the model-size power-law fit, the
bootstrap CI code, the diagnostic scripts above, and the dashboard. All of
this is exercised against real data and runs cleanly.

**Unfinished / known issues:**
- Two of the originally planned NLP datasets (Yahoo Answers, DBpedia) were
  never run.
- A subset of the AG News grid got stuck near random-chance accuracy at
  certain learning-rate/model-size combinations — likely an optimization
  failure at those specific hyperparameters rather than a data issue. This
  visibly distorts AG News's fitted curve; treat AG News results with extra
  skepticism.
- A Gaussian-process meta-model for the prior (`meta_model/`) was scoped
  but never implemented — it's a documented stub, not a bug.
- Timeseries support (LSTM/TCN) was planned but never started.

## Running the reference API/SDK

The `api/`, `sdk/`, and `dashboard`'s original API-backed pages (now
bypassed by `dashboard/data_source.py`, see above) are a complete,
functional reference implementation of "scaling-law fitting as a hosted
service" — a FastAPI backend with SQLite persistence, auth, and a Python
SDK client with offline buffering. We're including it because it's working
code that might be a useful template for similar projects, not because
this project is an active service.

To run it locally:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

```python
import sdk.client as tbound
tbound.init(api_key="...", project="my-project", api_url="http://localhost:8000")
# create a project first via POST /v1/projects if you don't have a key
```

## License

[Source-available, non-commercial](LICENSE). You're free to read, run,
modify, and learn from this for personal or academic purposes. Commercial
use requires our permission — we may build a commercial product on this
work in the future and are reserving that right. If you want to use this
commercially, reach out via the repo first.

## Background

This was originally planned as a two-person SaaS product: pip-installable
SDK, hosted API, a prior database meant to compound across customers. We
got through grid sweeps for 7 of 9 planned datasets (1,801 real runs) before
finding the extrapolation problem documented above, at which point
continuing to build out the product stopped making sense until the
underlying method was more reliable. We may revisit this in the future —
for now, it's released as-is.
