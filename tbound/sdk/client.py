"""
[sdk/client). — main customer interface for [t-bound).

DAYANCH: Build the public API that customers interact with.
This is the only file customers import. Keep it simple.

--- STANDARD USAGE (benchmark datasets, small data) ---

    import tbound

    tbound.init(
        api_key="tb_key_xxx",
        project="my-experiment",
        architecture="cnn",
        domain="vision",
    )

    tbound.log(
        params=342000,
        val_accuracy=0.847,
        num_steps=10000,
        learning_rate=0.001,
        batch_size=128,
        dataset_fraction=1.0,    # relative — fraction of training pool used
    )

--- LARGE DATA USAGE (500GB datasets, subsampling protocol) ---

    tbound.init(
        api_key="tb_key_xxx",
        project="my-experiment",
        architecture="transformer",
        domain="nlp",
        full_dataset_size=10_000_000,   # total samples in their full dataset
                                         # tells us the extrapolation target
    )

    # They run 3-4 fixed model size experiments varying only dataset_size.
    # They never touch most of their data — just small representative subsets.
    tbound.log(
        params=1_200_000,
        val_accuracy=0.847,
        num_steps=10_000,
        dataset_size=25_000,     # absolute sample count — alternative to dataset_fraction
    )

    tbound.log(
        params=1_200_000,
        val_accuracy=0.863,
        num_steps=10_000,
        dataset_size=50_000,
    )

    tbound.log(
        params=1_200_000,
        val_accuracy=0.871,
        num_steps=10_000,
        dataset_size=100_000,
    )

    # Recommendation extrapolates delta to their full 10M sample dataset
    rec = tbound.recommend(target_accuracy=0.90)
    rec = tbound.recommend(compute_budget_hours=10)

--- DATASET SIZE NORMALIZATION ---

    When dataset_size (absolute) is provided instead of dataset_fraction:
        dataset_fraction = dataset_size / full_dataset_size

    full_dataset_size must be set in tbound.init() for this to work.
    If full_dataset_size is not set and dataset_size is provided:
        store dataset_size as-is
        set dataset_fraction = None
        fitting_service handles None fractions by fitting on raw dataset_size values

    When dataset_size < 0.01 * full_dataset_size:
        log a warning in the run metadata: "subsampling_extrapolation=True"
        dashboard will show: "Extrapolating to full dataset — consider stratified sampling"

WHAT TO IMPLEMENT:

1. tbound.init(api_key, project, architecture, domain, full_dataset_size=None)
   - Store config in module-level _state dict
   - Validate api_key against API via GET /v1/health
   - Raise TBoundAuthError if invalid
   - Initialize the buffer (sdk/buffer.py)
   - full_dataset_size is optional — only needed for large-data customers

2. tbound.log(**kwargs)
   - Accept EITHER dataset_fraction OR dataset_size, not both
   - If dataset_size provided and full_dataset_size known:
       compute dataset_fraction = dataset_size / full_dataset_size
       set subsampling_extrapolation = dataset_size < 0.01 * full_dataset_size
   - If dataset_size provided and full_dataset_size NOT known:
       dataset_fraction = None
       pass dataset_size through as-is
   - Required fields: params, val_accuracy
   - Warn (do not raise) if num_steps is missing
   - Send to API: POST /v1/runs
   - If API unreachable: buffer locally via sdk/buffer.py
   - Never raise an exception — customer training loop must not break

3. tbound.recommend(target_accuracy=None, compute_budget_hours=None)
   - Exactly one of target_accuracy or compute_budget_hours must be set
   - Send to API: GET /v1/recommend
   - Return typed Recommendation object (see sdk/recommender.py)
   - If < 1 run logged: raise TBoundInsufficientRuns with helpful message

4. tbound.flush()
   - Replay buffered runs from sdk/buffer.py
   - Call automatically at start of tbound.log() and tbound.recommend()

IMPORTANT RULES:
- tbound.log() must NEVER raise an exception
- tbound.recommend() can raise TBoundInsufficientRuns — that's informative
- All API calls go through sdk/logger.py and sdk/recommender.py
- Module-level state: _state dict, _buffer Buffer instance

_state = {
    "api_key":            None,
    "project_id":         None,
    "api_url":            "https://api.tbound.ai",
    "architecture":       None,
    "domain":             None,
    "full_dataset_size":  None,   # NEW — for large-data subsampling
}
"""

# TODO: implement this file
raise NotImplementedError("sdk/client.py not yet implemented — see docstring")