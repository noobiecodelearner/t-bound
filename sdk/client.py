"""[sdk/client). — main customer interface for [t-bound)."""

import sys
from sdk.exceptions import TBoundAuthError, TBoundInsufficientRunsError, TBoundAPIError
from sdk.logger import log_run
from sdk.recommender import get_recommendation

_state = {
    "api_key": None,
    "project_id": None,
    "api_url": "https://api.tbound.ai",
    "architecture": None,
    "domain": None,
    "_run_count": 0,
}


def init(
    api_key: str,
    project: str,
    architecture: str = "cnn",
    domain: str = "vision",
    api_url: str = None,
):
    """Initialize tbound. Validates API key immediately."""
    if api_url:
        _state["api_url"] = api_url

    _state["api_key"] = api_key
    _state["project_id"] = project
    _state["architecture"] = architecture
    _state["domain"] = domain
    _state["_run_count"] = 0

    # Validate key immediately
    try:
        import httpx
        r = httpx.get(
            f"{_state['api_url']}/v1/health",
            timeout=5,
        )
        # Now verify key by hitting a protected endpoint
        r2 = httpx.get(
            f"{_state['api_url']}/v1/projects/{project}/runs",
            headers={"X-TBound-Key": api_key},
            timeout=5,
        )
        if r2.status_code == 401:
            raise TBoundAuthError("Invalid API key")
    except TBoundAuthError:
        raise
    except Exception:
        # API unreachable — allow offline buffering, don't fail init
        pass


def log(
    params: int,
    val_accuracy: float,
    num_steps: int,
    learning_rate: float,
    batch_size: int,
    dataset_fraction: float = 1.0,
):
    """Log a training run. Never raises — always safe to call in training loop."""
    try:
        run_data = {
            "params": params,
            "val_accuracy": val_accuracy,
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dataset_fraction": dataset_fraction,
        }
        result = log_run(
            api_url=_state["api_url"],
            api_key=_state["api_key"],
            project_id=_state["project_id"],
            run_data=run_data,
        )
        if not result.get("buffered"):
            _state["_run_count"] += 1
    except TBoundAuthError:
        print("[tbound] Warning: invalid API key. Run not logged.", file=sys.stderr)
    except Exception as e:
        print(f"[tbound] Warning: unexpected error logging run: {e}", file=sys.stderr)


def recommend(target_accuracy: float = 0.85):
    """Get scaling recommendation. Raises TBoundInsufficientRunsError if 0 runs."""
    if _state["_run_count"] < 1:
        raise TBoundInsufficientRunsError("Log at least 1 run before calling recommend().")
    return get_recommendation(
        api_url=_state["api_url"],
        api_key=_state["api_key"],
        target_accuracy=target_accuracy,
    )
