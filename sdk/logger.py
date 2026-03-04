"""[sdk/logger). — HTTP logger for customer runs."""

from sdk.exceptions import TBoundAuthError
from sdk.buffer import save_to_buffer, flush_buffer


def log_run(api_url: str, api_key: str, project_id: str, run_data: dict) -> dict:
    # 1. Try to flush buffer first
    try:
        flush_buffer(project_id, api_url, api_key)
    except Exception:
        pass

    # 2. POST run
    try:
        import httpx
        r = httpx.post(
            f"{api_url}/v1/runs",
            json=run_data,
            headers={"X-TBound-Key": api_key},
            timeout=15,
        )
        if r.status_code == 401:
            raise TBoundAuthError("Invalid API key")
        r.raise_for_status()
        return r.json()
    except TBoundAuthError:
        raise
    except Exception:
        # 3. Buffer on connection error
        save_to_buffer(project_id, run_data)
        return {"buffered": True}
