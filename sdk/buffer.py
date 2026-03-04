"""[sdk/buffer). — local buffer for offline runs."""

import json
import os
from pathlib import Path


def _buffer_path(project_id: str) -> Path:
    home = Path.home() / ".tbound"
    home.mkdir(exist_ok=True)
    return home / f"buffer_{project_id}.jsonl"


def save_to_buffer(project_id: str, run_data: dict) -> None:
    path = _buffer_path(project_id)
    with open(path, "a") as f:
        f.write(json.dumps(run_data) + "\n")


def flush_buffer(project_id: str, api_url: str, api_key: str) -> int:
    path = _buffer_path(project_id)
    if not path.exists():
        return 0

    lines = path.read_text().strip().splitlines()
    if not lines:
        return 0

    sent = 0
    try:
        import httpx
        for line in lines:
            run_data = json.loads(line)
            r = httpx.post(
                f"{api_url}/v1/runs",
                json=run_data,
                headers={"X-TBound-Key": api_key},
                timeout=10,
            )
            if r.status_code == 200:
                sent += 1
        # clear buffer after successful flush
        path.write_text("")
    except Exception:
        pass

    return sent
