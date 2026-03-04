"""
[sdk/logger). — HTTP logger for customer runs.

DAYANCH: Handles sending runs to the API.

WHAT TO IMPLEMENT:

class SDKLogger:
    def __init__(self, api_url: str, api_key: str, project_id: str):
        ...

    def send_run(self, run_data: dict) -> dict:
        '''
        POST /v1/runs with run_data.
        Headers: X-TBound-Key: {api_key}
        Body: JSON with all run fields

        On success: return response JSON {run_id, confirmation}
        On failure: raise TBoundAPIError (caller handles buffering)

        Timeout: 5 seconds. Do not wait longer.
        '''
        ...

NOTES:
- Use requests library or httpx
- Always include X-TBound-Key header
- Timeout must be short — customer training loop is waiting
- Do not retry here — retry logic is in sdk/buffer.py
"""

# TODO: implement this file
raise NotImplementedError("sdk/logger.py not yet implemented — see docstring")
