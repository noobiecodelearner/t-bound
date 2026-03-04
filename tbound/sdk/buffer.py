"""
[sdk/buffer). — local buffer for offline resilience.

DAYANCH: Stores runs locally when API is unreachable. Replays on reconnect.
Customer training loop must never break due to API downtime.

WHAT TO IMPLEMENT:

class Buffer:
    '''
    JSONL file buffer at ~/.tbound/buffer_{project_id}.jsonl

    Why JSONL: one run per line, append-only, survives crashes,
    human-readable, easy to replay.
    '''

    def __init__(self, project_id: str):
        self.path = Path.home() / ".tbound" / f"buffer_{project_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def push(self, run_data: dict) -> None:
        '''Append one run to the buffer file.'''
        ...

    def pop_all(self) -> list:
        '''Read all buffered runs and clear the file.'''
        ...

    def size(self) -> int:
        '''Number of buffered runs.'''
        ...

    def is_empty(self) -> bool:
        ...

REPLAY LOGIC (implement in sdk/client.py flush()):
    1. Read all runs from buffer
    2. Send each to API via sdk/logger.py
    3. If send succeeds: remove from buffer
    4. If send fails: leave in buffer, stop replaying
    5. Log how many were replayed

NOTES:
- Never raise exceptions from push() — buffer must always work
- pop_all() should be atomic: read file, clear file, return contents
- If buffer file is corrupted: delete it and start fresh (log warning)
"""

# TODO: implement this file
raise NotImplementedError("sdk/buffer.py not yet implemented — see docstring")
