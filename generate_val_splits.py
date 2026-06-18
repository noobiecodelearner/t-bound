
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from utils.logger import RUNS_FIELDS

RUNS_PATH = Path("results/runs.csv")


def main():
    if not RUNS_PATH.exists():
        print(f"[migrate] {RUNS_PATH} not found - nothing to do.")
        return

    # read with python engine - tolerates column count mismatches -----------
    # sep=',' with engine='python' reads all rows regardless of field count
    print(f"[migrate] Reading {RUNS_PATH} with flexible parser...")
    try:
        df = pd.read_csv(RUNS_PATH, engine="python", on_bad_lines="warn")
    except TypeError:
        # pandas < 1.3: on_bad_lines not supported, use error_bad_lines
        df = pd.read_csv(RUNS_PATH, engine="python", error_bad_lines=False)

    print(f"[migrate] Read {len(df)} rows, {len(df.columns)} columns in file.")
    print(f"[migrate] Target schema: {len(RUNS_FIELDS)} columns.")

    existing_cols = list(df.columns)
    missing_cols  = [c for c in RUNS_FIELDS if c not in existing_cols]
    extra_cols    = [c for c in existing_cols if c not in RUNS_FIELDS]

    if missing_cols:
        print(f"[migrate] Adding missing columns (filled with ''): {missing_cols}")
        for col in missing_cols:
            df[col] = ""

    if extra_cols:
        print(f"[migrate] WARNING: unexpected columns in file (kept): {extra_cols}")

    # reorder to canonical schema, keeping any extra cols at the end
    ordered = RUNS_FIELDS + [c for c in extra_cols if c not in RUNS_FIELDS]
    df = df.reindex(columns=ordered, fill_value="")

    # backup original -------------------------------------------------------
    ts     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup = RUNS_PATH.parent / f"runs_backup_{ts}.csv"
    shutil.copy2(RUNS_PATH, backup)
    print(f"[migrate] Backup written to {backup}")

    # write clean file ------------------------------------------------------
    df.to_csv(RUNS_PATH, index=False)
    print(f"[migrate] Wrote {len(df)} rows with {len(ordered)} columns to {RUNS_PATH}")

    # verify it reads back cleanly -----------------------------------------
    check = pd.read_csv(RUNS_PATH)
    assert len(check) == len(df), "Row count changed after migration!"
    assert list(check.columns[:len(RUNS_FIELDS)]) == RUNS_FIELDS, "Column order wrong!"
    print(f"[migrate] Verification passed. runs.csv is clean.")
    print(f"[migrate] You can now re-run --fit_surface and --optimize.")


if __name__ == "__main__":
    main()