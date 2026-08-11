"""Raw alert workbook loader.

Phase 1 scope: read-only ingestion for audit purposes. Does not clean,
impute, or transform anything -- every transformation belongs to Phase 2
(normalization) and must be separately documented there (master plan Rule:
"Do not silently remove, repair, impute, or transform suspicious data").
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SHEET_NAMES = ["CustomerViolation", "TransactionNameViolation", "Rule"]


def get_raw_data_path() -> Path:
    env_path = os.environ.get("RAW_DATA_PATH", "data/raw/Alerts_Samples.xlsx")
    path = Path(env_path)
    if not path.is_absolute():
        # resolve relative to repo root (this file lives in pipelines/ingestion/)
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / env_path
    return path


def load_raw_alerts(path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the three alert sheets with pandas' native type inference.

    No dedup, no imputation, no manual transformation -- audit must see the
    raw shape first. Pandas' default na_values list (which includes the
    literal string "NULL" used throughout this workbook, plus "", "N/A",
    etc.) is relied on for missingness -- verified by direct inspection to
    match a manual audit exactly, so a custom missing-token parser is not
    needed on top of it.
    """
    path = path or get_raw_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Raw workbook not found at {path}. This file is gitignored "
            "(client PII) -- place your local copy of Alerts_Samples.xlsx there."
        )
    sheets = pd.read_excel(path, sheet_name=SHEET_NAMES, engine="openpyxl")
    return sheets
