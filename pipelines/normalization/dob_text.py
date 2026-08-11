"""Handler for the `Hit Details (DOB)` field specifically.

This field is NOT a clean date column despite its name, confirmed by direct
inspection of Alerts_Samples.xlsx. In CustomerViolation it mixes:
  - genuine datetime values (718 rows)
  - bare integers that are sometimes a plausible year (e.g. 1965) and
    sometimes clearly garbage / a misparsed Excel serial (e.g. 1033879,
    which is not a valid year under any reasonable interpretation)
  - free text: multi-value year lists ("1948 | 1954 | 1960"), date ranges
    ("01 Jan 1963 to 31 Dec 1965"), qualifiers ("circa 1949"), and
    malformed dates ("1985/01/00" -- day 00 does not exist)

Running this column through a generic date parser (`pd.to_datetime`) would
silently misinterpret bare years as ordinal timestamps/serials -- a wrong
value that looks like a valid one, which is worse than a missing value.
Master plan rule: "Do not silently remove, repair, impute, or transform
suspicious data." This module classifies instead of guessing: a value only
becomes a resolved date/year when it is unambiguous; everything else is
flagged `is_unresolved=True` and left for the client/compliance team to
clarify, exactly as the master plan requires for the temporal-integrity
findings.
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd

_YEAR_ONLY_RE = re.compile(r"^\s*(1[89]\d{2}|20[0-2]\d)\s*$")
_PLAUSIBLE_YEAR_MIN, _PLAUSIBLE_YEAR_MAX = 1900, 2026
_MULTI_VALUE_RE = re.compile(r"[|;]|(?i:\bto\b)|(?i:circa)")


def classify_hit_dob(series: pd.Series) -> pd.DataFrame:
    """Returns a DataFrame indexed like `series` with:
      - parsed_date: datetime64, set only for an unambiguous single full date
      - year: nullable Int64, set for a resolved year (from a full date or
        a bare plausible-year value)
      - is_multi_value: True if the cell lists more than one date/year
      - is_unresolved: True if the value could not be safely classified
        (garbage integer, malformed date, qualifier like "circa", etc.)
    """
    parsed_dates: list = []
    years: list = []
    multi_flags: list = []
    unresolved_flags: list = []

    for raw in series:
        if pd.isna(raw):
            parsed_dates.append(pd.NaT)
            years.append(pd.NA)
            multi_flags.append(False)
            unresolved_flags.append(False)
            continue

        if isinstance(raw, (dt.datetime, dt.date, pd.Timestamp)):
            ts = pd.Timestamp(raw)
            parsed_dates.append(ts)
            years.append(ts.year)
            multi_flags.append(False)
            unresolved_flags.append(False)
            continue

        if isinstance(raw, int) and not isinstance(raw, bool):
            if _PLAUSIBLE_YEAR_MIN <= raw <= _PLAUSIBLE_YEAR_MAX:
                parsed_dates.append(pd.NaT)
                years.append(int(raw))
                multi_flags.append(False)
                unresolved_flags.append(False)
            else:
                # out-of-range integer -- e.g. 1033879. Almost certainly a
                # misparsed serial/garbage. Do not guess a year from it.
                parsed_dates.append(pd.NaT)
                years.append(pd.NA)
                multi_flags.append(False)
                unresolved_flags.append(True)
            continue

        raw_str = str(raw).strip()
        if _MULTI_VALUE_RE.search(raw_str):
            parsed_dates.append(pd.NaT)
            years.append(pd.NA)
            multi_flags.append(True)
            unresolved_flags.append(False)
            continue

        year_match = _YEAR_ONLY_RE.match(raw_str)
        if year_match:
            parsed_dates.append(pd.NaT)
            years.append(int(year_match.group(1)))
            multi_flags.append(False)
            unresolved_flags.append(False)
            continue

        parsed = pd.to_datetime(raw_str, errors="coerce")
        if pd.notna(parsed):
            parsed_dates.append(parsed)
            years.append(int(parsed.year))
            multi_flags.append(False)
            unresolved_flags.append(False)
        else:
            # malformed (e.g. "1985/01/00") or otherwise unparseable free text
            parsed_dates.append(pd.NaT)
            years.append(pd.NA)
            multi_flags.append(False)
            unresolved_flags.append(True)

    return pd.DataFrame(
        {
            "parsed_date": pd.array(parsed_dates, dtype="datetime64[ns]"),
            "year": pd.array(years, dtype="Int64"),
            "is_multi_value": multi_flags,
            "is_unresolved": unresolved_flags,
        },
        index=series.index,
    )
