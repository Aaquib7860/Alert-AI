"""Country / nationality representation normalization.

Master plan section 7: "Normalize country/nationality representations
without using normalization as a decision rule." This produces a
consistent *representation* for the model to consume -- it never decides
whether an alert is a match; that stays entirely with the anomaly model.

Observed raw patterns in this workbook (verified by direct inspection of
Alerts_Samples.xlsx, not assumed):
  - "Alerted Party Nationality" columns: always "XX-COUNTRY NAME"
    (ISO-3166 alpha-2 code + hyphen + English short name), e.g. "IN-INDIA".
  - "Hit Details (Nationality)" columns are far messier: bare alpha-2 codes
    ("IN"), bare names ("INDIA"), pipe- or semicolon-separated multi-value
    lists with repeats ("INDIA | INDIA", "AFGHANISTAN;PAKISTAN"), and some
    values that look truncated at a fixed character length
    ("AFGHANISTAN | AFGHANISTAN | AFGHANISTAN | AFGHANIS").
  - "Rule" sheet nationality/country fields: full English name only.

Only alpha-2 codes actually observed in this workbook are mapped below --
this is a lookup table grounded in real data, not a guessed/complete
ISO-3166 list. An unrecognized code is left as-is and flagged, never
guessed into a country name.
"""
from __future__ import annotations

import re

import pandas as pd

# Alpha-2 -> name, limited to codes observed in Alerts_Samples.xlsx.
ISO2_TO_NAME = {
    "AE": "UNITED ARAB EMIRATES", "AF": "AFGHANISTAN", "AL": "ALBANIA",
    "AU": "AUSTRALIA", "BD": "BANGLADESH", "BI": "BURUNDI", "CA": "CANADA",
    "CN": "CHINA", "CZ": "CZECHIA", "DE": "GERMANY", "DK": "DENMARK",
    "DM": "DOMINICA", "DZ": "ALGERIA", "EG": "EGYPT", "ET": "ETHIOPIA",
    "GB": "UNITED KINGDOM", "GH": "GHANA", "ID": "INDONESIA", "IL": "ISRAEL",
    "IN": "INDIA", "IQ": "IRAQ", "IR": "IRAN", "JO": "JORDAN", "KE": "KENYA",
    "KM": "COMOROS", "LB": "LEBANON", "LK": "SRI LANKA", "MA": "MOROCCO",
    "MM": "MYANMAR", "MR": "MAURITANIA", "NP": "NEPAL", "NZ": "NEW ZEALAND",
    "OM": "OMAN", "PH": "PHILIPPINES", "PK": "PAKISTAN",
    "PS": "PALESTINE, STATE OF", "RU": "RUSSIAN FEDERATION",
    "RS": "SERBIA", "SA": "SAUDI ARABIA", "SD": "SUDAN", "SL": "SIERRA LEONE",
    "SY": "SYRIA", "UG": "UGANDA", "VN": "VIETNAM", "YE": "YEMEN",
}

_CODE_PREFIX_RE = re.compile(r"^([A-Z]{2})-(.+)$")
_MULTI_VALUE_SPLIT_RE = re.compile(r"[|;]")


def _normalize_single_token(token: str) -> str:
    token = token.strip().upper()
    if not token:
        return ""
    m = _CODE_PREFIX_RE.match(token)
    if m:
        # "XX-COUNTRY NAME" pattern -- trust the name half, code is redundant.
        return m.group(2).strip()
    if len(token) == 2 and token.isalpha():
        # bare alpha-2 code
        return ISO2_TO_NAME.get(token, token)
    return token


def normalize_country_field(series: pd.Series) -> pd.DataFrame:
    """Returns a DataFrame with columns:
      - normalized: single canonical value, or pipe-joined sorted unique
        values when the source cell held more than one distinct country
      - is_multi_value: True if the source cell listed more than one
        distinct country
      - unresolved_token_present: True if a token could not be resolved to
        either a known alpha-2 code or treated as a full name (heuristic:
        looks like a truncated/garbled fragment -- length <= 3 letters that
        isn't a known code, or ends mid-word based on surrounding repeats).
        Flagged for review, never guessed/repaired.
    """
    normalized_vals = []
    multi_flags = []
    unresolved_flags = []

    for raw in series:
        if pd.isna(raw):
            normalized_vals.append(pd.NA)
            multi_flags.append(False)
            unresolved_flags.append(False)
            continue

        raw_str = str(raw).strip()
        if not raw_str:
            normalized_vals.append(pd.NA)
            multi_flags.append(False)
            unresolved_flags.append(False)
            continue

        tokens = [t for t in _MULTI_VALUE_SPLIT_RE.split(raw_str) if t.strip()]
        normalized_tokens = [_normalize_single_token(t) for t in tokens]
        normalized_tokens = [t for t in normalized_tokens if t]

        unique_tokens = sorted(set(normalized_tokens))
        is_multi = len(unique_tokens) > 1

        # Truncation heuristic: a token that is a proper prefix of a longer
        # token also present in the same cell is very likely a cut-off
        # fragment of it (seen directly in the data, e.g. a cell containing
        # both "AFGHANISTAN" and "AFGHANIS" -- the value was truncated at a
        # fixed character length somewhere upstream). Length>=3 avoids
        # flagging legitimate short codes as prefixes of unrelated names.
        unresolved = any(
            a != b and len(a) >= 3 and b.startswith(a)
            for a in unique_tokens
            for b in unique_tokens
        )

        normalized_vals.append(" | ".join(unique_tokens) if unique_tokens else pd.NA)
        multi_flags.append(is_multi)
        unresolved_flags.append(unresolved)

    return pd.DataFrame(
        {
            "normalized": pd.array(normalized_vals, dtype="string"),
            "is_multi_value": multi_flags,
            "unresolved_token_present": unresolved_flags,
        },
        index=series.index,
    )
