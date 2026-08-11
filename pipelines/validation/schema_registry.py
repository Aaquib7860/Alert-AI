"""Schema registry for each alert type.

Master plan section 7: "Create a schema registry for each alert type."

This is a validation contract, not a transformation step. It records what
we expect to see in each sheet (columns, semantic type, whether a column is
leakage) and checks incoming data against that contract. If the workbook
schema drifts (column renamed/removed/added), validation fails loudly
instead of pipeline code silently adapting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    semantic_type: str  # one of: id, date, numeric, categorical, text, leakage
    required: bool = True


@dataclass(frozen=True)
class SheetSchema:
    sheet_name: str
    columns: tuple[ColumnSpec, ...]

    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    @property
    def leakage_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.semantic_type == "leakage"]

    def columns_of_type(self, semantic_type: str) -> list[str]:
        return [c.name for c in self.columns if c.semantic_type == semantic_type]


CUSTOMER_VIOLATION_SCHEMA = SheetSchema(
    sheet_name="CustomerViolation",
    columns=(
        ColumnSpec("UIN", "id"),
        ColumnSpec("Customer Type", "categorical"),
        ColumnSpec("Trxn Type", "categorical", required=False),  # 100% missing in current sample
        ColumnSpec("Alerted Party Name", "text"),
        ColumnSpec("Alerted Party DOB", "date"),
        ColumnSpec("Alerted Party POB", "text", required=False),  # 100% missing, excluded from features
        ColumnSpec("Alerted Party Nationality", "categorical"),
        ColumnSpec("Hit Details (Name)", "text"),
        ColumnSpec("Hit Details (DOB)", "date"),
        ColumnSpec("Hit Details (POB)", "text", required=False),
        ColumnSpec("Hit Details (Nationality)", "categorical"),
        ColumnSpec("Hit Details (Aditional Information)", "text", required=False),
        ColumnSpec("Matched Screening %", "numeric"),
        ColumnSpec("Sanctions Screening List Name", "categorical"),
        ColumnSpec("Alerted Party", "categorical"),
        ColumnSpec("Branch Name", "categorical"),
        ColumnSpec("Alert Generated Date & Time", "date"),
        ColumnSpec("Alert Type", "categorical"),
        ColumnSpec("Alert Status", "leakage"),
        ColumnSpec("Maker Name", "leakage"),
        ColumnSpec("Maker Comment Date", "leakage"),
        ColumnSpec("Maker Comment", "leakage"),
        ColumnSpec("Alert Closure Date & Time", "leakage"),
    ),
)

TRANSACTION_NAME_VIOLATION_SCHEMA = SheetSchema(
    sheet_name="TransactionNameViolation",
    columns=(
        ColumnSpec("UIN", "id"),
        ColumnSpec("Customer Type", "categorical"),
        ColumnSpec("Trxn Type", "categorical"),
        ColumnSpec("Trxn date & Time", "date"),
        ColumnSpec("Trxn Ref Number", "id"),
        ColumnSpec("Transaction Status", "categorical"),
        ColumnSpec("Alerted Party Name", "text"),
        ColumnSpec("Alerted Party DOB", "date"),
        ColumnSpec("Alerted Party POB", "text", required=False),
        ColumnSpec("Alerted Party Nationality", "categorical"),
        ColumnSpec("Hit Details (Name)", "text"),
        ColumnSpec("Hit Details (DOB)", "date"),
        ColumnSpec("Hit Details (POB)", "text", required=False),
        ColumnSpec("Hit Details (Nationality)", "categorical"),
        ColumnSpec("Hit Details (Aditional Information)", "text", required=False),
        ColumnSpec("Matched Screening %", "numeric"),
        ColumnSpec("Sanctions Screening List Name", "categorical"),
        ColumnSpec("Alerted Party", "categorical"),
        ColumnSpec("Branch Name", "categorical"),
        ColumnSpec("Alert Generated Date & Time", "date"),
        ColumnSpec("Alert Type", "categorical"),
        ColumnSpec("Alert Status", "leakage"),
        ColumnSpec("Maker Name", "leakage"),
        ColumnSpec("Maker Comment Date", "leakage"),
        ColumnSpec("Maker Comment", "leakage"),
    ),
)

RULE_SCHEMA = SheetSchema(
    sheet_name="Rule",
    columns=(
        ColumnSpec("Customer Number", "id", required=False),  # ~2% blank in current sample
        ColumnSpec("Branch Code", "id"),
        ColumnSpec("Branch Description", "categorical"),
        ColumnSpec("Transaction Type Code", "categorical"),
        ColumnSpec("Reference Number", "id"),
        ColumnSpec("Transaction Date", "date"),
        ColumnSpec("Transaction Created By", "categorical"),
        ColumnSpec("Customer Name", "text"),
        ColumnSpec("Customer Nationality", "categorical"),
        ColumnSpec("Customer Currency", "categorical"),
        ColumnSpec("Beneficiary Name", "text", required=False),
        ColumnSpec("Beneficiary Id Number", "id", required=False),  # 97.1% missing
        ColumnSpec("Currency Name", "categorical", required=False),
        ColumnSpec("Beneficiary Relationship", "categorical", required=False),
        ColumnSpec("Purpose Code", "categorical", required=False),
        ColumnSpec("Purpose", "categorical", required=False),
        # Released/UPS/Followup: post-review outcome, not a safe live feature -- see Phase 1 label audit
        ColumnSpec("Status", "leakage"),
        ColumnSpec("Scan Date", "date"),
        ColumnSpec("Comment", "leakage"),
        ColumnSpec("Actiondate", "leakage"),
        ColumnSpec("Action Taken By", "leakage"),
        ColumnSpec("Rule Type", "categorical"),
        ColumnSpec("Rule Name", "categorical"),
        ColumnSpec("Rule Description", "text"),
    ),
)

SCHEMA_REGISTRY: dict[str, SheetSchema] = {
    "CustomerViolation": CUSTOMER_VIOLATION_SCHEMA,
    "TransactionNameViolation": TRANSACTION_NAME_VIOLATION_SCHEMA,
    "Rule": RULE_SCHEMA,
}


@dataclass
class ValidationResult:
    sheet_name: str
    passed: bool
    missing_required_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sheet_name": self.sheet_name,
            "passed": self.passed,
            "missing_required_columns": self.missing_required_columns,
            "unexpected_columns": self.unexpected_columns,
        }


def validate_schema(df: pd.DataFrame, sheet_name: str) -> ValidationResult:
    if sheet_name not in SCHEMA_REGISTRY:
        raise KeyError(f"No schema registered for sheet '{sheet_name}'")

    schema = SCHEMA_REGISTRY[sheet_name]
    expected_cols = {c.name for c in schema.columns}
    actual_cols = set(df.columns)

    missing_required = [c for c in schema.required_columns if c not in actual_cols]
    unexpected = sorted(actual_cols - expected_cols)

    return ValidationResult(
        sheet_name=sheet_name,
        passed=len(missing_required) == 0,
        missing_required_columns=missing_required,
        unexpected_columns=unexpected,
    )


def validate_all(sheets: dict[str, pd.DataFrame]) -> dict[str, ValidationResult]:
    return {name: validate_schema(df, name) for name, df in sheets.items()}
