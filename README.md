# Alert Intelligence Engine

Unsupervised / self-supervised ML PoC for AML/sanctions alert triage. No LLM, no paid AI API — client-hosted, scikit-learn/PyTorch based novelty detection over three alert families (Customer Name, Transaction Name, Transaction Rule).

Spec: see [`docs/Alert_Intelligence_Unsupervised_ML_Data_Science_Feasibility_Report.docx`](docs/Alert_Intelligence_Unsupervised_ML_Data_Science_Feasibility_Report.docx) (data audit) and [`docs/Alert_Intelligence_ML_PoC_AI_Agent_Development_Master_Plan.docx`](docs/Alert_Intelligence_ML_PoC_AI_Agent_Development_Master_Plan.docx) (build spec — authoritative for implementation rules, phases, and gates).

## Non-negotiable rules (see master plan §1 for full list)

- No LLM / paid AI inference API.
- No hand-weighted heuristic matching engine (no raw RapidFuzz/Levenshtein/phonetic thresholds as the decision layer).
- `Released` / `UPS` / `Followup` are **not** ground-truth labels — do not manufacture TRUE/FALSE from them.
- Maker/compliance comments, closure info, action-taken fields = leakage. Never live model input.
- Customer/UIN/transaction-reference grouping + time-forward validation only — no random row splits.
- Every score carries model version, feature version, schema version.

## Data

Raw client workbook (`Alerts_Samples.xlsx`) contains real customer PII. It is **never committed** — lives only at `data/raw/Alerts_Samples.xlsx` locally, excluded via `.gitignore`. Anyone re-running this repo must supply their own copy at that path.

| Sheet | Rows | Cols | Role |
|---|---|---|---|
| CustomerViolation | 2,397 | 23 | Entity pipeline |
| TransactionNameViolation | 2,000 | 25 | Entity pipeline |
| Rule | 2,244 | 24 | Transaction pipeline |

## Structure

```
app/            FastAPI service (scoring/batch-scoring/health/feedback)
pipelines/      ingestion, validation, normalization, entity, transaction, evaluation
features/       feature/representation builders
models/         trained artifacts + registry (gitignored contents)
notebooks/      experiment + training notebooks (Phase 1 audit, entity/transaction experiments)
evaluation/     validation manifests, metrics, model cards
feedback/       outcome/feedback store
monitoring/     drift + data-quality monitors
tests/
configs/
docker/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in local paths
```

## Status

Phase 0 (repo/environment) in progress. See master plan §20 for phase gates — each phase stops for human approval before continuing.
