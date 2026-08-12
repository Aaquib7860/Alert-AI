const API = "/api/v1";
let lastSample = null;
let lastScore = null;

async function getJSON(url, options) {
  const res = await fetch(url, options);
  const body = await res.json();
  if (!res.ok) {
    const detail = typeof body.detail === "object" ? JSON.stringify(body.detail) : body.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return body;
}

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v;
    else if (k === "class") e.className = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) e.appendChild(c);
  return e;
}

// ---------- 1. Data overview ----------
async function loadOverview() {
  const container = document.getElementById("overview-content");
  try {
    const data = await getJSON(`${API}/demo/data-overview`);
    container.innerHTML = "";
    const table = el("table");
    const thead = el("thead", {}, [
      el("tr", {}, ["Sheet", "Rows", "Columns", "Exact duplicates", "Near-duplicate candidates"].map(h => el("th", { text: h }))),
    ]);
    const tbody = el("tbody", {}, data.sheets.map(s => el("tr", {}, [
      el("td", { text: s.sheet_name }),
      el("td", { text: s.rows }),
      el("td", { text: s.columns }),
      el("td", { text: s.exact_duplicate_rows }),
      el("td", { text: s.near_duplicate_candidate_rows }),
    ])));
    table.appendChild(thead);
    table.appendChild(tbody);
    container.appendChild(table);
    container.appendChild(el("p", { class: "note", text: `Total: ${data.total_rows} rows across 3 alert families.` }));
    container.appendChild(el("p", { class: "note", text: data.label_audit_finding }));
  } catch (e) {
    container.innerHTML = "";
    container.appendChild(el("p", { class: "error", text: `Could not load data overview: ${e.message}` }));
  }
}

// ---------- 2. Score an alert ----------
// Which raw field (and which `evidence` key) each reason code points back
// to -- lets the UI highlight the exact value that drove the flag, in red,
// instead of just naming the code. Must stay in sync with
// app/services/scoring.py's _reason_codes()/_evidence_fields().
const REASON_CODE_TO_RAW_FIELD = {
  ALERTED_PARTY_DOB_MISSING: "Alerted Party DOB",
  HIT_DOB_UNRESOLVED: "Hit Details (DOB)",
  HIT_DOB_MULTI_VALUE: "Hit Details (DOB)",
  HIT_NATIONALITY_MISSING: "Hit Details (Nationality)",
  HIGH_SCREENING_MATCH_PERCENTAGE: "Matched Screening %",
  BENEFICIARY_NAME_MISSING: "Beneficiary Name",
  BENEFICIARY_RELATIONSHIP_MISSING: "Beneficiary Relationship",
  CURRENCY_NAME_MISSING: "Currency Name",
};

const REASON_CODE_TO_EVIDENCE_KEY = {
  HIT_NATIONALITY_MISSING: "hit_nationality",
  HIGH_SCREENING_MATCH_PERCENTAGE: "matched_screening_pct",
  BENEFICIARY_RELATIONSHIP_MISSING: "beneficiary_relationship",
  CURRENCY_NAME_MISSING: "currency",
};

function flaggedRawFieldNames(reasonCodes) {
  return new Set(reasonCodes.map(c => REASON_CODE_TO_RAW_FIELD[c]).filter(Boolean));
}

function flaggedEvidenceKeys(reasonCodes) {
  return new Set(reasonCodes.map(c => REASON_CODE_TO_EVIDENCE_KEY[c]).filter(Boolean));
}

function renderRawFields(fields, reasonCodes = []) {
  const table = document.getElementById("raw-fields-table");
  table.innerHTML = "";
  const flagged = flaggedRawFieldNames(reasonCodes);
  for (const [k, v] of Object.entries(fields)) {
    const isFlagged = flagged.has(k);
    const valueCell = el("td", { text: v === null ? "(missing)" : String(v) });
    if (isFlagged) valueCell.className = "flagged-value";
    const row = el("tr", {}, [el("td", { text: k }), valueCell]);
    if (isFlagged) row.className = "flagged-row";
    table.appendChild(row);
  }
  document.getElementById("raw-fields-panel").classList.remove("hidden");
}

const PLAIN_LABEL_CLASS = {
  "Needs Review": "rec-review",
  "Not Confident": "rec-borderline",
  "Looks Routine": "rec-lower",
};

function renderScoreResult(result) {
  const container = document.getElementById("score-result-content");
  container.innerHTML = "";

  const labelClass = PLAIN_LABEL_CLASS[result.plain_language_label] || "rec-borderline";
  container.appendChild(el("p", { class: "plain-label" }, [el("span", { class: labelClass, text: result.plain_language_label })]));
  container.appendChild(el("p", { text: result.plain_language_detail }));
  container.appendChild(el("p", { class: "note", text: `Technical recommendation: ${result.recommendation}` }));
  container.appendChild(el("p", { class: "note", text: result.recommendation_threshold_note }));

  container.appendChild(el("p", { text: `Global novelty: ${result.novelty.global}` }));
  const track = el("div", { class: "novelty-bar-track" });
  track.appendChild(el("div", { class: "novelty-bar-fill", style: `width:${result.novelty.global}%` }));
  container.appendChild(track);
  container.appendChild(el("p", {
    text: result.novelty.customer !== null
      ? `Customer-specific novelty (z-score vs. own history): ${result.novelty.customer}`
      : "Customer-specific novelty: no baseline (new/unseen customer)",
  }));
  container.appendChild(el("p", { class: "note", text: result.novelty_scale_note }));

  container.appendChild(el("p", { text: "Record evidence -- the actual field values behind this score. Values in red are the specific ones that drove a reason code below:" }));
  const evidenceKeys = Object.keys(result.evidence);
  const flaggedKeys = flaggedEvidenceKeys(result.reason_codes);
  if (evidenceKeys.length === 0) {
    container.appendChild(el("p", { class: "note", text: "(no evidence fields available)" }));
  } else {
    const evidenceTable = el("table");
    evidenceKeys.forEach(key => {
      const valueCell = el("td", { text: String(formatEvidenceValue(key, result.evidence[key])) });
      if (flaggedKeys.has(key)) valueCell.className = "flagged-value";
      evidenceTable.appendChild(el("tr", {}, [el("th", { text: EVIDENCE_LABELS[key] || key }), valueCell]));
    });
    container.appendChild(evidenceTable);
  }

  container.appendChild(el("p", { text: "What stood out:" }));
  if (result.reason_codes_plain.length === 0) {
    container.appendChild(el("p", { class: "note", text: "(nothing flagged)" }));
  } else {
    const chipRow = el("div");
    result.reason_codes_plain.forEach(c => chipRow.appendChild(el("span", { class: "chip", text: c })));
    container.appendChild(chipRow);
  }

  container.appendChild(el("p", {
    text: `Prior alerts for this customer: ${result.historical_context.customer_prior_alert_count} `
      + `(baseline known: ${result.historical_context.customer_baseline_known})`,
  }));

  container.appendChild(el("p", { class: "note", text:
    `model_version=${result.model_version} | feature_version=${result.feature_version} | `
    + `schema_version=${result.schema_version} | scored_at=${result.scored_at}` }));

  document.getElementById("score-result-panel").classList.remove("hidden");
}

function renderHistoricalOutcome(outcome, note) {
  const container = document.getElementById("historical-outcome-content");
  container.innerHTML = "";
  container.appendChild(el("p", { text: `Recorded historical status: ${outcome === null ? "(none)" : outcome}` }));
  container.appendChild(el("p", { class: "note", text: note }));
  document.getElementById("historical-outcome-panel").classList.remove("hidden");
}

document.getElementById("load-sample-btn").addEventListener("click", async () => {
  const alertType = document.getElementById("alert-type-select").value;
  try {
    lastSample = await getJSON(`${API}/demo/sample-alert?alert_type=${alertType}`);
    renderRawFields(lastSample.raw_fields);
    renderHistoricalOutcome(lastSample.historical_operational_outcome, lastSample._demo_note);
    document.getElementById("score-btn").disabled = false;
    document.getElementById("score-result-panel").classList.add("hidden");
  } catch (e) {
    alert(`Could not load sample alert: ${e.message}`);
  }
});

document.getElementById("score-btn").addEventListener("click", async () => {
  if (!lastSample) return;
  try {
    lastScore = await getJSON(`${API}/alerts/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        alert_id: lastSample.alert_id,
        alert_type: lastSample.alert_type,
        raw_fields: lastSample.raw_fields,
      }),
    });
    renderScoreResult(lastScore);
    // re-render raw fields now that reason codes are known, so the exact
    // value(s) that drove a flag show up in red right where they live
    renderRawFields(lastSample.raw_fields, lastScore.reason_codes);
    document.getElementById("feedback-alert-id").value = lastScore.alert_id;
    document.getElementById("feedback-alert-type").value = lastScore.alert_type;
  } catch (e) {
    alert(`Scoring failed: ${e.message}`);
  }
});

// ---------- 3. Review queue ----------
// One real table column per evidence field -- not chips crammed into a
// single cell -- so you can scan straight down a column (e.g. every row's
// "Screening %") and compare across the whole queue. The evidence field
// set differs by alert type (see app/services/scoring.py:_evidence_fields),
// so columns are built dynamically from whatever the API actually returns
// for the selected alert type, not hardcoded in the HTML.
const EVIDENCE_LABELS = {
  matched_screening_pct: "Screening %",
  alerted_party_nationality: "Alerted Nationality",
  hit_nationality: "Hit Nationality",
  sanctions_screening_list: "Watchlist",
  alert_type: "Alert Type",
  rule_name: "Rule",
  transaction_type: "Txn Type",
  currency: "Currency",
  beneficiary_relationship: "Beneficiary Rel.",
  customer_nationality: "Customer Nationality",
};

function formatEvidenceValue(key, value) {
  if (value === null || value === undefined) return "—"; // em dash for "not present"
  return key === "matched_screening_pct" ? `${value}%` : value;
}

const IDENTITY_COLS = ["Name", "DOB", "ID", "Country"];

function buildQueueHeader(evidenceKeys) {
  const thead = document.getElementById("queue-thead");
  thead.innerHTML = "";
  const fixedCols = ["Rank", "Alert ID", ...IDENTITY_COLS, "Global Novelty", "Status"];
  const evidenceCols = evidenceKeys.map(k => EVIDENCE_LABELS[k] || k);
  const trailingCols = ["What Stood Out"];
  const headerRow = el("tr", {}, [...fixedCols, ...evidenceCols, ...trailingCols].map(h => el("th", { text: h })));
  thead.appendChild(headerRow);
}

document.getElementById("build-queue-btn").addEventListener("click", async () => {
  const alertType = document.getElementById("queue-alert-type-select").value;
  const n = document.getElementById("queue-n-input").value;
  const tbody = document.querySelector("#queue-table tbody");
  tbody.innerHTML = "";
  try {
    const data = await getJSON(`${API}/demo/review-queue?alert_type=${alertType}&n=${n}`);
    const evidenceKeys = data.queue.length > 0 ? Object.keys(data.queue[0].evidence) : [];
    buildQueueHeader(evidenceKeys);

    data.queue.forEach((r, i) => {
      const labelClass = PLAIN_LABEL_CLASS[r.plain_language_label] || "rec-borderline";
      const identity = r.identity || {};
      const row = el("tr", {}, [
        el("td", { text: i + 1 }),
        el("td", { text: r.alert_id }),
        el("td", { text: identity.name ?? "—" }),
        el("td", { text: identity.dob ?? "—" }),
        el("td", { text: identity.id ?? "—" }),
        el("td", { text: identity.country ?? "—" }),
        el("td", { text: r.novelty.global }),
        el("td", {}, [el("span", { class: labelClass, text: r.plain_language_label })]),
      ]);
      const rowFlaggedKeys = flaggedEvidenceKeys(r.reason_codes);
      evidenceKeys.forEach(key => {
        const cell = el("td", { text: String(formatEvidenceValue(key, r.evidence[key])) });
        if (rowFlaggedKeys.has(key)) cell.className = "flagged-value";
        row.appendChild(cell);
      });
      row.appendChild(el("td", { text: r.reason_codes_plain.join("; ") || "—" }));
      tbody.appendChild(row);
    });
    document.getElementById("queue-table").classList.add("dense");
    document.getElementById("queue-panel").classList.remove("hidden");
  } catch (e) {
    alert(`Could not build review queue: ${e.message}`);
  }
});

// ---------- 4. Feedback ----------
document.getElementById("submit-feedback-btn").addEventListener("click", async () => {
  const resultDiv = document.getElementById("feedback-result");
  try {
    const body = {
      alert_id: document.getElementById("feedback-alert-id").value,
      alert_type: document.getElementById("feedback-alert-type").value,
      model_version: lastScore ? lastScore.model_version : "unknown",
      compliance_outcome: document.getElementById("feedback-outcome").value,
      notes: document.getElementById("feedback-notes").value || null,
    };
    const ack = await getJSON(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    resultDiv.textContent = `Recorded at ${ack.recorded_at}`;
  } catch (e) {
    resultDiv.textContent = `Failed: ${e.message}`;
  }
});

loadOverview();
