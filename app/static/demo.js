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
function renderRawFields(fields) {
  const table = document.getElementById("raw-fields-table");
  table.innerHTML = "";
  for (const [k, v] of Object.entries(fields)) {
    table.appendChild(el("tr", {}, [el("td", { text: k }), el("td", { text: v === null ? "(missing)" : String(v) })]));
  }
  document.getElementById("raw-fields-panel").classList.remove("hidden");
}

function renderScoreResult(result) {
  const container = document.getElementById("score-result-content");
  container.innerHTML = "";

  const recClass = result.recommendation === "REVIEW" ? "rec-review" : "rec-lower";
  container.appendChild(el("p", {}, [el("span", { class: recClass, text: result.recommendation })]));
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

  container.appendChild(el("p", { text: "Reason codes:" }));
  if (result.reason_codes.length === 0) {
    container.appendChild(el("p", { class: "note", text: "(none flagged)" }));
  } else {
    const chipRow = el("div");
    result.reason_codes.forEach(c => chipRow.appendChild(el("span", { class: "chip", text: c })));
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
    document.getElementById("feedback-alert-id").value = lastScore.alert_id;
    document.getElementById("feedback-alert-type").value = lastScore.alert_type;
  } catch (e) {
    alert(`Scoring failed: ${e.message}`);
  }
});

// ---------- 3. Review queue ----------
document.getElementById("build-queue-btn").addEventListener("click", async () => {
  const alertType = document.getElementById("queue-alert-type-select").value;
  const n = document.getElementById("queue-n-input").value;
  const tbody = document.querySelector("#queue-table tbody");
  tbody.innerHTML = "";
  try {
    const data = await getJSON(`${API}/demo/review-queue?alert_type=${alertType}&n=${n}`);
    data.queue.forEach((r, i) => {
      const recClass = r.recommendation === "REVIEW" ? "rec-review" : "rec-lower";
      tbody.appendChild(el("tr", {}, [
        el("td", { text: i + 1 }),
        el("td", { text: r.alert_id }),
        el("td", { text: r.novelty.global }),
        el("td", {}, [el("span", { class: recClass, text: r.recommendation })]),
        el("td", { text: r.reason_codes.join(", ") || "-" }),
      ]));
    });
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
