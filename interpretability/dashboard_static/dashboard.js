const state = { runs: null, payload: null };

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function text(value) {
  if (value === null || value === undefined || value === "") return "unavailable";
  return String(value);
}

function json(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function pct(value) {
  return `${Number(value || 0).toFixed(3)}`;
}

function card(label, value, cls = "") {
  return `<article class="card ${cls}"><div class="label">${label}</div><div class="value">${text(value)}</div></article>`;
}

function renderOverview(payload) {
  const score = payload.score;
  const warnings = payload.warnings || [];
  document.querySelector("#subtitle").textContent = payload.run.path;
  document.querySelector("#overview").innerHTML = [
    card("Model", payload.model.model_id),
    card("Run", payload.run.name),
    card("Accepted", `${score.accepted} - ${score.reason}`),
    card("Functionality", pct(score.functionality)),
    card("Explainability", pct(score.explainability)),
    card("Warnings", warnings.length, warnings.length ? "warning" : ""),
  ].join("");
}

function renderScores(details) {
  const entries = Object.entries(details || {});
  document.querySelector("#scoreBars").innerHTML = entries.length
    ? entries.map(([key, value]) => {
        const width = Math.max(0, Math.min(100, Number(value) * 100));
        return `<div class="bar"><span>${key}</span><div class="track"><div class="fill" style="width:${width}%"></div></div><strong>${pct(value)}</strong></div>`;
      }).join("")
    : `<p class="muted">No score component details were captured.</p>`;
}

function renderChecks(checks) {
  const table = document.querySelector("#checks");
  if (!checks.length) {
    table.innerHTML = `<tbody><tr><td class="muted">No check diagnostics captured.</td></tr></tbody>`;
    return;
  }
  table.innerHTML = `<thead><tr><th>Status</th><th>Type</th><th>Path</th><th>Expected</th><th>Excerpt</th></tr></thead><tbody>${
    checks.map(check => `<tr><td class="${check.passed ? "pass" : "fail"}">${check.passed ? "pass" : "fail"}</td><td>${text(check.type)}</td><td>${text(check.path)}</td><td>${text(check.expected_text)}</td><td>${text(check.final_excerpt)}</td></tr>`).join("")
  }</tbody>`;
}

function renderToolLog(events) {
  document.querySelector("#toolLog").innerHTML = events.length
    ? events.map((event, index) => `<div class="event"><strong>${index + 1}. ${text(event.type || event.command)}</strong><pre>${json(event)}</pre></div>`).join("")
    : `<p class="muted">No tool log events captured.</p>`;
}

function renderMechanistic(mech, warnings) {
  const backendRows = Object.entries(mech.deep_circuit_backends || {})
    .map(([key, value]) => `<p><strong>${key}</strong>: <span class="warning">${value}</span></p>`)
    .join("");
  const evidenceRows = (mech.evidence || []).map(item =>
    `<div class="evidence"><strong>${text(item.technique)}</strong> <span class="muted">${text(item.status)}</span><p>${text(item.summary)}</p><pre>${json(item.metrics || item.artifacts || {})}</pre></div>`
  ).join("");
  document.querySelector("#mechanistic").innerHTML = backendRows + (evidenceRows || `<p class="muted">No mechanistic evidence records captured.</p>`) +
    (warnings || []).map(w => `<p class="warning">${w}</p>`).join("");
}

function renderRecommendations(items) {
  document.querySelector("#recommendations").innerHTML = items.length
    ? items.map(item => `<div class="recommendation"><strong>${text(item.kind)}</strong> readiness ${pct(item.readiness)}<p>${text(item.summary)}</p><p class="muted">${text(item.artifact)}</p></div>`).join("")
    : `<p class="muted">No recommendations captured.</p>`;
}

function render(payload) {
  state.payload = payload;
  renderOverview(payload);
  renderScores(payload.score.details);
  renderChecks(payload.task_checks || []);
  document.querySelector("#trace").textContent = payload.decision_trace || "No decision trace captured.";
  document.querySelector("#action").textContent = json(payload.parsed_action);
  renderToolLog(payload.tool_log || []);
  document.querySelector("#snapshots").textContent = json(payload.file_snapshots || payload.work_files);
  renderMechanistic(payload.mechanistic || {}, payload.warnings || []);
  renderRecommendations((payload.mechanistic || {}).recommendations || []);
  document.querySelector("#adapter").textContent = json(payload.adapter_metadata);
}

async function loadRun(path) {
  const url = path ? `/api/run?path=${encodeURIComponent(path)}` : "/api/best";
  render(await getJson(url));
}

async function init() {
  state.runs = await getJson("/api/runs");
  const picker = document.querySelector("#runPicker");
  picker.innerHTML = (state.runs.candidates || []).map(candidate =>
    `<option value="${candidate.path}">${candidate.accepted ? "accepted" : "rejected"} ${pct(candidate.functionality)} / ${pct(candidate.explainability)} - ${candidate.path}</option>`
  ).join("");
  picker.addEventListener("change", () => loadRun(picker.value));
  if (state.runs.selected) picker.value = state.runs.selected.path;
  await loadRun();
}

init().catch(error => {
  document.body.innerHTML = `<main><section class="panel"><h2>Dashboard Error</h2><pre>${error.stack || error}</pre></section></main>`;
});
