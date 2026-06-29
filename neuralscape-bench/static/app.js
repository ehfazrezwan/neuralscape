"use strict";
const $ = (id) => document.getElementById(id);
// Escape untrusted strings before they go into innerHTML. Run metadata (labels,
// notes, urls, commits) comes from result JSON files on disk, so a crafted file
// would otherwise be a DOM-XSS vector in this local dashboard.
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = async (p, opts) => {
  const r = await fetch(p, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${p}`);
  if (!(r.headers.get("content-type") || "").includes("application/json"))
    throw new Error(`non-JSON response — ${p}`);
  return r.json();
};
const COLORS = { base: "#8b949e", cand: "#388bfd", p50: "#3fb950", p95: "#d29922", p99: "#f85149" };
let charts = [];
let selected = []; // single-run filenames selected for compare

function clearCharts() { charts.forEach((c) => c.destroy()); charts = []; }

function chart(parent, type, data, opts) {
  const c = document.createElement("canvas");
  parent.appendChild(c);
  charts.push(new Chart(c, { type, data, options: Object.assign({ responsive: true,
    plugins: { legend: { labels: { color: "#c9d1d9" } } },
    scales: { x: { ticks: { color: "#8b949e" } }, y: { ticks: { color: "#8b949e" }, beginAtZero: true } } }, opts || {}) }));
}

function card(title) {
  const d = document.createElement("div"); d.className = "card";
  if (title) { const h = document.createElement("strong"); h.textContent = title; d.appendChild(h); }
  $("view").appendChild(d); return d;
}

// ── Runs list ──
async function loadRuns() {
  const { runs } = await api("/api/runs");
  const box = $("runs"); box.innerHTML = "";
  if (!runs.length) box.innerHTML = '<p class="note">No runs yet.</p>';
  runs.forEach((r) => {
    const el = document.createElement("div"); el.className = "run";
    if (selected.includes(r.name)) el.classList.add("sel");
    if (r.kind === "compare") {
      el.innerHTML = `<div class="lbl">${esc(r.baseline)} <span class="muted">vs</span> ${esc(r.candidate)}</div>
        <div class="meta"><span class="pill compare">A/B</span> ${esc((r.timestamp||"").slice(0,19))}</div>`;
      el.onclick = () => { selected = []; loadRuns(); showCompareFile(r.name); };
    } else {
      el.innerHTML = `<div class="lbl">${esc(r.label)}</div>
        <div class="meta"><span class="pill">${esc(r.profile)}</span> ${esc((r.timestamp||"").slice(0,19))}</div>`;
      el.onclick = () => toggleSelect(r.name);
    }
    box.appendChild(el);
  });
}

function toggleSelect(name) {
  const i = selected.indexOf(name);
  if (i >= 0) selected.splice(i, 1); else selected.push(name);
  if (selected.length > 2) selected.shift();
  loadRuns();
  if (selected.length === 1) showRun(selected[0]);
  else if (selected.length === 2) compareTwo(selected[0], selected[1]);
  else $("view").innerHTML = '<p class="muted">Select a run, or a second run to compare.</p>';
}

// ── Single run view ──
async function showRun(name) {
  const run = await api(`/api/runs/${name}`);
  clearCharts(); $("view").innerHTML = "";
  const head = card(run.label);
  head.innerHTML += `<div class="note">${esc(run.target_url)} · profile ${esc(run.profile)} · ${esc((run.timestamp||"").slice(0,19))} · ${esc(run.git_commit||"")}</div>`;
  const m = run.metrics || {};

  if (m.read) {
    const ops = Object.keys(m.read);
    chart(card("Read latency (ms)"), "bar", {
      labels: ops,
      datasets: ["p50", "p95", "p99"].map((p) => ({ label: p, backgroundColor: COLORS[p],
        data: ops.map((o) => m.read[o][p]) })),
    });
  }
  if (m.write) {
    const w = m.write;
    chart(card("Write latency (ms): enqueue vs end-to-end"), "bar", {
      labels: ["p50", "p95", "p99"],
      datasets: [
        { label: "enqueue", backgroundColor: COLORS.p50, data: ["p50","p95","p99"].map((p)=>w.enqueue_ms[p]) },
        { label: "e2e", backgroundColor: COLORS.p99, data: ["p50","p95","p99"].map((p)=>w.e2e_ms[p]) },
      ],
    });
    const c = card("Write completion");
    c.innerHTML += `<div class="note">e2e completed ${w.e2e_completed}/${w.e2e_attempts}` +
      ` · timeouts ${w.e2e_timeouts} · enqueue errors ${w.enqueue_errors}</div>`;
  }
  if (m.contention) {
    const ct = m.contention;
    chart(card(`Read p95 under write load (×${ct.loaded_over_unloaded_p95 ?? "?"})`), "bar", {
      labels: ["read p95 (ms)"],
      datasets: [
        { label: "unloaded", backgroundColor: COLORS.base, data: [ct.unloaded.p95] },
        { label: "under write load", backgroundColor: COLORS.p99, data: [ct.loaded.p95] },
      ],
    });
  }
  if (m.throughput) {
    const t = m.throughput;
    chart(card("Throughput (ops/sec)"), "bar", {
      labels: ["reads/sec", "writes/sec"],
      datasets: [{ label: run.label, backgroundColor: COLORS.cand, data: [t.reads_per_sec, t.writes_per_sec] }],
    });
  }
  if (m.stress) {
    const s = m.stress, f = s.fairness || {};
    const sum = card("Multi-user stress");
    sum.innerHTML += `<div class="note">${s.users} users · <b>${s.ops_per_sec} ops/sec</b> · ` +
      `errors ${(s.error_rate*100).toFixed(1)}% · read p95 ${s.read.p95}ms · ` +
      `fairness spread ×${f.read_p95_spread ?? "?"} (cv ${f.read_p95_cv})</div>`;
    chart(card("Stress latency (ms)"), "bar", {
      labels: ["p50", "p95", "p99"],
      datasets: [
        { label: "read", backgroundColor: COLORS.p50, data: ["p50","p95","p99"].map((p)=>s.read[p]) },
        { label: "write enqueue", backgroundColor: COLORS.p99, data: ["p50","p95","p99"].map((p)=>s.write[p]) },
      ],
    });
    const users = Object.keys(s.per_user_read_p95 || {});
    if (users.length) {
      chart(card("Per-user read p95 — fairness (flat = fair)"), "bar", {
        labels: users,
        datasets: [{ label: "read p95 (ms)", backgroundColor: COLORS.cand,
          data: users.map((u) => s.per_user_read_p95[u]) }],
      });
    }
  }
  if (run.notes && run.notes.length) {
    const n = card("Notes"); n.innerHTML += "<ul class='note'>" + run.notes.map((x)=>`<li>${esc(x)}</li>`).join("") + "</ul>";
  }
}

// ── Comparison views ──
function renderCompare(baseLabel, candLabel, comparison) {
  clearCharts(); $("view").innerHTML = "";
  const head = card(`A/B: ${baseLabel} → ${candLabel}`);
  const HEAD = [
    ["write.e2e_ms.p95", "write e2e p95 (ms)"],
    ["write.enqueue_ms.p95", "write enqueue p95 (ms)"],
    ["throughput.writes_per_sec", "writes/sec"],
    ["throughput.reads_per_sec", "reads/sec"],
    ["contention.loaded_over_unloaded_p95", "read p95 loaded/unloaded"],
    ["read.search.p95", "search p95 (ms)"],
    ["read.graph_search.p95", "graph_search p95 (ms)"],
  ];
  const tbl = document.createElement("table");
  tbl.innerHTML = "<tr><th>metric</th><th>baseline</th><th>candidate</th><th>Δ%</th></tr>";
  HEAD.forEach(([path, label]) => {
    const d = comparison[path]; if (!d) return;
    const cls = d.improved ? "up" : "down";
    const sign = d.pct_change > 0 ? "+" : "";
    tbl.innerHTML += `<tr><td>${label}</td><td>${d.baseline}</td><td>${d.candidate}</td>` +
      `<td class="${cls}">${d.pct_change===null?"–":sign+d.pct_change+"%"}</td></tr>`;
  });
  card("Headline deltas").appendChild(tbl);

  // Grouped bars for the most important latency metrics.
  const bars = HEAD.filter(([p]) => comparison[p]);
  chart(card("Baseline vs candidate"), "bar", {
    labels: bars.map(([, l]) => l),
    datasets: [
      { label: "baseline", backgroundColor: COLORS.base, data: bars.map(([p]) => comparison[p].baseline) },
      { label: "candidate", backgroundColor: COLORS.cand, data: bars.map(([p]) => comparison[p].candidate) },
    ],
  }, { indexAxis: "y" });
}

async function showCompareFile(name) {
  const data = await api(`/api/runs/${name}`);
  renderCompare(data.baseline.label, data.candidate.label, data.comparison);
}
async function compareTwo(a, b) {
  const data = await api(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  renderCompare(data.baseline.label, data.candidate.label, data.comparison);
}

// ── Triggers ──
$("run-btn").onclick = async () => {
  $("run-btn").disabled = true; $("run-btn").textContent = "Running…";
  await api("/api/run", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target: $("t-url").value, label: $("t-label").value,
      profile: $("t-profile").value, live_baseline: $("t-live").value === "true" }) });
  setTimeout(() => { $("run-btn").disabled = false; $("run-btn").textContent = "Run"; loadRuns(); }, 1500);
  pollStatus();
};
$("stress-btn").onclick = async () => {
  $("stress-btn").disabled = true; $("stress-btn").textContent = "Running…";
  await api("/api/stress", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target: $("s-url").value, label: $("s-label").value, profile: $("s-profile").value,
      users: parseInt($("s-users").value) || null, duration: parseFloat($("s-dur").value) || null }) });
  setTimeout(() => { $("stress-btn").disabled = false; $("stress-btn").textContent = "Run stress"; loadRuns(); }, 1500);
  pollStatus();
};
$("ab-btn").onclick = async () => {
  await api("/api/ab", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate_branch: $("ab-cand").value,
      baseline_url: $("ab-baseurl").value || null, profile: $("ab-profile").value }) });
  alert("A/B started — it builds Docker stacks; watch the terminal. Results appear here when done.");
};

async function pollStatus() {
  const { running } = await api("/api/status");
  if (Object.values(running).some((s) => s === "running")) { setTimeout(pollStatus, 3000); }
  loadRuns();
}

loadRuns();
setInterval(loadRuns, 15000);
