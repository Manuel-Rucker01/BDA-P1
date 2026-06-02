"use strict";

// ─────────────────────────── helpers ───────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function api(path, opts) {
  const res = await fetch(path, opts);
  return res.json();
}
const fmtUsd = (v) =>
  v == null ? "—" : v.toLocaleString("en-US", { style: "currency", currency: "USD" });
const fmtPct = (v) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);
const fmtNum = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));
const cls = (v) => (v == null ? "" : v >= 0 ? "pos" : "neg");

function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast ${kind}`;
  setTimeout(() => t.classList.add("hidden"), 5000);
}

// ─────────────────────────── tabs ───────────────────────────
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.remove("active"));
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "portfolio") loadPortfolio();
    if (btn.dataset.tab === "history") loadHistory();
    if (btn.dataset.tab === "about") loadProfile();
  });
});

// ─────────────────────────── server status ───────────────────────────
let SERVER = {};
async function loadStatus() {
  try {
    SERVER = await api("/api/status");
    const pill = $("#server-pill");
    if (!SERVER.alpaca_configured) {
      pill.textContent = "Alpaca not configured";
      pill.className = "pill pill-warn";
    } else if (SERVER.paper_trading) {
      pill.textContent = "PAPER trading";
      pill.className = "pill pill-ok";
    } else {
      pill.textContent = "● LIVE trading";
      pill.className = "pill pill-live";
    }
    if (SERVER.defaults) $("#cfg-topk").value = SERVER.defaults.top_k;
  } catch {
    $("#server-pill").textContent = "server offline";
  }
}

// ─────────────────────────── run pipeline ───────────────────────────
let CURRENT_JOB = null;
let POLL_TIMER = null;

$("#btn-run").addEventListener("click", async () => {
  $("#btn-run").disabled = true;
  $("#proposal-card").classList.add("hidden");
  $("#execute-result").innerHTML = "";
  $("#run-progress").classList.remove("hidden");
  $("#run-stage").textContent = "Starting…";

  const body = {
    universe: $("#cfg-universe").value,
    strategy: $("#cfg-strategy").value,
    top_k: parseInt($("#cfg-topk").value, 10) || null,
    force_regime: $("#cfg-regime").value || null,
  };
  const r = await api("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    finishRun();
    toast(r.error || "Failed to start run.", "err");
    return;
  }
  CURRENT_JOB = r.job_id;
  POLL_TIMER = setInterval(pollJob, 1500);
});

async function pollJob() {
  if (!CURRENT_JOB) return;
  const r = await api(`/api/run/status/${CURRENT_JOB}`);
  if (!r.ok) return;
  const job = r.job;
  $("#run-stage").textContent = job.stage || job.status;
  if (job.status === "done") {
    clearInterval(POLL_TIMER);
    finishRun();
    renderProposal(job.result);
  } else if (job.status === "error") {
    clearInterval(POLL_TIMER);
    finishRun();
    toast(`Pipeline error: ${job.error}`, "err");
    console.error(job.traceback);
  }
}

function finishRun() {
  $("#btn-run").disabled = false;
  $("#run-progress").classList.add("hidden");
}

function renderProposal(res) {
  const regimeCls = res.regime === "bull" ? "regime-bull" : "regime-bear";
  $("#proposal-meta").innerHTML = `
    <span class="meta-chip">Regime <b class="${regimeCls}">${res.regime.toUpperCase()}</b></span>
    <span class="meta-chip">Strategy <b>${res.strategy}</b></span>
    <span class="meta-chip">Universe <b>${res.n_universe}</b></span>
    <span class="meta-chip">Holdings <b>${res.n_holdings}</b></span>
    <span class="meta-chip">Gross exposure <b>${res.gross_exposure_pct}%</b></span>`;

  const body = $("#proposal-body");
  body.innerHTML = "";
  res.proposals.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td><b>${p.ticker}</b></td>
      <td>${p.sector}</td>
      <td class="num">${p.weight_pct.toFixed(2)}%</td>
      <td class="num">${p.pred_rank == null ? "—" : p.pred_rank.toFixed(1) + "%"}</td>
      <td class="num">${fmtUsd(p.price)}</td>
      <td class="num">${fmtNum(p.kalman_beta)}</td>
      <td><span class="badge ${p.side.toLowerCase()}">${p.side}</span></td>`;
    body.appendChild(tr);
  });
  $("#proposal-card").classList.remove("hidden");
  $("#btn-execute").disabled = !SERVER.alpaca_configured;
}

// ─────────────────────────── execute ───────────────────────────
$("#btn-execute").addEventListener("click", async () => {
  const mode = SERVER.paper_trading ? "PAPER" : "LIVE";
  if (!confirm(`Submit this portfolio to your ${mode} Alpaca account now?`)) return;
  $("#btn-execute").disabled = true;
  $("#execute-result").innerHTML = `<div class="progress"><div class="spinner"></div><span>Submitting orders…</span></div>`;
  const r = await api("/api/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: CURRENT_JOB }),
  });
  if (r.ok) {
    $("#execute-result").innerHTML = `<p class="pos">✓ ${r.message}</p>`;
    toast("Orders submitted to Alpaca.", "ok");
  } else {
    $("#execute-result").innerHTML = `<p class="neg">✗ ${r.error}</p>`;
    $("#btn-execute").disabled = false;
    toast(r.error || "Execution failed.", "err");
  }
});

// ─────────────────────────── portfolio ───────────────────────────
$("#btn-refresh-portfolio").addEventListener("click", loadPortfolio);

async function loadPortfolio() {
  const r = await api("/api/portfolio");
  if (!r.ok) {
    $("#account-cards").innerHTML = `<div class="stat"><div class="label">Error</div><div class="value neg" style="font-size:14px">${r.error}</div></div>`;
    $("#holdings-body").innerHTML = "";
    return;
  }
  const a = r.account, s = r.summary;
  const dayPl = a.equity != null && a.last_equity != null ? a.equity - a.last_equity : null;
  $("#account-cards").innerHTML = `
    ${stat("Equity", fmtUsd(a.equity))}
    ${stat("Cash", fmtUsd(a.cash))}
    ${stat("Buying Power", fmtUsd(a.buying_power))}
    ${stat("Unrealized P/L", fmtUsd(s.total_unrealized_pl), cls(s.total_unrealized_pl))}
    ${stat("Total Return", fmtPct(s.total_unrealized_plpc), cls(s.total_unrealized_plpc))}
    ${stat("Today", dayPl == null ? "—" : fmtUsd(dayPl), cls(dayPl))}`;

  $("#portfolio-asof").textContent = `${s.n_positions} positions · ${a.paper ? "paper" : "LIVE"} · as of ${new Date(r.as_of).toLocaleString()}`;

  const body = $("#holdings-body");
  body.innerHTML = "";
  if (!r.positions.length) {
    $("#holdings-empty").classList.remove("hidden");
  } else {
    $("#holdings-empty").classList.add("hidden");
    r.positions.forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><b>${p.ticker}</b></td>
        <td class="num">${fmtNum(p.qty, 0)}</td>
        <td class="num">${fmtUsd(p.avg_entry_price)}</td>
        <td class="num">${fmtUsd(p.current_price)}</td>
        <td class="num">${fmtUsd(p.cost_basis)}</td>
        <td class="num">${fmtUsd(p.market_value)}</td>
        <td class="num ${cls(p.unrealized_pl)}">${fmtUsd(p.unrealized_pl)}</td>
        <td class="num ${cls(p.unrealized_plpc)}">${fmtPct(p.unrealized_plpc)}</td>`;
      body.appendChild(tr);
    });
  }
}

function stat(label, value, valueCls = "") {
  return `<div class="stat"><div class="label">${label}</div><div class="value ${valueCls}">${value}</div></div>`;
}

// ─────────────────────────── history ───────────────────────────
$("#btn-refresh-history").addEventListener("click", loadHistory);
let CHARTS = {};

async function loadHistory() {
  const r = await api("/api/history");
  if (!r.ok) { toast("Could not load history.", "err"); return; }

  const eqLabels = r.equity.map((e) => new Date(e.ts).toLocaleDateString());
  drawChart("chart-equity", "line", {
    labels: eqLabels,
    datasets: [
      { label: "Equity", data: r.equity.map((e) => e.equity), borderColor: "#2f81f7",
        backgroundColor: "rgba(47,129,247,0.1)", fill: true, tension: 0.25, pointRadius: 2 },
      { label: "Peak", data: r.equity.map((e) => e.peak), borderColor: "#8b949e",
        borderDash: [4, 4], pointRadius: 0, fill: false, tension: 0.25 },
    ],
  });

  drawChart("chart-drawdown", "line", {
    labels: eqLabels,
    datasets: [{ label: "Drawdown %", data: r.equity.map((e) => -e.drawdown_pct),
      borderColor: "#f85149", backgroundColor: "rgba(248,81,73,0.15)", fill: true,
      tension: 0.25, pointRadius: 0 }],
  });

  const at = r.attribution;
  drawChart("chart-attribution", "bar", {
    labels: at.map((a) => a.period_end),
    datasets: [
      { label: "Strategy", data: at.map((a) => a.realised_return_pct), backgroundColor: "#3fb950" },
      { label: "Benchmark (S&P)", data: at.map((a) => a.benchmark_return_pct), backgroundColor: "#8b949e" },
    ],
  });

  const body = $("#trades-body");
  body.innerHTML = "";
  r.trades.forEach((t) => {
    const tr = document.createElement("tr");
    const isBuy = (t.action || "").toUpperCase().includes("BUY");
    tr.innerHTML = `
      <td>${t.decision_date || "—"}</td>
      <td><b>${t.ticker}</b></td>
      <td class="${isBuy ? "pos" : "neg"}">${t.action}</td>
      <td class="num">${t.target_weight == null ? "—" : (t.target_weight * 100).toFixed(2) + "%"}</td>
      <td class="num">${fmtUsd(t.intended_notional_usd)}</td>
      <td class="num">${fmtUsd(t.ref_price)}</td>
      <td>${t.side || "—"}</td>
      <td>${t.dry_run ? "dry" : "live"}</td>`;
    body.appendChild(tr);
  });
}

function drawChart(id, type, data) {
  const ctx = document.getElementById(id);
  if (CHARTS[id]) CHARTS[id].destroy();
  CHARTS[id] = new Chart(ctx, {
    type,
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8b949e", boxWidth: 12 } } },
      scales: {
        x: { ticks: { color: "#8b949e", maxRotation: 0, autoSkip: true }, grid: { color: "#2a3441" } },
        y: { ticks: { color: "#8b949e" }, grid: { color: "#2a3441" } },
      },
    },
  });
}

// ─────────────────────────── init ───────────────────────────

// Profile / About view logic
async function loadProfile() {
  try {
    const r = await api('/api/profile');
    if (!r.ok) { toast('Could not load profile.', 'err'); return; }
    const p = r.profile || {};
    $('#profile-name').value = p.name || '';
    $('#profile-use-env').checked = !!p.use_env;
    $('#profile-paper-trading').checked = !!p.paper_trading;
    $('#env-info').textContent = p.env_available ? 'Environment Alpaca keys detected' : 'No Alpaca env keys detected';
    if (p.use_env) {
      $('#profile-alpaca-key').value = '';
      $('#profile-alpaca-secret').value = '';
      $('#profile-alpaca-key').disabled = true;
      $('#profile-alpaca-secret').disabled = true;
    } else {
      $('#profile-alpaca-key').value = (p.alpaca_key === '*****') ? '' : (p.alpaca_key || '');
      $('#profile-alpaca-secret').value = '';
      $('#profile-alpaca-key').disabled = false;
      $('#profile-alpaca-secret').disabled = false;
    }
    $('#profile-status').textContent = `Hello ${p.name || 'user'}. Alpaca: ${p.env_available ? 'env keys' : (p.alpaca_key ? 'stored keys' : 'not configured')}`;
  } catch (e) { console.error(e); toast('Error loading profile', 'err'); }
}

// Toggle form fields when checkbox changes
$('#profile-use-env').addEventListener('change', (e) => {
  const useEnv = e.target.checked;
  $('#profile-alpaca-key').disabled = useEnv;
  $('#profile-alpaca-secret').disabled = useEnv;
});

$('#btn-save-profile').addEventListener('click', async () => {
  const body = {
    name: $('#profile-name').value,
    use_env: $('#profile-use-env').checked,
    alpaca_key: $('#profile-alpaca-key').value,
    alpaca_secret: $('#profile-alpaca-secret').value,
    paper_trading: $('#profile-paper-trading').checked,
  };
  const r = await api('/api/profile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (r.ok) {
    toast('Profile saved', 'ok');
    await loadStatus();
    await loadProfile();
  } else {
    toast('Failed to save profile', 'err');
  }
});

loadStatus();
