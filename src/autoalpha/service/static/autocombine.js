const state = {
  bootstrap: null,
  taskId: null,
  detail: null,
  logMode: "events",
  timer: null,
  factorSelection: new Set(),
  pickerQuery: "",
};

const $ = (id) => document.getElementById(id);
const number = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? "--" : Number(value).toFixed(digits);
const percent = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? "--" : `${(Number(value) * 100).toFixed(digits)}%`;
const dateTime = (value) => value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "--";
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* response is not JSON */ }
    throw new Error(message);
  }
  return response.json();
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `show${error ? " error" : ""}`;
  window.clearTimeout(node._timer);
  node._timer = window.setTimeout(() => { node.className = ""; }, 4200);
}

function statusLabel(status) {
  return ({
    READY: "待启动", RUNNING: "运行中", STOPPING: "停止中", PAUSED: "已暂停",
    COMPLETED: "盲测通过", EXHAUSTED: "公开门禁耗尽", BLIND_REJECTED: "盲测未通过",
    PAUSED_FAILURE: "异常暂停",
  })[status] || status;
}

async function loadBootstrap() {
  state.bootstrap = await api("/api/bootstrap");
  renderSummary();
  renderTaskList();
  renderStrategies();
  if (!$("taskDialog").open) fillTaskDefaults();
}

function renderSummary() {
  const tasks = state.bootstrap.tasks;
  $("taskCount").textContent = tasks.length;
  $("runningCount").textContent = `${tasks.filter((task) => task.worker_alive).length} 个运行中`;
  $("experimentCount").textContent = tasks.reduce((sum, task) => sum + Number(task.experiment_count || 0), 0);
  $("visibleFactorCount").textContent = tasks.reduce((sum, task) => sum + Number(task.factor_count || 0), 0);
  $("strategyCount").textContent = state.bootstrap.strategies.length;
  $("providerState").textContent = state.bootstrap.provider_configured ? "READY" : "FALLBACK";
}

function renderTaskList() {
  const query = $("taskSearch").value.trim().toLowerCase();
  const status = $("taskStatusFilter").value;
  const tasks = state.bootstrap.tasks.filter((task) => {
    const text = `${task.name} ${task.task_id} ${task.snapshot_hash}`.toLowerCase();
    return (!query || text.includes(query)) && (!status || task.status === status);
  });
  $("taskList").innerHTML = tasks.length ? tasks.map((task) => `
    <button class="combine-task-item ${task.task_id === state.taskId ? "active" : ""}" data-task-id="${esc(task.task_id)}">
      <div class="task-item-top"><strong>${esc(task.name)}</strong><span class="state-pill small ${esc(task.status)}">${esc(statusLabel(task.status))}</span></div>
      <div class="task-item-meta"><span>${esc(task.market === "CN_A" ? "A 股" : task.market)} · ${task.factor_count} 因子</span><span>${task.iteration}/${task.budget.maximum_experiments}</span></div>
      <div class="task-item-foot"><code>${esc(task.task_id)}</code><div class="mini-progress"><i style="width:${Math.round(task.progress * 100)}%"></i></div></div>
    </button>`).join("") : `<div class="rail-empty">没有符合筛选条件的任务</div>`;
  document.querySelectorAll("[data-task-id]").forEach((button) => {
    button.onclick = () => selectTask(button.dataset.taskId, true);
  });
}

async function selectTask(taskId, updateHistory = false) {
  state.taskId = taskId;
  state.detail = await api(`/api/tasks/${encodeURIComponent(taskId)}`);
  if (updateHistory) history.pushState({}, "", `/tasks/${taskId}`);
  $("emptyWorkspace").hidden = true;
  $("strategyWorkspace").hidden = true;
  $("taskWorkspace").hidden = false;
  renderTaskList();
  renderTaskDetail();
}

function renderTaskDetail() {
  const { task, experiments, best, factor_snapshot: snapshot } = state.detail;
  const objectivePreset = state.bootstrap.objective_presets.find((item) => item.profile === task.objective.profile);
  $("taskTitle").textContent = task.name;
  $("taskIdentity").textContent = `${task.task_id} · ${task.market} · ${objectivePreset?.label || task.objective.profile} · ${task.snapshot_hash.slice(0, 16)}`;
  $("taskStatus").textContent = statusLabel(task.status);
  $("taskStatus").className = `state-pill ${task.status}`;
  $("taskPhase").textContent = task.phase;
  $("taskIteration").textContent = `${task.iteration} / ${task.budget.maximum_experiments}`;
  $("taskFactorCount").textContent = `${task.factor_count} 个`;
  $("taskFactorBounds").textContent = `${task.construction.min_factors} — ${task.construction.max_factors}`;
  $("taskWeightBounds").textContent = `${percent(task.construction.minimum_weight, 0)} — ${percent(task.construction.maximum_weight, 0)} · 步长 ${percent(task.construction.weight_step, 0)}`;
  $("taskWorker").textContent = task.worker_alive ? "WORKER ACTIVE" : "IDLE";
  $("taskProgress").style.width = `${Math.round(task.progress * 100)}%`;
  $("snapshotBadge").textContent = `SNAPSHOT ${task.snapshot_hash.slice(0, 10)}`;
  $("taskError").hidden = !task.last_error;
  $("taskError").textContent = task.last_error || "";
  $("startTask").disabled = task.worker_alive || task.status === "COMPLETED";
  $("stopTask").disabled = !task.worker_alive;
  const protocol = task.protocol;
  $("explorationRange").textContent = `${protocol.exploration_start} — ${protocol.exploration_end}`;
  $("validationRange").textContent = `${protocol.validation_start} — ${protocol.validation_end}`;
  $("holdoutRange").textContent = `${protocol.holdout_start} — ${protocol.holdout_end} · ${task.blind_verdict || "尚未提交"}`;
  $("dataPath").textContent = task.data_path;
  $("dataPath").title = task.data_path;
  renderFlow(task.phase, task.status);
  renderChampion(best, snapshot);
  renderExperiments(experiments, state.detail.frontier, snapshot);
  renderSnapshot(snapshot);
  renderLogs();
  drawChart();
}

function renderFlow(phase, status) {
  const order = ["PREFLIGHT", "SCREENING", "SEARCHING", "ROBUSTNESS", "BLIND_REVIEW", "DELIVERY"];
  let index = Math.max(0, order.indexOf(phase));
  if (["WAITING", "RECOVERED", "PAUSED"].includes(phase)) index = 0;
  if (status === "COMPLETED") index = order.length - 1;
  document.querySelectorAll("#flowGrid .flow-step").forEach((node, nodeIndex) => {
    node.classList.toggle("active", nodeIndex === index);
    node.classList.toggle("completed", nodeIndex < index);
  });
}

function factorName(factorId, snapshot = state.detail?.factor_snapshot || []) {
  return snapshot.find((item) => item.factor_id === factorId)?.name || factorId;
}

function renderChampion(best, snapshot) {
  const metrics = best?.metrics || {};
  $("bestSharpe").textContent = best ? number(metrics.portfolio_sharpe_ratio) : "--";
  $("bestAnnual").textContent = best ? percent(metrics.portfolio_simple_annual_return) : "--";
  $("bestActiveIr").textContent = best ? number(metrics.portfolio_active_information_ratio) : "--";
  $("bestDrawdown").textContent = best ? percent(metrics.portfolio_max_drawdown) : "--";
  $("bestWorstFold").textContent = best ? number(metrics.portfolio_walk_forward_worst_sharpe) : "--";
  $("bestTurnover").textContent = best ? number(metrics.portfolio_annual_turnover) : "--";
  $("championSubtitle").textContent = best ? `实验 #${best.iteration} · ${best.proposal_source} · 综合分 ${number(best.score, 3)}` : "尚无有效实验";
  $("promoteButton").disabled = !best || best.gate_status !== "PASSED";
  $("bestComposition").innerHTML = best ? best.factor_ids.map((factorId, index) => `
    <div class="component-row"><strong title="${esc(factorId)}">${esc(factorName(factorId, snapshot))}</strong><div class="weight-bar"><i style="width:${Math.min(100, best.weights[index] * 200)}%"></i></div><span>${percent(best.weights[index], 0)}</span></div>`).join("") : `<div class="empty-line">搜索开始后将在此显示最佳组合构成</div>`;
  const gates = $("bestGates");
  gates.classList.toggle("failed", Boolean(best && best.failed_gates.length));
  gates.textContent = !best ? "等待候选" : best.failed_gates.length ? `未通过：${best.failed_gates.join(" · ")}` : "全部公开组合门禁通过，可登记为 QUALIFIED 策略版本";
}

function renderExperiments(experiments, frontier, snapshot) {
  const frontierSet = new Set(frontier);
  $("experimentSummary").textContent = `${experiments.length} 个实验 · ${frontier.length} 个 Pareto 候选`;
  $("experimentRows").innerHTML = experiments.length ? experiments.map((item) => {
    const metrics = item.metrics || {};
    const factors = item.factor_ids.map((factorId, index) => `<span title="${esc(factorId)}">${esc(factorName(factorId, snapshot))} <b>${percent(item.weights[index], 0)}</b></span>`).join("");
    return `<tr class="${frontierSet.has(item.id) ? "frontier-row" : ""}">
      <td><strong>#${String(item.iteration).padStart(3, "0")}</strong></td>
      <td><strong>${esc(item.proposal_source)}</strong><br><code>${esc(item.action)}</code></td>
      <td><div class="factor-stack">${factors}</div></td>
      <td><span class="gate-badge ${item.gate_status === "PASSED" ? "" : "failed"}" title="${esc(item.failed_gates.join(", "))}">${item.gate_status === "PASSED" ? "通过" : `${item.failed_gates.length} 项`}</span></td>
      <td>${number(item.score, 3)}</td><td>${number(metrics.portfolio_sharpe_ratio)}</td><td>${percent(metrics.portfolio_simple_annual_return)}</td>
      <td>${number(metrics.portfolio_active_information_ratio)}</td><td>${percent(metrics.portfolio_max_drawdown)}</td><td>${number(metrics.portfolio_walk_forward_worst_sharpe)}</td>
      <td>${number(metrics.portfolio_annual_turnover, 1)}</td><td>${number(item.duration_seconds, 1)}s</td></tr>`;
  }).join("") : `<tr><td colspan="12" class="empty-table">尚未运行组合实验</td></tr>`;
}

function renderSnapshot(snapshot) {
  $("snapshotCount").textContent = `${snapshot.length} 个因子`;
  $("snapshotList").innerHTML = snapshot.map((item, index) => `
    <div class="snapshot-item ${item.holdout_contaminated ? "contaminated" : ""}"><code>#${index + 1}</code><strong title="${esc(item.factor_id)}">${esc(item.name)}</strong><span>${esc(item.family)}${item.required ? " · 必选" : ""}${item.holdout_contaminated ? " · 污染" : ""}</span><span>${number(item.prefilter_score, 2)}</span></div>`).join("");
}

function renderLogs() {
  if (!state.detail) return;
  const values = state.logMode === "events" ? state.detail.events : state.detail.memories;
  $("logList").innerHTML = values.length ? values.map((item) => {
    const code = state.logMode === "events" ? item.category : item.kind;
    const title = state.logMode === "events" ? item.title : `迭代 #${item.iteration}`;
    const message = state.logMode === "events" ? item.message : item.content;
    return `<div class="combine-log-item"><code>${esc(code)}</code><div><strong>${esc(title)}</strong><p>${esc(message)}</p></div><time>${dateTime(item.created_at)}</time></div>`;
  }).join("") : `<div class="empty-line">暂无记录</div>`;
}

function drawChart() {
  const canvas = $("metricChart");
  const key = $("chartMetric").value;
  const points = [...(state.detail?.experiments || [])].reverse().filter((item) => Number.isFinite(Number(item.metrics?.[key])));
  $("chartEmpty").hidden = points.length > 0;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!points.length) return;
  const pad = { left: 45, right: 16, top: 18, bottom: 28 };
  const width = rect.width - pad.left - pad.right;
  const height = rect.height - pad.top - pad.bottom;
  const values = points.map((item) => Number(item.metrics[key]));
  let min = Math.min(...values), max = Math.max(...values);
  if (min === max) { min -= 0.5; max += 0.5; }
  const margin = (max - min) * 0.12;
  min -= margin; max += margin;
  ctx.strokeStyle = "#e3e7ed"; ctx.lineWidth = 1; ctx.fillStyle = "#7b8799"; ctx.font = "9px sans-serif";
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (height * i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(rect.width - pad.right, y); ctx.stroke();
    const label = max - ((max - min) * i / 4);
    ctx.fillText(label.toFixed(Math.abs(label) < 1 ? 2 : 1), 5, y + 3);
  }
  const frontier = new Set(state.detail.frontier);
  points.forEach((item, index) => {
    const x = pad.left + (points.length === 1 ? width / 2 : width * index / (points.length - 1));
    const y = pad.top + (max - Number(item.metrics[key])) / (max - min) * height;
    if (index > 0) {
      const previous = points[index - 1];
      const px = pad.left + width * (index - 1) / Math.max(1, points.length - 1);
      const py = pad.top + (max - Number(previous.metrics[key])) / (max - min) * height;
      ctx.strokeStyle = "#a9b7cc"; ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x, y); ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(x, y, frontier.has(item.id) ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = item.gate_status === "PASSED" ? "#09845b" : "#be4b0a"; ctx.fill();
    if (frontier.has(item.id)) { ctx.strokeStyle = "#245eea"; ctx.lineWidth = 2; ctx.stroke(); }
  });
  ctx.fillStyle = "#7b8799"; ctx.fillText("实验次数", rect.width - 48, rect.height - 8);
}

function renderStrategies() {
  const strategies = state.bootstrap.strategies;
  $("strategyLibraryCount").textContent = `${strategies.length} 个版本`;
  $("strategyList").innerHTML = strategies.length ? strategies.map((item) => {
    const spec = item.specification;
    return `<div class="strategy-item"><div><strong>${esc(item.name)}</strong><code>${esc(item.strategy_id)} · V${item.version}</code></div><div class="strategy-spec-factors">${spec.factor_ids.map((id, index) => `<span>${esc(id)} ${percent(spec.factor_weights[index], 0)}</span>`).join("")}</div><div><strong>${esc(item.lifecycle)}</strong><span>${esc(item.market)}</span></div><div><strong>${dateTime(item.created_at)}</strong><code>${esc(item.evidence_hash.slice(0, 16))}</code></div></div>`;
  }).join("") : `<div class="empty-line">尚无策略版本。门禁通过的组合可从任务详情登记。</div>`;
}

function fillTaskDefaults() {
  const defaults = state.bootstrap.defaults;
  $("formDataPath").value = defaults.data_path || "";
  const protocol = defaults.protocol || {};
  const mapping = {
    formExplorationStart: "exploration_start", formExplorationEnd: "exploration_end",
    formValidationStart: "validation_start", formValidationEnd: "validation_end",
    formHoldoutStart: "holdout_start", formHoldoutEnd: "holdout_end", formMinimumFolds: "minimum_folds",
  };
  Object.entries(mapping).forEach(([id, key]) => { if (protocol[key] != null) $(id).value = protocol[key]; });
  $("formSourceTask").innerHTML = state.bootstrap.research_tasks.map((task) => {
    const counts = task.counts || {};
    return `<option value="${esc(task.task_id)}" selected>${esc(task.name)} · ${esc(statusLabel(task.status))} · ${Number(counts.total || 0)} 因子</option>`;
  }).join("");
  $("formObjectiveProfile").innerHTML = state.bootstrap.objective_presets.map((preset) => `<option value="${esc(preset.profile)}">${esc(preset.label)}</option>`).join("");
  $("formObjectiveProfile").value = "DRAWDOWN_FIRST";
  applyObjectivePreset();
  renderFactorPicker();
}

function openDialog() { fillTaskDefaults(); $("taskDialog").showModal(); renderFactorPicker(); lucide.createIcons(); }
function closeDialog() { $("taskDialog").close(); }
function splitIds(value) { return [...new Set(value.split(/[\s,，]+/).map((item) => item.trim()).filter(Boolean))]; }

function selectedValues(id) { return [...$(id).selectedOptions].map((option) => option.value); }

function applyObjectivePreset() {
  const preset = state.bootstrap?.objective_presets?.find((item) => item.profile === $("formObjectiveProfile").value);
  if (!preset) return;
  const fields = {
    formCoverage: "minimum_coverage", formPositiveFolds: "minimum_positive_fold_fraction",
    formWorstFold: "minimum_worst_fold_sharpe", formMaxDrawdown: "maximum_drawdown",
    formMaxTurnover: "maximum_annual_turnover", formMaxCorrelation: "maximum_factor_correlation",
    formMinAnnual: "minimum_simple_annual_return",
  };
  Object.entries(fields).forEach(([id, key]) => { $(id).value = preset[key]; });
  $("presetDescription").textContent = preset.description;
}

function factorPickerRecords() {
  const query = state.pickerQuery.toLowerCase();
  return (state.bootstrap?.factors || []).filter((factor) => {
    if (!query) return true;
    const task = state.bootstrap.research_tasks.find((item) => item.task_id === factor.source_task_id);
    return `${factor.name} ${factor.factor_id} ${factor.family} ${factor.mechanism} ${task?.name || ""}`.toLowerCase().includes(query);
  }).slice(0, 400);
}

function renderFactorPicker() {
  if (!state.bootstrap) return;
  const mode = $("formScopeMode").value;
  $("factorPicker").classList.toggle("inactive", mode === "SMART");
  const sourceNames = new Map(state.bootstrap.research_tasks.map((task) => [task.task_id, task.name]));
  const records = factorPickerRecords();
  $("factorPickerList").innerHTML = records.map((factor) => `
    <label class="picker-row ${state.factorSelection.has(factor.factor_id) ? "selected" : ""}">
      <input type="checkbox" data-picker-id="${esc(factor.factor_id)}" ${state.factorSelection.has(factor.factor_id) ? "checked" : ""} ${mode === "SMART" ? "disabled" : ""}>
      <span><strong>${esc(factor.name)}</strong><code>${esc(factor.factor_id)} · ${esc(sourceNames.get(factor.source_task_id) || factor.source_task_id || "--")}</code></span>
      <span><b>${esc(factor.mechanism || factor.family)}</b><small>${esc(factor.status)}${factor.holdout_contaminated ? " · 污染可用" : ""}</small></span>
      <span><b>${number(factor.sharpe)}</b><small>${percent(factor.annual_return)}</small></span>
    </label>`).join("") || `<div class="empty-line">没有匹配因子</div>`;
  document.querySelectorAll("[data-picker-id]").forEach((input) => {
    input.onchange = () => {
      if (input.checked) state.factorSelection.add(input.dataset.pickerId);
      else state.factorSelection.delete(input.dataset.pickerId);
      syncFactorIds(); renderFactorPicker();
    };
  });
  $("factorPickerCount").textContent = `${state.factorSelection.size} 已选${mode === "HYBRID" ? " · 全部必选" : ""}`;
}

function syncFactorIds() { $("formFactorIds").value = [...state.factorSelection].join("\n"); }
function importFactorIds() { state.factorSelection = new Set(splitIds($("formFactorIds").value)); renderFactorPicker(); }

async function submitTask(event) {
  event.preventDefault();
  const payload = {
    name: $("formName").value,
    market: $("formMarket").value,
    data_path: $("formDataPath").value,
    notes: $("formNotes").value,
    protocol: {
      exploration_start: $("formExplorationStart").value, exploration_end: $("formExplorationEnd").value,
      validation_start: $("formValidationStart").value, validation_end: $("formValidationEnd").value,
      holdout_start: $("formHoldoutStart").value, holdout_end: $("formHoldoutEnd").value,
      minimum_folds: Number($("formMinimumFolds").value),
    },
    scope: {
      mode: $("formScopeMode").value,
      factor_ids: $("formScopeMode").value === "MANUAL" ? [...state.factorSelection] : [],
      required_factor_ids: $("formScopeMode").value === "HYBRID" ? [...state.factorSelection] : [],
      excluded_factor_ids: [], families: [], source_task_ids: selectedValues("formSourceTask"),
      statuses: selectedValues("formStatuses"),
    },
    construction: {
      min_factors: Number($("formMinFactors").value), max_factors: Number($("formMaxFactors").value),
      minimum_weight: Number($("formMinWeight").value), maximum_weight: Number($("formMaxWeight").value),
      weight_step: Number($("formWeightStep").value), candidate_pool_limit: Number($("formPoolLimit").value),
      allow_negative_weights: false, maximum_same_family: Number($("formFamilyLimit").value),
    },
    objective: {
      profile: $("formObjectiveProfile").value, preset_version: 1, minimum_coverage: Number($("formCoverage").value),
      minimum_positive_fold_fraction: Number($("formPositiveFolds").value), minimum_worst_fold_sharpe: Number($("formWorstFold").value),
      maximum_drawdown: Number($("formMaxDrawdown").value), maximum_annual_turnover: Number($("formMaxTurnover").value),
      maximum_factor_correlation: Number($("formMaxCorrelation").value), minimum_cost_stress_ir: 0,
      minimum_simple_annual_return: Number($("formMinAnnual").value),
    },
    budget: {
      maximum_experiments: Number($("formExperiments").value), maximum_llm_proposals: Number($("formLlmProposals").value),
      maximum_runtime_minutes: Number($("formRuntime").value), maximum_holdout_submissions: 1,
      weight_evaluations_per_subset: Number($("formWeightEvaluations").value), iteration_interval_seconds: Number($("formInterval").value),
    },
  };
  try {
    const task = await api("/api/tasks", { method: "POST", body: JSON.stringify(payload) });
    closeDialog(); await loadBootstrap(); await selectTask(task.task_id, true); toast(`任务已创建并冻结 ${task.factor_count} 个因子`);
  } catch (error) { toast(error.message, true); }
}

async function taskCommand(command) {
  if (!state.taskId) return;
  try {
    await api(`/api/tasks/${state.taskId}/${command}`, { method: "POST", body: "{}" });
    await refreshCurrent(); toast(command === "start" ? "组合搜索已启动" : "停止请求已登记");
  } catch (error) { toast(error.message, true); }
}

async function promoteBest() {
  const best = state.detail?.best;
  if (!best) return;
  try {
    await api(`/api/tasks/${state.taskId}/promote`, { method: "POST", body: JSON.stringify({ experiment_id: best.id, name: state.detail.task.name }) });
    await loadBootstrap(); await selectTask(state.taskId); toast("候选已登记为 QUALIFIED 策略版本");
  } catch (error) { toast(error.message, true); }
}

async function refreshCurrent() {
  await loadBootstrap();
  if (state.taskId) await selectTask(state.taskId);
}

function showRoute() {
  if (location.pathname === "/strategies") {
    state.taskId = null; $("emptyWorkspace").hidden = true; $("taskWorkspace").hidden = true; $("strategyWorkspace").hidden = false; renderTaskList(); return;
  }
  const match = location.pathname.match(/^\/tasks\/([^/]+)/);
  if (match) selectTask(match[1]).catch((error) => toast(error.message, true));
  else if (state.bootstrap.tasks.length) selectTask(state.bootstrap.tasks[0].task_id);
}

function bind() {
  $("newTaskButton").onclick = openDialog; $("emptyNewTask").onclick = openDialog;
  $("closeDialog").onclick = closeDialog; $("cancelDialog").onclick = closeDialog;
  $("taskForm").onsubmit = submitTask;
  $("refreshButton").onclick = () => refreshCurrent().catch((error) => toast(error.message, true));
  $("taskSearch").oninput = renderTaskList; $("taskStatusFilter").onchange = renderTaskList;
  $("startTask").onclick = () => taskCommand("start"); $("stopTask").onclick = () => taskCommand("stop");
  $("promoteButton").onclick = promoteBest; $("chartMetric").onchange = drawChart;
  $("formScopeMode").onchange = renderFactorPicker;
  $("formObjectiveProfile").onchange = applyObjectivePreset;
  $("factorPickerSearch").oninput = (event) => { state.pickerQuery = event.target.value.trim(); renderFactorPicker(); };
  $("formFactorIds").onchange = importFactorIds;
  document.querySelectorAll("[data-log]").forEach((button) => { button.onclick = () => { state.logMode = button.dataset.log; document.querySelectorAll("[data-log]").forEach((item) => item.classList.toggle("active", item === button)); renderLogs(); }; });
  window.onresize = () => { if (state.detail) drawChart(); };
  window.onpopstate = showRoute;
}

async function init() {
  bind();
  try {
    await loadBootstrap(); showRoute(); $("liveState").classList.remove("offline");
    const params = new URLSearchParams(location.search);
    const factorIds = splitIds(params.get("factors") || "");
    if (factorIds.length) {
      state.factorSelection = new Set(factorIds);
      openDialog();
      $("formScopeMode").value = params.get("scope") === "HYBRID" ? "HYBRID" : "MANUAL";
      syncFactorIds(); renderFactorPicker();
    }
  }
  catch (error) { $("liveState").classList.add("offline"); toast(error.message, true); }
  lucide.createIcons();
  state.timer = window.setInterval(async () => {
    if (document.hidden) return;
    try { await refreshCurrent(); $("liveState").classList.remove("offline"); }
    catch (_) { $("liveState").classList.add("offline"); }
  }, 3000);
}

init();
