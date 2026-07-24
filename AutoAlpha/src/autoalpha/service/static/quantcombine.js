const quantState = {
  bootstrap: null,
  detail: null,
  selectedFactors: new Set(),
  taskQuery: "",
  taskStatus: "ALL",
  refreshTimer: null,
  actionCandidate: null,
};

const PHASES = ["PREFLIGHT", "SCREENING", "CLUSTERING", "SFFS", "EVOLUTION", "ADAPTIVE", "BLIND_REVIEW", "DELIVERY"];
const STATUS_LABELS = { READY: "待启动", RUNNING: "运行中", STOPPING: "正在停止", PAUSED: "已暂停", PAUSED_FAILURE: "异常暂停", EXHAUSTED: "公开预算耗尽", RESEARCH_COMPLETED: "研究完成未晋级", COMPLETED: "已完成", BLIND_REJECTED: "盲测未通过" };
const QUALIFICATION_LABELS = { NO_CANDIDATE: "尚无候选", RESEARCH_LEADER_ONLY: "仅研究领先", QUALIFIED_CHAMPION: "合格冠军", PRODUCTION_CANDIDATE: "生产候选", BLIND_REJECTED: "盲测未通过" };

document.addEventListener("DOMContentLoaded", async () => {
  bindControls();
  await loadBootstrap();
  quantState.refreshTimer = window.setInterval(() => {
    if (!document.hidden) refreshCurrent();
  }, 5000);
});

function bindControls() {
  document.getElementById("newTaskButton").onclick = openTaskDialog;
  document.getElementById("emptyCreateButton").onclick = openTaskDialog;
  document.getElementById("closeDialog").onclick = closeTaskDialog;
  document.getElementById("cancelDialog").onclick = closeTaskDialog;
  document.getElementById("refreshButton").onclick = refreshCurrent;
  document.getElementById("taskSearch").oninput = event => { quantState.taskQuery = event.target.value.trim().toLowerCase(); renderTaskList(); };
  document.getElementById("taskStatus").onchange = event => { quantState.taskStatus = event.target.value; renderTaskList(); };
  document.getElementById("startButton").onclick = () => command("start");
  document.getElementById("stopButton").onclick = () => command("stop");
  document.getElementById("promoteButton").onclick = promoteCurrent;
  document.getElementById("quickScreenButton").onclick = () => openCandidateWorkflow("screener");
  document.getElementById("quickBacktestButton").onclick = () => openCandidateWorkflow("backtest");
  document.getElementById("headerQuickScreenButton").onclick = () => openCandidateWorkflow("screener");
  document.getElementById("headerQuickBacktestButton").onclick = () => openCandidateWorkflow("backtest");
  document.getElementById("chartMetric").onchange = renderChart;
  document.getElementById("factorSearch").oninput = renderFactorPicker;
  document.getElementById("formObjective").onchange = applyObjectivePreset;
  document.getElementById("taskForm").onsubmit = createTask;
  window.addEventListener("resize", () => { if (quantState.detail) renderChart(); });
}

async function loadBootstrap() {
  try {
    quantState.bootstrap = await api("/api/bootstrap");
    hydrateSummary();
    hydrateForm();
    renderTaskList();
    renderStrategies();
    const pathname = window.location.pathname;
    if (pathname.startsWith("/strategies")) {
      showStrategies();
      return;
    }
    const taskId = pathname.startsWith("/tasks/") ? pathname.split("/").pop() : quantState.bootstrap.tasks[0]?.task_id;
    if (taskId) await selectTask(taskId, false);
    else showEmpty();
  } catch (error) {
    toast(error.message, true);
  }
}

async function refreshCurrent() {
  const currentId = quantState.detail?.task?.task_id;
  try {
    quantState.bootstrap = await api("/api/bootstrap");
    hydrateSummary();
    renderTaskList();
    renderStrategies();
    if (currentId) {
      quantState.detail = await api(`/api/tasks/${currentId}`);
      renderTaskDetail();
    }
  } catch (error) {
    toast(error.message, true);
  }
}

function hydrateSummary() {
  const tasks = quantState.bootstrap.tasks;
  text("taskCount", tasks.length);
  text("activeTaskCount", `${tasks.filter(task => task.worker_alive).length} 个运行中`);
  text("candidateCount", tasks.reduce((sum, task) => sum + Number(task.candidate_count || 0), 0));
  text("factorCount", quantState.bootstrap.factors.length);
  const candidates = quantState.detail?.candidates || [];
  text("paretoCount", candidates.filter(item => item.pareto_rank === 0).length || "--");
}

function renderTaskList() {
  const list = document.getElementById("taskList");
  list.replaceChildren();
  const activeId = quantState.detail?.task?.task_id;
  const tasks = (quantState.bootstrap?.tasks || []).filter(task => {
    const matchesQuery = !quantState.taskQuery || `${task.name} ${task.task_id}`.toLowerCase().includes(quantState.taskQuery);
    const matchesStatus = quantState.taskStatus === "ALL" || (quantState.taskStatus === "DELIVERY" ? task.phase === "DELIVERY" : task.status === quantState.taskStatus);
    return matchesQuery && matchesStatus;
  });
  tasks.forEach(task => {
    const button = element("button", `task-card${task.task_id === activeId ? " active" : ""}`);
    button.type = "button";
    button.innerHTML = `<strong>${escapeHtml(task.name)}</strong><span class="task-status">${escapeHtml(STATUS_LABELS[task.status] || task.status)}</span><small>${escapeHtml(task.market)} · ${task.factor_count} 因子 · ${task.evaluation_count}/${task.budget.maximum_evaluations}</small><code>${escapeHtml(task.task_id)}</code><div class="task-progress"><i style="width:${Math.round(task.progress * 100)}%"></i></div>`;
    button.onclick = () => selectTask(task.task_id, true);
    list.append(button);
  });
  if (!tasks.length) list.append(emptyLine("没有匹配任务"));
}

async function selectTask(taskId, updateHistory = true) {
  try {
    quantState.detail = await api(`/api/tasks/${taskId}`);
    if (updateHistory) window.history.pushState({}, "", `/tasks/${taskId}`);
    showTask();
    renderTaskList();
    renderTaskDetail();
  } catch (error) {
    toast(error.message, true);
  }
}

function showTask() {
  document.getElementById("emptyWorkspace").hidden = true;
  document.getElementById("strategyWorkspace").hidden = true;
  document.getElementById("taskWorkspace").hidden = false;
}

function showEmpty() {
  document.getElementById("emptyWorkspace").hidden = false;
  document.getElementById("taskWorkspace").hidden = true;
  document.getElementById("strategyWorkspace").hidden = true;
}

function showStrategies() {
  document.getElementById("emptyWorkspace").hidden = true;
  document.getElementById("taskWorkspace").hidden = true;
  document.getElementById("strategyWorkspace").hidden = false;
  renderStrategies();
}

function renderTaskDetail() {
  const { task, best, qualified, production } = quantState.detail;
  text("taskName", task.name);
  text("taskIdentity", `${task.task_id} · ${task.market} · ${task.snapshot_hash.slice(0, 16)}`);
  text("taskState", STATUS_LABELS[task.status] || task.status);
  text("taskPhase", task.phase);
  text("evaluationProgress", `${task.evaluation_count} / ${task.budget.maximum_evaluations}`);
  text("taskCandidateCount", task.candidate_count || 0);
  text("qualificationState", QUALIFICATION_LABELS[task.qualification_status] || task.qualification_status);
  text("progressText", `${Math.round(task.progress * 100)}%`);
  document.getElementById("progressFill").style.width = `${task.progress * 100}%`;
  document.getElementById("startButton").disabled = task.worker_alive || task.status === "COMPLETED";
  document.getElementById("stopButton").disabled = !task.worker_alive;
  renderPipeline(task.phase, task.engine.mode);
  text("explorationRange", `${task.protocol.exploration_start} — ${task.protocol.exploration_end}`);
  text("validationRange", `${task.protocol.validation_start} — ${task.protocol.validation_end}`);
  text("holdoutRange", `${task.protocol.holdout_start} — ${task.protocol.holdout_end} · ${task.blind_verdict || "未提交"}`);
  text("engineMode", `${task.engine.mode} · seed ${task.engine.random_seed}`);
  text("factorConstraint", `${task.construction.min_factors}—${task.construction.max_factors} 因子 · ${percent(task.construction.minimum_weight)}—${percent(task.construction.maximum_weight)}`);
  renderBest(production || qualified || best, { best, qualified, production });
  renderScreen();
  renderCandidates();
  renderEvents();
  renderConfig();
  renderChart();
  hydrateSummary();
  if (window.lucide) window.lucide.createIcons();
}

function renderPipeline(currentPhase, mode) {
  const current = PHASES.indexOf(currentPhase);
  const skipped = new Set();
  if (mode === "DETERMINISTIC") { skipped.add("EVOLUTION"); skipped.add("ADAPTIVE"); }
  if (mode === "EVOLUTIONARY") skipped.add("ADAPTIVE");
  if (mode === "BAYESIAN") skipped.add("EVOLUTION");
  document.querySelectorAll("#pipeline > div").forEach(item => {
    const index = PHASES.indexOf(item.dataset.phase);
    item.classList.toggle("skipped", skipped.has(item.dataset.phase));
    item.classList.toggle("done", !skipped.has(item.dataset.phase) && current >= 0 && index < current);
    item.classList.toggle("active", !skipped.has(item.dataset.phase) && item.dataset.phase === currentPhase);
  });
}

function renderBest(candidate, tiers) {
  quantState.actionCandidate = candidate || null;
  text("leaderState", tiers.best ? `#${tiers.best.iteration} · ${tiers.best.algorithm}` : "尚未产生");
  text("qualifiedState", tiers.qualified ? `#${tiers.qualified.iteration}` : "尚未产生");
  text("productionState", tiers.production ? `#${tiers.production.iteration}` : "尚未盲测");
  const promote = document.getElementById("promoteButton");
  promote.disabled = !tiers.production;
  promote.dataset.candidateId = tiers.production?.id || "";
  const workflowTitles = {
    quickScreenButton: "使用当前最佳组合在最新交易日快速选股",
    headerQuickScreenButton: "使用当前最佳组合在最新交易日快速选股",
    quickBacktestButton: "将当前最佳组合及任务协议带入手动回测",
    headerQuickBacktestButton: "将当前最佳组合及任务协议带入手动回测",
  };
  Object.entries(workflowTitles).forEach(([id, readyTitle]) => {
    const button = document.getElementById(id);
    button.disabled = !candidate;
    button.title = candidate ? readyTitle : "该任务尚未产生可用候选组合";
  });
  if (!candidate) {
    text("bestSubtitle", "等待候选");
    ["bestSharpe", "bestAnnual", "bestDrawdown", "bestWorstFold", "bestBets", "bestMechanisms", "bestCorrelation", "bestTurnover"].forEach(id => text(id, "--"));
    document.getElementById("bestComposition").replaceChildren();
    text("bestGates", "等待评价");
    return;
  }
  const metrics = candidate.metrics || {};
  text("bestSubtitle", `候选 #${candidate.iteration} · ${candidate.algorithm} · ${candidate.qualification}`);
  text("bestSharpe", number(metrics.portfolio_sharpe_ratio));
  text("bestAnnual", percent(metrics.portfolio_simple_annual_return));
  text("bestDrawdown", percent(metrics.portfolio_max_drawdown));
  text("bestWorstFold", number(metrics.portfolio_walk_forward_worst_sharpe));
  text("bestBets", number(metrics.portfolio_effective_factor_bets));
  text("bestMechanisms", number(metrics.portfolio_effective_mechanisms));
  text("bestCorrelation", number(metrics.portfolio_max_factor_correlation, 3));
  text("bestTurnover", number(metrics.portfolio_annual_turnover, 1));
  const names = factorMap();
  const composition = document.getElementById("bestComposition");
  composition.replaceChildren(...candidate.factor_ids.map((factorId, index) => {
    const item = element("span", "factor-weight");
    item.innerHTML = `<span>${escapeHtml(names[factorId]?.name || factorId)}</span><b>${percent(candidate.weights[index])}</b>`;
    return item;
  }));
  const gates = document.getElementById("bestGates");
  gates.className = `gate-summary ${candidate.gate_status === "PASSED" ? "passed" : "failed"}`;
  gates.textContent = candidate.gate_status === "PASSED" ? "全部公开门禁通过" : `未通过：${candidate.failed_gates.join(" · ")} · 门禁距离 ${number(candidate.gate_distance, 3)}`;
}

function openCandidateWorkflow(target) {
  const candidate = quantState.actionCandidate;
  const task = quantState.detail?.task;
  if (!candidate || !task) return;
  const metrics = candidate.metrics || {};
  const url = new URL(target === "screener" ? "/screener" : "/backtest", window.location.href);
  url.port = "8788";
  const common = {
    factors: candidate.factor_ids.join(","),
    weights: candidate.weights.join(","),
    source_task: task.task_id,
    source_candidate: String(candidate.id),
  };
  const parameters = target === "screener"
    ? {
      ...common,
      as_of_date: quantState.bootstrap?.defaults?.data_range?.end || task.protocol.validation_end,
      selection_count: String(metrics.portfolio_maximum_positions || 30),
      selection_side: "TOP",
      run: "1",
    }
    : {
      ...common,
      start_date: task.protocol.validation_start,
      end_date: task.protocol.validation_end,
      backtest_preset: "A_SHARE_NON_PIT_PROXY_WEEKLY_V1",
      backtest_engine: "EVENT_LEDGER",
      execution_data_mode: "NON_PIT_PROXY",
      product_template: "LONG_ONLY_CAPITAL",
      gross_exposure: String(metrics.portfolio_target_gross_exposure || 0.9),
      holding_period_days: "5",
      rebalance_schedule: metrics.portfolio_rebalance_schedule || "WEEKLY_FIRST_SESSION",
      maximum_positions: String(metrics.portfolio_maximum_positions || 30),
      selection_fraction: "0.10",
    };
  url.search = new URLSearchParams(parameters).toString();
  window.open(url.toString(), "_blank", "noopener");
}

function renderScreen() {
  const rows = document.getElementById("screenRows");
  rows.replaceChildren();
  const factors = factorMap();
  const screen = quantState.detail.factor_screen || [];
  text("screenSummary", `${screen.length} 已评价 · ${new Set(screen.map(item => item.cluster_id).filter(Boolean)).size} 个收益簇`);
  screen.forEach((item, index) => {
    const factor = factors[item.factor_id] || {};
    const metrics = item.metrics || {};
    const row = element("tr");
    row.innerHTML = `<td>#${index + 1}</td><td><strong>${escapeHtml(factor.name || item.factor_id)}</strong><br><code>${escapeHtml(item.factor_id)}</code></td><td>${escapeHtml(factor.mechanism || "OTHER")}</td><td><code>${escapeHtml(item.cluster_id || "--")}</code></td><td><span class="role-pill ${item.cluster_leader ? "leader" : "member"}">${item.cluster_leader ? "LEADER" : "MEMBER"}</span></td><td>${number(item.stability_score, 3)}</td><td>${number(metrics.portfolio_sharpe_ratio)}</td><td>${percent(metrics.portfolio_simple_annual_return)}</td><td>${percent(metrics.portfolio_max_drawdown)}</td><td>${number(metrics.portfolio_walk_forward_worst_sharpe)}</td><td>${number(metrics.portfolio_annual_turnover, 1)}</td><td>${escapeHtml(item.exclusion_reason || "保留")}</td>`;
    rows.append(row);
  });
  if (!screen.length) rows.append(tableEmpty(12, "等待稳定性筛选"));
}

function renderCandidates() {
  const rows = document.getElementById("candidateRows");
  rows.replaceChildren();
  const names = factorMap();
  const candidates = quantState.detail.candidates || [];
  text("candidateSummary", `${candidates.length} 个候选 · ${quantState.detail.pareto_frontier.length} 个 Pareto 前沿`);
  candidates.forEach(candidate => {
    const metrics = candidate.metrics || {};
    const algorithmClass = candidate.algorithm === "NSGA2" ? "evolution" : candidate.algorithm === "BAYESIAN_INCLUSION" ? "adaptive" : "";
    const composition = candidate.factor_ids.map((factorId, index) => `${escapeHtml(names[factorId]?.name || factorId)} <b>${percent(candidate.weights[index])}</b>`).join(" · ");
    const row = element("tr");
    row.innerHTML = `<td><strong>#${String(candidate.iteration).padStart(3, "0")}</strong></td><td><span class="algorithm-tag ${algorithmClass}">${escapeHtml(candidate.algorithm)}</span><br><small>${escapeHtml(candidate.action)}</small></td><td class="factor-cell">${composition}</td><td>${candidate.pareto_rank === 0 ? `<span class="pareto-pill">FRONTIER</span>` : candidate.pareto_rank ?? "--"}</td><td><span class="gate-pill ${candidate.gate_status === "PASSED" ? "passed" : "failed"}">${candidate.gate_status}</span><br><small>${candidate.failed_gates.length} 项</small></td><td>${number(candidate.score, 3)}</td><td>${number(metrics.portfolio_sharpe_ratio)}</td><td>${percent(metrics.portfolio_simple_annual_return)}</td><td>${percent(metrics.portfolio_max_drawdown)}</td><td>${number(metrics.portfolio_walk_forward_worst_sharpe)}</td><td>${number(metrics.portfolio_effective_factor_bets)}</td><td>${number(metrics.portfolio_effective_mechanisms)}</td><td>${number(metrics.portfolio_max_factor_correlation, 3)}</td><td>${number(metrics.portfolio_annual_turnover, 1)}</td>`;
    rows.append(row);
  });
  if (!candidates.length) rows.append(tableEmpty(14, "等待组合搜索"));
}

function renderEvents() {
  const list = document.getElementById("eventList");
  list.replaceChildren();
  (quantState.detail.events || []).forEach(event => {
    const item = element("article", "event-item");
    item.innerHTML = `<time>${formatTime(event.created_at)}</time><div><strong>${escapeHtml(event.title)}</strong><p>${escapeHtml(event.message)}</p></div><small>${escapeHtml(event.category)} · ${escapeHtml(event.level)}</small>`;
    list.append(item);
  });
  if (!list.children.length) list.append(emptyLine("尚无日志"));
}

function renderConfig() {
  const task = quantState.detail.task;
  const values = [
    ["数据路径", task.data_path], ["快照哈希", task.snapshot_hash], ["引擎", task.engine.mode],
    ["聚类阈值", task.engine.cluster_correlation_threshold], ["SFFS 束宽", task.engine.sffs_beam_width],
    ["进化预算", `${task.engine.evolution_population} × ${task.engine.evolution_generations}`],
    ["自适应试验", task.engine.adaptive_trials], ["协方差收缩", task.engine.covariance_shrinkage],
    ["目标预设", task.objective.profile], ["评价预算", task.budget.maximum_evaluations],
  ];
  const list = document.getElementById("configList");
  list.replaceChildren(...values.map(([label, value]) => {
    const wrapper = element("div"); wrapper.innerHTML = `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd>`; return wrapper;
  }));
}

function renderChart() {
  const canvas = document.getElementById("trajectoryChart");
  const candidates = [...(quantState.detail?.candidates || [])].sort((a, b) => a.iteration - b.iteration);
  const metric = document.getElementById("chartMetric").value;
  const values = candidates.map(item => metric === "gate_distance" ? Number(item.gate_distance) : Number(item.metrics?.[metric])).filter(Number.isFinite);
  document.getElementById("chartEmpty").hidden = values.length > 0;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio)); canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, rect.width, rect.height);
  if (!values.length) return;
  const pad = { left: 46, right: 18, top: 18, bottom: 28 }; const width = rect.width - pad.left - pad.right; const height = rect.height - pad.top - pad.bottom;
  let min = Math.min(...values), max = Math.max(...values); if (min === max) { min -= 1; max += 1; } const margin = (max - min) * .08; min -= margin; max += margin;
  ctx.font = "9px system-ui"; ctx.fillStyle = "#7a8699"; ctx.strokeStyle = "#e3e7ef"; ctx.lineWidth = 1;
  for (let step = 0; step <= 4; step += 1) { const y = pad.top + height * step / 4; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(rect.width - pad.right, y); ctx.stroke(); const value = max - (max - min) * step / 4; ctx.fillText(formatMetricAxis(value, metric), 4, y + 3); }
  candidates.forEach((candidate, index) => { const value = metric === "gate_distance" ? Number(candidate.gate_distance) : Number(candidate.metrics?.[metric]); if (!Number.isFinite(value)) return; const x = pad.left + width * (candidates.length === 1 ? .5 : index / (candidates.length - 1)); const y = pad.top + height * (max - value) / (max - min); const color = candidate.algorithm === "NSGA2" ? "#7357c7" : candidate.algorithm === "BAYESIAN_INCLUSION" ? "#087f8c" : "#1e5ee7"; ctx.beginPath(); ctx.arc(x, y, candidate.pareto_rank === 0 ? 4.5 : 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); if (candidate.pareto_rank === 0) { ctx.strokeStyle = "#e2a52d"; ctx.lineWidth = 2; ctx.stroke(); } });
  ctx.fillStyle = "#7a8699"; ctx.fillText("实验序号", rect.width - 52, rect.height - 6);
}

function hydrateForm() {
  const defaults = quantState.bootstrap.defaults;
  document.getElementById("formDataPath").value = defaults.data_path || "";
  const protocol = defaults.protocol || {};
  [["formExplorationStart", "exploration_start"], ["formExplorationEnd", "exploration_end"], ["formValidationStart", "validation_start"], ["formValidationEnd", "validation_end"], ["formHoldoutStart", "holdout_start"], ["formHoldoutEnd", "holdout_end"], ["formMinimumFolds", "minimum_folds"]].forEach(([id, key]) => { document.getElementById(id).value = protocol[key] ?? ""; });
  const objective = document.getElementById("formObjective"); objective.replaceChildren(...quantState.bootstrap.objective_presets.map(preset => option(preset.profile, preset.label))); objective.value = defaults.objective.profile;
  const sources = document.getElementById("formSourceTasks"); sources.replaceChildren(...quantState.bootstrap.research_tasks.map(task => option(task.task_id, `${task.name} · ${task.status}`)));
  applyObjectivePreset();
  const fromUrl = new URLSearchParams(window.location.search).get("factors"); if (fromUrl) fromUrl.split(",").filter(Boolean).forEach(value => quantState.selectedFactors.add(value));
  renderFactorPicker();
}

function openTaskDialog() {
  if (!quantState.bootstrap) return;
  document.getElementById("formName").value ||= `统计组合优化 · ${new Date().toLocaleDateString("zh-CN")}`;
  renderFactorPicker(); document.getElementById("taskDialog").showModal(); if (window.lucide) window.lucide.createIcons();
}
function closeTaskDialog() { document.getElementById("taskDialog").close(); }

function renderFactorPicker() {
  if (!quantState.bootstrap) return;
  const query = document.getElementById("factorSearch").value.trim().toLowerCase();
  const list = document.getElementById("factorPickerList"); list.replaceChildren();
  quantState.bootstrap.factors.filter(factor => !query || `${factor.name} ${factor.factor_id} ${factor.mechanism} ${factor.source_task_id}`.toLowerCase().includes(query)).slice(0, 160).forEach(factor => {
    const label = element("label", "factor-option"); const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = quantState.selectedFactors.has(factor.factor_id); checkbox.onchange = () => { checkbox.checked ? quantState.selectedFactors.add(factor.factor_id) : quantState.selectedFactors.delete(factor.factor_id); text("selectedFactorCount", `${quantState.selectedFactors.size} 已选`); };
    const description = element("span"); description.innerHTML = `<strong>${escapeHtml(factor.name)}</strong><small>${escapeHtml(factor.mechanism)} · ${escapeHtml(factor.source_task_id || "legacy")}</small>`;
    const code = element("code"); code.textContent = factor.factor_id; label.append(checkbox, description, code); list.append(label);
  });
  text("selectedFactorCount", `${quantState.selectedFactors.size} 已选`);
}

function applyObjectivePreset() {
  if (!quantState.bootstrap) return;
  const profile = document.getElementById("formObjective").value;
  const preset = quantState.bootstrap.objective_presets.find(item => item.profile === profile) || quantState.bootstrap.objective_presets[0];
  text("presetNote", preset.description || "");
  const fields = { formCoverage: "minimum_coverage", formPositiveFolds: "minimum_positive_fold_fraction", formWorstFold: "minimum_worst_fold_sharpe", formMaxDrawdown: "maximum_drawdown", formTurnover: "maximum_annual_turnover", formCorrelation: "maximum_factor_correlation", formBets: "minimum_effective_factor_bets", formMechanisms: "minimum_effective_mechanisms", formMechanismWeight: "maximum_mechanism_weight", formDsr: "minimum_deflated_sharpe_probability" };
  Object.entries(fields).forEach(([id, key]) => { document.getElementById(id).value = preset[key]; });
}

async function createTask(event) {
  event.preventDefault();
  const profile = document.getElementById("formObjective").value;
  const preset = quantState.bootstrap.objective_presets.find(item => item.profile === profile);
  const selected = [...quantState.selectedFactors]; const mode = document.getElementById("formScopeMode").value;
  const body = {
    name: value("formName"), market: value("formMarket"), data_path: value("formDataPath"), notes: value("formNotes"),
    protocol: { exploration_start: value("formExplorationStart"), exploration_end: value("formExplorationEnd"), validation_start: value("formValidationStart"), validation_end: value("formValidationEnd"), holdout_start: value("formHoldoutStart"), holdout_end: value("formHoldoutEnd"), minimum_folds: integer("formMinimumFolds") },
    scope: { mode, factor_ids: selected, required_factor_ids: mode === "HYBRID" ? selected : [], excluded_factor_ids: [], source_task_ids: selectedOptions("formSourceTasks"), statuses: ["ELIGIBLE", "SCREENED_OUT", "ACTIVE"], families: [] },
    construction: { min_factors: integer("formMinFactors"), max_factors: integer("formMaxFactors"), minimum_weight: numeric("formMinWeight"), maximum_weight: numeric("formMaxWeight"), weight_step: numeric("formWeightStep"), candidate_pool_limit: integer("formPoolLimit"), allow_negative_weights: false, maximum_same_family: 2, maximum_same_semantic_cluster: 1 },
    objective: { ...preset, profile, minimum_coverage: numeric("formCoverage"), minimum_positive_fold_fraction: numeric("formPositiveFolds"), minimum_worst_fold_sharpe: numeric("formWorstFold"), maximum_drawdown: numeric("formMaxDrawdown"), maximum_annual_turnover: numeric("formTurnover"), maximum_factor_correlation: numeric("formCorrelation"), minimum_effective_factor_bets: numeric("formBets"), minimum_effective_mechanisms: numeric("formMechanisms"), maximum_mechanism_weight: numeric("formMechanismWeight"), minimum_deflated_sharpe_probability: numeric("formDsr") },
    engine: { mode: value("formEngineMode"), cluster_correlation_threshold: numeric("formClusterThreshold"), minimum_stability_score: -2, sffs_beam_width: integer("formBeamWidth"), evolution_population: integer("formPopulation"), evolution_generations: integer("formGenerations"), adaptive_trials: integer("formAdaptiveTrials"), covariance_shrinkage: numeric("formShrinkage"), weight_regularization: numeric("formRegularization"), random_seed: 20260718 },
    budget: { maximum_evaluations: integer("formEvaluations"), maximum_runtime_minutes: integer("formRuntime"), weight_candidates_per_subset: integer("formWeightCandidates"), iteration_interval_seconds: 0 },
  };
  delete body.objective.label; delete body.objective.description;
  try { const task = await api("/api/tasks", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); closeTaskDialog(); toast("统计组合任务已创建"); await loadBootstrap(); await selectTask(task.task_id, true); } catch (error) { toast(error.message, true); }
}

async function command(action) {
  const taskId = quantState.detail?.task?.task_id; if (!taskId) return;
  try { await api(`/api/tasks/${taskId}/${action}`, { method: "POST" }); toast(action === "start" ? "任务已启动" : "停止请求已提交"); await refreshCurrent(); } catch (error) { toast(error.message, true); }
}

async function promoteCurrent() {
  const task = quantState.detail?.task; const candidateId = Number(document.getElementById("promoteButton").dataset.candidateId); if (!task || !candidateId) return;
  const name = window.prompt("策略名称", `${task.name} · 生产候选`); if (!name) return;
  try { await api(`/api/tasks/${task.task_id}/promote`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ candidate_id: candidateId, name }) }); toast("策略版本已交付"); await loadBootstrap(); showStrategies(); window.history.pushState({}, "", "/strategies"); } catch (error) { toast(error.message, true); }
}

function renderStrategies() {
  if (!quantState.bootstrap) return; const strategies = quantState.bootstrap.strategies || []; text("strategyCount", `${strategies.length} 个版本`); const list = document.getElementById("strategyList"); list.replaceChildren(...strategies.map(strategy => { const card = element("article", "strategy-card"); const spec = strategy.specification; card.innerHTML = `<span class="state-pill small">${escapeHtml(strategy.lifecycle)}</span><h3>${escapeHtml(strategy.name)}</h3><code>${escapeHtml(strategy.strategy_id)} · VERSION ${strategy.version}</code><p>${spec.factor_ids.length} 因子 · ${escapeHtml(spec.engine.mode)} · ${formatTime(strategy.created_at)}</p>`; return card; })); if (!strategies.length) list.append(emptyLine("尚无通过隔离盲测的策略版本"));
}

function factorMap() { return Object.fromEntries((quantState.detail?.factor_snapshot || []).map(item => [item.factor_id, item])); }
async function api(path, options = {}) { const response = await fetch(path, options); const data = await response.json().catch(() => ({})); if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`); return data; }
function text(id, value) { const node = document.getElementById(id); if (node) node.textContent = value ?? "--"; }
function value(id) { return document.getElementById(id).value.trim(); } function numeric(id) { return Number(document.getElementById(id).value); } function integer(id) { return Math.trunc(numeric(id)); }
function selectedOptions(id) { return [...document.getElementById(id).selectedOptions].map(item => item.value); }
function element(tag, className = "") { const node = document.createElement(tag); if (className) node.className = className; return node; }
function option(value, label) { const node = document.createElement("option"); node.value = value; node.textContent = label; return node; }
function emptyLine(label) { const node = element("p", "table-empty"); node.textContent = label; return node; }
function tableEmpty(span, label) { const row = element("tr"); const cell = element("td", "table-empty"); cell.colSpan = span; cell.textContent = label; row.append(cell); return row; }
function number(value, digits = 2) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--"; }
function percent(value) { return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "--"; }
function formatTime(value) { if (!value) return "--"; return new Date(value).toLocaleString("zh-CN", { hour12: false }); }
function formatMetricAxis(value, metric) { return metric.includes("return") || metric.includes("drawdown") ? `${(value * 100).toFixed(1)}%` : value.toFixed(2); }
function toast(message, error = false) { const node = document.getElementById("toast"); node.textContent = message; node.classList.toggle("error", error); node.classList.add("show"); window.setTimeout(() => node.classList.remove("show"), 3200); }
function escapeHtml(value) { const node = document.createElement("span"); node.textContent = value ?? ""; return node.innerHTML; }
