const state = {
  taskId: taskIdFromPath(),
  snapshot: null,
  chartMetric: "sharpe_ratio",
  logCategory: "all",
  stream: null,
  refreshTimer: null,
  settingsHydrated: false,
};

const chartMetrics = {
  sharpe_ratio: { label: "夏普", format: "number", prefer: "max" },
  simple_annual_return: { label: "简单年化", format: "percent", prefer: "max" },
  pearson_ic_mean: { label: "IC", format: "number4", prefer: "max" },
  rank_ic_mean: { label: "Rank IC", format: "number4", prefer: "max" },
  rank_ic_ir: { label: "Rank IC IR", format: "number", prefer: "max" },
  pearson_ic_ir: { label: "IC IR", format: "number", prefer: "max" },
  compound_annual_return: { label: "复合年化", format: "percent", prefer: "max" },
  incremental_net_ir: { label: "增量净 IR", format: "number", prefer: "max" },
  incremental_max_drawdown: { label: "最大回撤变化", format: "percent", prefer: "max" },
  return_drawdown_efficiency_change: { label: "收益回撤效率", format: "number", prefer: "max" },
  cost_stress_net_ir: { label: "成本压力净 IR", format: "number", prefer: "max" },
  annual_turnover: { label: "年化换手", format: "number", prefer: "min" },
  coverage: { label: "覆盖率", format: "percent", prefer: "max" },
  capacity_cny: { label: "容量估计", format: "currency", prefer: "max" },
  positive_year_ratio: { label: "正收益年份占比", format: "percent", prefer: "max" },
  annual_return_dispersion: { label: "年收益离散", format: "percent", prefer: "min" },
  walk_forward_positive_fraction: { label: "滚动正折比例", format: "percent", prefer: "max" },
  walk_forward_worst_sharpe: { label: "最差折夏普", format: "number", prefer: "max" },
  deflated_sharpe_probability: { label: "Deflated Sharpe 概率", format: "percent", prefer: "max" },
  probability_backtest_overfitting: { label: "PBO", format: "percent", prefer: "min" },
  parameter_stability_positive_fraction: { label: "参数邻域正向比例", format: "percent", prefer: "max" },
  portfolio_sharpe_ratio: { label: "组合夏普", format: "number", prefer: "max" },
  portfolio_simple_annual_return: { label: "组合简单年化", format: "percent", prefer: "max" },
  portfolio_max_drawdown: { label: "组合最大回撤", format: "percent", prefer: "max" },
  portfolio_incremental_net_ir: { label: "组合边际净 IR", format: "number", prefer: "max" },
  portfolio_annual_turnover: { label: "组合年化换手", format: "number", prefer: "min" },
  portfolio_max_factor_correlation: { label: "最大因子相关性", format: "number4", prefer: "min" },
  portfolio_walk_forward_positive_fraction: { label: "组合滚动正折比例", format: "percent", prefer: "max" },
  portfolio_walk_forward_worst_sharpe: { label: "组合最差折夏普", format: "number", prefer: "max" },
  portfolio_deflated_sharpe_probability: { label: "组合 Deflated Sharpe", format: "percent", prefer: "max" },
};

const flowPhases = ["CONFIGURE", "MEMORY", "DIRECTION", "PROPOSAL", "SEMANTICS", "EVALUATION", "PORTFOLIO", "HOLDOUT", "CAPITAL_SIMULATION", "EVIDENCE", "DELIVERY", "WAITING"];
const flowLabels = {
  CONFIGURE: "等待 API、模型与数据配置",
  MEMORY: "正在载入连续研究记忆",
  DIRECTION: "正在自检公开证据并冻结本轮优化方向",
  PROPOSAL: "Researcher 正在生成候选",
  SEMANTICS: "正在校验 DSL、时序与重复表达式",
  EVALUATION: "正在运行真实价量回测",
  PORTFOLIO: "正在枚举组合保留、加入、删除与替换动作",
  HOLDOUT: "冻结组合正在隔离边界内执行一次性盲测",
  CAPITAL_SIMULATION: "正在执行带交易限制的人民币资金仿真",
  EVIDENCE: "正在执行研究门禁与证据入库",
  DELIVERY: "正在发布哈希制品与交付日志",
  WAITING: "本轮完成，等待下一轮",
  RETRY: "本轮异常，正在退避重试",
  BLOCKED: "连续错误触发熔断，等待人工检查",
  STOPPING: "当前轮次完成后停止",
  STOPPED: "研究循环已停止",
};

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  bindControls();
  try {
    await refresh();
    connectStream();
  } catch (error) {
    handleError(error);
  }
  new ResizeObserver(drawChart).observe(document.querySelector(".chart-frame"));
});

function bindControls() {
  document.getElementById("startBtn").onclick = startResearch;
  document.getElementById("stopBtn").onclick = () => command(taskApi("/stop"), "停止请求已登记");
  document.getElementById("refreshBtn").onclick = refreshWithToast;
  document.getElementById("dataSyncBtn").onclick = () => command("/api/data-sync/start", "数据增量同步已启动");
  document.getElementById("experimentRefreshBtn").onclick = refreshWithToast;
  document.getElementById("verifyBtn").onclick = () => command(taskApi("/audit/verify"), "审计哈希链验证通过");
  document.getElementById("settingsForm").addEventListener("submit", saveSettings);
  document.getElementById("manualLogForm").addEventListener("submit", appendManualLog);
  document.getElementById("chartMetric").addEventListener("change", event => {
    state.chartMetric = event.target.value;
    renderChartSummary(state.snapshot?.metrics.at(-1) || null);
    drawChart();
  });
  document.querySelectorAll("#logFilters button").forEach(button => {
    button.onclick = () => {
      state.logCategory = button.dataset.category;
      document.querySelectorAll("#logFilters button").forEach(item => item.classList.toggle("active", item === button));
      renderLogs();
    };
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.detail || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function refresh() {
  state.snapshot = await api(taskApi("/workspace"));
  render();
}

async function refreshWithToast() {
  try {
    await refresh();
    toast("数据已刷新");
  } catch (error) {
    handleError(error);
  }
}

async function command(path, message) {
  try {
    await api(path, { method: "POST", body: "{}" });
    toast(message);
    await refresh();
  } catch (error) {
    handleError(error);
  }
}

async function startResearch() {
  if (!state.snapshot?.settings.api_key_configured) {
    document.getElementById("apiKey").focus();
    toast("请先填写并保存 API Key", true);
    return;
  }
  if (!state.snapshot?.research_task?.readiness?.runnable) {
    toast(state.snapshot?.research_task?.readiness?.blockers?.join("；") || "当前任务暂不可启动", true);
    return;
  }
  if (state.snapshot?.data_sync?.running) {
    toast("市场数据正在更新，完成后再启动研究", true);
    return;
  }
  await command(taskApi("/start"), "持续研究已启动");
}

async function saveSettings(event) {
  event.preventDefault();
  try {
    const serviceToken = document.getElementById("serviceToken").value;
    if (serviceToken) {
      await api("/api/session", { method: "POST", body: JSON.stringify({ token: serviceToken }) });
    }
    await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify({
        base_url: document.getElementById("baseUrl").value,
        api_key: document.getElementById("apiKey").value || null,
        tushare_token: document.getElementById("tushareToken").value || null,
        model: document.getElementById("model").value,
        temperature: Number(document.getElementById("temperature").value),
        data_path: state.snapshot.settings.data_path,
        iteration_interval_seconds: Number(document.getElementById("interval").value),
        maximum_active_factors: Number(document.getElementById("maximumActiveFactors").value),
        market_data_root: document.getElementById("marketDataRoot").value,
        data_auto_update_enabled: document.getElementById("dataAutoUpdate").checked,
        data_update_hour: Number(document.getElementById("dataUpdateHour").value),
      }),
    });
    document.getElementById("apiKey").value = "";
    document.getElementById("tushareToken").value = "";
    toast("服务配置已保存");
    await refresh();
  } catch (error) {
    handleError(error);
  }
}

async function appendManualLog(event) {
  event.preventDefault();
  const content = document.getElementById("manualContent").value.trim();
  if (!content) return;
  try {
    await api(taskApi("/logs/manual"), {
      method: "POST",
      body: JSON.stringify({
        category: document.getElementById("manualCategory").value,
        content,
      }),
    });
    document.getElementById("manualContent").value = "";
    toast("人工日志已写入审计流");
    await refresh();
  } catch (error) {
    handleError(error);
  }
}

function connectStream() {
  if (state.stream) state.stream.close();
  const stream = new EventSource(taskApi("/events"), { withCredentials: true });
  state.stream = stream;
  stream.onopen = () => document.getElementById("streamState").classList.remove("offline");
  stream.onerror = () => document.getElementById("streamState").classList.add("offline");
  stream.onmessage = () => {
    clearTimeout(state.refreshTimer);
    state.refreshTimer = setTimeout(() => refresh().catch(() => {}), 180);
  };
}

function render() {
  const snapshot = state.snapshot;
  const service = snapshot.state;
  const settings = snapshot.settings;
  const task = snapshot.research_task || {};
  const latest = snapshot.metrics.at(-1) || null;
  renderTaskIdentity(task);
  const badge = document.getElementById("runStatus");
  badge.textContent = service.state;
  badge.className = `state-pill ${service.state}`;
  text("serviceSummary", `running=${snapshot.worker_alive} · iteration=${service.iteration} · phase=${service.phase} · last_error=${service.last_error || "none"}`);
  text("credentialState", settings.api_key_configured ? "Keychain 已配置" : "缺少密钥");
  const dataSync = snapshot.data_sync || {};
  const syncTargetsTask = task.data_path === settings.data_path;
  text("dataSyncState", syncTargetsTask
    ? `数据同步：${dataSync.state || "IDLE"}${dataSync.message ? ` · ${dataSync.message}` : ""}`
    : "当前任务使用独立数据目录，请在数据中心维护对应数据源");
  document.getElementById("dataSyncBtn").disabled = Boolean(dataSync.running) || snapshot.any_worker_alive || !syncTargetsTask;
  text("memoryCount", `${snapshot.memory_count} 条`);
  const runnable = task.readiness?.runnable !== false;
  document.getElementById("startBtn").disabled = snapshot.worker_alive || Boolean(dataSync.running) || !runnable;
  document.getElementById("startBtn").title = runnable ? "启动当前任务" : (task.readiness?.blockers?.join("；") || "当前任务暂不可启动");
  document.getElementById("stopBtn").disabled = !snapshot.worker_alive;
  hydrateSettings(settings, task);
  renderProtocol();
  renderMemory();
  renderChartSummary(latest);
  renderFlow(service);
  renderPortfolio();
  renderExperiments();
  renderLogs();
  drawChart();
}

function renderTaskIdentity(task) {
  const name = task.name || state.taskId;
  const market = ({ CN_A: "A 股", HK: "港股", US: "美股" })[task.market] || task.market || "--";
  text("researchWorkspaceTitle", `${name} · 自动研究`);
  const identity = document.getElementById("taskWorkspaceIdentity");
  identity.textContent = `${market} · ${task.task_id || state.taskId}`;
  identity.href = `/research-tasks/${encodeURIComponent(task.task_id || state.taskId)}`;
  identity.title = task.data_path || "查看任务配置";
  document.getElementById("autoResearchNav").href = `/research/${encodeURIComponent(task.task_id || state.taskId)}`;
  document.title = `${name} · AutoAlpha`;
}

function hydrateSettings(settings, task) {
  if (state.settingsHydrated) return;
  document.getElementById("baseUrl").value = settings.base_url || "https://api.deepseek.com";
  document.getElementById("model").value = settings.model || "deepseek-v4-pro";
  document.getElementById("temperature").value = settings.temperature || "0.7";
  document.getElementById("interval").value = settings.iteration_interval_seconds || "5";
  document.getElementById("maximumActiveFactors").value = settings.maximum_active_factors || "5";
  document.getElementById("dataPath").value = task.data_path || settings.data_path || "";
  document.getElementById("marketDataRoot").value = settings.market_data_root || "~/MarketData/Ashare";
  document.getElementById("dataAutoUpdate").checked = settings.data_auto_update_enabled === "true";
  document.getElementById("dataUpdateHour").value = settings.data_update_hour || "18";
  document.getElementById("tokenLabel").hidden = !settings.service_token_required;
  state.settingsHydrated = true;
}

function renderMemory() {
  const memories = state.snapshot.memories.slice().reverse().slice(0, 8);
  const list = document.getElementById("memoryList");
  if (!memories.length) {
    list.replaceChildren(emptyState("暂无连续记忆"));
    return;
  }
  list.replaceChildren(...memories.map(memory => {
    const item = element("article", "memory-item");
    const header = document.createElement("header");
    header.append(element("strong", "", memory.kind.toUpperCase()), element("span", "", `ITER ${memory.iteration}`));
    item.append(header, element("p", "", formatMemory(memory.content)));
    return item;
  }));
}

function formatMemory(content) {
  try {
    const memory = JSON.parse(content);
    const single = memory.single_factor || {};
    const portfolio = memory.portfolio || {};
    const best = portfolio.best_option;
    const parts = [
      `${memory.name || "候选"} · ${memory.family || "未分类"}`,
      `夏普 ${number(single.sharpe)} · 年化 ${percent(single.annual_return)} · 换手 ${number(single.turnover)}`,
    ];
    if (single.exploratory_failures?.length) {
      parts.push(`单因子未过：${single.exploratory_failures.join(", ")}`);
    }
    if (Number.isFinite(Number(single.walk_forward_positive_fraction))) {
      parts.push(`滚动正折 ${percent(single.walk_forward_positive_fraction)} · 最差折夏普 ${number(single.walk_forward_worst_sharpe)}`);
    }
    if (Number.isFinite(Number(single.deflated_sharpe_probability))) {
      parts.push(`DSR ${percent(single.deflated_sharpe_probability)} · PBO ${percent(single.probability_backtest_overfitting)}`);
    }
    if (portfolio.accepted) {
      parts.push(`组合已接受 ${portfolio.action}`);
    }
    if (best) {
      const candidateWeight = best.weights?.at(-1);
      const bestLabel = best.failed_gates?.length
        ? (portfolio.accepted ? "备选近失" : "最佳近失")
        : "已采用方案";
      parts.push(
        `${bestLabel} ${best.action}${Number.isFinite(Number(candidateWeight)) ? ` @ ${percent(candidateWeight)}` : ""}`,
        `选择差值 ${number(best.utility_change)} · 未过：${best.failed_gates?.join(", ") || "无"}`,
      );
    } else if (portfolio.action) {
      parts.push(`${portfolio.action}${portfolio.accepted ? " 已接受" : " 保持"}`);
    }
    if (portfolio.holdout_verdict && !portfolio.holdout_verdict.startsWith("NOT_EVALUATED")) parts.push(`盲测 ${portfolio.holdout_verdict}`);
    if (portfolio.capital_verdict && !portfolio.capital_verdict.startsWith("NOT_EVALUATED")) parts.push(`资金门禁 ${portfolio.capital_verdict}`);
    return parts.join("；");
  } catch (_) {
    return content;
  }
}

function renderProtocol() {
  const protocol = state.snapshot.research_protocol || {};
  const generation = state.snapshot.research_generation || {};
  const walk = protocol.walk_forward || {};
  const portfolio = protocol.portfolio || {};
  text("protocolSummary", `${protocol.version || "--"} · ${protocol.generation || "--"}`);
  text("generationStatus", generation.status || "NOT STARTED");
  text("protocolExploration", protocol.exploration ? `${protocol.exploration.start} — ${protocol.exploration.end}` : "--");
  text("protocolWalkForward", walk.first_validation_year ? `${walk.train_years}Y → ${walk.validation_years}Y · ${walk.first_validation_year}—${walk.last_validation_year}` : "--");
  text("candidateBudget", generation.maximum_candidates == null ? "--" : `${generation.candidate_attempts} / ${generation.maximum_candidates}`);
  text("holdoutBudget", generation.maximum_holdout_attempts == null ? "--" : `${generation.holdout_attempts} / ${generation.maximum_holdout_attempts}`);
  text("holdingPeriod", portfolio.holding_period_days ? `${portfolio.holding_period_days} 日` : "--");
  text("targetExposure", percent(portfolio.target_gross_exposure));
  text("holdoutBoundary", protocol.holdout ? `${protocol.holdout.start} — ${protocol.holdout.end} · 仅分级结论与证据哈希` : "隐藏指标不进入模型上下文");
  const adaptive = state.snapshot.adaptive_direction || {};
  const campaign = adaptive.active || adaptive.latest;
  if (!campaign) {
    text("directionTitle", adaptive.config?.enabled ? "等待首次公开方向诊断" : "自适应方向已禁用");
    text("directionObjective", "方向只由公开研究证据决定");
    text("directionRationale", "尚无方向战役记录");
    text("directionProgress", `0 / ${adaptive.config?.maximum_attempts_per_campaign || 3}`);
    text("directionRemaining", "未开始");
    document.getElementById("directionProgressBar").style.width = "0%";
  } else if (portfolio.protocol_stale) {
    text("portfolioSummary", `历史组合协议已失效 · 当前 ${portfolio.current_protocol} · 下一轮将从当前合格因子池重建`);
    text("portfolioVersion", `HISTORICAL VERSION ${portfolio.id}`);
    ["portfolioSharpe", "portfolioAnnual", "portfolioDrawdown", "portfolioTurnover", "portfolioCorrelation"].forEach(id => text(id, "--"));
    text("portfolioFactorCount", "0");
    members.replaceChildren(emptyState("旧组合仅保留审计记录，不再作为当前冠军"));
  } else {
    const maximum = Number(campaign.maximum_attempts) || 1;
    const used = Number(campaign.attempts_used) || 0;
    const remaining = Math.max(0, maximum - used);
    text("directionTitle", `${campaign.title} · ${campaign.status}`);
    text("directionObjective", campaign.objective);
    text("directionRationale", campaign.rationale?.join("；") || "公开门禁与近期失败聚合诊断");
    text("directionProgress", `${used} / ${maximum}`);
    text("directionRemaining", campaign.status === "ACTIVE" ? `剩余 ${remaining} 次` : (campaign.closure_reason || campaign.status));
    document.getElementById("directionProgressBar").style.width = `${Math.min(100, 100 * used / maximum)}%`;
  }
  const records = state.snapshot.blind_evaluations || [];
  const box = document.getElementById("blindVerdicts");
  if (!records.length) {
    box.replaceChildren(emptyState("当前世代尚未消耗盲测额度"));
    return;
  }
  box.replaceChildren(...records.slice(0, 6).map(record => {
    const row = element("article", `blind-verdict ${record.holdout_passed ? "passed" : "failed"}`);
    row.append(
      element("strong", "", `ITER ${record.iteration} · ${record.holdout_verdict}`),
      element("span", "", record.capital_verdict || "资本门禁未执行"),
      element("code", "", record.holdout_evidence_hash.slice(0, 16)),
    );
    return row;
  }));
}

function renderChartSummary(latest) {
  const key = state.chartMetric;
  const config = chartMetrics[key];
  const completed = state.snapshot.iterations.filter(item => item.metrics && Number.isFinite(Number(item.metrics[key])));
  const values = completed.map(item => Number(item.metrics[key]));
  const best = values.length ? (config.prefer === "min" ? Math.min(...values) : Math.max(...values)) : null;
  text("chartTitle", `${config.label}迭代曲线`);
  text("latestMetricLabel", `latest · ${config.label}`);
  text("bestMetricLabel", `${config.prefer === "min" ? "lowest" : "best"} · ${config.label}`);
  text("latestMetric", latest ? formatMetric(latest[key], config.format) : "--");
  text("bestMetric", best == null ? "--" : formatMetric(best, config.format));
  text("bestLegend", config.prefer === "min" ? "RUNNING LOWEST" : "RUNNING BEST");
  text("totalIterations", state.snapshot.iteration_stats.total);
  if (!latest) {
    text("chartSubtitle", "等待首个有效实验");
    return;
  }
  text("chartSubtitle", `latest #${latest.iteration} · ${outcomeFromMetrics(latest)} · ${config.label}=${formatMetric(latest[key], config.format)} · gates=${latest.exploratory_gate_failure_count ?? "--"}`);
}

function renderFlow(service) {
  const phase = service.phase || "CONFIGURE";
  const index = flowPhases.indexOf(phase);
  text("flowStatusText", flowLabels[phase] || phase);
  text("flowIteration", `ITER ${service.iteration}`);
  document.querySelectorAll(".flow-step").forEach((step, stepIndex) => {
    step.classList.toggle("completed", index >= 0 && stepIndex < index);
    const fallbackActive = ["RETRY", "BLOCKED"].includes(phase) && step.dataset.phase === "EVALUATION";
    step.classList.toggle("active", step.dataset.phase === phase || fallbackActive);
  });
}

function renderPortfolio() {
  const portfolio = state.snapshot.portfolio;
  const members = document.getElementById("portfolioMembers");
  const actions = document.getElementById("portfolioActions");
  if (!portfolio) {
    text("portfolioSummary", "尚未建立组合，下一轮将从合格因子池初始化");
    text("portfolioVersion", "VERSION --");
    ["portfolioSharpe", "portfolioAnnual", "portfolioDrawdown", "portfolioTurnover", "portfolioCorrelation"].forEach(id => text(id, "--"));
    text("portfolioFactorCount", "0");
    members.replaceChildren(emptyState("暂无活跃因子"));
  } else {
    const metrics = portfolio.metrics || {};
    const evaluationProtocol = metrics.portfolio_evaluation_protocol || metrics.evaluation_protocol;
    const protocolLabel = evaluationProtocol || "LEGACY / UNVERSIONED";
    text("portfolioSummary", `第 ${portfolio.iteration} 轮 · ${portfolio.action} · ${protocolLabel} · ${portfolio.reason}`);
    text("portfolioVersion", `VERSION ${portfolio.id}`);
    text("portfolioSharpe", number(metrics.portfolio_sharpe_ratio));
    text("portfolioAnnual", percent(metrics.portfolio_simple_annual_return));
    text("portfolioDrawdown", percent(metrics.portfolio_max_drawdown));
    text("portfolioTurnover", number(metrics.portfolio_annual_turnover));
    text("portfolioCorrelation", number4(metrics.portfolio_max_factor_correlation));
    text("portfolioFactorCount", String(portfolio.members.length));
    members.replaceChildren(...portfolio.members.map(member => {
      const row = element("article", "portfolio-member");
      const identity = element("div");
      identity.append(element("strong", "", member.name), element("span", "", `${member.family} · ITER ${member.source_iteration}`));
      row.append(identity, element("strong", "member-weight", percent(member.weight)));
      return row;
    }));
  }
  const history = state.snapshot.portfolio_history || [];
  if (!history.length) {
    actions.replaceChildren(emptyState("暂无组合动作"));
    return;
  }
  actions.replaceChildren(...history.slice(0, 8).map(item => {
    const row = element("article", `portfolio-action ${item.accepted ? "accepted" : "held"}`);
    row.append(
      element("strong", "", `${item.action}${item.accepted ? " · 已更新" : " · 保持"}`),
      element("span", "", `ITER ${item.iteration} · ${formatDateTime(item.created_at)}`),
      element("p", "", item.reason),
    );
    return row;
  }));
}

function renderExperiments() {
  const iterations = state.snapshot.iterations;
  const stats = state.snapshot.iteration_stats;
  text("experimentSummary", `${iterations.length} 条最近记录 · ${stats.completed} 完成 · ${stats.failed} 失败 · 成功率 ${percent(stats.success_rate)}`);
  const list = document.getElementById("experimentList");
  if (!iterations.length) {
    list.replaceChildren(emptyState("暂无实验记录"));
    return;
  }
  list.replaceChildren(...iterations.slice(0, 50).map(renderExperimentCard));
}

function renderExperimentCard(iteration) {
  const outcome = iterationOutcome(iteration);
  const card = element("article", `record-card ${outcome.className}`);
  const header = document.createElement("header");
  header.append(
    element("h3", "", `Run #${String(iteration.iteration).padStart(4, "0")} · ${outcome.label}`),
    element("time", "", formatDateTime(iteration.finished_at || iteration.started_at)),
  );
  const metrics = iteration.metrics;
  const proposal = iteration.proposal;
  const line = metrics
    ? `sharpe ${number(metrics.sharpe_ratio)} · annual ${percent(metrics.simple_annual_return)} · rank_ic ${number4(metrics.rank_ic_mean)} · portfolio ${metrics.portfolio_action || "--"}`
    : iteration.error || proposal?.hypothesis || "候选正在处理";
  const tags = element("div", "tags");
  [
    iteration.candidate_id,
    metrics && `return ${percent(metrics.simple_annual_return)}`,
    metrics && `drawdown ${percent(metrics.incremental_max_drawdown)}`,
    metrics && `coverage ${percent(metrics.coverage)}`,
    metrics && `gates ${metrics.exploratory_gate_failure_count ?? "--"}`,
    metrics?.portfolio_action && `${metrics.portfolio_action}${metrics.portfolio_action_accepted ? " accepted" : " hold"}`,
    metrics?.portfolio_factor_count && `${metrics.portfolio_factor_count} factors`,
    proposal?.family,
  ].filter(Boolean).forEach(value => tags.append(element("span", "tag", value)));
  const details = document.createElement("details");
  details.append(element("summary", "", "实验详情"));
  details.append(element("pre", "", JSON.stringify({ proposal, decision: iteration.decision, error: iteration.error, metrics }, null, 2)));
  card.append(header, element("p", "record-line", line), tags, details);
  return card;
}

function renderLogs() {
  const events = state.snapshot.events
    .filter(event => state.logCategory === "all" || event.category === state.logCategory)
    .slice()
    .reverse();
  text("logCount", `${events.length} 条`);
  const list = document.getElementById("logList");
  if (!events.length) {
    list.replaceChildren(emptyState("当前分类暂无日志"));
    return;
  }
  list.replaceChildren(...events.map(event => {
    const card = element("article", "record-card log-card");
    card.dataset.category = event.category;
    const header = document.createElement("header");
    header.append(element("h3", "", `#${event.id} · ${event.title}`), element("time", "", formatDateTime(event.timestamp_utc)));
    const tags = element("div", "tags");
    [event.category.toUpperCase(), event.iteration && `ITER ${event.iteration}`, event.level, event.event].filter(Boolean).forEach(value => tags.append(element("span", "tag", value)));
    const details = document.createElement("details");
    details.append(element("summary", "", "原始详情"));
    details.append(element("pre", "", JSON.stringify(event.payload, null, 2)));
    card.append(header, element("p", "record-line", event.message), tags, details);
    return card;
  }));
}

function drawChart() {
  if (!state.snapshot) return;
  const key = state.chartMetric;
  const config = chartMetrics[key];
  const rows = state.snapshot.iterations.slice().reverse();
  const completed = rows.filter(row => row.metrics && Number.isFinite(Number(row.metrics[key])));
  const canvas = document.getElementById("metricChart");
  const empty = document.getElementById("chartEmpty");
  empty.hidden = completed.length > 0;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  if (!completed.length || !width || !height) return;

  const values = completed.map(row => Number(row.metrics[key]));
  let min = Math.min(...values, 0);
  let max = Math.max(...values, 0);
  const spread = Math.max(max - min, 1);
  min -= spread * 0.18;
  max += spread * 0.18;
  const pad = { left: 48, right: 20, top: 16, bottom: 32 };
  const xFor = index => pad.left + (width - pad.left - pad.right) * (rows.length === 1 ? 0.5 : index / (rows.length - 1));
  const yFor = value => pad.top + (height - pad.top - pad.bottom) * (1 - (value - min) / (max - min));

  ctx.font = "10px system-ui";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const value = max - (max - min) * i / 4;
    const y = yFor(value);
    ctx.strokeStyle = "#e3e8ef";
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillStyle = "#7d899c"; ctx.textAlign = "right"; ctx.fillText(formatAxis(value, config.format), pad.left - 7, y + 3);
  }

  const completedPoints = rows.map((row, index) => row.metrics ? { row, index, value: Number(row.metrics[key]) } : null).filter(point => point && Number.isFinite(point.value));
  ctx.strokeStyle = "#263247"; ctx.lineWidth = 2; ctx.beginPath();
  completedPoints.forEach((point, index) => {
    const x = xFor(point.index), y = yFor(point.value);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  let runningBest = config.prefer === "min" ? Infinity : -Infinity;
  ctx.strokeStyle = "#245eea"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]); ctx.beginPath();
  completedPoints.forEach((point, index) => {
    runningBest = config.prefer === "min" ? Math.min(runningBest, point.value) : Math.max(runningBest, point.value);
    const x = xFor(point.index), y = yFor(runningBest);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke(); ctx.setLineDash([]);

  rows.forEach((row, index) => {
    const x = xFor(index);
    if (row.status === "FAILED") {
      ctx.fillStyle = "#c33832"; ctx.fillRect(x - 3, yFor(min) - 6, 6, 6); return;
    }
    if (!row.metrics || !Number.isFinite(Number(row.metrics[key]))) return;
    ctx.fillStyle = row.metrics.exploratory_gate_passed ? "#09845b" : "#be4b0a";
    ctx.beginPath(); ctx.arc(x, yFor(Number(row.metrics[key])), 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  });

  ctx.fillStyle = "#7d899c"; ctx.textAlign = "center";
  const labelCount = Math.min(8, rows.length);
  for (let i = 0; i < labelCount; i += 1) {
    const index = Math.round(i * (rows.length - 1) / Math.max(1, labelCount - 1));
    ctx.fillText(`#${rows[index].iteration}`, xFor(index), height - 10);
  }
}

function iterationOutcome(iteration) {
  if (iteration.status === "RUNNING") return { label: "RUNNING", className: "running" };
  if (iteration.status === "FAILED") return { label: "CRASH / ERROR", className: "failed" };
  if (iteration.metrics?.exploratory_gate_passed) return { label: "GATE PASS", className: "accepted" };
  return { label: "REJECTED", className: "rejected" };
}

function outcomeFromMetrics(metrics) {
  return metrics.exploratory_gate_passed ? "GATE PASS" : "REJECTED";
}

function element(tag, className = "", content = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== "") node.textContent = content;
  return node;
}

function emptyState(message) {
  return element("p", "sidebar-note", message);
}

function text(id, value) {
  document.getElementById(id).textContent = value;
}

function taskIdFromPath() {
  const parts = location.pathname.split("/").filter(Boolean);
  return parts[0] === "research" && parts[1]
    ? decodeURIComponent(parts[1])
    : "legacy-ashare";
}

function taskApi(suffix) {
  return `/api/research-tasks/${encodeURIComponent(state.taskId)}${suffix}`;
}

function formatDateTime(value) {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "--";
}

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(2) : "--";
}

function number4(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(4) : "--";
}

function percent(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(2)}%` : "--";
}

function formatMetric(value, format) {
  if (format === "percent") return percent(value);
  if (format === "number4") return number4(value);
  if (format === "currency") return currency(value);
  return number(value);
}

function formatAxis(value, format) {
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "number4") return value.toFixed(3);
  if (format === "currency") return compactNumber(value);
  return value.toFixed(2);
}

function currency(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "--";
  return `${compactNumber(parsed)} 元`;
}

function compactNumber(value) {
  const absolute = Math.abs(value);
  if (absolute >= 1e8) return `${(value / 1e8).toFixed(2)}亿`;
  if (absolute >= 1e4) return `${(value / 1e4).toFixed(1)}万`;
  return value.toFixed(0);
}

function handleError(error) {
  if (error.status === 401) toast("服务访问令牌无效", true);
  else toast(error.message, true);
}

let toastTimer;
function toast(message, isError = false) {
  const node = document.getElementById("toast");
  node.textContent = message;
  node.style.background = isError ? "#a93430" : "#202a3b";
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 2800);
}
