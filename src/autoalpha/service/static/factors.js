const libraryState = {
  data: null,
  refreshedAt: null,
  refreshing: false,
  selected: new Set(),
  query: "",
  category: "all",
  task: "all",
  status: "all",
  lifecycle: "all",
  ranking: "overall",
  detailFactor: null,
};

const LIBRARY_REFRESH_INTERVAL_MS = 15_000;

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  bindLibraryControls();
  await loadLibrary({ announce: false });
  window.setInterval(() => {
    if (!document.hidden) loadLibrary({ announce: false });
  }, LIBRARY_REFRESH_INTERVAL_MS);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) loadLibrary({ announce: false });
  });
  window.addEventListener("mathjax-ready", () => typesetFormula());
});

function bindLibraryControls() {
  document.getElementById("factorSearch").addEventListener("input", event => {
    libraryState.query = event.target.value.trim().toLowerCase();
    renderFactorTable();
  });
  document.getElementById("categoryFilter").addEventListener("change", event => {
    libraryState.category = event.target.value;
    renderFactorTable();
  });
  document.getElementById("taskFilter").addEventListener("change", event => {
    libraryState.task = event.target.value;
    renderFactorTable();
  });
  document.getElementById("statusFilter").addEventListener("change", event => {
    libraryState.status = event.target.value;
    renderFactorTable();
  });
  document.getElementById("lifecycleFilter").addEventListener("change", event => {
    libraryState.lifecycle = event.target.value;
    renderFactorTable();
  });
  document.getElementById("rankingMetric").addEventListener("change", event => {
    libraryState.ranking = event.target.value;
    renderFactorTable();
  });
  document.getElementById("libraryRefresh").onclick = () => loadLibrary({ announce: true });
  document.getElementById("clearSelection").onclick = () => {
    libraryState.selected.clear();
    renderFactorTable();
    renderSelection();
  };
  document.getElementById("openBacktest").onclick = () => {
    const factors = [...libraryState.selected].join(",");
    window.location.href = `/backtest?factors=${encodeURIComponent(factors)}`;
  };
  document.getElementById("selectVisible").addEventListener("change", event => {
    filteredFactors().forEach(factor => {
      if (event.target.checked) libraryState.selected.add(factor.factor_id);
      else libraryState.selected.delete(factor.factor_id);
    });
    renderFactorTable();
    renderSelection();
  });
  document.getElementById("closeFactorDetail").onclick = closeFactorDetail;
  document.getElementById("factorDetailDialog").addEventListener("click", event => {
    if (event.target === event.currentTarget) closeFactorDetail();
  });
  document.getElementById("copyDsl").onclick = () => copyDetailText("DSL", JSON.stringify(libraryState.detailFactor?.expression || {}, null, 2));
  document.getElementById("copyLatex").onclick = () => copyDetailText("LaTex", expressionToLatex(libraryState.detailFactor?.expression));
}

async function loadLibrary({ announce = true } = {}) {
  if (libraryState.refreshing) return;
  libraryState.refreshing = true;
  const refreshButton = document.getElementById("libraryRefresh");
  refreshButton.disabled = true;
  try {
    libraryState.data = await api("/api/factors");
    libraryState.refreshedAt = new Date();
    hydrateFilters();
    renderSummary();
    renderCategories();
    renderFactorTable();
    renderSelection();
    if (announce) toast("因子库已刷新");
  } catch (error) {
    toast(error.message, true);
  } finally {
    libraryState.refreshing = false;
    refreshButton.disabled = false;
  }
}

function hydrateFilters() {
  const select = document.getElementById("categoryFilter");
  const current = select.value;
  select.replaceChildren(option("all", "全部分类"));
  libraryState.data.categories.forEach(category => {
    select.append(option(category.name, `${category.name} · ${category.count}`));
  });
  select.value = [...select.options].some(item => item.value === current) ? current : "all";
  const taskSelect = document.getElementById("taskFilter");
  const currentTask = taskSelect.value;
  taskSelect.replaceChildren(option("all", "全部任务"));
  (libraryState.data.research_tasks || []).forEach(task => {
    taskSelect.append(option(task.task_id, `${task.name} · ${marketLabel(task.market)}`));
  });
  taskSelect.value = [...taskSelect.options].some(item => item.value === currentTask) ? currentTask : "all";
}

function renderSummary() {
  const summary = libraryState.data.summary;
  text("factorCount", summary.factor_count);
  text("activeCount", summary.active_count);
  text("qualifiedCount", summary.eligible_count);
  text("observedCount", summary.observed_count);
  text("categoryCount", summary.category_count);
  text("clusterCount", summary.cluster_count);
  text("shadowCount", summary.shadow_count);
  text("contaminatedCount", summary.contaminated_count);
  text("dataEnd", libraryState.data.data.last_trade_date);
  text(
    "librarySummary",
    `${summary.factor_count} 个持久化候选 · ${summary.stale_protocol_count} 个等待当前协议重评 · ${formatClock(libraryState.refreshedAt)} 已同步`,
  );
}

function renderCategories() {
  const strip = document.getElementById("categoryStrip");
  strip.replaceChildren(...libraryState.data.categories.map(category => {
    const button = element("button", "category-chip");
    button.type = "button";
    button.append(element("strong", "", category.count), element("span", "", category.name));
    button.onclick = () => {
      libraryState.category = category.name;
      document.getElementById("categoryFilter").value = category.name;
      renderFactorTable();
    };
    return button;
  }));
}

function filteredFactors() {
  if (!libraryState.data) return [];
  return libraryState.data.factors
    .filter(factor => libraryState.category === "all" || factor.category === libraryState.category)
    .filter(factor => libraryState.task === "all" || factor.source_task_id === libraryState.task)
    .filter(factor => libraryState.status === "all" || factor.research_state === libraryState.status)
    .filter(factor => libraryState.lifecycle === "all" || factor.lifecycle_state === libraryState.lifecycle)
    .filter(factor => {
      if (!libraryState.query) return true;
      const haystack = [factor.factor_id, factor.name, factor.family, factor.category, factor.hypothesis, factor.cluster_id, factor.lifecycle_state, factor.source_task_name, factor.source_market, factor.origin, ...(factor.tags || [])]
        .join(" ").toLowerCase();
      return haystack.includes(libraryState.query);
    })
    .sort((left, right) => right.scores[libraryState.ranking] - left.scores[libraryState.ranking]);
}

function renderFactorTable() {
  const factors = filteredFactors();
  const body = document.getElementById("factorTableBody");
  body.replaceChildren(...factors.map((factor, index) => factorRow(factor, index + 1)));
  document.getElementById("factorEmpty").hidden = factors.length > 0;
  const selectedVisible = factors.filter(factor => libraryState.selected.has(factor.factor_id)).length;
  const selectVisible = document.getElementById("selectVisible");
  selectVisible.checked = factors.length > 0 && selectedVisible === factors.length;
  selectVisible.indeterminate = selectedVisible > 0 && selectedVisible < factors.length;
  renderSelection();
}

function factorRow(factor, rank) {
  const historicalMetrics = Boolean(factor.protocol_stale);
  const metrics = historicalMetrics
    ? (factor.historical_metric_summary || {})
    : (factor.metric_summary || {});
  const row = element("tr", libraryState.selected.has(factor.factor_id) ? "selected" : "");
  const selector = document.createElement("input");
  selector.type = "checkbox";
  selector.checked = libraryState.selected.has(factor.factor_id);
  selector.setAttribute("aria-label", `选择 ${factor.name}`);
  selector.onchange = () => {
    if (selector.checked) libraryState.selected.add(factor.factor_id);
    else libraryState.selected.delete(factor.factor_id);
    row.classList.toggle("selected", selector.checked);
    renderSelection();
  };
  row.append(cell(selector), cell(element("strong", "rank-value", `#${rank}`)));

  const identity = element("div", "factor-identity");
  const detailButton = element("button", "factor-detail-trigger");
  detailButton.type = "button";
  detailButton.title = `查看 ${factor.name} 的公式与研究档案`;
  detailButton.append(element("strong", "", factor.name), element("code", "", `${factor.factor_id} · ITER ${factor.source_iteration}`));
  detailButton.onclick = () => openFactorDetail(factor);
  identity.append(detailButton, element("p", "factor-hypothesis", factor.hypothesis || "未记录经济假设"));
  row.append(cell(identity));
  const source = element("a", "factor-source-task", factor.source_task_name || factor.source_task_id);
  source.href = `/research/${encodeURIComponent(factor.source_task_id)}`;
  source.title = `打开来源任务工作台 ${factor.source_task_id}`;
  row.append(cell(source));
  row.append(cell(tag(factor.category)));
  const cluster = element("div", "cluster-cell");
  cluster.append(element("strong", "", factor.cluster_id), element("small", "", `${factor.cluster_role} · ${factor.cluster_size}`));
  row.append(cell(cluster));
  const researchStatus = statusPill(factor.research_state);
  researchStatus.title = historicalMetrics
    ? `评价协议已过期：${factor.metric_protocol || "未知"}；当前任务协议：${factor.current_protocol || "未知"}`
    : (factor.status_reason || "当前任务协议评价");
  row.append(cell(researchStatus));
  const lifecycle = statusPill(factor.lifecycle_state);
  if (factor.holdout_contaminated) lifecycle.title = "已人工查看当前隐藏期；更换研究世代不能清除污染";
  row.append(cell(lifecycle));
  row.append(cell(element("strong", "score-value", historicalMetrics ? "--" : number(factor.scores[libraryState.ranking]))));
  const marginalCell = metricCell(number(factor.marginal_contribution?.incremental_net_ir), historicalMetrics);
  if (!factor.marginal_contribution && !historicalMetrics) {
    marginalCell.title = "未通过单因子门禁或未进入组合增删评估，因此没有边际 IR";
  }
  row.append(marginalCell);
  row.append(metricCell(number(metrics.sharpe_ratio), historicalMetrics));
  row.append(metricCell(percent(metrics.simple_annual_return), historicalMetrics));
  row.append(metricCell(percent(metrics.max_drawdown), historicalMetrics));
  row.append(metricCell(number(metrics.walk_forward_worst_sharpe), historicalMetrics));
  row.append(metricCell(number4(metrics.rank_ic_mean), historicalMetrics));
  row.append(metricCell(number(metrics.annual_turnover), historicalMetrics));
  return row;
}

function renderSelection() {
  const dock = document.getElementById("selectionDock");
  const selected = libraryState.data
    ? libraryState.data.factors.filter(factor => libraryState.selected.has(factor.factor_id))
    : [];
  dock.hidden = selected.length === 0;
  text("selectedCount", selected.length);
  text("selectedNames", selected.slice(0, 4).map(factor => factor.name).join(" · ") + (selected.length > 4 ? ` · +${selected.length - 4}` : ""));
}

async function openFactorDetail(factor) {
  libraryState.detailFactor = factor;
  text("factorDetailCategory", factor.category || "未分类");
  text("factorDetailTitle", factor.name);
  text("factorDetailId", `${factor.factor_id} · ITER ${factor.source_iteration} · ${factor.source_task_name || factor.source_task_id}`);
  text("factorDetailHypothesis", factor.hypothesis || "该因子尚未记录经济假设。");
  document.getElementById("factorDetailTags").replaceChildren(tag(factor.family || "未分类家族"), tag(factor.origin || "AUTO_LLM_RESEARCH"), ...(factor.tags || []).map(value => tag(value)), tag(`簇 ${factor.cluster_id || "--"}`), statusPill(factor.research_state), statusPill(factor.lifecycle_state));
  const latex = expressionToLatex(factor.expression);
  const stats = expressionStats(factor.expression);
  document.getElementById("factorLatex").textContent = `\\[${latex}\\]`;
  document.getElementById("factorLatexSource").textContent = latex;
  text("formulaSummary", `${stats.nodes} 个节点 · 估计回看 ${stats.lookback} 期`);
  renderDefinitionList("expressionAudit", [["输入字段", [...stats.fields].join(" · ") || "--"], ["算子", [...stats.operators].join(" · ") || "--"], ["表达式节点", `${stats.nodes}`], ["估计最大回看", `${stats.lookback} 个交易期`], ["截面处理", stats.crossSectional ? "是" : "否"]]);
  const historicalMetrics = Boolean(factor.protocol_stale);
  const metrics = historicalMetrics ? (factor.historical_metric_summary || {}) : (factor.metric_summary || {}), marginal = factor.marginal_contribution || {};
  renderDefinitionList("factorMetrics", [["评价口径", historicalMetrics ? "历史协议，仅供诊断" : "当前任务协议"], ["单因子门禁", factor.status === "SCREENED_OUT" ? `未通过：${factor.status_reason || "未记录原因"}` : "通过"], ["综合机构分", historicalMetrics ? "--" : number(factor.scores?.overall)], ["夏普比率", number(metrics.sharpe_ratio)], ["简单年化", percent(metrics.simple_annual_return)], ["最大回撤", percent(metrics.max_drawdown)], ["Rank IC / IR", `${number4(metrics.rank_ic_mean)} / ${number(metrics.rank_ic_ir)}`], ["Pearson IC", number4(metrics.pearson_ic_mean)], ["年化换手", number(metrics.annual_turnover)], ["覆盖率", percent(metrics.coverage)], ["最差滚动夏普", number(metrics.walk_forward_worst_sharpe)], ["正收益窗口", percent(metrics.walk_forward_positive_fraction)], ["DSR 概率", percent(metrics.deflated_sharpe_probability)], ["边际净 IR", marginal.incremental_net_ir == null ? "未进入组合评估" : number(marginal.incremental_net_ir)]]);
  document.getElementById("factorRawExpression").textContent = JSON.stringify(factor.expression, null, 2);
  const factorQuery = encodeURIComponent(factor.factor_id);
  document.getElementById("openFactorScreener").href = `/screener?factors=${factorQuery}`;
  document.getElementById("openFactorBacktest").href = `/backtest?factors=${factorQuery}`;
  renderLifecycleLoading();
  const dialog = document.getElementById("factorDetailDialog");
  if (!dialog.open) dialog.showModal();
  if (window.lucide) window.lucide.createIcons();
  await typesetFormula();
  try {
    const lifecycle = await api(`/api/factors/${encodeURIComponent(factor.factor_id)}/lifecycle`);
    if (libraryState.detailFactor?.factor_id === factor.factor_id) renderLifecycle(lifecycle.events || []);
  } catch (error) {
    if (libraryState.detailFactor?.factor_id === factor.factor_id) renderLifecycleError(error.message);
  }
}

function closeFactorDetail() { const dialog = document.getElementById("factorDetailDialog"); if (dialog.open) dialog.close(); }
async function typesetFormula() {
  const node = document.getElementById("factorLatex");
  if (!node || !libraryState.detailFactor || !window.MathJax?.typesetPromise) return;
  try { window.MathJax.typesetClear?.([node]); await window.MathJax.typesetPromise([node]); } catch (_) { /* TeX source remains readable without MathJax. */ }
}
function renderDefinitionList(id, rows) { document.getElementById(id).replaceChildren(...rows.flatMap(([label, value]) => [element("dt", "", label), element("dd", "", value)])); }
function renderLifecycleLoading() { text("lifecycleCount", "正在读取"); document.getElementById("factorLifecycle").replaceChildren(element("li", "lifecycle-placeholder", "正在读取该因子的治理轨迹...")); }
function renderLifecycle(events) {
  text("lifecycleCount", `${events.length} 条记录`);
  const list = document.getElementById("factorLifecycle");
  if (!events.length) { list.replaceChildren(element("li", "lifecycle-placeholder", "暂未记录生命周期事件")); return; }
  list.replaceChildren(...events.map(event => {
    const item = element("li", "lifecycle-event"), meta = element("div", "lifecycle-meta");
    meta.append(element("strong", "", `${event.previous_state ? `${event.previous_state} → ` : ""}${event.state}`), element("span", "", `${event.actor || "SYSTEM"} · ${formatDate(event.created_at)}`));
    item.append(statusPill(event.state), meta, element("p", "", event.reason || "未记录原因"));
    return item;
  }));
}
function renderLifecycleError(message) { text("lifecycleCount", "读取失败"); document.getElementById("factorLifecycle").replaceChildren(element("li", "lifecycle-placeholder error", message)); }

function expressionStats(expression) {
  const stats = { nodes: 0, fields: new Set(), operators: new Set(), crossSectional: false };
  const walk = node => {
    if (!node || typeof node !== "object") return 0;
    stats.nodes += 1;
    const operator = String(node.operator || "unknown"), parameters = node.parameters || {};
    stats.operators.add(operator);
    if (operator === "field" && parameters.name) stats.fields.add(String(parameters.name));
    if (["cs_rank", "cs_zscore", "winsorize_mad", "neutralize"].includes(operator)) stats.crossSectional = true;
    const base = Math.max(0, ...(node.arguments || []).map(walk));
    const window = Number(parameters.window ?? parameters.periods ?? 0);
    return base + (Number.isFinite(window) ? Math.max(0, window) : 0);
  };
  return { ...stats, lookback: walk(expression) };
}

function expressionToLatex(expression) {
  if (!expression || typeof expression !== "object") return "\\text{invalid expression}";
  const args = (expression.arguments || []).map(expressionToLatex), parameters = expression.parameters || {};
  const first = args[0] || "\\varnothing", second = args[1] || "\\varnothing";
  const period = Number(parameters.window ?? parameters.periods ?? 1);
  const fieldNames = { open: "O_t", high: "H_t", low: "L_t", close: "C_t", adj_close: "C_t^{adj}", vol: "V_t", amount: "A_t" };
  switch (expression.operator) {
    case "field": return fieldNames[parameters.name] || `\\mathrm{${latexIdentifier(parameters.name)}}_t`;
    case "constant": return latexNumber(parameters.value);
    case "negate": return `-\\left(${first}\\right)`;
    case "absolute": return `\\left|${first}\\right|`;
    case "add": return `\\left(${first} + ${second}\\right)`;
    case "subtract": return `\\left(${first} - ${second}\\right)`;
    case "multiply": return `\\left(${first} \\cdot ${second}\\right)`;
    case "divide": return `\\frac{${first}}{${second}}`;
    case "delay": return `\\operatorname{Delay}_{${period}}\\left(${first}\\right)`;
    case "delta": return `\\Delta_{${period}}\\left(${first}\\right)`;
    case "returns": return `\\operatorname{Ret}_{${period}}\\left(${first}\\right)`;
    case "rolling_mean": return `\\operatorname{MA}_{${period}}\\left(${first}\\right)`;
    case "rolling_sum": return `\\operatorname{Sum}_{${period}}\\left(${first}\\right)`;
    case "rolling_std": return `\\operatorname{Std}_{${period}}\\left(${first}\\right)`;
    case "rolling_min": return `\\operatorname{Min}_{${period}}\\left(${first}\\right)`;
    case "rolling_max": return `\\operatorname{Max}_{${period}}\\left(${first}\\right)`;
    case "cs_rank": return `\\operatorname{Rank}^{CS}\\left(${first}\\right)`;
    case "cs_zscore": return `\\operatorname{ZScore}^{CS}\\left(${first}\\right)`;
    case "winsorize_mad": return `\\operatorname{WinsorMAD}_{${latexNumber(parameters.threshold ?? 3)}}\\left(${first}\\right)`;
    case "neutralize": return `\\operatorname{Neutralize}\\left(${first}; ${second}\\right)`;
    default: return `\\operatorname{${latexIdentifier(expression.operator)}}\\left(${args.join(", ")}\\right)`;
  }
}
function latexIdentifier(value) { return String(value ?? "unknown").replace(/[^a-zA-Z0-9_]/g, ""); }
function latexNumber(value) { const parsed = Number(value); return Number.isFinite(parsed) ? `${parsed}` : "0"; }
async function copyDetailText(label, value) { if (!value) return; try { await navigator.clipboard.writeText(value); toast(`${label} 已复制`); } catch (_) { toast(`无法复制 ${label}`, true); } }
function formatDate(value) { const date = new Date(value); return Number.isNaN(date.valueOf()) ? "--" : date.toLocaleString("zh-CN", { hour12: false }); }

async function api(path) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function cell(content) {
  const node = document.createElement("td");
  node.append(content);
  return node;
}

function cellText(content) {
  const node = document.createElement("td");
  node.textContent = content;
  return node;
}

function metricCell(content, historical = false) {
  const node = cellText(content);
  if (historical) {
    node.classList.add("historical-metric");
    node.title = "历史协议指标，仅供诊断，不能参与当前排行榜";
  }
  return node;
}

function statusPill(status) {
  const labels = { ACTIVE: "冠军", QUALIFIED: "合格", OBSERVE: "观察", STALE_PROTOCOL: "待重评", HOLDOUT_CONTAMINATED: "盲测污染", RESEARCH: "研究", SHADOW: "影子", PAPER: "仿真", PRODUCTION: "生产", WATCH: "观察", SUSPENDED: "暂停", RETIRED: "退役", REJECTED: "否决" };
  return element("span", `research-status ${status.toLowerCase()}`, labels[status] || status);
}

function tag(value) { return element("span", "tag", value); }
function option(value, label) { const node = document.createElement("option"); node.value = value; node.textContent = label; return node; }
function marketLabel(value) { return ({ CN_A: "A 股", HK: "港股", US: "美股" })[value] || value || "未知市场"; }
function element(tagName, className = "", content = "") { const node = document.createElement(tagName); if (className) node.className = className; if (content !== "") node.textContent = content; return node; }
function text(id, value) { document.getElementById(id).textContent = value; }
function number(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed.toFixed(2) : "--"; }
function number4(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed.toFixed(4) : "--"; }
function percent(value) { const parsed = Number(value); return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(2)}%` : "--"; }
function formatClock(value) { return value instanceof Date && !Number.isNaN(value.valueOf()) ? value.toLocaleTimeString("zh-CN", { hour12: false }) : "--:--:--"; }

let toastTimer;
function toast(message, isError = false) {
  const node = document.getElementById("toast");
  node.textContent = message;
  node.style.background = isError ? "#a93430" : "#202a3b";
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 2400);
}
