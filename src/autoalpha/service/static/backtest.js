const backtestState = {
  library: null,
  selected: new Map(),
  result: null,
  query: "",
  templates: [],
  presets: [],
  history: [],
  compare: new Set(),
  favoriteOnly: false,
  tradeOffset: 0,
  tradePageSize: 100,
};

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  bindBacktestControls();
  try {
    await Promise.all([loadTemplates(), loadPresets(), loadFactors(), loadHistory()]);
  } catch (error) {
    toast(error.message, true);
  }
  new ResizeObserver(drawEquityChart).observe(document.querySelector(".equity-chart-frame"));
});

function bindBacktestControls() {
  document.getElementById("backtestForm").addEventListener("submit", runBacktest);
  document.getElementById("pickerSearch").addEventListener("input", event => {
    backtestState.query = event.target.value.trim().toLowerCase();
    renderPicker();
  });
  document.getElementById("equalWeights").onclick = () => {
    backtestState.selected.forEach((_, factorId) => backtestState.selected.set(factorId, 1));
    renderSelectedEditor();
  };
  document.getElementById("historyRefresh").onclick = loadHistory;
  document.getElementById("productTemplate").onchange = applyTemplateDefaults;
  document.getElementById("backtestEngine").onchange = applyEngineDefaults;
  document.getElementById("executionDataMode").onchange = applyEngineDefaults;
  document.getElementById("backtestPreset").onchange = applyBacktestPreset;
  document.getElementById("favoriteResult").onclick = toggleCurrentFavorite;
  document.getElementById("saveMetadata").onclick = saveCurrentMetadata;
  document.getElementById("favoritesOnly").onchange = event => {
    backtestState.favoriteOnly = event.target.checked;
    loadHistory();
  };
  document.getElementById("compareSelected").onclick = renderComparison;
  document.getElementById("applyTradeFilters").onclick = () => loadTradeStatement(true);
  document.getElementById("tradePrevious").onclick = () => {
    backtestState.tradeOffset = Math.max(0, backtestState.tradeOffset - backtestState.tradePageSize);
    loadTradeStatement(false);
  };
  document.getElementById("tradeNext").onclick = () => {
    backtestState.tradeOffset += backtestState.tradePageSize;
    loadTradeStatement(false);
  };
  [
    "backtestEngine", "executionDataMode", "vectorCostModel", "grossExposure", "holdingPeriod",
    "rebalanceSchedule", "productTemplate", "selectionFraction", "maximumPositions",
    "maximumVolumeParticipation", "lotSize", "openingLimitThreshold",
    "costStressMultiplier", "commissionBps", "stampDutyBps", "transferFeeBps",
    "minimumCommission", "slippageBps", "historicalFeeSchedule",
  ].forEach(id => document.getElementById(id).addEventListener("input", markPresetCustom));
}

async function loadPresets() {
  const payload = await api("/api/backtest-presets");
  backtestState.presets = payload.presets || [];
  const select = document.getElementById("backtestPreset");
  select.replaceChildren(
    option("CUSTOM", "自定义"),
    ...backtestState.presets.map(preset => option(preset.preset_id, preset.name)),
  );
  select.value = payload.default || "CUSTOM";
  if (select.value !== "CUSTOM") applyBacktestPreset();
}

function applyBacktestPreset() {
  const presetId = document.getElementById("backtestPreset").value;
  const preset = backtestState.presets.find(item => item.preset_id === presetId);
  if (!preset) {
    text("presetStatus", "自定义口径 · 修改生产预设参数后会自动回到此模式");
    return;
  }
  const ids = {
    backtest_engine: "backtestEngine",
    execution_data_mode: "executionDataMode",
    vector_cost_model: "vectorCostModel",
    product_template: "productTemplate",
    rebalance_schedule: "rebalanceSchedule",
    gross_exposure: "grossExposure",
    holding_period_days: "holdingPeriod",
    selection_fraction: "selectionFraction",
    maximum_positions: "maximumPositions",
    lot_size: "lotSize",
    maximum_volume_participation: "maximumVolumeParticipation",
    opening_limit_threshold: "openingLimitThreshold",
    commission_bps_each_side: "commissionBps",
    stamp_duty_bps_sell: "stampDutyBps",
    transfer_fee_bps_each_side: "transferFeeBps",
    minimum_commission_cny: "minimumCommission",
    slippage_bps_each_side: "slippageBps",
    use_historical_fee_schedule: "historicalFeeSchedule",
    cost_stress_multiplier: "costStressMultiplier",
  };
  Object.entries(preset.settings).forEach(([key, value]) => {
    const control = document.getElementById(ids[key]);
    if (!control) return;
    if (control.type === "checkbox") control.checked = Boolean(value);
    else control.value = value;
  });
  applyEngineDefaults(false);
  const template = backtestState.templates.find(item => item.template_id === preset.settings.product_template);
  text("productLimitation", template?.limitation || "");
  const basis = backtestState.library?.data?.execution_basis;
  const proxy = preset.settings.execution_data_mode === "NON_PIT_PROXY";
  const ready = proxy ? basis?.capital_ledger_proxy_ready : basis?.capital_ledger_ready;
  const blockers = proxy ? basis?.proxy_blockers : basis?.blockers;
  const readiness = basis && !ready
    ? ` 当前数据存在 ${(blockers || []).length} 项资金账本阻断，运行时显示完整原因。`
    : " 数据不满足账本要求时拒绝运行。";
  text("presetStatus", `${preset.description}${readiness}`);
  toast(`已应用：${preset.name}`);
}

function markPresetCustom() {
  const select = document.getElementById("backtestPreset");
  if (select.value === "CUSTOM") return;
  select.value = "CUSTOM";
  text("presetStatus", "参数已修改，当前为自定义口径");
}

async function loadTemplates() {
  const payload = await api("/api/product-templates");
  backtestState.templates = payload.templates;
  const select = document.getElementById("productTemplate");
  select.replaceChildren(...payload.templates.map(template => option(template.template_id, template.name)));
  select.value = "MARKET_NEUTRAL_RESEARCH";
  applyTemplateDefaults();
}

function applyTemplateDefaults() {
  const selected = backtestState.templates.find(template => template.template_id === document.getElementById("productTemplate").value);
  if (!selected) return;
  document.getElementById("grossExposure").value = selected.default_gross_exposure;
  document.getElementById("maximumPositions").value = selected.maximum_positions;
  text("productLimitation", selected.limitation);
  applyEngineDefaults(false);
}

function applyEngineDefaults(changeTemplate = true) {
  const ledger = document.getElementById("backtestEngine").value === "EVENT_LEDGER";
  const proxy = document.getElementById("executionDataMode").value === "NON_PIT_PROXY";
  document.getElementById("vectorCostModel").disabled = ledger;
  document.getElementById("executionDataMode").disabled = !ledger;
  if (ledger && changeTemplate) {
    const selected = backtestState.templates.find(template => template.template_id === document.getElementById("productTemplate").value);
    if (selected?.portfolio_mode !== "long_only") {
      document.getElementById("productTemplate").value = "LONG_ONLY_CAPITAL";
      applyTemplateDefaults();
      return;
    }
  }
  text("engineLimitation", ledger
    ? (proxy
      ? "原始成交价、现金与整手约束 · 非 PIT 代理，生成研究级交割单"
      : "逐笔现金、整手、涨跌停与成交量约束 · 生成交割单")
    : "矩阵计算 · 次日开盘执行 · 可复现研究口径");
}

async function loadFactors() {
  backtestState.library = await api("/api/factors");
  const data = backtestState.library.data;
  const startInput = document.getElementById("startDate");
  const endInput = document.getElementById("endDate");
  startInput.min = data.first_trade_date;
  startInput.max = data.last_trade_date;
  endInput.min = data.first_trade_date;
  endInput.max = data.last_trade_date;
  startInput.value = data.first_trade_date < "2020-01-02" ? "2020-01-02" : data.first_trade_date;
  endInput.value = data.last_trade_date;
  text("dataRange", `${data.first_trade_date} — ${data.last_trade_date}`);
  text("backtestSummary", `${backtestState.library.summary.factor_count} 个因子 · ${data.fingerprint.slice(0, 12)} · MANUAL / NON-GOVERNANCE`);
  const queryFactors = new URLSearchParams(window.location.search).get("factors");
  if (queryFactors) {
    const available = new Set(backtestState.library.factors.map(factor => factor.factor_id));
    queryFactors.split(",").filter(factorId => available.has(factorId)).slice(0, 12)
      .forEach(factorId => backtestState.selected.set(factorId, 1));
  }
  renderPicker();
  renderSelectedEditor();
}

function renderPicker() {
  if (!backtestState.library) return;
  const factors = backtestState.library.factors.filter(factor => {
    if (!backtestState.query) return true;
    return [factor.name, factor.factor_id, factor.family, factor.category]
      .join(" ").toLowerCase().includes(backtestState.query);
  });
  const picker = document.getElementById("factorPicker");
  picker.replaceChildren(...factors.map(factor => {
    const label = element("label", "picker-row");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = backtestState.selected.has(factor.factor_id);
    checkbox.onchange = () => {
      if (checkbox.checked) {
        if (backtestState.selected.size >= 12) {
          checkbox.checked = false;
          toast("单次最多选择 12 个因子", true);
          return;
        }
        backtestState.selected.set(factor.factor_id, 1);
      } else {
        backtestState.selected.delete(factor.factor_id);
      }
      renderSelectedEditor();
      text("pickerCount", `${backtestState.selected.size} 已选`);
    };
    const copy = element("span", "picker-copy");
    const score = factor.protocol_stale ? "待重评" : factor.scores.overall.toFixed(1);
    copy.append(element("strong", "", factor.name), element("small", "", `${factor.category} · #${factor.rank} · ${score}`));
    label.append(checkbox, copy, statusPill(factor.research_state));
    return label;
  }));
  text("pickerCount", `${backtestState.selected.size} 已选`);
}

function renderSelectedEditor() {
  const editor = document.getElementById("selectedFactorEditor");
  if (!backtestState.library || !backtestState.selected.size) {
    editor.replaceChildren(element("p", "table-empty", "尚未选择因子"));
    text("pickerCount", "0 已选");
    return;
  }
  const byId = new Map(backtestState.library.factors.map(factor => [factor.factor_id, factor]));
  editor.replaceChildren(...[...backtestState.selected].map(([factorId, weight]) => {
    const factor = byId.get(factorId);
    const row = element("div", "weight-row");
    const copy = element("div", "");
    copy.append(element("strong", "", factor.name), element("small", "", factorId));
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0.01";
    input.max = "100";
    input.step = "0.01";
    input.value = weight;
    input.setAttribute("aria-label", `${factor.name} 权重`);
    input.onchange = () => backtestState.selected.set(factorId, Number(input.value));
    const remove = element("button", "icon-button small-icon");
    remove.type = "button";
    remove.title = "移除因子";
    remove.innerHTML = '<i data-lucide="x"></i>';
    remove.onclick = () => {
      backtestState.selected.delete(factorId);
      renderPicker();
      renderSelectedEditor();
    };
    row.append(copy, input, remove);
    return row;
  }));
  text("pickerCount", `${backtestState.selected.size} 已选`);
  if (window.lucide) window.lucide.createIcons();
}

async function runBacktest(event) {
  event.preventDefault();
  if (!backtestState.selected.size) {
    toast("请至少选择一个因子", true);
    return;
  }
  const button = document.getElementById("runBacktest");
  button.disabled = true;
  button.textContent = "正在回测";
  text("resultStatus", "RUNNING");
  document.getElementById("resultStatus").className = "state-pill small RUNNING";
  const entries = [...backtestState.selected];
  try {
    const result = await api("/api/manual-backtests", {
      method: "POST",
      body: JSON.stringify({
        factor_ids: entries.map(([factorId]) => factorId),
        weights: entries.map(([, weight]) => Number(weight)),
        start_date: document.getElementById("startDate").value,
        end_date: document.getElementById("endDate").value,
        initial_cash_cny: Number(document.getElementById("initialCash").value),
        gross_exposure: Number(document.getElementById("grossExposure").value),
        holding_period_days: Number(document.getElementById("holdingPeriod").value),
        backtest_preset: document.getElementById("backtestPreset").value,
        backtest_engine: document.getElementById("backtestEngine").value,
        execution_data_mode: document.getElementById("executionDataMode").value,
        rebalance_schedule: document.getElementById("rebalanceSchedule").value,
        vector_cost_model: document.getElementById("vectorCostModel").value,
        product_template: document.getElementById("productTemplate").value,
        selection_fraction: Number(document.getElementById("selectionFraction").value),
        maximum_positions: Number(document.getElementById("maximumPositions").value),
        lot_size: Number(document.getElementById("lotSize").value),
        maximum_volume_participation: Number(document.getElementById("maximumVolumeParticipation").value),
        opening_limit_threshold: Number(document.getElementById("openingLimitThreshold").value),
        commission_bps_each_side: Number(document.getElementById("commissionBps").value),
        stamp_duty_bps_sell: Number(document.getElementById("stampDutyBps").value),
        transfer_fee_bps_each_side: Number(document.getElementById("transferFeeBps").value),
        minimum_commission_cny: Number(document.getElementById("minimumCommission").value),
        slippage_bps_each_side: Number(document.getElementById("slippageBps").value),
        use_historical_fee_schedule: document.getElementById("historicalFeeSchedule").checked,
        cost_stress_multiplier: Number(document.getElementById("costStressMultiplier").value),
      }),
    });
    renderResult(result);
    await loadHistory();
    toast(`回测 #${result.id} 已完成`);
  } catch (error) {
    text("resultStatus", "FAILED");
    document.getElementById("resultStatus").className = "state-pill small STOPPED";
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.innerHTML = '<i data-lucide="play"></i>运行手动回测';
    if (window.lucide) window.lucide.createIcons();
  }
}

function renderResult(result) {
  backtestState.result = result;
  const metrics = result.metrics;
  text("resultStatus", `RUN #${result.id}`);
  document.getElementById("resultStatus").className = "state-pill small RUNNING";
  text("resultPeriod", `${metrics.backtest_start} — ${metrics.backtest_end} · ${result.factors.length} 因子 · ${engineLabel(metrics.backtest_engine)} · ${modeLabel(metrics.portfolio_mode)}`);
  text("metricAnnual", percent(metrics.simple_annual_return));
  text("metricSharpe", number(metrics.sharpe_ratio));
  text("metricDrawdown", percent(metrics.max_drawdown));
  text("metricTotalReturn", percent(metrics.total_return));
  text("metricProfit", currency(metrics.net_profit_cny));
  text("metricSortino", number(metrics.sortino_ratio));
  text("metricCalmar", number(metrics.calmar_ratio));
  text("metricTurnover", number(metrics.annual_turnover));
  text("metricRankIc", number4(metrics.rank_ic_mean));
  text("metricCorrelation", number4(metrics.maximum_factor_correlation));
  text("metricCoverage", percent(metrics.coverage));
  text("metricDsr", percent(metrics.deflated_sharpe_probability));
  text("metricBenchmark", percent(metrics.benchmark_simple_annual_return));
  text("metricActive", percent(metrics.active_simple_annual_return));
  text("metricIr", number(metrics.information_ratio));
  text("metricTe", percent(metrics.tracking_error));
  text("finalEquity", currency(metrics.final_equity_cny));
  text("transactionCost", currency(metrics.transaction_cost_cny));
  text("observations", metrics.observations);
  text("equitySubtitle", `${engineLabel(metrics.backtest_engine)} · ${currency(metrics.final_equity_cny)} · ${scheduleLabel(metrics.rebalance_schedule)} · ${(metrics.gross_exposure * 100).toFixed(0)}% 仓位`);
  text("contaminationNotice", result.product?.limitation || "覆盖隐藏期时会登记污染，并阻断同世代盲测");
  document.getElementById("equityEmpty").hidden = true;
  renderAnnualReturns(result.annual_returns);
  renderCorrelations(result.factor_correlations, result.factors);
  renderExecutionAssumptions(result);
  renderMetadata(result.metadata || {});
  if (result.trade_statement?.available) loadTradeStatement(true);
  else document.getElementById("tradeStatementPanel").hidden = true;
  drawEquityChart();
}

function renderExecutionAssumptions(result) {
  const config = result.configuration || {};
  const metrics = result.metrics || {};
  const recorded = result.execution_assumptions || {};
  const holding = Number(recorded.holding_period_trading_sessions || config.holding_period_days || metrics.holding_period_days || 1);
  const vector = (recorded.engine || metrics.backtest_engine || config.backtest_engine || "VECTOR") === "VECTOR";
  const gross = Number(recorded.target_gross_exposure ?? config.gross_exposure ?? metrics.gross_exposure);
  const fraction = Number(recorded.selection_fraction_each_side ?? config.selection_fraction ?? metrics.selection_fraction);
  const maximum = Number(recorded.maximum_positions_each_side ?? config.maximum_positions ?? metrics.maximum_positions_per_side);
  const priceBasis = recorded.price_adjustment || "forward_adjusted";
  const costModel = recorded.cost_model || config.vector_cost_model || metrics.vector_cost_model;
  const schedule = recorded.rebalance_schedule || config.rebalance_schedule || metrics.rebalance_schedule || "DAILY_ROLLING";
  const dailyRolling = schedule === "DAILY_ROLLING";
  const executionDataMode = recorded.execution_data_mode || config.execution_data_mode || metrics.execution_data_mode || (vector ? "RESEARCH_VECTOR" : "STRICT_PIT");
  const rows = [
    ["信号形成", "T 日收盘后"],
    ["假设成交", "T+1 交易日 09:30 开盘"],
    ["收益结算", vector ? "T+1 开盘至 T+2 开盘" : "开盘成交后逐日收盘盯市"],
    ["再平衡频率", scheduleLabel(schedule)],
    [dailyRolling ? "单袖套持有" : "目标持有", dailyRolling ? `${holding} 个交易日` : "至下个计划调仓日"],
    ["订单失败", vector ? "无逐笔订单状态" : "后续交易日逐日重试"],
    ["固定星期", schedule === "WEEKLY_FIRST_SESSION" ? "否 · 节假日顺延至首个交易日" : "否"],
    ["选股范围", `${metrics.portfolio_mode === "long_only" ? "多头" : "多空双侧"} ${percent(fraction)} · 每侧最多 ${maximum} 只`],
    ["目标总仓位", percent(gross)],
    ["价格口径", vector ? priceBasisLabel(priceBasis) : priceBasisLabel(recorded.execution_price_adjustment || "unadjusted")],
    ["成交状态", executionDataMode === "NON_PIT_PROXY" ? "非 PIT 代理（非生产）" : (vector ? "研究向量口径" : "严格 PIT")],
    ["持仓模型", vector ? "等权目标的滚动袖套均值" : "独立现金袖套与整数股"],
    ["费用模型", vector ? costModelLabel(costModel) : "逐笔费用与最低佣金"],
    ["开盘滑点", `${Number(recorded.slippage_bps_each_side ?? config.slippage_bps_each_side ?? 0).toFixed(1)} bps / 边`],
    ["历史费率", recorded.historical_fee_schedule ?? config.use_historical_fee_schedule ? "按成交日切换" : "固定当前设置"],
  ];
  const box = document.getElementById("executionAssumptions");
  box.replaceChildren(...rows.map(([label, value]) => {
    const cell = element("div", "assumption-cell");
    cell.append(element("span", "", label), element("strong", "", value));
    return cell;
  }));
  text("assumptionProtocol", `${scheduleLabel(schedule)} · ${recorded.preset && recorded.preset !== "CUSTOM" ? "生产预检预设" : "自定义口径"}`);
  text("assumptionEngine", vector ? "VECTOR" : "LEDGER");
  const missing = recorded.constraints_not_modeled || (vector ? [
    "整手与现金约束",
    "集合竞价滑点",
    "停牌持仓冻结与退市处置",
    "涨跌停成交限制",
    "融券可得性、费率与召回",
    "市场冲击与订单簿容量",
    "逐笔最低佣金",
  ] : ["订单簿排队", "显式费用之外的市场冲击"]);
  const translated = missing.map(assumptionConstraintLabel);
  const basisWarning = executionDataMode === "NON_PIT_PROXY"
    ? "当前为非 PIT 成交代理：历史 ST、停牌、上市退市与精确涨跌停状态未被验证，结果不能进入生产。"
    : (vector && ["forward_adjusted", "event_adjusted_pit"].includes(priceBasis)
      ? "当前使用事件调整研究价，只适合研究收益，不等同于可成交现金价格。"
      : "");
  text("assumptionWarning", `${basisWarning}${basisWarning ? " " : ""}未建模：${translated.join("、")}。`);
  if (window.lucide) window.lucide.createIcons();
}

function priceBasisLabel(value) {
  if (value === "forward_adjusted") return "前复权日开盘价";
  if (value === "event_adjusted_pit") return "复权因子事件调整研究价";
  if (["raw", "unadjusted"].includes(value)) return "未复权日开盘价";
  return String(value || "未知");
}

function costModelLabel(value) {
  return value === "legacy_half_turnover" ? "历史半换手费用" : "买卖侧精确线性费用";
}

function assumptionConstraintLabel(value) {
  const labels = {
    "integer lots and cash": "整手与现金约束",
    "opening auction slippage": "集合竞价滑点",
    "suspension carry and forced delisting treatment": "停牌持仓冻结与退市处置",
    "limit-up/limit-down fills": "涨跌停成交限制",
    "short borrow availability, borrow fees, and recalls": "融券可得性、费率与召回",
    "market impact and order-book capacity": "市场冲击与订单簿容量",
    "minimum commission per order": "逐笔最低佣金",
    "intraday order-book queue": "订单簿排队",
    "market impact beyond explicit fees": "显式费用之外的市场冲击",
    "market impact beyond fixed opening slippage": "固定开盘滑点之外的市场冲击",
  };
  return labels[value] || value;
}

async function loadTradeStatement(resetOffset) {
  if (!backtestState.result) return;
  if (resetOffset) backtestState.tradeOffset = 0;
  const parameters = new URLSearchParams({
    limit: String(backtestState.tradePageSize),
    offset: String(backtestState.tradeOffset),
    side: document.getElementById("tradeSideFilter").value,
  });
  const symbol = document.getElementById("tradeSymbolFilter").value.trim();
  const startDate = document.getElementById("tradeStartDate").value;
  const endDate = document.getElementById("tradeEndDate").value;
  if (symbol) parameters.set("symbol", symbol);
  if (startDate) parameters.set("start_date", startDate);
  if (endDate) parameters.set("end_date", endDate);
  try {
    const payload = await api(`/api/manual-backtests/${backtestState.result.id}/trades?${parameters}`);
    const panel = document.getElementById("tradeStatementPanel");
    panel.hidden = !payload.available;
    if (!payload.available) return;
    document.getElementById("downloadTradeStatement").href = `/api/manual-backtests/${backtestState.result.id}/trades.csv`;
    const statement = payload.statement;
    text("tradeStatementSummary", `${statement.row_count} 笔 · 买入 ${statement.buy_count} · 卖出 ${statement.sell_count} · 费用 ${currency(statement.total_fees_cny)} · ${String(statement.sha256 || "").slice(0, 12)}`);
    const body = document.getElementById("tradeStatementBody");
    body.replaceChildren(...payload.rows.map(trade => {
      const row = document.createElement("tr");
      const security = element("div", "trade-security");
      security.append(element("strong", "", trade.symbol), element("small", "", trade.security_name || ""));
      [
        trade.trade_date,
        trade.signal_date,
        security,
        element("span", `trade-side ${trade.side.toLowerCase()}`, trade.side === "BUY" ? "买入" : "卖出"),
        String(trade.quantity),
        money2(trade.price_cny),
        money2(trade.notional_cny),
        money2(trade.commission_cny),
        money2(trade.transfer_fee_cny),
        money2(trade.stamp_duty_cny),
        money2(trade.total_fees_cny),
        money2(trade.net_cash_flow_cny),
        String(trade.sleeve),
      ].forEach(value => row.append(value instanceof Node ? cellNode(value) : element("td", "", value)));
      return row;
    }));
    if (!payload.rows.length) {
      const emptyRow = document.createElement("tr");
      const emptyCell = element("td", "table-empty", "没有符合筛选条件的成交");
      emptyCell.colSpan = 13;
      emptyRow.append(emptyCell);
      body.append(emptyRow);
    }
    const start = payload.total ? payload.offset + 1 : 0;
    const end = Math.min(payload.offset + payload.rows.length, payload.total);
    text("tradePageStatus", `${start}–${end} / ${payload.total}`);
    document.getElementById("tradePrevious").disabled = payload.offset === 0;
    document.getElementById("tradeNext").disabled = !payload.has_more;
    if (window.lucide) window.lucide.createIcons();
  } catch (error) {
    toast(error.message, true);
  }
}

function cellNode(content) {
  const cell = document.createElement("td");
  cell.append(content);
  return cell;
}

function renderMetadata(metadata) {
  const favorite = Boolean(metadata.favorite);
  const button = document.getElementById("favoriteResult");
  button.disabled = false;
  button.classList.toggle("active", favorite);
  button.title = favorite ? "取消收藏" : "收藏结果";
  document.getElementById("resultMetadata").hidden = !favorite;
  document.getElementById("favoriteTitle").value = metadata.title || "";
  document.getElementById("favoriteTags").value = (metadata.tags || []).join(", ");
  document.getElementById("favoriteNotes").value = metadata.notes || "";
  if (window.lucide) window.lucide.createIcons();
}

async function toggleCurrentFavorite() {
  if (!backtestState.result) return;
  const current = backtestState.result.metadata || {};
  await updateMetadata({
    favorite: !current.favorite,
    title: current.title || null,
    notes: current.notes || "",
    tags: current.tags || [],
  });
}

async function saveCurrentMetadata() {
  if (!backtestState.result) return;
  const tags = document.getElementById("favoriteTags").value
    .split(/[,，]/).map(tag => tag.trim()).filter(Boolean).slice(0, 8);
  await updateMetadata({
    favorite: true,
    title: document.getElementById("favoriteTitle").value.trim() || null,
    notes: document.getElementById("favoriteNotes").value.trim(),
    tags,
  });
}

async function updateMetadata(payload) {
  try {
    const record = await api(`/api/manual-backtests/${backtestState.result.id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    backtestState.result.metadata = {
      favorite: record.favorite,
      title: record.title,
      notes: record.notes,
      tags: record.tags,
      updated_at: record.updated_at,
    };
    renderMetadata(backtestState.result.metadata);
    await loadHistory();
    toast(record.favorite ? "收藏信息已保存" : "已取消收藏");
  } catch (error) {
    toast(error.message, true);
  }
}

function renderAnnualReturns(returns) {
  const box = document.getElementById("annualReturns");
  const entries = Object.entries(returns);
  box.replaceChildren(...entries.map(([year, value]) => {
    const row = element("div", `year-return ${value >= 0 ? "positive" : "negative"}`);
    row.append(element("span", "", year), element("strong", "", percent(value)));
    return row;
  }));
  if (!entries.length) box.append(element("p", "table-empty", "暂无年度数据"));
}

function renderCorrelations(correlations, factors) {
  const box = document.getElementById("correlationList");
  const names = new Map(factors.map(factor => [factor.factor_id, factor.name]));
  const entries = Object.entries(correlations);
  box.replaceChildren(...entries.map(([pair, value]) => {
    const [left, right] = pair.split(":");
    const row = element("div", "correlation-row");
    row.append(element("span", "", `${names.get(left) || left} / ${names.get(right) || right}`), element("strong", "", number4(value)));
    return row;
  }));
  if (!entries.length) box.append(element("p", "table-empty", "单因子组合无相关性矩阵"));
}

async function loadHistory() {
  const payload = await api(`/api/manual-backtests?limit=50&favorite_only=${backtestState.favoriteOnly}`);
  backtestState.history = payload.backtests;
  const availableIds = new Set(payload.backtests.map(record => record.id));
  [...backtestState.compare].filter(id => !availableIds.has(id))
    .forEach(id => backtestState.compare.delete(id));
  const box = document.getElementById("manualHistory");
  box.replaceChildren(...payload.backtests.map(record => {
    const row = element("div", "history-row");
    const request = record.request;
    const metrics = record.metrics || {};
    const open = element("button", "history-open");
    open.type = "button";
    const copy = element("span", "history-copy");
    const title = record.title || `回测 #${record.id}`;
    copy.append(
      element("strong", "", `${record.favorite ? "★ " : ""}${title} · ${request.factor_ids.length} 因子`),
      element("small", "", record.status === "COMPLETED" ? `${engineLabel(request.backtest_engine)} · ${percent(metrics.simple_annual_return)} 年化 · ${number(metrics.sharpe_ratio)} 夏普 · ${String(record.result_hash || "").slice(0, 12)}` : (record.error || record.status)),
    );
    open.append(copy);
    if (record.status === "COMPLETED") {
      open.onclick = async () => {
        try { renderResult(await api(`/api/manual-backtests/${record.id}`)); }
        catch (error) { toast(error.message, true); }
      };
    }
    const controls = element("div", "history-controls");
    const compare = document.createElement("input");
    compare.type = "checkbox";
    compare.title = "加入比较";
    compare.checked = backtestState.compare.has(record.id);
    compare.disabled = record.status !== "COMPLETED";
    compare.onchange = () => {
      if (compare.checked && backtestState.compare.size >= 4) {
        compare.checked = false;
        toast("最多比较 4 个结果", true);
        return;
      }
      if (compare.checked) backtestState.compare.add(record.id);
      else backtestState.compare.delete(record.id);
      syncCompareButton();
    };
    const reuse = iconButton("rotate-ccw", "复用参数", () => reuseRequest(record));
    const favorite = iconButton("star", record.favorite ? "取消收藏" : "收藏", async () => {
      try {
        await api(`/api/manual-backtests/${record.id}`, {
          method: "PATCH",
          body: JSON.stringify({
            favorite: !record.favorite,
            title: record.title,
            notes: record.notes || "",
            tags: record.tags || [],
          }),
        });
        await loadHistory();
      } catch (error) { toast(error.message, true); }
    });
    favorite.classList.toggle("active", record.favorite);
    controls.append(compare, reuse, favorite, statusPill(record.status));
    row.append(open, controls);
    return row;
  }));
  if (!payload.backtests.length) box.append(element("p", "table-empty", "尚无手动回测记录"));
  syncCompareButton();
  if (window.lucide) window.lucide.createIcons();
}

function reuseRequest(record) {
  if (!backtestState.library) return;
  const request = record.request;
  const available = new Set(backtestState.library.factors.map(factor => factor.factor_id));
  backtestState.selected.clear();
  request.factor_ids.forEach((factorId, index) => {
    if (available.has(factorId)) backtestState.selected.set(factorId, request.weights?.[index] || 1);
  });
  const values = {
    startDate: request.start_date,
    endDate: request.end_date,
    initialCash: request.initial_cash_cny,
    grossExposure: request.gross_exposure,
    holdingPeriod: request.holding_period_days,
    backtestPreset: request.backtest_preset || "CUSTOM",
    backtestEngine: request.backtest_engine || "VECTOR",
    executionDataMode: request.execution_data_mode || "STRICT_PIT",
    rebalanceSchedule: request.rebalance_schedule || "DAILY_ROLLING",
    vectorCostModel: request.vector_cost_model || "legacy_half_turnover",
    productTemplate: request.product_template || "MARKET_NEUTRAL_RESEARCH",
    selectionFraction: request.selection_fraction ?? 0.10,
    maximumPositions: request.maximum_positions ?? 30,
    lotSize: request.lot_size ?? 100,
    maximumVolumeParticipation: request.maximum_volume_participation ?? 0.05,
    openingLimitThreshold: request.opening_limit_threshold ?? 0.095,
    commissionBps: request.commission_bps_each_side ?? 1.5,
    stampDutyBps: request.stamp_duty_bps_sell ?? 5,
    transferFeeBps: request.transfer_fee_bps_each_side ?? 0.1,
    minimumCommission: request.minimum_commission_cny ?? 5,
    slippageBps: request.slippage_bps_each_side ?? 0,
    costStressMultiplier: request.cost_stress_multiplier ?? 2,
  };
  Object.entries(values).forEach(([id, value]) => { document.getElementById(id).value = value; });
  document.getElementById("historicalFeeSchedule").checked = request.use_historical_fee_schedule ?? false;
  applyEngineDefaults(false);
  const template = backtestState.templates.find(item => item.template_id === values.productTemplate);
  text("productLimitation", template?.limitation || "");
  const preset = backtestState.presets.find(item => item.preset_id === values.backtestPreset);
  const basis = backtestState.library?.data?.execution_basis;
  const proxy = values.executionDataMode === "NON_PIT_PROXY";
  const ready = proxy ? basis?.capital_ledger_proxy_ready : basis?.capital_ledger_ready;
  const blockers = proxy ? basis?.proxy_blockers : basis?.blockers;
  const readiness = basis && !ready
    ? ` 当前数据存在 ${(blockers || []).length} 项资金账本阻断，运行时显示完整原因。`
    : " 数据不满足账本要求时拒绝运行。";
  text("presetStatus", preset ? `${preset.description}${readiness}` : "自定义口径");
  renderPicker();
  renderSelectedEditor();
  document.querySelector(".backtest-config").scrollIntoView({ behavior: "smooth", block: "start" });
  toast(`已载入回测 #${record.id} 的完整参数`);
}

function syncCompareButton() {
  const button = document.getElementById("compareSelected");
  button.disabled = backtestState.compare.size < 2;
  button.innerHTML = `<i data-lucide="columns-3"></i>比较 ${backtestState.compare.size || ""}`;
  if (window.lucide) window.lucide.createIcons();
}

function renderComparison() {
  const records = [...backtestState.compare]
    .map(id => backtestState.history.find(record => record.id === id)).filter(Boolean);
  if (records.length < 2) return;
  const panel = document.getElementById("comparisonPanel");
  panel.hidden = false;
  const head = document.getElementById("comparisonHead");
  const headRow = document.createElement("tr");
  headRow.append(element("th", "", "指标"));
  records.forEach(record => headRow.append(element("th", "", record.title || `#${record.id}`)));
  head.replaceChildren(headRow);
  const rows = [
    ["回测引擎", "backtest_engine", engineLabel],
    ["简单年化", "simple_annual_return", percent],
    ["夏普", "sharpe_ratio", number],
    ["最大回撤", "max_drawdown", percent],
    ["总收益", "total_return", percent],
    ["信息比率", "information_ratio", number],
    ["跟踪误差", "tracking_error", percent],
    ["年化换手", "annual_turnover", number],
    ["交易成本", "transaction_cost_cny", currency],
    ["Rank IC", "rank_ic_mean", number4],
  ];
  const body = document.getElementById("comparisonBody");
  body.replaceChildren(...rows.map(([label, key, formatter]) => {
    const row = document.createElement("tr");
    row.append(element("th", "", label));
    records.forEach(record => row.append(element("td", "", formatter(record.metrics?.[key]))));
    return row;
  }));
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function iconButton(icon, title, handler) {
  const button = element("button", "icon-button small-icon");
  button.type = "button";
  button.title = title;
  button.innerHTML = `<i data-lucide="${icon}"></i>`;
  button.onclick = handler;
  return button;
}

function drawEquityChart() {
  const canvas = document.getElementById("equityChart");
  const frame = canvas.parentElement;
  const curve = backtestState.result?.equity_curve || [];
  const width = frame.clientWidth;
  const height = frame.clientHeight;
  if (!width || !height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  context.clearRect(0, 0, width, height);
  if (!curve.length) return;
  const pad = { left: 66, right: 18, top: 18, bottom: 34 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const values = curve.flatMap(point => [point.equity, point.benchmark_equity].filter(Number.isFinite));
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  const spread = Math.max(1, maximum - minimum);
  minimum -= spread * 0.08;
  maximum += spread * 0.08;
  const x = index => pad.left + plotWidth * index / Math.max(1, curve.length - 1);
  const y = value => pad.top + plotHeight * (maximum - value) / (maximum - minimum);
  context.font = "10px Inter, system-ui, sans-serif";
  context.fillStyle = "#758197";
  context.strokeStyle = "#e2e7ee";
  context.lineWidth = 1;
  for (let line = 0; line <= 4; line += 1) {
    const value = maximum - (maximum - minimum) * line / 4;
    const lineY = pad.top + plotHeight * line / 4;
    context.beginPath(); context.moveTo(pad.left, lineY); context.lineTo(width - pad.right, lineY); context.stroke();
    context.fillText(compactCurrency(value), 6, lineY + 3);
  }
  const negative = curve.map(point => Math.max(0, -point.drawdown));
  const maximumDrawdown = Math.max(...negative, 0.0001);
  context.beginPath();
  curve.forEach((point, index) => {
    const areaY = pad.top + plotHeight - plotHeight * 0.25 * negative[index] / maximumDrawdown;
    if (index === 0) context.moveTo(x(index), pad.top + plotHeight);
    context.lineTo(x(index), areaY);
  });
  context.lineTo(x(curve.length - 1), pad.top + plotHeight);
  context.closePath();
  context.fillStyle = "rgba(195,56,50,.10)";
  context.fill();
  context.beginPath();
  curve.forEach((point, index) => index === 0 ? context.moveTo(x(index), y(point.equity)) : context.lineTo(x(index), y(point.equity)));
  context.strokeStyle = "#245eea";
  context.lineWidth = 2;
  context.stroke();
  if (curve.some(point => Number.isFinite(point.benchmark_equity))) {
    context.beginPath();
    curve.forEach((point, index) => index === 0 ? context.moveTo(x(index), y(point.benchmark_equity)) : context.lineTo(x(index), y(point.benchmark_equity)));
    context.strokeStyle = "#758197";
    context.lineWidth = 1.5;
    context.setLineDash([5, 4]);
    context.stroke();
    context.setLineDash([]);
  }
  context.fillStyle = "#758197";
  [0, Math.floor((curve.length - 1) / 2), curve.length - 1].forEach((index, position) => {
    const label = curve[index].date;
    const offset = position === 0 ? 0 : position === 2 ? 54 : 27;
    context.fillText(label, x(index) - offset, height - 11);
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
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function modeLabel(mode) { return mode === "long_only" ? "仅多头" : "多空截面"; }
function engineLabel(engine) { return engine === "EVENT_LEDGER" ? "事件账本" : "向量引擎"; }
function scheduleLabel(schedule) {
  if (schedule === "WEEKLY_FIRST_SESSION") return "每周首个交易日";
  if (schedule === "MONTHLY_FIRST_SESSION") return "每月首个交易日";
  return "每日滚动袖套";
}
function option(value, label) { const node = document.createElement("option"); node.value = value; node.textContent = label; return node; }
function statusPill(status) { const labels = { ACTIVE: "冠军", QUALIFIED: "合格", OBSERVE: "观察", STALE_PROTOCOL: "待重评", HOLDOUT_CONTAMINATED: "盲测污染", COMPLETED: "完成", FAILED: "失败", RUNNING: "运行中" }; return element("span", `research-status ${String(status).toLowerCase()}`, labels[status] || status); }
function element(tagName, className = "", content = "") { const node = document.createElement(tagName); if (className) node.className = className; if (content !== "") node.textContent = content; return node; }
function text(id, value) { document.getElementById(id).textContent = value; }
function number(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed.toFixed(2) : "--"; }
function number4(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed.toFixed(4) : "--"; }
function percent(value) { const parsed = Number(value); return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(2)}%` : "--"; }
function currency(value) { const parsed = Number(value); return Number.isFinite(parsed) ? new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 0 }).format(parsed) : "--"; }
function money2(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "--"; }
function compactCurrency(value) { const absolute = Math.abs(value); if (absolute >= 1e8) return `${(value / 1e8).toFixed(1)}亿`; if (absolute >= 1e4) return `${(value / 1e4).toFixed(0)}万`; return value.toFixed(0); }

let toastTimer;
function toast(message, isError = false) {
  const node = document.getElementById("toast");
  node.textContent = message;
  node.style.background = isError ? "#a93430" : "#202a3b";
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 2800);
}
