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
  behavior: "all",
  ranking: "long_only_overall",
  rankingAscending: false,
  favoriteOnly: false,
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
  document.getElementById("canonicalLeaderboard").onclick = () => selectLeaderboard("long_only_overall");
  document.getElementById("recentLeaderboard").onclick = () => selectLeaderboard("recent_long_only_overall");
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
  document.getElementById("behaviorFilter").addEventListener("change", event => {
    libraryState.behavior = event.target.value;
    renderFactorTable();
  });
  document.getElementById("rankingMetric").addEventListener("change", event => {
    libraryState.ranking = event.target.value;
    libraryState.rankingAscending = !rankingDefinition().higher_is_better;
    renderLeaderboardSwitch();
    renderFactorTable();
  });
  document.getElementById("rankingDirection").onclick = () => {
    libraryState.rankingAscending = !libraryState.rankingAscending;
    renderFactorTable();
  };
  document.getElementById("favoriteFactorFilter").addEventListener("change", event => {
    libraryState.favoriteOnly = event.target.checked;
    renderFactorTable();
  });
  document.getElementById("favoriteFactorBtn").onclick = () => {
    if (libraryState.detailFactor) toggleFactorFavorite(libraryState.detailFactor.factor_id);
  };
  document.getElementById("libraryRefresh").onclick = () => queueLibraryRefresh();
  document.getElementById("clearSelection").onclick = () => {
    libraryState.selected.clear();
    renderFactorTable();
    renderSelection();
  };
  document.getElementById("openBacktest").onclick = () => {
    const factors = [...libraryState.selected].join(",");
    window.location.href = `/backtest?factors=${encodeURIComponent(factors)}`;
  };
  document.getElementById("openAutoCombine").onclick = async () => {
    const factors = [...libraryState.selected];
    if (!factors.length) return;
    const button = document.getElementById("openAutoCombine");
    button.disabled = true;
    try {
      const result = await api("/api/autocombine/quick-task", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ factor_ids: factors, objective_profile: document.getElementById("quickObjectiveProfile").value, maximum_factors: libraryState.data?.autocombine_defaults?.maximum_factors || 5, start_immediately: true }),
      });
      toast(result.started ? "组合优化任务已创建并启动" : "任务已创建，等待在 AutoCombine 启动");
      window.location.href = result.task_url;
    } catch (error) {
      toast(error.message, true);
      button.disabled = false;
    }
  };
  document.getElementById("openQuantCombine").onclick = () => {
    const factors = [...libraryState.selected];
    if (!factors.length) return;
    const origin = `${window.location.protocol}//${window.location.hostname}:8889`;
    window.location.href = `${origin}/?factors=${encodeURIComponent(factors.join(","))}`;
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

async function queueLibraryRefresh() {
  if (libraryState.refreshing) return;
  libraryState.refreshing = true;
  const refreshButton = document.getElementById("libraryRefresh");
  refreshButton.disabled = true;
  try {
    const result = await api("/api/factors/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (result.queued) {
      toast(result.deduplicated ? "因子库刷新作业已在队列中" : "因子库刷新作业已入队");
      const jobId = result.job?.job_id;
      window.location.href = `/jobs?queue=${encodeURIComponent(result.job?.queue || "system")}${jobId ? `&job=${encodeURIComponent(jobId)}` : ""}`;
      return;
    }
    await loadLibrary({ announce: false });
    toast("因子库已刷新");
  } catch (error) {
    toast(error.message, true);
  } finally {
    libraryState.refreshing = false;
    refreshButton.disabled = false;
  }
}

async function loadLibrary({ announce = true } = {}) {
  if (libraryState.refreshing) return;
  libraryState.refreshing = true;
  const refreshButton = document.getElementById("libraryRefresh");
  refreshButton.disabled = true;
  try {
    libraryState.data = await api("/api/factors");
    const objective = libraryState.data.autocombine_defaults?.objective_profile;
    if (objective && [...document.getElementById("quickObjectiveProfile").options].some(option => option.value === objective)) {
      document.getElementById("quickObjectiveProfile").value = objective;
    }
    libraryState.refreshedAt = new Date();
    hydrateFilters();
    renderSummary();
    renderDiagnostics();
    renderResearchMap();
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
  const rankingSelect = document.getElementById("rankingMetric");
  const currentRanking = libraryState.ranking;
  rankingSelect.replaceChildren();
  const rankingGroups = new Map();
  (libraryState.data.ranking_options || []).forEach(definition => {
    if (!rankingGroups.has(definition.group)) {
      const group = document.createElement("optgroup");
      group.label = definition.group;
      rankingGroups.set(definition.group, group);
      rankingSelect.append(group);
    }
    rankingGroups.get(definition.group).append(option(definition.id, definition.label));
  });
  const hasRanking = [...rankingSelect.options].some(item => item.value === currentRanking);
  libraryState.ranking = hasRanking ? currentRanking : "long_only_overall";
  rankingSelect.value = libraryState.ranking;
}

function renderSummary() {
  const summary = libraryState.data.summary;
  text("factorCount", summary.factor_count);
  text("activeCount", summary.active_count);
  text("qualifiedCount", summary.eligible_count);
  text("observedCount", summary.observed_count);
  text("categoryCount", summary.category_count);
  text("clusterCount", summary.cluster_count);
  const behavior = libraryState.data.behavior_clustering || {};
  const behaviorCount = summary.behavior_cluster_count || 0;
  text(
    "behaviorClusterCount",
    behavior.status === "RUNNING"
      ? `${behavior.completed_count || 0}/${behavior.total_count || summary.factor_count}`
      : behaviorCount || "待计算",
  );
  text("crowdedClusterCount", summary.crowded_behavior_cluster_count || 0);
  text("shadowCount", summary.shadow_count);
  text("contaminatedCount", summary.contaminated_count);
  text("dataEnd", libraryState.data.data.last_trade_date);
  text(
    "librarySummary",
    `${summary.factor_count} 个持久化候选 · 主榜 ${summary.long_only_evaluated_count || 0} · 行为分类 ${behavior.status || "NOT_STARTED"}${behavior.status === "RUNNING" ? ` ${behavior.completed_count || 0}/${behavior.total_count || summary.factor_count}` : ""} · ${summary.stale_protocol_count} 个待统一重评 · ${formatClock(libraryState.refreshedAt)} 已同步`,
  );
}

function renderDiagnostics() {
  const materialization = libraryState.data.materialization || {};
  const homogeneity = materialization.factor_homogeneity_backfill;
  const integrity = libraryState.data.knowledge_integrity || {};
  const gate = libraryState.data.gate_funnel || {};
  const knowledgeNode = document.getElementById("knowledgeBackfillStatus").closest("div");
  const gateNode = document.getElementById("gateFunnelStatus").closest("div");
  const categoryNode = document.getElementById("gateFunnelTopCategory").closest("div");
  const actionNode = document.getElementById("gateFunnelAction").closest("div");
  setDiagnosticState(knowledgeNode, integrity.complete ? "good" : (homogeneity ? "warn" : "bad"));
  text("knowledgeBackfillStatus", integrity.complete ? `${integrity.knowledge_count || 0}/${integrity.factor_count || 0} 完整` : `${integrity.missing_count || 0} 缺失`);
  text("knowledgeBackfillMeta", integrity.complete ? `${homogeneity?.behavior_snapshot_id || "behavior"} · ${formatDate(homogeneity?.updated_at)}` : `建议作业 ${integrity.recommended_job_type || "factor_homogeneity_backfill"}`);
  const total = Number(gate.total_candidates || 0), passed = Number(gate.passed_candidates || 0);
  setDiagnosticState(gateNode, total && passed === 0 ? "bad" : (passed > 0 ? "good" : "warn"));
  text("gateFunnelStatus", total ? `${passed}/${total} 通过` : "无候选");
  text("gateFunnelMeta", gate.materialized ? `物化 ${formatDate(gate.materialized_at)}` : "实时诊断，未缓存");
  const top = (gate.failure_categories || [])[0];
  setDiagnosticState(categoryNode, top ? "warn" : "good");
  text("gateFunnelTopCategory", top ? gateCategoryLabel(top.key) : "暂无");
  text("gateFunnelTopMeta", top ? `${top.count} 次失败 · ${percent(top.count / Math.max(1, total - passed))}` : "没有失败样本");
  const action = gateAction(gate);
  setDiagnosticState(actionNode, action.level);
  text("gateFunnelAction", action.title);
  text("gateFunnelActionMeta", action.note);
}

function renderResearchMap() {
  const map = libraryState.data?.research_map || {};
  const mechanisms = map.mechanism_map || map.mechanism_clusters || [];
  const folded = map.homogeneity_fold_groups || map.crowded_clusters || [];
  const parameterFamilies = map.parameter_families || [];
  text(
    "researchMapSummary",
    `${map.factor_count || 0} 个因子 · ${map.mechanism_cluster_count || mechanisms.length || 0} 个机制源 · ${map.behavior_cluster_count || 0} 个行为簇 · ${map.parameter_family_count || parameterFamilies.length || 0} 个参数家族 · ${map.near_duplicate_count || 0} 个近重复`,
  );
  text("researchMapState", map.research_map_protocol || map.protocol ? "READY" : "PENDING");
  renderMapCards(
    "mechanismMapList",
    mechanisms.slice(0, 8),
    cluster => mapClusterCard(cluster, { filterType: "mechanism" }),
  );
  renderMapCards(
    "crowdedClusterList",
    folded.slice(0, 8),
    cluster => mapClusterCard(cluster, { filterType: "behavior" }),
  );
  renderMapCards(
    "parameterFamilyList",
    parameterFamilies.slice(0, 8),
    family => parameterFamilyCard(family),
  );
  renderMapCards(
    "nearDuplicateList",
    (map.near_duplicates || []).slice(0, 8),
    duplicate => nearDuplicateCard(duplicate),
  );
  renderAnnualHeatmap(map.annual_heatmap || {});
}

function renderMapCards(id, items, renderer) {
  const node = document.getElementById(id);
  node.replaceChildren();
  if (!items.length) {
    node.append(element("p", "empty-data-state", "暂无可折叠对象"));
    return;
  }
  node.replaceChildren(...items.map(renderer));
}

function mapClusterCard(cluster, { filterType }) {
  const card = element("button", "factor-map-card");
  card.type = "button";
  const clusterId = cluster.cluster_id || cluster.mechanism || cluster.parameter_family || "--";
  const size = cluster.size || cluster.factor_count || 0;
  const score = cluster.average_score ?? cluster.average_long_only_score ?? cluster.leader_score;
  card.innerHTML = `
    <strong>${escapeHtml(clusterId)}</strong>
    <span>${escapeHtml(cluster.leader_name)} · ${escapeHtml(cluster.leader_factor_id)}</span>
    <small>${size} 个因子 · 均分 ${number(score, 2)} · 近重复 ${cluster.near_duplicate_count || 0}</small>
  `;
  card.onclick = () => {
    if (filterType === "mechanism") {
      libraryState.category = "all";
      libraryState.query = String(clusterId).toLowerCase();
      document.getElementById("factorSearch").value = clusterId;
    } else {
      libraryState.behavior = "all";
      libraryState.query = String(clusterId).toLowerCase();
      document.getElementById("factorSearch").value = clusterId;
    }
    renderFactorTable();
  };
  return card;
}

function parameterFamilyCard(family) {
  const card = element("button", "factor-map-card");
  card.type = "button";
  const familyId = family.parameter_family || "--";
  card.innerHTML = `
    <strong>${escapeHtml(familyId)}</strong>
    <span>${escapeHtml(family.leader_name || "--")} · ${escapeHtml(family.leader_factor_id || "--")}</span>
    <small>${family.factor_count || 0} 个因子 · ${escapeHtml((family.mechanisms || []).slice(0, 2).join(", ") || "机制待识别")}</small>
  `;
  card.onclick = () => {
    libraryState.query = String(familyId).toLowerCase();
    document.getElementById("factorSearch").value = familyId;
    renderFactorTable();
  };
  return card;
}

function nearDuplicateCard(duplicate) {
  const card = element("button", "factor-map-card warning");
  card.type = "button";
  card.innerHTML = `
    <strong>${escapeHtml(duplicate.name)}</strong>
    <span>${escapeHtml(duplicate.factor_id)} ↔ ${escapeHtml(duplicate.nearest_factor_id || "--")}</span>
    <small>${escapeHtml(duplicate.cluster_id || "--")} · 相似度 ${number4(duplicate.similarity)}</small>
  `;
  card.onclick = () => {
    libraryState.query = duplicate.factor_id.toLowerCase();
    document.getElementById("factorSearch").value = duplicate.factor_id;
    renderFactorTable();
  };
  return card;
}

function renderAnnualHeatmap(heatmap) {
  const node = document.getElementById("annualHeatmap");
  const years = heatmap.years || [];
  const rows = heatmap.rows || [];
  node.replaceChildren();
  if (!years.length || !rows.length) {
    node.append(element("p", "empty-data-state", "暂无年度折叠数据"));
    return;
  }
  const columns = `minmax(160px, 1.2fr) repeat(${years.length}, minmax(48px, 1fr))`;
  const header = element("div", "heatmap-row heatmap-header");
  header.style.gridTemplateColumns = columns;
  header.append(element("strong", "", "机制"));
  years.forEach(year => header.append(element("span", "", String(year))));
  node.append(header);
  rows.forEach(row => {
    const line = element("button", "heatmap-row");
    line.type = "button";
    line.style.gridTemplateColumns = columns;
    line.title = `${row.mechanism} · ${row.factor_count} 因子 · 弱年份 ${(row.weak_years || []).join(", ") || "无"}`;
    line.append(element("strong", "", row.mechanism));
    years.forEach(year => {
      const value = row.annual_returns?.[year];
      const cellNode = element("span", `heat-cell ${heatClass(value)}`, value == null ? "--" : percent(value, 0));
      line.append(cellNode);
    });
    line.onclick = () => {
      libraryState.query = row.mechanism.toLowerCase();
      document.getElementById("factorSearch").value = row.mechanism;
      renderFactorTable();
    };
    node.append(line);
  });
}

function heatClass(value) {
  if (value == null || !Number.isFinite(Number(value))) return "missing";
  if (Number(value) < 0) return "cold";
  if (Number(value) < 0.05) return "flat";
  if (Number(value) < 0.15) return "warm";
  return "hot";
}

function setDiagnosticState(node, state) {
  node.classList.remove("good", "warn", "bad");
  node.classList.add(state);
}

function gateCategoryLabel(value) {
  return ({
    SAMPLE_OR_COVERAGE: "样本/覆盖",
    MULTIPLE_TESTING_OR_DSR: "DSR/多重检验",
    DRAWDOWN_OR_RISK: "回撤/风险",
    CORRELATION_OR_INDEPENDENCE: "相关性/独立性",
    TURNOVER: "换手",
    RETURN_OR_SHARPE: "收益/夏普",
    CAPACITY_OR_LIQUIDITY: "容量/流动性",
    OTHER: "其他",
  })[value] || value || "--";
}

function gateAction(gate) {
  const backendAction = (gate.operator_actions || [])[0];
  if (backendAction) {
    return {
      level: backendAction.priority === "P0" ? "bad" : "warn",
      title: gateActionLabel(backendAction.action),
      note: backendAction.reason || backendAction.action,
    };
  }
  const categories = (gate.failure_categories || []).map(item => item.key);
  if (!gate.total_candidates) return { level: "warn", title: "先运行组合任务", note: "无组合候选可诊断" };
  if (Number(gate.passed_candidates || 0) > 0) return { level: "good", title: "进入策略晋级", note: "冻结候选并走盲测/影子/模拟链路" };
  if (categories.includes("SAMPLE_OR_COVERAGE")) return { level: "bad", title: "先修验证折数", note: "降低样本不足噪音，再比较搜索质量" };
  if (categories.includes("MULTIPLE_TESTING_OR_DSR")) return { level: "warn", title: "收缩试验预算", note: "减少同质候选和参数换皮惩罚" };
  if (categories.includes("CORRELATION_OR_INDEPENDENCE")) return { level: "warn", title: "提高独立性", note: "跨机制簇选因子，降低同源暴露" };
  if (categories.includes("DRAWDOWN_OR_RISK")) return { level: "warn", title: "切换风险优先", note: "降低回撤目标和单因子集中度" };
  return { level: "warn", title: "复查门禁", note: "查看失败项分布后调整约束" };
}

function gateActionLabel(value) {
  return ({
    RUN_COMBINATION_EXPERIMENTS: "先运行组合任务",
    PROMOTE_GATE_PASSING_CANDIDATES_TO_STRATEGY_LIBRARY: "进入策略晋级",
    REPAIR_WALK_FORWARD_CAPACITY_OR_COVERAGE: "先修验证折数",
    REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE: "压缩同质试验",
    FORCE_CROSS_MECHANISM_AND_CROSS_CLUSTER_SELECTION: "强制跨机制选因子",
    SWITCH_OBJECTIVE_TO_DRAWDOWN_AND_TAIL_RISK_FIRST: "切换风险优先",
    EXPAND_OR_RESEED_FACTOR_MECHANISMS: "扩展收益来源",
    INSPECT_UNCATEGORIZED_GATE_FAILURES: "复查门禁遥测",
  })[value] || value || "--";
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
      if (libraryState.behavior === "all") return true;
      if (libraryState.behavior === "LEADER") return factor.behavior_cluster_role === "LEADER";
      if (libraryState.behavior === "CROWDED") {
        return Number(factor.behavior_cluster_size || factor.homogeneity_cluster_size || 0) >= 8;
      }
      return (factor.behavior_redundancy || "PENDING") === libraryState.behavior;
    })
    .filter(factor => !libraryState.favoriteOnly || factor.favorite)
    .filter(factor => {
      if (!libraryState.query) return true;
      const haystack = [factor.factor_id, factor.name, factor.family, factor.category, factor.canonical_mechanism, factor.mechanism_summary, factor.hypothesis, factor.cluster_id, factor.behavior_cluster_id, factor.behavior_cluster_label, factor.behavior_redundancy, factor.lifecycle_state, factor.source_task_name, factor.source_market, factor.origin, ...(factor.tags || [])]
        .join(" ").toLowerCase();
      return haystack.includes(libraryState.query);
    })
    .sort(compareFactors);
}

function rankingDefinition() {
  return (libraryState.data?.ranking_options || []).find(item => item.id === libraryState.ranking)
    || { id: "long_only_overall", label: "纯多综合分", format: "score", higher_is_better: true };
}

function rankingValue(factor) {
  const value = factor.ranking_values?.[libraryState.ranking];
  const parsed = Number(value);
  return value !== null && value !== undefined && Number.isFinite(parsed) ? parsed : null;
}

function selectLeaderboard(ranking) {
  libraryState.ranking = ranking;
  libraryState.rankingAscending = false;
  const select = document.getElementById("rankingMetric");
  if ([...select.options].some(item => item.value === ranking)) select.value = ranking;
  renderLeaderboardSwitch();
  renderFactorTable();
}

function renderLeaderboardSwitch() {
  const recent = boardMetricPrefix() === "recent_";
  document.getElementById("canonicalLeaderboard")?.classList.toggle("active", !recent);
  document.getElementById("recentLeaderboard")?.classList.toggle("active", recent);
}

function boardMetricPrefix() {
  return libraryState.ranking.startsWith("recent_long_only") ? "recent_" : "";
}

function compareFactors(left, right) {
  const leftValue = rankingValue(left), rightValue = rankingValue(right);
  if (leftValue === null && rightValue !== null) return 1;
  if (leftValue !== null && rightValue === null) return -1;
  if (leftValue !== null && rightValue !== null && leftValue !== rightValue) {
    return libraryState.rankingAscending ? leftValue - rightValue : rightValue - leftValue;
  }
  const overallDifference = Number(right.ranking_values?.long_only_overall ?? -Infinity) - Number(left.ranking_values?.long_only_overall ?? -Infinity);
  if (Number.isFinite(overallDifference) && overallDifference !== 0) return overallDifference;
  const iterationDifference = Number(left.source_iteration) - Number(right.source_iteration);
  return iterationDifference || String(left.factor_id).localeCompare(String(right.factor_id));
}

function renderFactorTable() {
  const factors = filteredFactors();
  const definition = rankingDefinition();
  renderLeaderboardSwitch();
  const recent = boardMetricPrefix() === "recent_";
  text("boardSharpeHeader", recent ? "近期纯多夏普" : "主榜纯多夏普");
  text("boardAnnualHeader", recent ? "近期纯多年化" : "主榜纯多年化");
  text("boardDrawdownHeader", recent ? "近期纯多回撤" : "主榜纯多回撤");
  text("boardWorstHeader", recent ? "近期纯多最差折" : "主榜纯多最差折");
  text("rankingValueHeader", definition.label);
  const direction = document.getElementById("rankingDirection");
  direction.title = libraryState.rankingAscending ? `${definition.label}：升序，点击切换为降序` : `${definition.label}：降序，点击切换为升序`;
  direction.setAttribute("aria-label", direction.title);
  direction.innerHTML = `<i data-lucide="${libraryState.rankingAscending ? "arrow-up-wide-narrow" : "arrow-down-wide-narrow"}"></i>`;
  if (window.lucide) window.lucide.createIcons({ nodes: [direction] });
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
  const favorite = favoriteButton(factor.favorite, `收藏因子 ${factor.name}`);
  favorite.onclick = event => { event.stopPropagation(); toggleFactorFavorite(factor.factor_id); };
  const detailButton = element("button", "factor-detail-trigger");
  detailButton.type = "button";
  detailButton.title = `查看 ${factor.name} 的公式与研究档案`;
  detailButton.append(element("strong", "", factor.name), element("code", "", `${factor.factor_id} · ITER ${factor.source_iteration}`));
  detailButton.onclick = () => openFactorDetail(factor);
  const identityHead = element("div", "factor-identity-head");
  identityHead.append(favorite, detailButton);
  identity.append(identityHead, element("p", "factor-hypothesis", factor.hypothesis || "未记录经济假设"));
  row.append(cell(identity));
  const source = element("a", "factor-source-task", factor.source_task_name || factor.source_task_id);
  source.href = `/research/${encodeURIComponent(factor.source_task_id)}`;
  source.title = `打开来源任务工作台 ${factor.source_task_id}`;
  row.append(cell(source));
  const mechanism = element("div", "cluster-cell");
  mechanism.append(
    tag(factor.category),
    element("small", "", factor.canonical_mechanism || factor.family || "未规范化"),
  );
  row.append(cell(mechanism));
  const cluster = element("div", "cluster-cell");
  cluster.append(element("strong", "", factor.cluster_id), element("small", "", `${factor.cluster_role} · ${factor.cluster_size}`));
  row.append(cell(cluster));
  const behaviorCluster = element("div", "cluster-cell");
  behaviorCluster.append(
    element("strong", "", factor.behavior_cluster_id || "待计算"),
    element(
      "small",
      "",
      factor.behavior_cluster_id
        ? `${factor.behavior_cluster_role} · ${factor.behavior_cluster_size} · ${behaviorLabel(factor.behavior_redundancy)}`
        : "向量行为指纹尚未完成",
    ),
  );
  if (factor.behavior_nearest_factor_id) {
    behaviorCluster.title = `最近因子 ${factor.behavior_nearest_factor_id} · 综合相似度 ${number4(factor.behavior_nearest_similarity)}`;
  }
  row.append(cell(behaviorCluster));
  const researchStatus = statusPill(factor.research_state);
  researchStatus.title = historicalMetrics
    ? `评价协议已过期：${factor.metric_protocol || "未知"}；当前任务协议：${factor.current_protocol || "未知"}`
    : (factor.status_reason || "当前任务协议评价");
  row.append(cell(researchStatus));
  const lifecycle = statusPill(factor.lifecycle_state);
  if (factor.holdout_contaminated) lifecycle.title = "已人工查看当前隐藏期；更换研究世代不能清除污染";
  row.append(cell(lifecycle));
  const selectedRankingCell = metricCell(formatRankingValue(rankingValue(factor), rankingDefinition().format), rankingUsesHistoricalEvidence(factor));
  selectedRankingCell.classList.add("score-value");
  if (rankingValue(factor) === null) selectedRankingCell.title = factor.protocol_stale ? "当前协议尚未重评该指标" : "该因子尚未生成此指标";
  row.append(selectedRankingCell);
  const marginalCell = metricCell(number(factor.marginal_contribution?.incremental_net_ir), historicalMetrics);
  if (!factor.marginal_contribution && !historicalMetrics) {
    marginalCell.title = "未通过单因子门禁或未进入组合增删评估，因此没有边际 IR";
  }
  row.append(marginalCell);
  const prefix = boardMetricPrefix();
  row.append(metricCell(number(metrics[`${prefix}long_only_sharpe_ratio`]), historicalMetrics));
  row.append(metricCell(percent(metrics[`${prefix}long_only_simple_annual_return`]), historicalMetrics));
  row.append(metricCell(percent(metrics[`${prefix}long_only_max_drawdown`]), historicalMetrics));
  row.append(metricCell(number(metrics[`${prefix}long_only_walk_forward_worst_sharpe`]), historicalMetrics));
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
  updateFavoriteButton(document.getElementById("favoriteFactorBtn"), factor.favorite, `收藏因子 ${factor.name}`);
  text("factorDetailCategory", factor.category || "未分类");
  text("factorDetailTitle", factor.name);
  text("factorDetailId", `${factor.factor_id} · ITER ${factor.source_iteration} · ${factor.source_task_name || factor.source_task_id}`);
  text("factorDetailHypothesis", factor.hypothesis || "该因子尚未记录经济假设。");
  document.getElementById("factorDetailTags").replaceChildren(tag(factor.family || "未分类家族"), tag(factor.canonical_mechanism || "机制待规范"), tag(factor.origin || "AUTO_LLM_RESEARCH"), ...(factor.tags || []).map(value => tag(value)), ...(factor.knowledge_tags || []).slice(0, 5).map(value => tag(value)), tag(`结构簇 ${factor.cluster_id || "--"}`), tag(`行为簇 ${factor.behavior_cluster_id || "待计算"}`), statusPill(factor.research_state), statusPill(factor.lifecycle_state));
  const latex = expressionToLatex(factor.expression);
  const stats = expressionStats(factor.expression);
  document.getElementById("factorLatex").textContent = `\\[${latex}\\]`;
  document.getElementById("factorLatexSource").textContent = latex;
  text("formulaSummary", `${stats.nodes} 个节点 · 估计回看 ${stats.lookback} 期`);
  renderDefinitionList("expressionAudit", [["输入字段", [...stats.fields].join(" · ") || "--"], ["算子", [...stats.operators].join(" · ") || "--"], ["表达式节点", `${stats.nodes}`], ["估计最大回看", `${stats.lookback} 个交易期`], ["截面处理", stats.crossSectional ? "是" : "否"]]);
  const historicalMetrics = Boolean(factor.protocol_stale);
  const metrics = historicalMetrics ? (factor.historical_metric_summary || {}) : (factor.metric_summary || {}), marginal = factor.marginal_contribution || {};
  renderDefinitionList("factorMetrics", [
    ["主评价口径", historicalMetrics ? "历史协议，仅供诊断" : "统一主榜 2015–2024 · A股纯多周频代理"],
    ["规范机制标签", factor.canonical_mechanism || "待知识回填"],
    ["机制摘要", factor.mechanism_summary || "暂无机制知识摘要"],
    ["单因子纯多门禁", factor.status === "SCREENED_OUT" ? `未通过：${factor.status_reason || "未记录原因"}` : "通过"],
    ["统一主榜综合分", factor.long_only_score_available ? number(factor.scores?.long_only_overall) : "待统一重评"],
    ["主榜夏普 / 年化", `${number(metrics.long_only_sharpe_ratio)} / ${percent(metrics.long_only_simple_annual_return)}`],
    ["主榜回撤 / 最差折", `${percent(metrics.long_only_max_drawdown)} / ${number(metrics.long_only_walk_forward_worst_sharpe)}`],
    ["近期榜综合分", factor.recent_long_only_score_available ? number(factor.scores?.recent_long_only_overall) : "待统一重评"],
    ["近期夏普 / 年化", `${number(metrics.recent_long_only_sharpe_ratio)} / ${percent(metrics.recent_long_only_simple_annual_return)}`],
    ["近期回撤 / 最差折", `${percent(metrics.recent_long_only_max_drawdown)} / ${number(metrics.recent_long_only_walk_forward_worst_sharpe)}`],
    ["主榜正收益窗口", percent(metrics.long_only_walk_forward_positive_fraction)],
    ["主榜 DSR 概率", percent(metrics.long_only_deflated_sharpe_probability)],
    ["主榜年化换手", number(metrics.long_only_annual_turnover)],
    ["主榜容量", currency(metrics.long_only_capacity_cny)],
    ["组合边际净 IR", marginal.incremental_net_ir == null ? "未进入策略晋级" : number(marginal.incremental_net_ir)],
    ["同质化门禁", metrics.homogeneity_gate_passed === false ? `拦截：${metrics.homogeneity_gate_failure || "INSUFFICIENT_BEHAVIOR_NOVELTY"}` : (metrics.homogeneity_gate_passed === true ? "通过" : "未执行")],
    ["同质化新颖度", metrics.homogeneity_novelty_score == null ? "--" : number(metrics.homogeneity_novelty_score, 3)],
    ["同质化最近因子", metrics.homogeneity_nearest_factor_id ? `${metrics.homogeneity_nearest_factor_id} · ${number4(metrics.homogeneity_nearest_similarity)}` : "--"],
    ["行为分类", factor.behavior_cluster_id ? `${factor.behavior_cluster_label} · ${factor.behavior_cluster_role} · ${behaviorLabel(factor.behavior_redundancy)}` : "待向量行为重算"],
    ["最近行为因子", factor.behavior_nearest_factor_id ? `${factor.behavior_nearest_factor_id} · ${number4(factor.behavior_nearest_similarity)}` : "--"],
    ["信号 / 残差收益相关", `${number4(factor.behavior_signal_correlation)} / ${number4(factor.behavior_return_correlation)}`],
    ["诊断 · Alpha多空夏普", number(metrics.sharpe_ratio)],
    ["诊断 · Alpha多空年化", percent(metrics.simple_annual_return)],
    ["诊断 · Rank IC / IR", `${number4(metrics.rank_ic_mean)} / ${number(metrics.rank_ic_ir)}`],
    ["诊断 · Pearson IC", number4(metrics.pearson_ic_mean)],
  ]);
  document.getElementById("factorRawExpression").textContent = JSON.stringify(factor.expression, null, 2);
  const factorQuery = encodeURIComponent(factor.factor_id);
  document.getElementById("openFactorScreener").href = `/screener?factors=${factorQuery}`;
  document.getElementById("openFactorBacktest").href = `/backtest?factors=${factorQuery}`;
  renderLifecycleLoading();
  document.getElementById("factorIntelligence").innerHTML = `<p class="table-empty">正在读取结构化研究档案</p>`;
  text("factorIntelligenceCount", "--");
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
  try {
    const intelligence = await api(`/api/factors/${encodeURIComponent(factor.factor_id)}/intelligence`);
    if (libraryState.detailFactor?.factor_id === factor.factor_id) renderFactorIntelligence(intelligence);
  } catch (error) {
    if (libraryState.detailFactor?.factor_id === factor.factor_id) document.getElementById("factorIntelligence").innerHTML = `<p class="table-empty">${escapeHtml(error.message)}</p>`;
  }
}

function behaviorLabel(value) {
  return ({ NEAR_DUPLICATE: "近重复", SUBSTITUTE: "可替代", RELATED: "相关", DISTINCT: "独立", PENDING: "待计算" })[value] || value || "待计算";
}

async function toggleFactorFavorite(factorId) {
  const factor = libraryState.data?.factors.find(item => item.factor_id === factorId);
  if (!factor) return;
  try {
    await api(`/api/favorites/factor/${encodeURIComponent(factorId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ favorite: !factor.favorite, label: factor.name, context: { source_task_id: factor.source_task_id, family: factor.family } }),
    });
    factor.favorite = !factor.favorite;
    if (libraryState.detailFactor?.factor_id === factorId) {
      libraryState.detailFactor = factor;
      updateFavoriteButton(document.getElementById("favoriteFactorBtn"), factor.favorite, `收藏因子 ${factor.name}`);
    }
    renderFactorTable();
    toast(factor.favorite ? "因子已收藏" : "因子已取消收藏");
  } catch (error) { toast(error.message, true); }
}

function renderFactorIntelligence(intelligence) {
  const target = document.getElementById("factorIntelligence");
  const knowledge = intelligence.knowledge;
  const artifacts = intelligence.role_artifacts || [];
  text("factorIntelligenceCount", `${artifacts.length} 个角色制品`);
  if (!knowledge && !artifacts.length) {
    target.innerHTML = `<p class="table-empty">该因子产生于标准研究链路，尚无 FULL LLM 档案</p>`;
    return;
  }
  const knowledgeHtml = knowledge ? `<div class="factor-intelligence-summary"><span>${escapeHtml(knowledge.canonical_mechanism)}</span><strong>${escapeHtml(knowledge.mechanism_summary || "暂无机制摘要")}</strong><div>${(knowledge.tags || []).map(tag => `<em>${escapeHtml(tag)}</em>`).join("")}</div></div>` : "";
  const artifactHtml = artifacts.map(item => `<details><summary><span>${escapeHtml(item.role.replaceAll("_", " "))}</span><strong>${escapeHtml(item.status)}</strong></summary><pre>${escapeHtml(JSON.stringify(item.artifact, null, 2))}</pre></details>`).join("");
  target.innerHTML = `${knowledgeHtml}<div class="factor-intelligence-artifacts">${artifactHtml}</div>`;
}

function closeFactorDetail() { const dialog = document.getElementById("factorDetailDialog"); if (dialog.open) dialog.close(); }
function favoriteButton(active, label) { const node = element("button", `favorite-button${active ? " is-favorite" : ""}`); node.type = "button"; node.title = active ? "取消收藏" : label; node.setAttribute("aria-label", node.title); node.innerHTML = '<i data-lucide="star"></i>'; return node; }
function updateFavoriteButton(node, active, label) { node.classList.toggle("is-favorite", Boolean(active)); node.title = active ? "取消收藏" : label; node.setAttribute("aria-label", node.title); if (window.lucide) window.lucide.createIcons({ nodes: [node] }); }
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

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...options });
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
    node.title = "历史协议指标，仅供诊断；可排序，但不参与当前协议复合评分";
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
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]); }
function numeric(value) { if (value === null || value === undefined || value === "" || typeof value === "boolean") return null; const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null; }
function number(value) { const parsed = numeric(value); return parsed === null ? "--" : parsed.toFixed(2); }
function number4(value) { const parsed = numeric(value); return parsed === null ? "--" : parsed.toFixed(4); }
function percent(value) { const parsed = numeric(value); return parsed === null ? "--" : `${(parsed * 100).toFixed(2)}%`; }
function integer(value) { const parsed = numeric(value); return parsed === null ? "--" : Math.round(parsed).toLocaleString("zh-CN"); }
function currency(value) { const parsed = numeric(value); return parsed === null ? "--" : new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", notation: "compact", maximumFractionDigits: 2 }).format(parsed); }
function formatRankingValue(value, format) { return ({ percent, number4, integer, currency, score: number, number }[format] || number)(value); }
function rankingUsesHistoricalEvidence(factor) { return Boolean(factor.protocol_stale && !["overall", "robustness", "return", "risk", "execution", "information", "long_only_overall", "long_only_return", "long_only_robustness", "long_only_risk", "long_only_execution", "marginal_incremental_net_ir", "source_iteration"].includes(libraryState.ranking)); }
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
