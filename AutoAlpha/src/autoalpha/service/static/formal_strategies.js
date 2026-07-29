const strategyState = { snapshot: null };

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refreshStrategies").onclick = loadStrategies;
  document.getElementById("syncStrategyBus").onclick = syncStrategyBus;
  document.getElementById("seedGateRepairJob").onclick = seedGateRepairJob;
  document.getElementById("seedStrategyCandidates").onclick = seedStrategyCandidates;
  document.getElementById("freezeReadyStrategies").onclick = freezeReadyStrategies;
  document.getElementById("closePackageDialog").onclick = () => {
    document.getElementById("executionPackageDialog").close();
  };
  document.getElementById("closeDossierDialog").onclick = () => {
    document.getElementById("releaseDossierDialog").close();
  };
  document.getElementById("closeLineageDialog").onclick = () => {
    document.getElementById("lineageDialog").close();
  };
  loadStrategies();
});

async function loadStrategies() {
  try {
    strategyState.snapshot = await api("/api/strategy-library");
    renderStrategies();
    renderProductionFunnel();
    renderCandidates();
    if (window.lucide) window.lucide.createIcons();
  } catch (error) {
    toast(error.message, true);
  }
}

async function syncStrategyBus() {
  const button = document.getElementById("syncStrategyBus");
  button.disabled = true;
  try {
    const result = await api("/api/strategy-bus/sync", {
      method: "POST",
      body: JSON.stringify({ run_now: false }),
    });
    const job = result.job || {};
    toast(result.deduplicated
      ? `已有同步作业：${job.job_id || "--"}`
      : `策略总线同步已入队：${job.job_id || "--"}`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* ignore */ }
    throw new Error(message);
  }
  return response.json();
}

function renderStrategies() {
  const snapshot = strategyState.snapshot || {};
  const strategies = snapshot.strategies || [];
  text("strategyCount", strategies.length);
  text("candidateCount", snapshot.promotion_candidate_count || 0);
  text("readyCount", strategies.filter(item => item.lifecycle_readiness?.ready).length);
  text("productionCount", strategies.filter(item => item.lifecycle === "PRODUCTION_CANDIDATE").length);
  text("strategyLibrarySummary", `${strategies.length} 个正式版本 · ${snapshot.promotion_candidate_count || 0} 个可入库候选`);
  const list = document.getElementById("strategyList");
  list.replaceChildren();
  if (!strategies.length) {
    list.append(emptyState("尚无正式策略版本"));
    return;
  }
  strategies.forEach(strategy => {
    const readiness = strategy.lifecycle_readiness || {};
    const evidenceSummary = strategy.production_evidence_summary || {};
    const publicGap = readiness.public_validation_gap || null;
    const gapSummary = publicGap
      ? `${publicGap.source_system || "--"} · ${publicGap.source_status || "--"} / ${publicGap.gate_status || "--"}`
      : "";
    const gapRoots = publicGap ? (publicGap.root_causes || []).join(" · ") : "";
    const article = element("article", "formal-strategy-card");
    article.innerHTML = `
      <div class="formal-card-head">
        <div>
          <span class="state-pill small">${escapeHtml(strategy.lifecycle)}</span>
          <h3>${escapeHtml(strategy.name)}</h3>
          <code>${escapeHtml(strategy.strategy_uid)} · VERSION ${strategy.version}</code>
        </div>
        <button class="button ${readiness.ready ? "primary" : ""}" data-approve="${escapeHtml(strategy.strategy_uid)}" data-version="${strategy.version}" data-target="${escapeHtml(readiness.next_lifecycle || "")}" ${readiness.ready ? "" : "disabled"}>
          <i data-lucide="badge-check"></i>${readiness.ready ? "审批并推进" : "证据不足"}
        </button>
        <button class="button" data-package="${escapeHtml(strategy.strategy_uid)}" data-version="${strategy.version}">
          <i data-lucide="file-json"></i>执行包
        </button>
        <button class="button" data-dossier="${escapeHtml(strategy.strategy_uid)}" data-version="${strategy.version}">
          <i data-lucide="clipboard-list"></i>发布档案
        </button>
        <button class="button" data-lineage="${escapeHtml(strategy.source_experiment_id || "")}" ${strategy.source_experiment_id ? "" : "disabled"}>
          <i data-lucide="git-branch"></i>血缘
        </button>
        <button class="button" data-export-dossier="${escapeHtml(strategy.strategy_uid)}" data-version="${strategy.version}">
          <i data-lucide="download"></i>导出档案
        </button>
      </div>
      <div class="formal-strategy-grid">
        <div><span>下一步</span><strong>${escapeHtml(readiness.transition_label || "--")}</strong></div>
        <div><span>目标阶段</span><strong>${escapeHtml(readiness.next_lifecycle || "终态")}</strong></div>
        <div><span>因子数</span><strong>${strategy.signal_policy?.factor_ids?.length || 0}</strong></div>
        <div><span>成交协议</span><strong>${escapeHtml(strategy.execution_policy?.execution_time || "--")}</strong></div>
      </div>
      <div class="evidence-box ${readiness.ready ? "ready" : "blocked"}">
        <strong>${escapeHtml(evidenceSummary.evidence_state || (readiness.ready ? "证据齐全" : "缺失证据"))}</strong>
        <p>${escapeHtml((evidenceSummary.missing_or_blocking_evidence || readiness.missing_evidence || []).join(" · ") || "可进入下一阶段")}</p>
        <p><b>下一阶段</b> ${escapeHtml(evidenceSummary.next_lifecycle || readiness.next_lifecycle || "终态")}</p>
        ${publicGap ? `<p><b>来源门禁</b> ${escapeHtml(gapSummary)}</p><p><b>根因</b> ${escapeHtml(gapRoots || "MISSING_GATE_TELEMETRY")}</p><p><b>建议</b> ${escapeHtml(publicGap.operator_hint || "INSPECT_SOURCE_CANDIDATE_GATE_TELEMETRY")}</p>` : ""}
      </div>
      <div class="strategy-factor-list">${(strategy.signal_policy?.factor_ids || []).map((factorId, index) => `<span><code>${escapeHtml(factorId)}</code><b>${percent(strategy.signal_policy.weights?.[index])}</b></span>`).join("")}</div>
    `;
    list.append(article);
  });
  document.querySelectorAll("[data-approve]").forEach(button => {
    button.onclick = () => approveStrategyTransition(
      button.dataset.approve,
      Number(button.dataset.version),
      button.dataset.target,
    );
  });
  document.querySelectorAll("[data-package]").forEach(button => {
    button.onclick = () => openExecutionPackage(button.dataset.package, Number(button.dataset.version));
  });
  document.querySelectorAll("[data-dossier]").forEach(button => {
    button.onclick = () => openReleaseDossier(button.dataset.dossier, Number(button.dataset.version));
  });
  document.querySelectorAll("[data-lineage]").forEach(button => {
    button.onclick = () => openLineage(button.dataset.lineage);
  });
  document.querySelectorAll("[data-export-dossier]").forEach(button => {
    button.onclick = () => exportReleaseDossier(button.dataset.exportDossier, Number(button.dataset.version));
  });
}

function renderProductionFunnel() {
  const funnel = strategyState.snapshot?.production_funnel || {};
  text("funnelProtocol", funnel.protocol || "--");
  const container = document.getElementById("productionFunnel");
  const stages = funnel.stages || [];
  container.replaceChildren();
  if (!stages.length) {
    container.append(emptyState("暂无策略生产漏斗数据"));
  } else {
    stages.forEach(stage => {
      const card = element("article", "strategy-funnel-card");
      const conversion = stage.conversion_from_previous == null
        ? "起点"
        : `${(Number(stage.conversion_from_previous) * 100).toFixed(1)}%`;
      card.innerHTML = `
        <span>${escapeHtml(stage.label)}</span>
        <strong>${number(stage.count, 0)}</strong>
        <small>${escapeHtml(conversion)}</small>
        <p>${escapeHtml(stage.description)}</p>
      `;
      container.append(card);
    });
  }
  const bottleneckList = document.getElementById("funnelBottlenecks");
  const bottlenecks = funnel.bottlenecks || [];
  const rootCauses = funnel.top_root_causes || [];
  const operatorHints = funnel.top_operator_hints || [];
  const repairTasks = funnel.repair_tasks || [];
  bottleneckList.replaceChildren();
  if (!bottlenecks.length && !rootCauses.length && !operatorHints.length && !repairTasks.length) {
    bottleneckList.append(element("p", "empty-data-state", "当前没有显著漏斗瓶颈"));
    return;
  }
  bottlenecks.forEach(item => {
    const row = element("article", "strategy-bottleneck-item");
    row.innerHTML = `
      <span class="state-pill small">${escapeHtml(item.severity)}</span>
      <div>
        <strong>${escapeHtml(item.title)}</strong>
        <p>${escapeHtml(item.detail)}</p>
      </div>
    `;
    bottleneckList.append(row);
  });
  rootCauses.slice(0, 5).forEach(([rootCause, count]) => {
    const row = element("article", "strategy-bottleneck-item");
    row.innerHTML = `
      <span class="state-pill small">ROOT</span>
      <div>
        <strong>${escapeHtml(rootCause)}</strong>
        <p>${number(count, 0)} 个组合候选暴露该根因。</p>
      </div>
    `;
    bottleneckList.append(row);
  });
  operatorHints.slice(0, 3).forEach(([hint, count]) => {
    const row = element("article", "strategy-bottleneck-item");
    row.innerHTML = `
      <span class="state-pill small">ACTION</span>
      <div>
        <strong>${escapeHtml(hint)}</strong>
        <p>${number(count, 0)} 个候选建议采用该修复方向。</p>
      </div>
    `;
    bottleneckList.append(row);
  });
  repairTasks.slice(0, 5).forEach(task => {
    const row = element("article", "strategy-bottleneck-item");
    const url = quantCombineTaskUrl(task.task_id);
    row.innerHTML = `
      <span class="state-pill small">REPAIR</span>
      <div>
        <strong>${escapeHtml(task.task_id)} · ${escapeHtml(task.status)} / ${escapeHtml(task.phase)}</strong>
        <p>${escapeHtml(task.objective_profile || "--")} · ${number(task.factor_count, 0)} 因子 · ${number(task.evaluation_count, 0)} 评价</p>
        <p><a class="inline-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">打开 QuantCombine 任务</a></p>
      </div>
    `;
    bottleneckList.append(row);
  });
}

function quantCombineTaskUrl(taskId) {
  const url = new URL(window.location.href);
  url.port = "8889";
  url.pathname = `/tasks/${taskId}`;
  url.search = "";
  url.hash = "";
  return url.toString();
}

function renderCandidates() {
  const candidates = strategyState.snapshot?.promotion_candidates || [];
  const list = document.getElementById("candidateList");
  list.replaceChildren();
  if (!candidates.length) {
    list.append(emptyState("暂无可入库候选"));
    return;
  }
  candidates.slice(0, 20).forEach(candidate => {
    const evidenceSummary = candidate.production_evidence_summary || {};
    const publicGap = candidate.public_validation_gap || null;
    const rootCauses = publicGap ? (publicGap.root_causes || []).join(" · ") : "";
    const gateSummary = publicGap
      ? `${publicGap.source_status || "--"} / ${publicGap.gate_status || "--"}`
      : "公开验证已通过";
    const evidenceClass = candidate.freeze_ready_after_creation ? "ready" : "blocked";
    const evidenceTitle = candidate.freeze_ready_after_creation
      ? "创建后可冻结"
      : "创建后仍需公开验证复核";
    const article = element("article", "formal-strategy-card candidate");
    article.innerHTML = `
      <div class="formal-card-head">
        <div>
          <span class="state-pill small">${escapeHtml(candidate.candidate_class)}</span>
          <h3>${escapeHtml(candidate.title)}</h3>
          <code>${escapeHtml(candidate.experiment_id)} · ${escapeHtml(candidate.source_system)}</code>
        </div>
        <button class="button" data-create="${escapeHtml(candidate.experiment_id)}"><i data-lucide="package-plus"></i>创建研究版本</button>
        <button class="button" data-lineage="${escapeHtml(candidate.experiment_id)}"><i data-lucide="git-branch"></i>血缘</button>
      </div>
      <div class="formal-strategy-grid">
        <div><span>候选分</span><strong>${number(candidate.score, 3)}</strong></div>
        <div><span>夏普</span><strong>${number(candidate.metrics?.portfolio_sharpe_ratio)}</strong></div>
        <div><span>年化</span><strong>${percent(candidate.metrics?.portfolio_simple_annual_return)}</strong></div>
        <div><span>回撤</span><strong>${percent(candidate.metrics?.portfolio_max_drawdown)}</strong></div>
      </div>
      <div class="evidence-box ${evidenceClass}">
        <strong>${escapeHtml(evidenceSummary.evidence_state || evidenceTitle)}</strong>
        <p><b>下一步</b> ${escapeHtml(candidate.next_action || "CREATE_RESEARCH_VERSION_FOR_REVIEW")}</p>
        <p><b>来源门禁</b> ${escapeHtml(gateSummary)}</p>
        <p><b>失败门禁</b> ${escapeHtml((candidate.failed_gates || []).join(" · ") || "无显式失败门禁")}</p>
        <p><b>生产缺口</b> ${escapeHtml((evidenceSummary.missing_or_blocking_evidence || []).join(" · ") || "无")}</p>
        <p><b>根因</b> ${escapeHtml(rootCauses || (candidate.freeze_ready_after_creation ? "READY_TO_FREEZE" : "MISSING_GATE_TELEMETRY"))}</p>
        <p><b>建议</b> ${escapeHtml(evidenceSummary.public_gate?.operator_hint || candidate.operator_hint || "INSPECT_SOURCE_CANDIDATE_GATE_TELEMETRY")}</p>
      </div>
    `;
    list.append(article);
  });
  document.querySelectorAll("[data-create]").forEach(button => {
    button.onclick = () => createStrategy(button.dataset.create);
  });
  document.querySelectorAll("[data-lineage]").forEach(button => {
    button.onclick = () => openLineage(button.dataset.lineage);
  });
}

async function createStrategy(experimentId) {
  try {
    await api("/api/strategy-library", {
      method: "POST",
      body: JSON.stringify({ experiment_id: experimentId }),
    });
    toast("已创建 RESEARCH 策略版本");
    await loadStrategies();
  } catch (error) {
    toast(error.message, true);
  }
}

async function seedGateRepairJob() {
  try {
    const result = await api("/api/gate-feedback/seed-quant-repair", {
      method: "POST",
      body: JSON.stringify({}),
    });
    const statusText = result.status === "EXISTING" ? "修复任务已存在" : result.status === "SKIPPED" ? "暂无可执行反馈" : "修复任务已入队";
    const reference = result.repair_task_id || result.job?.job_id || "";
    toast(`${statusText}${reference ? ` · ${reference}` : ""}`);
    await loadStrategies();
  } catch (error) {
    toast(error.message, true);
  }
}

async function seedStrategyCandidates() {
  try {
    const result = await api("/api/strategy-library/seed-candidates", {
      method: "POST",
      body: JSON.stringify({ limit: 20 }),
    });
    const text = result.status === "SKIPPED"
      ? "暂无可入库候选"
      : result.status === "EXISTING"
        ? "候选入库作业已存在"
        : "候选入库作业已入队";
    toast(`${text}${result.job?.job_id ? ` · ${result.job.job_id}` : ""}`);
    await loadStrategies();
  } catch (error) {
    toast(error.message, true);
  }
}

async function freezeReadyStrategies() {
  try {
    const result = await api("/api/strategy-library/freeze-ready", {
      method: "POST",
      body: JSON.stringify({ limit: 100 }),
    });
    const text = result.status === "SKIPPED"
      ? "暂无公开验证可冻结策略"
      : result.status === "EXISTING"
        ? "冻结推进作业已存在"
        : "冻结推进作业已入队";
    toast(`${text}${result.job?.job_id ? ` · ${result.job.job_id}` : ""}`);
    await loadStrategies();
  } catch (error) {
    toast(error.message, true);
  }
}

async function approveStrategyTransition(strategyUid, version, targetLifecycle) {
  try {
    await api(`/api/strategy-library/${strategyUid}/versions/${version}/approve`, {
      method: "POST",
      body: JSON.stringify({
        target_lifecycle: targetLifecycle || null,
        approval_type: approvalTypeFor(targetLifecycle),
        approver: window.localStorage.getItem("autoalpha_operator") || "local-operator",
        notes: "operator-approved-from-strategy-library-ui",
        evidence: {},
      }),
    });
    toast("策略审批已记录并完成晋级");
    await loadStrategies();
  } catch (error) {
    toast(error.message, true);
  }
}

function approvalTypeFor(targetLifecycle) {
  return ({
    FROZEN: "PUBLIC_VALIDATION_REVIEW",
    HIDDEN_HOLDOUT: "HIDDEN_HOLDOUT_REVIEW",
    SHADOW: "SHADOW_EXECUTION_REVIEW",
    PAPER: "PAPER_TRADING_REVIEW",
    PRODUCTION_CANDIDATE: "RISK_APPROVAL",
  })[targetLifecycle] || "PUBLIC_VALIDATION_REVIEW";
}

async function openExecutionPackage(strategyUid, version) {
  try {
    const pkg = await api(`/api/strategy-library/${strategyUid}/versions/${version}/execution-package`);
    document.getElementById("packageTitle").textContent = pkg.name;
    document.getElementById("packageSubtitle").textContent = `${pkg.strategy_uid} · VERSION ${pkg.version} · ${pkg.lifecycle}`;
    const body = document.getElementById("packageBody");
    body.replaceChildren(
      detailBlock("生产状态", [
        ["可生产", pkg.production_ready ? "YES" : "NO"],
        ["阻断", (pkg.production_blockers || []).join(" · ") || "无"],
        ["规格哈希", pkg.specification_hash],
      ]),
      detailBlock("信号规则", [
        ["信号时间", pkg.signal_contract?.signal_time],
        ["方向", pkg.signal_contract?.ranking_side],
        ["打分", pkg.signal_contract?.score_method],
        ["因子数", pkg.signal_contract?.factor_ids?.length],
      ]),
      detailBlock("调仓规则", [
        ["调仓", pkg.rebalance_contract?.schedule],
        ["持有期", pkg.rebalance_contract?.holding_period_days],
        ["卖出", pkg.rebalance_contract?.sell_rule],
        ["买入", pkg.rebalance_contract?.buy_rule],
      ]),
      detailBlock("成交约束", [
        ["成交时间", pkg.execution_contract?.execution_time],
        ["价格", pkg.execution_contract?.price_basis],
        ["可买", pkg.execution_contract?.tradability?.buy],
        ["可卖", pkg.execution_contract?.tradability?.sell],
      ]),
      detailBlock("交易说明书", [
        ["组合模式", pkg.trading_playbook?.portfolio_mode],
        ["信号截止", pkg.trading_playbook?.signal_cutoff],
        ["调仓触发", pkg.trading_playbook?.rebalance_trigger],
        ["目标权重", percent(pkg.trading_playbook?.capital_allocation?.per_position_target_weight)],
        ["买入受限", pkg.trading_playbook?.blocked_order_policy?.buy_blocked],
        ["卖出受限", pkg.trading_playbook?.blocked_order_policy?.sell_blocked],
        ["停用条件", (pkg.trading_playbook?.disable_conditions || []).slice(0, 3).join(" · ")],
      ]),
    );
    document.getElementById("executionPackageDialog").showModal();
  } catch (error) {
    toast(error.message, true);
  }
}

async function openReleaseDossier(strategyUid, version) {
  try {
    const dossier = await api(`/api/strategy-library/${strategyUid}/versions/${version}/release-dossier`);
    document.getElementById("dossierTitle").textContent = dossier.strategy?.name || "策略发布档案";
    document.getElementById("dossierSubtitle").textContent = `${dossier.strategy?.strategy_uid} · VERSION ${dossier.strategy?.version} · ${dossier.strategy?.lifecycle}`;
    const factors = dossier.factors || [];
    const body = document.getElementById("dossierBody");
    body.replaceChildren(
      detailBlock("发布判断", [
        ["协议", dossier.dossier_protocol],
        ["结论", dossier.audit?.release_decision],
        ["阻断", (dossier.production_blockers || []).join(" · ") || "无"],
        ["规格哈希", dossier.strategy?.specification_hash],
      ]),
      detailBlock("来源候选", [
        ["系统", dossier.source?.system],
        ["候选", dossier.source?.source_id],
        ["状态", dossier.source?.status],
        ["门禁", dossier.source?.gate_status],
        ["失败项", (dossier.source?.failed_gates || []).join(" · ") || "无"],
      ]),
      detailBlock("来源指标", [
        ["夏普", number(dossier.source?.metrics?.portfolio_sharpe_ratio)],
        ["年化", percent(dossier.source?.metrics?.portfolio_simple_annual_return)],
        ["回撤", percent(dossier.source?.metrics?.portfolio_max_drawdown)],
        ["最差折夏普", number(dossier.source?.metrics?.portfolio_walk_forward_worst_sharpe)],
      ]),
      factorDossierBlock(factors),
    );
    document.getElementById("releaseDossierDialog").showModal();
  } catch (error) {
    toast(error.message, true);
  }
}

async function exportReleaseDossier(strategyUid, version) {
  try {
    const result = await api(`/api/strategy-library/${strategyUid}/versions/${version}/release-dossier/export`, { method: "POST" });
    const artifact = result.artifact || {};
    toast(`发布档案已导出：${artifact.artifact_id || "--"}`);
    if (result.download_url) {
      window.open(result.download_url, "_blank", "noopener,noreferrer");
    }
  } catch (error) {
    toast(error.message, true);
  }
}

async function openLineage(experimentId) {
  if (!experimentId) return;
  try {
    const lineage = await api(`/api/strategy-experiments/${encodeURIComponent(experimentId)}/lineage?depth=3`);
    const summary = lineage.evidence_summary || {};
    document.getElementById("lineageTitle").textContent = lineage.center?.title || experimentId;
    document.getElementById("lineageSubtitle").textContent = `${lineage.center?.stage || "--"} · ${lineage.center?.source_system || "--"} · ${lineage.protocol}`;
    document.getElementById("lineageBody").replaceChildren(
      detailBlock("中心对象", [
        ["实验 ID", lineage.experiment_id],
        ["状态", lineage.center?.status],
        ["市场", lineage.center?.market],
        ["来源", `${lineage.center?.source_system || "--"} / ${lineage.center?.source_id || "--"}`],
      ]),
      detailBlock("证据摘要", [
        ["节点数", number(summary.node_count, 0)],
        ["边数", number(summary.edge_count, 0)],
        ["门禁", summary.gate_status || "--"],
        ["失败项", (summary.failed_gates || []).join(" · ") || "无"],
        ["正式入库", summary.has_formal_strategy_version ? "YES" : "NO"],
        ["策略生命周期", formatLifecycleCounts(summary.formal_strategy_lifecycles || {})],
      ]),
      formalStrategyRefsBlock(lineage.center?.formal_strategy_refs || []),
      lineageCollectionBlock("上游", lineage.nodes || [], lineage.upstream_experiment_ids || []),
      lineageCollectionBlock("下游", lineage.nodes || [], lineage.downstream_experiment_ids || []),
      lineageEdgesBlock(lineage.edges || []),
    );
    document.getElementById("lineageDialog").showModal();
  } catch (error) {
    toast(error.message, true);
  }
}

function formalStrategyRefsBlock(refs) {
  const section = element("section", "detail-block wide");
  section.append(element("h3", "", "正式策略引用"));
  if (!refs.length) {
    section.append(element("p", "empty-data-state", "尚未进入正式策略库"));
    return section;
  }
  refs.forEach(ref => {
    const row = element("div", "metric-row lineage-row");
    row.append(
      element("span", "", `${ref.strategy_uid} · VERSION ${ref.version}`),
      element("strong", "", ref.lifecycle || "--"),
    );
    row.title = ref.specification_hash || "";
    section.append(row);
  });
  return section;
}

function lineageCollectionBlock(title, nodes, ids) {
  const lookup = new Map(nodes.map(node => [node.experiment_id, node]));
  const section = element("section", "detail-block wide");
  section.append(element("h3", "", title));
  if (!ids.length) {
    section.append(element("p", "empty-data-state", "无关联对象"));
    return section;
  }
  ids.forEach(id => {
    const node = lookup.get(id) || { experiment_id: id, title: id, stage: "MISSING", status: "--" };
    const row = element("div", "metric-row lineage-row");
    row.append(
      element("span", "", `${node.stage} · ${node.title}`),
      element("strong", "", node.status),
    );
    row.title = id;
    section.append(row);
  });
  return section;
}

function formatLifecycleCounts(value) {
  const entries = Object.entries(value || {});
  if (!entries.length) return "--";
  return entries.map(([key, count]) => `${key}:${count}`).join(" · ");
}

function lineageEdgesBlock(edges) {
  const section = element("section", "detail-block wide");
  section.append(element("h3", "", "关系边"));
  if (!edges.length) {
    section.append(element("p", "empty-data-state", "无关系边"));
    return section;
  }
  edges.slice(0, 30).forEach(edge => {
    const row = element("div", "metric-row lineage-edge-row");
    row.append(
      element("span", "", `${edge.source_experiment_id} → ${edge.target_experiment_id}`),
      element("strong", "", edge.relation),
    );
    section.append(row);
  });
  return section;
}

function factorDossierBlock(factors) {
  const section = element("section", "detail-block wide");
  section.append(element("h3", "", "因子构成"));
  if (!factors.length) {
    section.append(element("p", "empty-data-state", "无因子明细"));
    return section;
  }
  factors.forEach(factor => {
    const row = element("div", "metric-row dossier-factor-row");
    row.append(
      element("span", "", `${factor.name || factor.factor_id} · ${factor.canonical_mechanism || "--"}`),
      element("strong", "", percent(factor.weight)),
    );
    section.append(row);
  });
  return section;
}

function detailBlock(title, rows) {
  const section = element("section", "detail-block");
  section.append(element("h3", "", title));
  rows.forEach(([label, value]) => {
    const row = element("div", "metric-row");
    row.append(element("span", "", label), element("strong", "", value == null ? "--" : String(value)));
    section.append(row);
  });
  return section;
}

function element(tag, className = "", textContent = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (textContent) node.textContent = textContent;
  return node;
}

function emptyState(message) {
  return element("p", "empty-data-state", message);
}

function text(id, value) {
  document.getElementById(id).textContent = value == null ? "--" : String(value);
}

function number(value, digits = 2) {
  return value == null || !Number.isFinite(Number(value)) ? "--" : Number(value).toFixed(digits);
}

function percent(value, digits = 2) {
  return value == null || !Number.isFinite(Number(value)) ? "--" : `${(Number(value) * 100).toFixed(digits)}%`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  }[char]));
}

function toast(message, error = false) {
  const node = document.getElementById("toast");
  node.textContent = message;
  node.className = `show${error ? " error" : ""}`;
  window.clearTimeout(node._timer);
  node._timer = window.setTimeout(() => { node.className = ""; }, 3600);
}
