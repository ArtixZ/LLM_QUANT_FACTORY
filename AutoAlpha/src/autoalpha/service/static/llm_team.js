const ROLE_LABELS = {
  INDEPENDENT_REVIEWER: "独立复核员",
  FALSIFICATION_DESIGNER: "证伪实验设计师",
  ROOT_CAUSE_ANALYST: "失败根因分析师",
  FACTOR_LIBRARIAN: "因子知识馆员",
  PORTFOLIO_MECHANISM_RESEARCHER: "组合机制研究员",
  TCA_PAPER_OBSERVER: "TCA 与仿真观察员",
};

const ROLE_ICONS = {
  INDEPENDENT_REVIEWER: "scan-search",
  FALSIFICATION_DESIGNER: "flask-conical",
  ROOT_CAUSE_ANALYST: "git-pull-request-draft",
  FACTOR_LIBRARIAN: "library-big",
  PORTFOLIO_MECHANISM_RESEARCHER: "network",
  TCA_PAPER_OBSERVER: "receipt-text",
};

const state = { snapshot: null, view: "artifacts", detailArtifact: null };

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refreshTeam").onclick = refresh;
  document.getElementById("closeArtifact").onclick = () => document.getElementById("artifactDialog").close();
  ["roleFilter", "statusFilter", "artifactSearch"].forEach(id => {
    document.getElementById(id).addEventListener("input", renderCurrentView);
  });
  document.getElementById("favoriteArtifactFilter").addEventListener("change", renderCurrentView);
  document.getElementById("favoriteArtifactBtn").onclick = () => {
    if (state.detailArtifact) toggleArtifactFavorite(state.detailArtifact.id);
  };
  document.getElementById("taskFilter").addEventListener("change", refresh);
  document.querySelectorAll(".llm-content-tabs button").forEach(button => {
    button.onclick = () => {
      state.view = button.dataset.view;
      document.querySelectorAll(".llm-content-tabs button").forEach(item => item.classList.toggle("active", item === button));
      renderCurrentView();
    };
  });
  refresh();
});

async function refresh() {
  const taskId = document.getElementById("taskFilter").value;
  try {
    const response = await fetch(`/api/llm-team${taskId ? `?task_id=${encodeURIComponent(taskId)}` : ""}`);
    if (!response.ok) throw new Error(await response.text());
    state.snapshot = await response.json();
    hydrateFilters();
    renderSummary();
    renderDomains();
    renderRoles();
    renderFlow();
    renderCurrentView();
    window.lucide?.createIcons();
  } catch (error) {
    toast(`读取失败：${error.message}`, true);
  }
}

function hydrateFilters() {
  const taskSelect = document.getElementById("taskFilter");
  const selectedTask = taskSelect.value;
  taskSelect.innerHTML = `<option value="">全部任务</option>${state.snapshot.tasks.map(task => `<option value="${escapeHtml(task.task_id)}">${escapeHtml(task.name)} · ${escapeHtml(task.market)}</option>`).join("")}`;
  taskSelect.value = selectedTask;
  const roleSelect = document.getElementById("roleFilter");
  const selectedRole = roleSelect.value;
  roleSelect.innerHTML = `<option value="">全部角色</option>${state.snapshot.roles.map(item => `<option value="${item.role}">${ROLE_LABELS[item.role] || item.role}</option>`).join("")}`;
  roleSelect.value = selectedRole;
}

function renderSummary() {
  const snapshot = state.snapshot;
  document.getElementById("variantName").textContent = snapshot.variant;
  document.getElementById("artifactCount").textContent = snapshot.summary.artifact_count;
  document.getElementById("knowledgeCount").textContent = snapshot.knowledge.length;
  document.getElementById("teamState").textContent = snapshot.enabled ? "ENABLED" : "DISABLED";
  document.getElementById("teamState").className = `state-pill ${snapshot.enabled ? "running" : ""}`;
  const failed = Object.values(snapshot.summary.roles).reduce((sum, item) => sum + item.failed, 0);
  document.getElementById("teamSummary").textContent = `${snapshot.roles.length} 个受限研究角色 · ${failed} 次失败开放 · 隐藏指标隔离`;
}

function renderDomains() {
  const matrix = state.snapshot.domain_matrix || {};
  const domains = matrix.domains || [];
  const root = document.getElementById("domainGrid");
  if (!domains.length) {
    root.innerHTML = "";
    return;
  }
  root.innerHTML = domains.map(domain => `
    <article class="llm-domain-card ${String(domain.status || "").toLowerCase()}">
      <header>
        <div><span>${escapeHtml(domain.domain)}</span><h2>${escapeHtml(domain.label)}</h2></div>
        <strong>${escapeHtml(domain.status || "WAITING")}</strong>
      </header>
      <p>${escapeHtml(domain.responsibility)}</p>
      <div class="llm-domain-metrics">
        <div><b>${domain.completed || 0}</b><small>完成</small></div>
        <div><b>${domain.failed || 0}</b><small>失败开放</small></div>
        <div><b>${Math.round(Number(domain.coverage_ratio || 0) * 100)}%</b><small>角色覆盖</small></div>
      </div>
      <footer>
        <span>${escapeHtml(domain.latest_headline || "等待结构化制品")}</span>
        <code>${escapeHtml(domain.latest_role || domain.roles.join(" / "))}</code>
      </footer>
    </article>
  `).join("");
}

function renderRoles() {
  const summaries = state.snapshot.summary.roles;
  document.getElementById("roleGrid").innerHTML = state.snapshot.roles.map(spec => {
    const summary = summaries[spec.role] || { completed: 0, failed: 0, total_tokens: 0, latest_at: null };
    return `<article class="llm-role-card" data-role="${spec.role}">
      <header><i data-lucide="${ROLE_ICONS[spec.role] || "bot"}"></i><div><span>${stageLabel(spec.stage)}</span><h2>${ROLE_LABELS[spec.role] || spec.role}</h2></div></header>
      <div class="llm-role-metrics"><div><strong>${summary.completed}</strong><span>完成</span></div><div><strong>${summary.failed}</strong><span>失败开放</span></div><div><strong>${formatNumber(summary.total_tokens)}</strong><span>Token</span></div></div>
      <footer><span>${summary.latest_at ? formatDate(summary.latest_at) : "尚无运行记录"}</span><code>ADVISORY ONLY</code></footer>
    </article>`;
  }).join("");
  document.querySelectorAll(".llm-role-card").forEach(card => {
    card.onclick = () => {
      document.getElementById("roleFilter").value = card.dataset.role;
      state.view = "artifacts";
      document.querySelectorAll(".llm-content-tabs button").forEach(item => item.classList.toggle("active", item.dataset.view === "artifacts"));
      renderCurrentView();
      document.querySelector(".llm-artifact-panel").scrollIntoView({ behavior: "smooth" });
    };
  });
}

function renderFlow() {
  const byStage = state.snapshot.roles.reduce((groups, item) => {
    (groups[item.stage] ||= []).push(item);
    return groups;
  }, {});
  document.getElementById("preRoleFlow").innerHTML = flowRoles(byStage.PRE_EVALUATION || []);
  document.getElementById("portfolioRoleFlow").innerHTML = flowRoles(byStage.PORTFOLIO_ADVISORY || []);
  document.getElementById("postRoleFlow").innerHTML = flowRoles(byStage.POST_EVALUATION || []);
}

function flowRoles(roles) {
  return roles.map(item => `<span title="${ROLE_LABELS[item.role] || item.role}"><i data-lucide="${ROLE_ICONS[item.role] || "bot"}"></i>${ROLE_LABELS[item.role] || item.role}</span>`).join("");
}

function renderCurrentView() {
  const artifacts = document.getElementById("artifactList");
  const knowledge = document.getElementById("knowledgeList");
  artifacts.hidden = state.view !== "artifacts";
  knowledge.hidden = state.view !== "knowledge";
  if (!state.snapshot) return;
  state.view === "artifacts" ? renderArtifacts() : renderKnowledge();
  window.lucide?.createIcons();
}

function renderArtifacts() {
  const role = document.getElementById("roleFilter").value;
  const status = document.getElementById("statusFilter").value;
  const query = document.getElementById("artifactSearch").value.trim().toLowerCase();
  const favoriteOnly = document.getElementById("favoriteArtifactFilter").checked;
  const records = state.snapshot.artifacts.filter(item => {
    const haystack = `${item.candidate_id || ""} ${item.iteration} ${item.role} ${JSON.stringify(item.artifact)}`.toLowerCase();
    return (!role || item.role === role) && (!status || item.status === status) && (!query || haystack.includes(query)) && (!favoriteOnly || item.favorite);
  });
  const container = document.getElementById("artifactList");
  if (!records.length) {
    container.innerHTML = `<p class="table-empty">当前筛选下没有角色制品</p>`;
    return;
  }
  container.innerHTML = records.map((item, index) => `<div class="llm-artifact-row" data-index="${index}" role="button" tabindex="0">
    <span class="llm-role-icon"><i data-lucide="${ROLE_ICONS[item.role] || "bot"}"></i></span>
    <span class="llm-artifact-main"><small>${stageLabel(item.stage)} · ITER ${item.iteration}</small><strong>${ROLE_LABELS[item.role] || item.role}</strong><em>${artifactHeadline(item)}</em></span>
    <span class="llm-artifact-factor"><code>${escapeHtml(item.candidate_id || "NO CANDIDATE")}</code><small>${formatDate(item.created_at)}</small></span>
    <span class="llm-status ${item.status.toLowerCase()}">${item.status === "COMPLETED" ? "已完成" : "失败开放"}</span>
    <button class="favorite-button${item.favorite ? " is-favorite" : ""}" type="button" data-favorite-id="${item.id}" title="${item.favorite ? "取消收藏" : "收藏制品"}" aria-label="${item.favorite ? "取消收藏" : "收藏制品"}"><i data-lucide="star"></i></button>
    <i data-lucide="chevron-right"></i>
  </div>`).join("");
  container.querySelectorAll(".llm-artifact-row").forEach((button, index) => {
    button.onclick = () => openArtifact(records[index]);
    button.onkeydown = event => { if (event.key === "Enter" || event.key === " ") openArtifact(records[index]); };
  });
  container.querySelectorAll("[data-favorite-id]").forEach(button => {
    button.onclick = event => { event.stopPropagation(); toggleArtifactFavorite(button.dataset.favoriteId); };
  });
}

function renderKnowledge() {
  const query = document.getElementById("artifactSearch").value.trim().toLowerCase();
  const records = state.snapshot.knowledge.filter(item => !query || `${item.name} ${item.factor_id} ${item.canonical_mechanism} ${item.tags.join(" ")} ${item.mechanism_summary}`.toLowerCase().includes(query));
  const container = document.getElementById("knowledgeList");
  if (!records.length) {
    container.innerHTML = `<p class="table-empty">当前任务尚未形成结构化机制档案</p>`;
    return;
  }
  container.innerHTML = records.map(item => `<article class="llm-knowledge-row">
    <div><span>${escapeHtml(item.canonical_mechanism)}</span><h3>${escapeHtml(item.name)}</h3><code>${escapeHtml(item.factor_id)}</code></div>
    <p>${escapeHtml(item.mechanism_summary || "暂无机制摘要")}</p>
    <div class="llm-tags">${item.tags.map(tag => `<span>${escapeHtml(tag)}</span>`).join("")}</div>
    <a class="icon-button" href="/factors?factor=${encodeURIComponent(item.factor_id)}" title="打开因子档案"><i data-lucide="square-arrow-out-up-right"></i></a>
  </article>`).join("");
}

function openArtifact(item) {
  state.detailArtifact = item;
  updateFavoriteButton(document.getElementById("favoriteArtifactBtn"), item.favorite);
  document.getElementById("dialogStage").textContent = `${stageLabel(item.stage)} · ${item.status}`;
  document.getElementById("dialogTitle").textContent = ROLE_LABELS[item.role] || item.role;
  document.getElementById("dialogIdentity").textContent = `${item.candidate_id || "NO CANDIDATE"} · ITER ${item.iteration}`;
  document.getElementById("dialogMeta").innerHTML = `<span>任务 ${escapeHtml(item.task_id)}</span><span>${formatDate(item.created_at)}</span><span>${formatNumber(item.usage.total_tokens || 0)} tokens</span><span>只读审计制品</span>`;
  document.getElementById("dialogPayload").textContent = JSON.stringify(item.artifact, null, 2);
  document.getElementById("artifactDialog").showModal();
  window.lucide?.createIcons();
}

async function toggleArtifactFavorite(artifactId) {
  const item = state.snapshot?.artifacts.find(record => String(record.id) === String(artifactId));
  if (!item) return;
  try {
    const response = await fetch(`/api/favorites/llm_artifact/${encodeURIComponent(item.id)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ favorite: !item.favorite, label: `${ROLE_LABELS[item.role] || item.role} · ITER ${item.iteration}`, context: { task_id: item.task_id, candidate_id: item.candidate_id } }),
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`);
    item.favorite = !item.favorite;
    if (String(state.detailArtifact?.id) === String(item.id)) {
      state.detailArtifact = item;
      updateFavoriteButton(document.getElementById("favoriteArtifactBtn"), item.favorite);
    }
    renderCurrentView();
    toast(item.favorite ? "角色制品已收藏" : "角色制品已取消收藏");
  } catch (error) { toast(error.message, true); }
}

function updateFavoriteButton(node, active) {
  node.classList.toggle("is-favorite", Boolean(active));
  node.title = active ? "取消收藏" : "收藏制品";
  node.setAttribute("aria-label", node.title);
  window.lucide?.createIcons({ nodes: [node] });
}

function artifactHeadline(item) {
  const artifact = item.artifact || {};
  const value = artifact.verdict || artifact.canonical_mechanism || artifact.primary_cause || artifact.preferred_action || artifact.execution_diagnosis || artifact.null_hypothesis || item.error || "结构化研究意见";
  return escapeHtml(String(value).slice(0, 180));
}

function stageLabel(stage) {
  return ({ PRE_EVALUATION: "提案前复核", PORTFOLIO_ADVISORY: "组合研究", POST_EVALUATION: "事后诊断" })[stage] || stage;
}

function formatDate(value) {
  if (!value) return "--";
  return new Intl.DateTimeFormat("zh-CN", { dateStyle: "short", timeStyle: "medium", hour12: false }).format(new Date(value));
}

function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN", { notation: value > 9999 ? "compact" : "standard" }).format(value || 0);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function toast(message, isError = false) {
  const target = document.getElementById("toast");
  target.textContent = message;
  target.className = isError ? "show error" : "show";
  setTimeout(() => { target.className = ""; }, 3200);
}
