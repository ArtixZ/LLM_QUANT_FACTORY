const state = { snapshot: null, original: null, draft: null, secrets: { api_key: "", tushare_token: "" }, revision: null };
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
const labels = {
  base_url: "API Base URL", model: "模型名称", temperature: "生成温度", full_llm_enabled: "完整 LLM 团队",
  iteration_interval_seconds: "轮次间隔", proposal_batch_size: "同方向批量提案数", maximum_active_factors: "冠军组合因子上限", research_concurrency: "并发研究轮次",
  data_path: "研究数据工作区", market_data_root: "市场数据下载器", data_auto_update_enabled: "自动增量更新", data_update_hour: "更新时间", data_product_ids: "数据产品",
  autocombine_default_objective: "默认优化目标", autocombine_default_min_factors: "最少因子", autocombine_default_max_factors: "最多因子",
  autocombine_default_minimum_weight: "最小权重", autocombine_default_maximum_weight: "最大权重", autocombine_default_weight_step: "权重步长",
  autocombine_default_pool_limit: "候选池上限", autocombine_default_maximum_experiments: "实验预算", autocombine_default_llm_proposals: "LLM 提议预算",
  autocombine_default_iteration_interval_seconds: "组合实验间隔", autocombine_concurrency: "并发组合任务",
};

document.addEventListener("DOMContentLoaded", async () => {
  bind();
  await load();
  if (window.lucide) window.lucide.createIcons();
});

function bind() {
  $("saveSettings").onclick = save;
  $("discardSettings").onclick = discard;
  $("refreshSettings").onclick = () => load(true);
  $("closeRevision").onclick = closeRevision;
  $("cancelRevision").onclick = closeRevision;
  $("restoreRevision").onclick = restoreRevision;
  $("revisionDialog").addEventListener("click", event => { if (event.target === event.currentTarget) closeRevision(); });
  window.addEventListener("beforeunload", event => { if (isDirty()) { event.preventDefault(); event.returnValue = ""; } });
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, credentials: "same-origin", cache: "no-store", headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `HTTP ${response.status}`); }
  return response.json();
}

async function load(announce = false) {
  try {
    const snapshot = await api("/api/control-settings");
    state.snapshot = snapshot;
    state.original = structuredClone(snapshot.values);
    state.draft = structuredClone(snapshot.values);
    state.secrets = { api_key: "", tushare_token: "" };
    render();
    if (announce) toast("配置状态已刷新");
  } catch (error) { toast(error.message, true); }
}

function render() {
  renderStatus(); renderNavigation(); renderActivation(); renderGroups(); renderRuntime(); renderGovernance(); renderRevisions(); renderDirty();
  if (window.lucide) window.lucide.createIcons();
}

function renderStatus() {
  const services = state.snapshot.services;
  const credentials = state.snapshot.credentials;
  const operational = state.snapshot.operational;
  const items = [
    ["AutoAlpha", services.autoalpha.status === "ok" ? "ONLINE" : "OFFLINE", `${services.autoalpha.active_tasks} 个活动任务 · ${services.autoalpha.task_count} 个任务`, services.autoalpha.status === "ok" ? "good" : "bad"],
    ["AutoCombine", services.autocombine.status === "ok" ? "ONLINE" : "UNREACHABLE", `${services.autocombine.active_tasks} 个活动任务 · ${services.autocombine.task_count} 个任务`, services.autocombine.status === "ok" ? "good" : "warn"],
    ["研究数据", operational.valid ? `${operational.data_end}` : "INVALID", operational.valid ? `${operational.data_start} 起 · ${shortHash(operational.data_fingerprint)}` : operational.error, operational.valid ? "good" : "bad"],
    ["AI 凭证", credentials.api_key_configured ? "CONFIGURED" : "MISSING", credentials.api_key_configured ? sourceLabel(credentials.api_key_source) : "自动研究无法调用 Provider", credentials.api_key_configured ? "good" : "bad"],
    ["Tushare", credentials.tushare_token_configured ? "CONFIGURED" : "MISSING", credentials.tushare_token_configured ? sourceLabel(credentials.tushare_token_source) : "数据增量更新不可用", credentials.tushare_token_configured ? "good" : "warn"],
  ];
  $("statusBand").innerHTML = items.map(([label, value, detail, tone]) => `<div class="settings-status-item ${tone}"><span>${esc(label)}</span><strong title="${esc(value)}">${esc(value)}</strong><small title="${esc(detail)}">${esc(detail)}</small></div>`).join("");
}

function renderNavigation() {
  const icons = { provider: "bot", autoalpha: "flask-conical", autocombine: "network", data: "database-zap" };
  const groups = [...state.snapshot.catalog.map(group => ({ key: group.key, title: group.title, icon: icons[group.key] })), { key: "runtime", title: "运行环境", icon: "server-cog" }, { key: "governance", title: "研究与交易边界", icon: "landmark" }, { key: "revisions", title: "配置版本", icon: "history" }];
  $("settingsNavigation").innerHTML = groups.map((group, index) => `<button type="button" class="${index === 0 ? "active" : ""}" data-target="${esc(group.key)}-section"><i data-lucide="${group.icon}"></i>${esc(group.title)}</button>`).join("");
  document.querySelectorAll("[data-target]").forEach(button => button.onclick = () => {
    document.querySelectorAll("[data-target]").forEach(item => item.classList.toggle("active", item === button));
    document.getElementById(button.dataset.target)?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function renderActivation() {
  const pending = state.snapshot.activation.pending_restart_keys || [];
  $("activationBanner").className = `activation-banner${pending.length ? " pending" : ""}`;
  $("activationBanner").innerHTML = pending.length
    ? `<i data-lucide="power"></i><div><strong>${pending.length} 项设置等待重启后生效</strong><small>${pending.map(key => labels[key] || key).join("、")}。当前任务继续使用已冻结协议，不会被静默改写。</small></div><a href="#runtime-section">查看当前进程值</a>`
    : `<i data-lucide="check-circle-2"></i><div><strong>配置与当前进程一致</strong><small>即时项按标注生效；新任务默认值不会反向改变正在运行的研究任务。</small></div><a href="#governance-section">查看治理边界</a>`;
}

function renderGroups() {
  $("settingsGroups").innerHTML = state.snapshot.catalog.map(group => `
    <section class="settings-section" id="${esc(group.key)}-section">
      <div class="settings-section-head"><div><span>${esc(group.key.toUpperCase())}</span><h2>${esc(group.title)}</h2><p>${esc(group.description)}</p></div><i data-lucide="${{ provider: "bot", autoalpha: "flask-conical", autocombine: "network", data: "database-zap" }[group.key]}"></i></div>
      <div class="settings-field-grid">${group.fields.map(field => fieldHtml(field)).join("")}</div>
    </section>`).join("");
  state.snapshot.catalog.flatMap(group => group.fields).forEach(bindField);
}

function fieldHtml(field) {
  const full = field.kind === "multiselect" || field.key === "data_path" || field.key === "market_data_root" || field.key === "base_url";
  const meta = `<div class="field-meta"><span>${esc(field.source)}</span><span class="${field.effect.includes("重启") ? "effect-restart" : ""}">${esc(field.effect)}</span></div>`;
  let control = "";
  if (field.kind === "boolean") {
    control = `<label class="toggle-control"><input id="setting-${esc(field.key)}" type="checkbox" ${state.draft[field.key] ? "checked" : ""}><span class="toggle-track"></span><span class="toggle-label">${state.draft[field.key] ? "已启用" : "已停用"}</span></label>`;
  } else if (field.kind === "secret") {
    const ready = field.key === "api_key" ? state.snapshot.credentials.api_key_configured : state.snapshot.credentials.tushare_token_configured;
    control = `<div class="secret-control"><input id="setting-${esc(field.key)}" type="password" value="" placeholder="${ready ? "留空则保留现有凭证" : "输入凭证"}" autocomplete="new-password"><span class="credential-badge ${ready ? "ready" : ""}"><i data-lucide="${ready ? "key-round" : "key"}"></i>${ready ? "Keychain 已配置" : "尚未配置"}</span></div>`;
  } else if (field.kind === "select") {
    control = `<select id="setting-${esc(field.key)}">${field.options.map(option => `<option value="${esc(option.value)}" ${state.draft[field.key] === option.value ? "selected" : ""}>${esc(option.label)}</option>`).join("")}</select>`;
  } else if (field.kind === "multiselect") {
    const selected = new Set(state.draft[field.key] || []);
    control = `<div class="product-selector" id="setting-${esc(field.key)}">${field.options.map(option => `<label class="product-option ${selected.has(option.value) ? "selected" : ""} ${option.disabled ? "locked" : ""}"><input type="checkbox" value="${esc(option.value)}" ${selected.has(option.value) ? "checked" : ""} ${option.disabled ? "disabled" : ""}><span><strong>${esc(option.label)}</strong><small>${esc(option.description)}</small></span></label>`).join("")}</div>`;
  } else {
    control = `<input id="setting-${esc(field.key)}" type="${field.kind === "number" ? "number" : field.kind === "url" ? "url" : "text"}" value="${esc(state.draft[field.key])}" ${field.minimum != null ? `min="${field.minimum}"` : ""} ${field.maximum != null ? `max="${field.maximum}"` : ""} ${field.step != null ? `step="${field.step}"` : ""}>`;
  }
  return `<div class="settings-field ${full ? "full" : ""}" data-field="${esc(field.key)}"><div class="field-head"><label for="setting-${esc(field.key)}">${esc(field.label)}</label>${meta}</div>${control}</div>`;
}

function bindField(field) {
  const node = $(`setting-${field.key}`);
  if (!node) return;
  if (field.kind === "multiselect") {
    node.querySelectorAll("input").forEach(input => input.onchange = () => {
      const values = [...node.querySelectorAll("input")].filter(item => item.checked).map(item => item.value);
      state.draft[field.key] = [...new Set(values)];
      renderGroups(); renderDirty();
    });
    return;
  }
  const event = field.kind === "boolean" || field.kind === "select" ? "change" : "input";
  node.addEventListener(event, () => {
    if (field.kind === "secret") state.secrets[field.key] = node.value;
    else if (field.kind === "boolean") state.draft[field.key] = node.checked;
    else if (field.kind === "number") state.draft[field.key] = node.value === "" ? null : Number(node.value);
    else state.draft[field.key] = node.value;
    if (field.kind === "boolean") node.closest(".toggle-control").querySelector(".toggle-label").textContent = node.checked ? "已启用" : "已停用";
    node.classList.toggle("changed", field.kind !== "secret" && !same(state.draft[field.key], state.original[field.key]));
    renderDirty();
  });
}

function renderRuntime() {
  $("runtimeTable").innerHTML = state.snapshot.runtime.runtime.map(item => `<div class="runtime-row"><strong>${esc(item.label)}</strong><code title="${esc(item.value)}">${esc(item.value)}</code><span>${esc(item.source)}</span><span>${esc(item.effect)}</span></div>`).join("");
}

function renderGovernance() {
  const names = { protocol_version: "研究协议版本", portfolio_mode: "组合方向", execution_protocol: "执行引擎协议", execution_data_mode: "成交数据口径", rebalance_schedule: "调仓规则", gross_exposure: "目标总仓位", maximum_positions: "最大持仓数", holding_period_days: "持有周期", holdout_budget: "世代盲测预算" };
  $("governanceGrid").innerHTML = Object.entries(state.snapshot.runtime.governance).map(([key, value]) => `<div class="governance-item"><span>${esc(names[key] || key)}</span><strong>${esc(formatGovernance(key, value))}</strong></div>`).join("");
}

function renderRevisions() {
  const revisions = state.snapshot.revisions || [];
  $("revisionList").innerHTML = revisions.length ? revisions.map(revision => `<button type="button" class="revision-row" data-revision="${revision.id}"><span>REV ${revision.id}</span><div><strong>${esc(revision.change_note)}</strong><small>${esc(revision.changed_by)}</small></div><code>${esc(revision.changed_keys.map(key => labels[key] || key).join(" · "))}</code><time>${esc(dateTime(revision.created_at))}</time><i data-lucide="chevron-right"></i></button>`).join("") : `<div class="empty-settings">尚无配置修订。首次有效保存后将在这里建立版本。</div>`;
  document.querySelectorAll("[data-revision]").forEach(button => button.onclick = () => openRevision(Number(button.dataset.revision)));
}

function openRevision(id) {
  state.revision = state.snapshot.revisions.find(item => item.id === id);
  if (!state.revision) return;
  $("revisionTitle").textContent = `REV ${id} · ${state.revision.change_note}`;
  $("revisionDetail").innerHTML = `<div class="revision-summary"><p>${esc(state.revision.change_note)}</p><small>${esc(dateTime(state.revision.created_at))} · ${esc(state.revision.changed_by)} · ${esc(shortHash(state.revision.fingerprint))}</small></div>${state.revision.changed_keys.map(key => `<div class="revision-change"><strong>${esc(labels[key] || key)}</strong><code>${esc(formatValue(state.revision.previous_values[key]))} → ${esc(formatValue(state.revision.values[key]))}</code></div>`).join("")}`;
  $("revisionDialog").showModal(); if (window.lucide) window.lucide.createIcons();
}

function closeRevision() { $("revisionDialog").close(); state.revision = null; }

async function restoreRevision() {
  if (!state.revision || !window.confirm(`确认恢复 REV ${state.revision.id}？凭证不会被改变。`)) return;
  const revisionId = state.revision.id;
  try {
    const snapshot = await api(`/api/control-settings/revisions/${revisionId}/restore`, { method: "POST", body: JSON.stringify({ change_note: `恢复配置版本 REV ${revisionId}` }) });
    closeRevision(); adopt(snapshot); toast(`已恢复 REV ${revisionId}`);
  } catch (error) { toast(error.message, true); }
}

async function save() {
  const errors = localValidation();
  if (errors.length) { toast(errors[0], true); return; }
  $("saveSettings").disabled = true;
  try {
    const snapshot = await api("/api/control-settings", { method: "PUT", body: JSON.stringify({ values: state.draft, api_key: state.secrets.api_key || null, tushare_token: state.secrets.tushare_token || null, change_note: $("changeNote").value.trim() || "更新全局设置" }) });
    adopt(snapshot); toast("配置已验证、保存并写入审计日志");
  } catch (error) { toast(error.message, true); renderDirty(); }
}

function adopt(snapshot) {
  state.snapshot = snapshot; state.original = structuredClone(snapshot.values); state.draft = structuredClone(snapshot.values); state.secrets = { api_key: "", tushare_token: "" }; $("changeNote").value = ""; render();
}
function discard() { state.draft = structuredClone(state.original); state.secrets = { api_key: "", tushare_token: "" }; $("changeNote").value = ""; render(); toast("待保存变更已放弃"); }
function changedKeys() { return Object.keys(state.draft || {}).filter(key => !same(state.draft[key], state.original[key])); }
function isDirty() { return changedKeys().length > 0 || Boolean(state.secrets.api_key || state.secrets.tushare_token); }
function renderDirty() { const count = changedKeys().length + Number(Boolean(state.secrets.api_key)) + Number(Boolean(state.secrets.tushare_token)); $("dirtyState").textContent = count ? `${count} 项待保存变更` : "没有待保存变更"; $("dirtyState").parentElement.classList.toggle("dirty", Boolean(count)); $("saveSettings").disabled = !count; $("discardSettings").disabled = !count; }
function localValidation() { const v = state.draft; const errors = []; if (!v.base_url?.match(/^https?:\/\//)) errors.push("API Base URL 必须是完整的 HTTP(S) 地址"); if (!v.model?.trim()) errors.push("模型名称不能为空"); if (v.autocombine_default_min_factors > v.autocombine_default_max_factors) errors.push("AutoCombine 最少因子不能大于最多因子"); if (v.autocombine_default_minimum_weight > v.autocombine_default_maximum_weight) errors.push("AutoCombine 最小权重不能大于最大权重"); if (!v.data_product_ids.includes("core_market")) errors.push("核心行情数据不能停用"); return errors; }
function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
function formatValue(value) { if (value == null) return "--"; if (typeof value === "string" && value.length > 120) return `${value.slice(0, 117)}...`; return Array.isArray(value) ? value.join(", ") : String(value); }
function formatGovernance(key, value) { if (key === "gross_exposure") return `${(Number(value) * 100).toFixed(0)}%`; if (key === "maximum_positions") return `${value} 只`; if (key === "holding_period_days") return `${value} 个交易日`; if (key === "holdout_budget") return `${value} 次 / 世代`; return value; }
function sourceLabel(value) { return value === "environment" ? "环境变量优先" : "系统 Keychain"; }
function shortHash(value) { return value ? String(value).slice(0, 12) : "--"; }
function dateTime(value) { return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "--"; }
function toast(message, error = false) { const node = $("toast"); node.textContent = message; node.className = `show${error ? " error" : ""}`; window.clearTimeout(node._timer); node._timer = window.setTimeout(() => { node.className = ""; }, 4200); }
