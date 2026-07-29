const taskState = { data: null, selectedId: taskIdFromPath(), editingId: null, polling: null };

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  bindControls();
  await loadTasks();
  taskState.polling = window.setInterval(refreshRunningTasks, 4000);
});

function bindControls() {
  document.getElementById("newTaskBtn").onclick = () => openTaskDialog();
  document.getElementById("refreshTasks").onclick = loadTasks;
  document.getElementById("taskSearch").addEventListener("input", renderRows);
  document.getElementById("marketFilter").addEventListener("change", renderRows);
  document.getElementById("statusFilter").addEventListener("change", renderRows);
  document.getElementById("favoriteTaskFilter").addEventListener("change", renderRows);
  document.getElementById("favoriteTaskBtn").onclick = () => toggleTaskFavorite(taskState.selectedId);
  document.getElementById("closeTaskDialog").onclick = closeTaskDialog;
  document.getElementById("cancelTaskDialog").onclick = closeTaskDialog;
  document.getElementById("taskForm").addEventListener("submit", saveTask);
  document.getElementById("editTaskBtn").onclick = () => {
    const task = taskState.data?.tasks.find(item => item.task_id === taskState.selectedId);
    if (task) openTaskDialog(task);
  };
  document.getElementById("startTaskBtn").onclick = () => commandTask("start");
  document.getElementById("stopTaskBtn").onclick = () => commandTask("stop");
  document.getElementById("refreshActivity").onclick = loadActivity;
  document.getElementById("autoSplitBtn").onclick = autoSplitProtocol;
  document.getElementById("protocolDesign").addEventListener("change", () => {
    const design = value("protocolDesign");
    if (design === "REGIME_COVERAGE_BACKWARD") {
      document.getElementById("protocolValidationYears").value = 3;
    } else if (design === "RECENT_FIVE_YEAR_BACKWARD") {
      document.getElementById("protocolValidationYears").value = 2;
    }
    updateProtocolDesign();
  });
  ["protocolExplorationYears", "protocolValidationYears", "protocolHoldoutMonths", "protocolEmbargoDays", "taskDataEnd"].forEach(id => {
    document.getElementById(id).addEventListener("input", updateProtocolDesign);
  });
  ["explorationStart", "explorationEnd", "validationStart", "validationEnd", "holdoutStart", "holdoutEnd"].forEach(id => {
    document.getElementById(id).addEventListener("input", () => {
      document.getElementById("protocolDesign").value = "CUSTOM";
      updateProtocolDesign();
    });
  });
}

async function loadTasks() {
  try {
    taskState.data = await api("/api/research-tasks");
    buildFilters();
    renderSummary();
    renderRows();
    if (taskState.selectedId) { renderDetail(taskState.selectedId); await loadActivity(); }
  } catch (error) { toast(error.message, true); }
}

function buildFilters() {
  const market = document.getElementById("marketFilter");
  if (market.options.length === 1) market.append(...taskState.data.markets.map(item => option(item.value, item.label)));
  const status = document.getElementById("statusFilter");
  const statuses = [...new Set(taskState.data.tasks.map(task => task.status))].sort();
  const selected = status.value;
  status.replaceChildren(option("all", "全部状态"), ...statuses.map(value => option(value, value)));
  status.value = statuses.includes(selected) ? selected : "all";
}

function renderSummary() {
  const summary = taskState.data.summary;
  text("taskCount", summary.task_count);
  text("runningCount", summary.running_count);
  text("readyCount", summary.ready_count);
  text("sourceFactorCount", formatNumber(summary.factor_count));
  text("taskPageSummary", `${summary.task_count} 个任务 · ${summary.running_count} 个运行中 · ${summary.ready_count} 个数据就绪`);
}

function renderRows() {
  if (!taskState.data) return;
  const query = document.getElementById("taskSearch").value.trim().toLowerCase();
  const market = document.getElementById("marketFilter").value;
  const status = document.getElementById("statusFilter").value;
  const favoriteOnly = document.getElementById("favoriteTaskFilter").checked;
  const rows = taskState.data.tasks.filter(task => {
    const corpus = `${task.name} ${task.task_id} ${task.data_path}`.toLowerCase();
    return (!query || corpus.includes(query)) && (market === "all" || task.market === market) && (status === "all" || task.status === status) && (!favoriteOnly || task.favorite);
  });
  const body = document.getElementById("taskRows");
  if (!rows.length) {
    const row = document.createElement("tr"), cell = document.createElement("td");
    cell.colSpan = 10; cell.className = "table-empty"; cell.textContent = "没有符合条件的研究任务"; row.append(cell); body.replaceChildren(row); return;
  }
  body.replaceChildren(...rows.map(task => taskRow(task)));
  if (window.lucide) window.lucide.createIcons();
}

function taskRow(task) {
  const row = document.createElement("tr");
  if (task.task_id === taskState.selectedId) row.classList.add("selected");
  const link = element("a", "task-name-link", task.name); link.href = `/research/${encodeURIComponent(task.task_id)}`; link.title = "打开该任务的自动研究工作台";
  const identity = element("div", "task-identity");
  const favorite = favoriteButton(task.favorite, `收藏任务 ${task.name}`);
  favorite.onclick = event => { event.preventDefault(); event.stopPropagation(); toggleTaskFavorite(task.task_id); };
  identity.append(favorite, link, element("code", "", task.task_id));
  const phase = element("div", "task-phase"); phase.append(element("strong", "", task.phase || "WAITING"), element("span", "", `第 ${formatNumber(task.iteration || 0)} 轮`));
  row.append(cell(identity), cell(marketLabel(task.market)), cell(statusPill(task.status)), cell(phase), cell(taskRange(task)), cell(pathBlock(task.data_path)), cell(formatNumber(task.factor_count)), cell(formatNumber(task.iteration_count)), cell(formatDate(task.updated_at)));
  const action = element("a", "icon-button", ""); action.href = `/research-tasks/${encodeURIComponent(task.task_id)}`; action.title = "查看任务详情"; action.innerHTML = '<i data-lucide="chevron-right"></i>'; row.append(cell(action));
  return row;
}

function renderDetail(taskId) {
  const task = taskState.data.tasks.find(item => item.task_id === taskId);
  if (!task) { document.getElementById("taskDetailPanel").hidden = true; return; }
  taskState.selectedId = taskId;
  document.getElementById("taskDetailPanel").hidden = false;
  text("detailTaskName", task.name);
  text("detailTaskIdentity", `${marketLabel(task.market)} · ${task.status} · ${task.task_id}`);
  text("detailSnapshotHash", `DATA ${task.snapshot_hash || "--"} · PROTOCOL ${task.protocol_hash || "--"}`);
  text("detailNotes", task.notes || "未填写任务备注");
  document.getElementById("taskDetailGrid").replaceChildren(...[
    ["市场", marketLabel(task.market)], ["AI 可见数据", taskRange(task)], ["证据等级", task.readiness?.research_evidence_tier === "PRIMARY_DISCOVERY" ? "主研究" : "市场状态切片"], ["协议模式", protocolDesignLabel(task.protocol?.design)], ["协议版本", `REV ${task.protocol_revision || 1}`], ["当前轮次", formatNumber(task.iteration || 0)], ["运行 ID", task.run_id || "尚未启动"],
  ].map(([label, value]) => stat(label, value)));
  const running = ["RUNNING", "RETRYING", "STOPPING"].includes(task.status);
  const readiness = task.readiness || { runnable: false, blockers: ["研究运行条件尚未计算"] };
  const readinessNode = document.getElementById("taskReadiness");
  readinessNode.hidden = false; readinessNode.classList.toggle("ready", readiness.runnable);
  readinessNode.replaceChildren(element("strong", "", readiness.runnable ? `任务级研究协议已冻结 · ${protocolDesignLabel(task.protocol?.design)}` : "任务暂不可启动"), element("span", "", readiness.runnable ? `探索 ${task.protocol?.exploration_start} 至 ${task.protocol?.exploration_end} · 公开验证 ${task.protocol?.validation_start} 至 ${task.protocol?.validation_end} · 隐藏测试 ${task.protocol?.holdout_start} 至 ${task.protocol?.holdout_end}` : readiness.blockers.join("；")));
  document.getElementById("startTaskBtn").disabled = running || !readiness.runnable;
  document.getElementById("stopTaskBtn").disabled = !running || task.status === "STOPPING";
  document.getElementById("editTaskBtn").disabled = running;
  updateFavoriteButton(document.getElementById("favoriteTaskBtn"), task.favorite, `收藏任务 ${task.name}`);
  const workspaceLink = document.getElementById("openCurrentResearch");
  workspaceLink.hidden = false;
  workspaceLink.href = `/research/${encodeURIComponent(task.task_id)}`;
  if (window.lucide) window.lucide.createIcons();
}

async function toggleTaskFavorite(taskId) {
  const task = taskState.data?.tasks.find(item => item.task_id === taskId);
  if (!task) return;
  try {
    await api(`/api/favorites/research_task/${encodeURIComponent(taskId)}`, {
      method: "PUT",
      body: JSON.stringify({ favorite: !task.favorite, label: task.name, context: { market: task.market } }),
    });
    task.favorite = !task.favorite;
    renderRows();
    if (taskId === taskState.selectedId) renderDetail(taskId);
    toast(task.favorite ? "研究任务已收藏" : "研究任务已取消收藏");
  } catch (error) { toast(error.message, true); }
}

async function commandTask(action) {
  if (!taskState.selectedId) return;
  const button = document.getElementById(action === "start" ? "startTaskBtn" : "stopTaskBtn"); button.disabled = true;
  try {
    await api(`/api/research-tasks/${encodeURIComponent(taskState.selectedId)}/${action}`, { method: "POST" });
    toast(action === "start" ? "研究任务已启动" : "停止请求已登记"); await loadTasks();
  } catch (error) { toast(error.message, true); } finally { button.disabled = false; }
}

async function loadActivity() {
  if (!taskState.selectedId) return;
  try {
    const data = await api(`/api/research-tasks/${encodeURIComponent(taskState.selectedId)}/activity`);
    const list = document.getElementById("taskActivityList");
    text("activitySummary", data.task.run_id ? `${data.events.length} 条近期记录 · ${data.metrics.length} 轮指标` : "尚未运行");
    if (!data.events.length) { const empty = element("p", "table-empty", "任务启动后将在这里显示行动、研究与审计记录"); list.replaceChildren(empty); return; }
    list.replaceChildren(...[...data.events].reverse().map(activityItem));
  } catch (error) { toast(error.message, true); }
}

function activityItem(event) {
  const node = element("article", "task-activity-item");
  node.append(element("code", "", `${event.category.toUpperCase()} · ${event.event}`), element("strong", "", event.title), element("p", "", event.message), element("time", "", formatDate(event.timestamp_utc)));
  return node;
}

async function refreshRunningTasks() {
  if (document.hidden || !taskState.data) return;
  if (!taskState.data.tasks.some(task => ["RUNNING", "RETRYING", "STOPPING"].includes(task.status))) return;
  await loadTasks();
}

function openTaskDialog(task = null) {
  taskState.editingId = task?.task_id || null;
  text("taskDialogEyebrow", task ? "EDIT TASK" : "NEW TASK");
  text("taskDialogTitle", task ? "编辑自动研究任务" : "新建自动研究任务");
  document.getElementById("taskName").value = task?.name || "";
  document.getElementById("taskMarket").value = task?.market || "CN_A";
  document.getElementById("taskDataPath").value =
    task?.data_path || taskState.data?.defaults?.data_path || "";
  const reference = task || taskState.data?.tasks.find(item => item.task_id === "legacy-ashare") || null;
  document.getElementById("taskDataStart").value = task?.data_start || reference?.data_start || "";
  document.getElementById("taskDataEnd").value = task?.data_end || reference?.data_end || "";
  setProtocolFields(task?.protocol || null);
  updateProtocolDesign();
  if (!task?.protocol) autoSplitProtocol();
  document.getElementById("taskNotes").value = task?.notes || "";
  document.getElementById("taskDialog").showModal();
}

function closeTaskDialog() { document.getElementById("taskDialog").close(); taskState.editingId = null; }

async function saveTask(event) {
  event.preventDefault();
  const button = document.getElementById("saveTaskBtn"); button.disabled = true;
  const body = { name: document.getElementById("taskName").value.trim(), market: document.getElementById("taskMarket").value, data_path: document.getElementById("taskDataPath").value.trim(), data_start: document.getElementById("taskDataStart").value || null, data_end: document.getElementById("taskDataEnd").value || null, protocol: protocolFromForm(), notes: document.getElementById("taskNotes").value.trim() };
  try {
    const path = taskState.editingId ? `/api/research-tasks/${encodeURIComponent(taskState.editingId)}` : "/api/research-tasks";
    const task = await api(path, { method: taskState.editingId ? "PUT" : "POST", body: JSON.stringify(body) });
    taskState.selectedId = task.task_id; closeTaskDialog(); await loadTasks(); history.replaceState({}, "", `/research-tasks/${encodeURIComponent(task.task_id)}`); toast(task.data_ready ? "任务与数据快照已保存" : "任务已建档，等待数据源接入");
  } catch (error) { toast(error.message, true); } finally { button.disabled = false; }
}

function setProtocolFields(protocol) {
  const mapping = { explorationStart: "exploration_start", explorationEnd: "exploration_end", validationStart: "validation_start", validationEnd: "validation_end", holdoutStart: "holdout_start", holdoutEnd: "holdout_end" };
  Object.entries(mapping).forEach(([id, key]) => { document.getElementById(id).value = protocol?.[key] || ""; });
  document.getElementById("minimumFolds").value = protocol?.minimum_folds || 1;
  document.getElementById("protocolDesign").value = protocol?.design || "CUSTOM";
  document.getElementById("protocolExplorationYears").value = protocol?.exploration_years || 5;
  document.getElementById("protocolValidationYears").value = protocol?.validation_years || (protocol?.design === "REGIME_COVERAGE_BACKWARD" ? 3 : 2);
  document.getElementById("protocolHoldoutMonths").value = protocol?.holdout_months || 6;
  document.getElementById("protocolEmbargoDays").value = protocol?.embargo_days ?? 30;
}

function protocolFromForm() {
  const design = value("protocolDesign");
  const protocol = { exploration_start: value("explorationStart"), exploration_end: value("explorationEnd"), validation_start: value("validationStart"), validation_end: value("validationEnd"), holdout_start: value("holdoutStart"), holdout_end: value("holdoutEnd"), minimum_folds: Number(value("minimumFolds")), design };
  if (design === "RECENT_FIVE_YEAR_BACKWARD") {
    protocol.anchor_date = value("taskDataEnd");
    protocol.exploration_years = Number(value("protocolExplorationYears"));
    protocol.validation_years = Number(value("protocolValidationYears"));
    protocol.holdout_months = Number(value("protocolHoldoutMonths"));
  } else if (design === "REGIME_COVERAGE_BACKWARD") {
    protocol.anchor_date = value("taskDataEnd");
    protocol.validation_years = Number(value("protocolValidationYears"));
    protocol.holdout_months = Number(value("protocolHoldoutMonths"));
    protocol.embargo_days = Number(value("protocolEmbargoDays"));
  }
  return protocol;
}

async function autoSplitProtocol() {
  const startText = value("taskDataStart"), endText = value("taskDataEnd");
  if (!startText || !endText) { toast("请先填写任务数据起止日", true); return; }
  try {
    const preview = await api("/api/research-protocol/preset", {
      method: "POST",
      body: JSON.stringify({ data_path: value("taskDataPath"), data_start: startText, data_end: endText, design: value("protocolDesign"), exploration_years: Number(value("protocolExplorationYears")), validation_years: Number(value("protocolValidationYears")), holdout_months: Number(value("protocolHoldoutMonths")), embargo_days: Number(value("protocolEmbargoDays")) }),
    });
    setProtocolFields(preview.protocol);
    updateProtocolDesign();
    if (!preview.valid) throw new Error(preview.blockers.join("；"));
    toast(`${protocolDesignLabel(preview.protocol.design)}已应用 · ${preview.walk_forward_capacity.maximum_folds} 个有效验证折`);
  } catch (error) {
    toast(`协议模板应用失败：${error.message}`, true);
  }
}

function updateProtocolDesign() {
  const design = value("protocolDesign");
  const recent = design === "RECENT_FIVE_YEAR_BACKWARD";
  const regime = design === "REGIME_COVERAGE_BACKWARD";
  document.getElementById("protocolExplorationYears").disabled = !recent;
  document.getElementById("protocolValidationYears").disabled = !(recent || regime);
  document.getElementById("protocolHoldoutMonths").disabled = !(recent || regime);
  document.getElementById("protocolEmbargoDays").disabled = !regime;
  const summary = document.getElementById("protocolDesignSummary");
  if (recent) {
    summary.textContent = `${value("protocolExplorationYears")} 年探索 · ${value("protocolValidationYears")} 年公开验证 · ${value("protocolHoldoutMonths")} 个月隐藏测试 · 锚定 ${value("taskDataEnd") || "最新日"}`;
  } else if (regime) {
    summary.textContent = `全历史探索（自数据起始日） · ${value("protocolValidationYears")} 年公开验证 · ${value("protocolEmbargoDays")} 天隔离带 · ${value("protocolHoldoutMonths")} 个月隐藏测试 · 锚定 ${value("taskDataEnd") || "最新日"}`;
  } else {
    summary.textContent = "按任务完整区间进行均衡自动划分；日期可继续手动调整";
  }
}

function value(id) { return document.getElementById(id).value; }

async function api(path, options = {}) { const response = await fetch(path, { ...options, credentials: "same-origin", headers: { "Content-Type": "application/json", ...(options.headers || {}) } }); if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `HTTP ${response.status}`); } return response.json(); }
function taskIdFromPath() { const parts = location.pathname.split("/").filter(Boolean); return parts.length === 2 ? decodeURIComponent(parts[1]) : null; }
function taskRange(task) { const protocol = task.protocol || {}; return protocol.exploration_start && protocol.validation_end ? `${protocol.exploration_start} — ${protocol.validation_end}` : task.data_start && task.data_end ? `${task.data_start} — ${task.data_end}` : "等待数据"; }
function protocolDesignLabel(value) { return value === "RECENT_FIVE_YEAR_BACKWARD" ? "近期五年倒推" : value === "REGIME_COVERAGE_BACKWARD" ? "全历史制度覆盖·隔离带" : "均衡 / 自定义"; }
function marketLabel(value) { return ({ CN_A: "A 股", HK: "港股", US: "美股" })[value] || value; }
function pathBlock(value) { const node = element("code", "task-path", value || "--"); node.title = value || ""; return node; }
function statusPill(value) { return element("span", `state-pill small ${value}`, value); }
function stat(label, value) { const node = element("div", ""); node.append(element("span", "", label), element("strong", "", value)); return node; }
function cell(content) { const node = document.createElement("td"); node.append(content instanceof Node ? content : document.createTextNode(String(content))); return node; }
function option(value, label) { const node = document.createElement("option"); node.value = value; node.textContent = label; return node; }
function element(tag, className = "", content = "") { const node = document.createElement(tag); if (className) node.className = className; if (content !== "") node.textContent = content; return node; }
function favoriteButton(active, label) { const node = element("button", `favorite-button${active ? " is-favorite" : ""}`); node.type = "button"; node.title = active ? "取消收藏" : label; node.setAttribute("aria-label", node.title); node.innerHTML = '<i data-lucide="star"></i>'; return node; }
function updateFavoriteButton(node, active, label) { node.classList.toggle("is-favorite", Boolean(active)); node.title = active ? "取消收藏" : label; node.setAttribute("aria-label", node.title); }
function text(id, value) { document.getElementById(id).textContent = value; }
function formatNumber(value) { const number = Number(value); return Number.isFinite(number) ? number.toLocaleString() : "--"; }
function formatDate(value) { const date = new Date(value); return Number.isNaN(date.valueOf()) ? "--" : date.toLocaleString("zh-CN", { hour12: false }); }
let toastTimer; function toast(message, error = false) { const node = document.getElementById("toast"); node.textContent = message; node.style.background = error ? "#a93430" : "#202a3b"; node.classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => node.classList.remove("show"), 2600); }
