const dataCenterState = { snapshot: null, timer: null };

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  buildHourOptions();
  bindControls();
  await loadDataCenter();
  dataCenterState.timer = setInterval(loadDataCenter, 15000);
});

function bindControls() {
  document.getElementById("dataSettingsForm").addEventListener("submit", saveSettings);
  document.getElementById("refreshDataCenter").onclick = loadDataCenter;
  document.getElementById("startDataSync").onclick = startSync;
  document.getElementById("dataProductCategory").onchange = renderProducts;
  document.getElementById("dataProductState").onchange = renderProducts;
  document.getElementById("dataProductSearch").oninput = renderProducts;
  document.getElementById("selectRecommendedProducts").onclick = selectRecommendedProducts;
}

function buildHourOptions() {
  const select = document.getElementById("dataUpdateHour");
  select.replaceChildren(...Array.from({ length: 24 }, (_, hour) => option(hour, `${String(hour).padStart(2, "0")}:00`)));
}

async function loadDataCenter() {
  try {
    dataCenterState.snapshot = await api("/api/data-center");
    renderDataCenter(dataCenterState.snapshot);
  } catch (error) {
    text("dataCenterSummary", error.message);
    text("dataState", "FAILED");
    toast(error.message, true);
  }
}

function renderDataCenter(snapshot) {
  const workspace = snapshot.workspace;
  const basis = snapshot.execution_basis;
  const sync = snapshot.sync || {};
  const hasWorkspace = Boolean(workspace);
  text("dataState", sync.state || (hasWorkspace ? "READY" : "FAILED"));
  text("dataCenterSummary", hasWorkspace ? `${formatNumber(workspace.rows)} 行 · ${workspace.symbols || "--"} 只股票 · 指纹 ${workspace.fingerprint.slice(0, 12)}` : snapshot.workspace_error || "数据工作区不可用");
  renderReadiness("research", hasWorkspace && workspace.price_research_ready, hasWorkspace ? "价格研究可用" : "不可用", hasWorkspace ? `${workspace.columns.length} 个字段` : "请检查目录");
  renderReadiness("proxy", Boolean(basis?.capital_ledger_proxy_ready), basis?.capital_ledger_proxy_ready ? "READY" : "BLOCKED", basis?.capital_ledger_proxy_ready ? "原始成交代理可用" : "缺少成交条件");
  renderReadiness("pit", Boolean(basis?.capital_ledger_ready), basis?.capital_ledger_ready ? "READY" : "BLOCKED", basis?.capital_ledger_ready ? "严格状态可用" : "需 PIT 市场状态");
  text("dataEnd", workspace?.last_trade_date || "--");
  text("dataRange", workspace ? `${workspace.first_trade_date} - ${workspace.last_trade_date}` : "--");
  text("dataRows", workspace ? formatNumber(workspace.rows) : "--");
  text("dataFiles", workspace ? `${workspace.files} 个 Parquet 分区` : "--");
  text("syncState", sync.state || "IDLE");
  text("syncUpdatedAt", formatDate(sync.updated_at));
  renderSettings(snapshot);
  renderWorkspace(snapshot);
  renderProducts();
  renderDownloader(snapshot.downloader);
  renderEvents(snapshot.recent_events || []);
  renderSyncProgress(sync.download_progress, Boolean(sync.running));
  document.getElementById("startDataSync").disabled = Boolean(sync.running) || !snapshot.credentials.tushare_token_configured || !snapshot.downloader.sync_cli_available;
  text("syncActionNote", sync.running ? "同步正在运行；研究与手动回测会在同步期间保持隔离。" : sync.migration_message || sync.message || "增量下载不复权截面与复权因子；历史覆盖完整后原子重建研究面板。");
  if (window.lucide) window.lucide.createIcons();
}

function renderSyncProgress(progress, running) {
  const panel = document.getElementById("syncProgressPanel");
  const tasks = progress?.tasks || [];
  panel.hidden = !tasks.length;
  if (!tasks.length) return;
  const active = progress.active_checkpoint || tasks[0];
  const percent = Number(active.checkpoint_percent || 0);
  const label = active.adjustment === "raw_plus_adj_factor" ? "原始行情与复权因子" : active.adjustment === "feature" ? `${active.dataset_id || "扩展"} 数据产品` : active.adjustment === "qfq" ? "前复权" : "未复权";
  text("syncProgressTitle", running ? `${label}数据同步中` : "最近可恢复断点");
  text("syncProgressPercent", `${percent.toFixed(2)}%`);
  document.getElementById("syncProgressBar").style.width = `${Math.max(0, Math.min(percent, 100))}%`;
  text("syncProgressDetail", active.last_message || "正在读取断点状态");
  document.getElementById("syncPhaseList").replaceChildren(...tasks.map(task => {
    const row = element("div", "sync-phase");
    const phase = task.adjustment === "raw_plus_adj_factor" ? "原始行情 + 复权因子" : task.adjustment === "feature" ? `扩展 · ${task.dataset_id || task.task_key}` : task.adjustment === "qfq" ? "qfq 研究源" : "未复权执行源";
    row.append(element("strong", "", phase), element("span", "", `${formatNumber(task.completed)} 完成 · ${formatNumber(task.failed)} 待重试`), element("small", "", `${Number(task.checkpoint_percent || 0).toFixed(2)}% · ${formatDate(task.updated_at)}`));
    return row;
  }));
}

function renderReadiness(id, ready, value, note) {
  text(`${id}Readiness`, value);
  text(`${id}ReadinessNote`, note);
  document.getElementById(`${id}Readiness`).closest("div").classList.toggle("ready", ready);
  document.getElementById(`${id}Readiness`).closest("div").classList.toggle("blocked", !ready);
}

function renderSettings(snapshot) {
  const settings = snapshot.schedule;
  const workspace = snapshot.workspace;
  document.getElementById("dataPath").value = workspace?.root_path || "";
  document.getElementById("marketDataRoot").value = snapshot.downloader.root_path || "";
  document.getElementById("autoUpdateEnabled").checked = settings.enabled;
  document.getElementById("dataUpdateHour").value = `${settings.hour}`;
  if (!document.getElementById("featureStartDate").value) document.getElementById("featureStartDate").value = workspace?.first_trade_date || "2020-01-01";
  if (!document.getElementById("featureEndDate").value) document.getElementById("featureEndDate").value = workspace?.last_trade_date || new Date().toISOString().slice(0, 10);
  const token = snapshot.credentials.tushare_token_configured;
  const tokenState = document.getElementById("tokenState");
  tokenState.classList.toggle("configured", token);
  tokenState.querySelector("span").textContent = token ? "Tushare Token 已配置（不会在页面回显）" : "尚未配置 Tushare Token，无法下载增量数据";
}

function renderProducts() {
  const catalog = dataCenterState.snapshot?.data_products;
  if (!catalog) return;
  const products = catalog.products || [];
  const categorySelect = document.getElementById("dataProductCategory");
  if (categorySelect.options.length === 1) {
    Object.keys(catalog.categories || {}).sort().forEach(category => categorySelect.append(option(category, categoryLabel(category))));
  }
  const category = categorySelect.value;
  const storageState = document.getElementById("dataProductState").value;
  const query = document.getElementById("dataProductSearch").value.trim().toLowerCase();
  const visible = products.filter(product => (category === "ALL" || product.category === category) && (storageState === "ALL" || product.storage_state === storageState) && (!query || [product.label, product.api_name, product.description, product.feature_family].join(" ").toLowerCase().includes(query)));
  text("selectedProductCount", `${products.filter(product => product.selected).length} ENABLED`);
  text("dataProductSummary", `${catalog.ready_count} 个已落盘 · ${catalog.panel_field_count} 个字段可供研究 · ${catalog.staged_field_count || 0} 个字段已暂存 · ${products.length} 个已登记接口`);
  const root = document.getElementById("dataProductRows");
  if (!visible.length) { root.replaceChildren(tableEmpty("没有符合筛选条件的数据产品", 10)); return; }
  root.replaceChildren(...visible.map(product => {
    const row = document.createElement("tr");
    const toggle = document.createElement("input"); toggle.type = "checkbox"; toggle.checked = Boolean(product.selected); toggle.disabled = !product.download_selectable; toggle.title = product.selection_lock_reason || (product.integration_state === "CATALOG" ? "下载到原始层；完成 PIT 治理前不会进入因子研究" : "启用数据产品"); toggle.setAttribute("aria-label", `${product.label}：${toggle.title}`);
    toggle.onchange = () => { product.selected = toggle.checked; text("selectedProductCount", `${products.filter(item => item.selected).length} ENABLED`); };
    row.append(cell(toggle));
    const identity = document.createElement("td"); identity.append(element("strong", "", product.label), element("code", "", product.api_name), element("small", "", product.description), element("small", `research-state ${String(product.research_state).toLowerCase()}`, researchStateLabel(product.research_state))); row.append(identity);
    row.append(cell(categoryLabel(product.category)), cell(product.feature_family), cell(`${product.grain}\n${product.cadence}`), cell(product.availability), cell(pitLabel(product.pit_policy)));
    const coverage = document.createElement("td"); coverage.append(element("strong", `product-state ${String(product.storage_state).toLowerCase()}`, product.storage_state), element("small", "", product.completed ? `${formatNumber(product.completed)} 期 · ${product.first_date || "--"} — ${product.last_date || "--"}` : "尚无本地分区")); row.append(coverage);
    row.append(cell(product.panel_ready ? `${product.panel_available_fields.length} 字段` : product.panel_fields.length ? "待发布" : "原始层"));
    const docs = document.createElement("a"); docs.href = product.documentation_url; docs.target = "_blank"; docs.rel = "noreferrer"; docs.className = "product-doc-link"; docs.title = "打开 Tushare 官方文档"; docs.append(document.createTextNode("官方")); row.append(cell(docs));
    return row;
  }));
}

function selectRecommendedProducts() {
  const recommended = new Set(["core_market", "daily_basic", "moneyflow", "stk_limit"]);
  (dataCenterState.snapshot?.data_products?.products || []).forEach(product => { product.selected = recommended.has(product.dataset_id); });
  renderProducts();
  toast("已选择首批生产推荐数据产品，保存配置后生效");
}

function researchStateLabel(value) {
  return ({
    RESEARCH_READY: "研究可用",
    WAITING_FOR_COVERAGE: "已接入 · 等待历史覆盖",
    RAW_DATA_ONLY: "原始层可用 · 尚未接入因子",
    RAW_DOWNLOAD_ONLY_REQUIRES_PIT_INTEGRATION: "可下载原始数据 · 研究接入待治理",
    CATALOG_ONLY: "仅登记 · 下载契约待实现",
  })[value] || value || "--";
}

function renderWorkspace(snapshot) {
  const workspace = snapshot.workspace, basis = snapshot.execution_basis;
  text("workspacePath", workspace?.panel_path || snapshot.workspace_error || "--");
  text("qualityState", workspace?.quality_passed === false ? "QUALITY FAILED" : workspace?.source_integrity_passed ? "INTEGRITY OK" : "CHECK REQUIRED");
  const stats = workspace ? [["股票数", workspace.symbols || "--"], ["数据行", formatNumber(workspace.rows)], ["文件数", workspace.files], ["起始日", workspace.first_trade_date || "--"], ["最新日", workspace.last_trade_date || "--"], ["质量报告", workspace.quality_passed === null ? "缺失" : workspace.quality_passed ? "通过" : "失败"]] : [["状态", "工作区不可用"]];
  document.getElementById("workspaceStats").replaceChildren(...stats.map(([label, value]) => statCard(label, value)));
  text("researchBasis", workspace?.price_research_ready ? "研究价格与活动字段完整，可用于截面因子研究。" : "研究面板缺少必要字段，当前不允许研究。 ");
  renderMessageList("researchWarnings", [...(workspace?.warnings || []), ...(workspace?.blockers || [])]);
  text("executionBasis", basis ? `研究价格：${basis.price_adjustment}；执行价格：${basis.execution_price_adjustment}；成交量：${basis.volume_unit}；成交额：${basis.amount_unit}。` : "--");
  renderMessageList("executionWarnings", basis ? [...basis.proxy_blockers, ...basis.blockers] : []);
}

function renderDownloader(downloader) {
  text("downloaderPath", downloader.root_path || "--");
  text("downloaderState", downloader.sync_cli_available && downloader.cross_sectional_available ? "READY" : "CHECK SETUP");
  const tasks = downloader.download_tasks || [];
  const root = document.getElementById("downloadTasks");
  if (!tasks.length) { root.replaceChildren(element("p", "empty-data-state", "未发现可用的 A 股下载任务。")); return; }
  root.replaceChildren(...tasks.map(task => {
    const item = element("article", "download-task");
    const label = task.adjustment === "raw_plus_adj_factor" ? "原始截面 + 复权因子" : task.adjustment === "qfq" ? "前复权遗留源" : task.adjustment === "none" ? "未复权遗留源" : "未知口径";
    item.append(element("strong", "", label), element("code", "", task.name), element("span", "", `${formatNumber(task.parquet_files)} 个文件 · ${formatDate(task.updated_at)}`));
    return item;
  }));
}

function renderEvents(events) {
  text("eventCount", `${events.length} 条`);
  const root = document.getElementById("dataEvents");
  if (!events.length) { root.replaceChildren(element("p", "empty-data-state", "尚无数据中心操作记录。")); return; }
  root.replaceChildren(...events.map(event => {
    const item = element("article", `data-event ${String(event.level || "INFO").toLowerCase()}`);
    const head = element("header", "");
    head.append(element("strong", "", event.title || event.event), element("time", "", formatDate(event.timestamp_utc)));
    item.append(head, element("p", "", event.message || "--"), element("code", "", event.event || "--"));
    return item;
  }));
}

async function saveSettings(event) {
  event.preventDefault();
  const button = document.getElementById("saveDataSettings");
  button.disabled = true;
  try {
    const selected = (dataCenterState.snapshot?.data_products?.products || []).filter(product => product.selected).map(product => product.dataset_id);
    dataCenterState.snapshot = await api("/api/data-center/settings", { method: "PUT", body: JSON.stringify({ data_path: document.getElementById("dataPath").value.trim(), market_data_root: document.getElementById("marketDataRoot").value.trim(), tushare_token: document.getElementById("tushareToken").value, data_auto_update_enabled: document.getElementById("autoUpdateEnabled").checked, data_update_hour: Number(document.getElementById("dataUpdateHour").value), data_product_ids: selected }) });
    document.getElementById("tushareToken").value = "";
    renderDataCenter(dataCenterState.snapshot);
    toast("数据配置已保存并记录审计日志");
  } catch (error) { toast(error.message, true); } finally { button.disabled = false; }
}

async function startSync() {
  const button = document.getElementById("startDataSync");
  button.disabled = true;
  try {
    const selected = (dataCenterState.snapshot?.data_products?.products || []).filter(product => product.selected).map(product => product.dataset_id);
    await api("/api/data-sync/start", { method: "POST", body: JSON.stringify({ dataset_ids: selected, start_date: document.getElementById("featureStartDate").value || null, end_date: document.getElementById("featureEndDate").value || null }) }); toast("多数据产品增量同步已启动"); await loadDataCenter();
  }
  catch (error) { toast(error.message, true); button.disabled = false; }
}

async function api(path, options = {}) { const response = await fetch(path, { ...options, credentials: "same-origin", headers: { "Content-Type": "application/json", ...(options.headers || {}) } }); if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `HTTP ${response.status}`); } return response.json(); }
function renderMessageList(id, messages) { document.getElementById(id).replaceChildren(...(messages.length ? messages.map(message => element("li", "", message)) : [element("li", "ok", "未发现阻断项") ])); }
function statCard(label, value) { const node = element("div", ""); node.append(element("span", "", label), element("strong", "", value)); return node; }
function option(value, label) { const node = document.createElement("option"); node.value = value; node.textContent = label; return node; }
function cell(content) { const node = document.createElement("td"); if (content instanceof Node) node.append(content); else String(content).split("\n").forEach((line, index) => { if (index) node.append(document.createElement("br")); node.append(document.createTextNode(line)); }); return node; }
function tableEmpty(message, columns) { const row = document.createElement("tr"), node = element("td", "table-empty", message); node.colSpan = columns; row.append(node); return row; }
function categoryLabel(value) { return ({ MARKET: "行情", VALUATION: "估值市值", FLOW: "资金流", TRADING_STATE: "交易状态", LEVERAGE: "杠杆", FUNDAMENTAL: "财务", OWNERSHIP: "股东行为", CORPORATE_ACTION: "公司行为", EVENT: "事件", CLASSIFICATION: "行业分类", BENCHMARK: "基准" })[value] || value; }
function pitLabel(value) { return ({ AFTER_CLOSE: "收盘后", SESSION_STATE: "会话状态", EVENT_TIME_REQUIRED: "需事件时点", REVISION_AWARE: "需版本治理", VERSIONED_REFERENCE: "版本化", EFFECTIVE_DATED: "按生效区间" })[value] || value; }
function element(tag, className = "", content = "") { const node = document.createElement(tag); if (className) node.className = className; if (content !== "") node.textContent = content; return node; }
function text(id, value) { document.getElementById(id).textContent = value; }
function formatNumber(value) { const number = Number(value); return Number.isFinite(number) ? number.toLocaleString() : "--"; }
function formatDate(value) { const date = new Date(value); return Number.isNaN(date.valueOf()) ? "--" : date.toLocaleString("zh-CN", { hour12: false }); }
let toastTimer; function toast(message, error = false) { const node = document.getElementById("toast"); node.textContent = message; node.style.background = error ? "#a93430" : "#202a3b"; node.classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => node.classList.remove("show"), 2600); }
