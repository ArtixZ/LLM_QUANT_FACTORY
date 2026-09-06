const state = { bootstrap: null, job: null, results: [], selectedFactorId: null, poll: null };

document.addEventListener("DOMContentLoaded", async () => {
  bindControls();
  if (window.lucide) window.lucide.createIcons();
  await loadBootstrap();
  state.poll = window.setInterval(pollActiveJob, 3000);
});

function bindControls() {
  document.getElementById("refreshBtn").onclick = loadSelectedJob;
  document.getElementById("newJobBtn").onclick = openJobDialog;
  document.getElementById("closeDialog").onclick = closeJobDialog;
  document.getElementById("cancelDialog").onclick = closeJobDialog;
  document.getElementById("jobForm").addEventListener("submit", createJob);
  document.getElementById("jobSelect").addEventListener("change", loadSelectedJob);
  document.getElementById("startJobBtn").onclick = () => commandJob("start");
  document.getElementById("pauseJobBtn").onclick = () => commandJob("pause");
  document.getElementById("factorSearch").addEventListener("input", renderResults);
  document.getElementById("resultFilter").addEventListener("change", renderResults);
}

async function loadBootstrap() {
  try {
    state.bootstrap = await api("/api/bootstrap");
    const service = state.bootstrap.service || {};
    document.title = `AutoAlpha ${service.page_title || "全因子批量回测"}`;
    text("pageHeading", service.page_title || "全因子批量回测");
    text("leaderboardTitle", service.leaderboard_title || "因子稳健性排行榜");
    text("batchDatabase", state.bootstrap.batch_database);
    populateJobSelect();
    if (state.bootstrap.jobs.length) await loadSelectedJob();
    else { renderEmpty(); openJobDialog(); }
  } catch (error) { toast(error.message, true); }
}

function populateJobSelect(preferredId = null) {
  const select = document.getElementById("jobSelect");
  const current = preferredId || select.value || state.bootstrap.jobs[0]?.job_id;
  select.replaceChildren(...state.bootstrap.jobs.map(job => option(job.job_id, `${job.name} · ${job.status}`)));
  if (current && state.bootstrap.jobs.some(job => job.job_id === current)) select.value = current;
}

async function loadSelectedJob() {
  const jobId = document.getElementById("jobSelect").value;
  if (!jobId) return;
  try {
    const data = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    state.job = data.job;
    state.results = data.results;
    state.events = data.events;
    renderJob(); renderResults(); renderEvents();
    if (state.selectedFactorId && state.results.some(item => item.factor_id === state.selectedFactorId && item.status === "SUCCESS")) await loadFactor(state.selectedFactorId);
  } catch (error) { toast(error.message, true); }
}

function renderJob() {
  const job = state.job, config = job.config;
  const service = state.bootstrap?.service || {};
  text("headerSummary", `${job.factor_count} 个冻结因子 · ${job.completed_count} 个已提交 · 独立端口 ${service.port || "--"}`);
  text("jobStatus", job.status); text("jobPhase", job.phase); text("jobId", job.job_id);
  text("systemJobId", job.system_job_id || "--");
  text("factorProgress", `${formatInt(job.completed_count)} / ${formatInt(job.factor_count)}`);
  text("failedCount", `失败 ${formatInt(job.failed_count)}`);
  text("progressPercent", formatPercent(job.progress));
  text("throughput", Number(job.factors_per_hour || 0).toFixed(2));
  text("etaText", job.eta_seconds ? `预计剩余 ${formatDuration(job.eta_seconds)}` : job.status === "COMPLETED" ? "全部结果已落盘" : "正在建立速度估计");
  text("monteCarloScale", formatInt(config.monte_carlo_samples));
  text("windowScale", `${config.window_months}月 / ${config.step_months}月`);
  text("dataEnd", config.end_date); text("dataFingerprint", `DATA ${(job.data_fingerprint || "--").slice(0, 12)}`);
  text("progressLabel", phaseLabel(job.phase));
  text("activeFactorHint", job.status === "RUNNING" ? `${config.workers} 个共享面板线程正在运行` : statusLabel(job.status));
  document.getElementById("progressBar").style.width = `${Math.max(0, Math.min(100, job.progress * 100))}%`;
  const realistic = config.protocol === "US_EQUITY_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1";
  const assumptions = realistic ? [
    ["回测区间", `${config.start_date} — ${config.end_date}`], ["执行时点", "收盘信号 · 下个计划开盘"], ["组合", `US 股票纯多 · ${formatPercent(config.gross_exposure)}`], ["调仓", "每周首个交易日"], ["选股", `目标最多 ${config.maximum_positions_per_side} 只`], ["费用", "配置费率 · 最低佣金 · 5bps滑点"], ["数据边界", "非PIT成交资格代理 · 不可生产晋级"],
  ] : [
    ["回测区间", `${config.start_date} — ${config.end_date}`], ["执行时点", "收盘信号 · 次日开盘"], ["组合", `多空 · ${formatPercent(config.gross_exposure)}`], ["持有期", `${config.holding_period_days} 日滚动`], ["选股", `双边前 ${formatPercent(config.selection_fraction)}`], ["压力测试", `持有 1/20日 · 参数 x0.5/x2`],
  ];
  document.getElementById("assumptionGrid").replaceChildren(...assumptions.map(([label, value]) => stat(label, value)));
  const active = job.active || ["RUNNING", "PAUSING"].includes(job.status);
  document.getElementById("startJobBtn").disabled = active || job.status === "COMPLETED";
  document.getElementById("pauseJobBtn").disabled = !active || job.status === "PAUSING";
  const error = document.getElementById("jobError"); error.hidden = !job.last_error; error.textContent = job.last_error || "";
}

function renderResults() {
  const body = document.getElementById("factorRows");
  const query = document.getElementById("factorSearch").value.trim().toLowerCase();
  const filter = document.getElementById("resultFilter").value;
  const rows = state.results.filter(item => {
    const corpus = `${item.factor_id} ${item.name} ${item.family}`.toLowerCase();
    return (!query || corpus.includes(query)) && (filter === "ALL" || item.status === filter);
  });
  text("leaderboardSummary", `${rows.length} / ${state.results.length} 个因子 · 按最差大窗口夏普优先排序`);
  if (!rows.length) { const row = document.createElement("tr"), cell = document.createElement("td"); cell.colSpan = 10; cell.className = "table-empty"; cell.textContent = "没有符合条件的因子结果"; row.append(cell); body.replaceChildren(row); return; }
  body.replaceChildren(...rows.map(resultRow));
}

function resultRow(item) {
  const row = document.createElement("tr");
  if (item.factor_id === state.selectedFactorId) row.classList.add("selected");
  const metrics = item.metrics || {}, mc = item.monte_carlo || {};
  const factor = element("div", "factor-cell"); factor.append(element("strong", "", item.name), element("code", "", `${item.factor_id} · ${item.family} · ${item.source_status || "UNKNOWN"}`));
  const computedStatus = item.status === "SUCCESS" ? "计算完成" : item.status;
  row.append(cell(item.rank || "--"), cell(factor), cell(element("span", `state ${item.status}`, computedStatus)), cell(formatMetric(long_only_metric(metrics, "sharpe_ratio"))), cell(formatPercent(long_only_metric(metrics, "simple_annual_return"))), cell(formatPercent(long_only_metric(metrics, "max_drawdown"))), cell(formatMetric(long_only_metric(metrics, "large_window_worst_sharpe"))), cell(formatPercent(metrics.robustness_pass_fraction)), cell(formatPercent(mc.probability_positive_annual_return)), cell(item.elapsed_seconds ? formatDuration(item.elapsed_seconds) : "--"));
  if (item.status === "SUCCESS") row.onclick = () => loadFactor(item.factor_id);
  else if (item.status === "FAILED") row.title = item.error || "因子执行失败";
  return row;
}

async function loadFactor(factorId) {
  if (!state.job) return;
  try {
    const detail = await api(`/api/jobs/${encodeURIComponent(state.job.job_id)}/factors/${encodeURIComponent(factorId)}`);
    state.selectedFactorId = factorId; renderResults(); renderFactorDetail(detail);
  } catch (error) { toast(error.message, true); }
}

function renderFactorDetail(detail) {
  const metrics = detail.metrics || {}, mc = detail.monte_carlo || {};
  const ranked = state.results.find(item => item.factor_id === detail.factor_id);
  text("detailRank", ranked?.rank ? `RANK #${ranked.rank}` : detail.status);
  text("detailName", detail.name); text("detailIdentity", `${detail.factor_id} · ${detail.family} · ITER ${detail.source_iteration || "--"} · ${detail.source_status || "UNKNOWN"}`);
  const values = [["纯多全期夏普", formatMetric(long_only_metric(metrics, "sharpe_ratio"))], ["纯多简单年化", formatPercent(long_only_metric(metrics, "simple_annual_return"))], ["纯多最大回撤", formatPercent(long_only_metric(metrics, "max_drawdown"))], ["纯多年化换手", formatMetric(long_only_metric(metrics, "annual_turnover"))], ["纯多最差窗口", formatMetric(long_only_metric(metrics, "large_window_worst_sharpe"))], ["窗口胜率", formatPercent(metrics.large_window_positive_fraction)], ["Rank IC", formatMetric(metrics.rank_ic_mean, 4)], ["MC正收益", formatPercent(mc.probability_positive_annual_return)]];
  if (metrics.engine_protocol === "US_EQUITY_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1") values.push(["累计交易成本", formatCurrency(metrics.total_transaction_cost_usd)], ["平均仓位", formatPercent(metrics.average_gross_exposure)], ["平均持股", formatMetric(metrics.average_positions, 1)], ["调仓次数", formatInt(metrics.rebalance_count)]);
  document.getElementById("detailMetrics").replaceChildren(...values.map(([label, value]) => stat(label, value)));
  text("curveCaption", `${metrics.backtest_start || "--"} — ${metrics.backtest_end || "--"}`);
  text("mcCaption", `${formatInt(mc.samples || 0)} 次 · ${mc.block_size_sessions || "--"} 日移动块`);
  drawLineChart(document.getElementById("equityChart"), detail.curve.map(item => item.equity), "#245eea");
  drawHistogram(document.getElementById("mcChart"), detail.monte_carlo_histogram?.sharpe_ratio, "#08785a");
  renderWindowTable(detail.windows || []); renderRobustnessTable(detail.robustness || []);
}

function long_only_metric(metrics, key) {
  return metrics[`long_only_${key}`] ?? metrics[key];
}

function renderWindowTable(windows) {
  const node = document.getElementById("windowTable");
  if (!windows.length) { node.replaceChildren(element("div", "table-empty", "暂无窗口结果")); return; }
  node.replaceChildren(...windows.map(item => miniRow([`${item.window_id} · ${item.period_start.slice(0, 7)}—${item.period_end.slice(0, 7)}`, formatMetric(item.metrics.sharpe_ratio), formatPercent(item.metrics.simple_annual_return), formatPercent(item.metrics.max_drawdown), formatMetric(item.metrics.annual_turnover)])));
}

function renderRobustnessTable(items) {
  const node = document.getElementById("robustnessTable");
  if (!items.length) { node.replaceChildren(element("div", "table-empty", "暂无压力测试结果")); return; }
  node.replaceChildren(...items.map(item => miniRow([`${item.test_type} · ${item.variant}`, item.metrics ? formatMetric(item.metrics.sharpe_ratio) : "失败", item.metrics ? formatPercent(item.metrics.simple_annual_return) : "--", item.metrics ? formatPercent(item.metrics.max_drawdown) : "--", item.error || "OK"])));
}

function renderEvents() {
  const list = document.getElementById("eventList"), events = state.events || [];
  if (!events.length) { list.replaceChildren(element("div", "table-empty", "任务启动后显示断点提交记录")); return; }
  list.replaceChildren(...events.map(item => { const node = element("article", "event-item"); node.append(element("code", "", item.event), element("span", `state ${item.level === "ERROR" ? "FAILED" : "SUCCESS"}`, item.level), element("div", "", item.message), element("time", "", formatDate(item.created_at))); return node; }));
}

function openJobDialog() {
  const defaults = state.bootstrap?.defaults || {};
  document.getElementById("jobName").value = defaults.name || "2020-2026 全因子大规模稳健性回测";
  document.getElementById("jobDataPath").value = defaults.data_path || "";
  document.getElementById("jobStart").value = defaults.start_date || "2020-01-01";
  document.getElementById("jobEnd").value = defaults.end_date || "";
  document.getElementById("jobWorkers").value = defaults.workers || 4;
  document.getElementById("jobWindow").value = defaults.window_months || 36;
  document.getElementById("jobStep").value = defaults.step_months || 12;
  document.getElementById("jobMcSamples").value = defaults.monte_carlo_samples || 10000;
  document.getElementById("jobMcBlock").value = defaults.monte_carlo_block_days || 20;
  document.getElementById("jobDialog").showModal();
}
function closeJobDialog() { document.getElementById("jobDialog").close(); }

async function createJob(event) {
  event.preventDefault();
  const body = { name: value("jobName"), data_path: value("jobDataPath"), start_date: value("jobStart"), end_date: value("jobEnd"), workers: Number(value("jobWorkers")), holding_period_days: Number(value("jobHolding")), window_months: Number(value("jobWindow")), step_months: Number(value("jobStep")), monte_carlo_samples: Number(value("jobMcSamples")), monte_carlo_block_days: Number(value("jobMcBlock")), parameter_multipliers: [0.5, 2.0], holding_period_tests: [1, 20] };
  try {
    const job = await api("/api/jobs", { method:"POST", body:JSON.stringify(body) });
    closeJobDialog(); await loadBootstrap(); document.getElementById("jobSelect").value = job.job_id; await loadSelectedJob(); toast(`已冻结 ${job.factor_count} 个因子`);
  } catch (error) { toast(error.message, true); }
}

async function commandJob(action) {
  if (!state.job) return;
  try { await api(`/api/jobs/${encodeURIComponent(state.job.job_id)}/${action}`, {method:"POST"}); await loadSelectedJob(); toast(action === "start" ? "批量任务已启动" : "暂停请求已登记"); }
  catch (error) { toast(error.message, true); }
}

async function pollActiveJob() { if (!document.hidden && state.job && ["RUNNING","PAUSING"].includes(state.job.status)) await loadSelectedJob(); }

function drawLineChart(canvas, values, color) {
  const context = canvas.getContext("2d"); context.clearRect(0,0,canvas.width,canvas.height); if (!values.length) return;
  const finite = values.filter(Number.isFinite), min = Math.min(...finite), max = Math.max(...finite), range = max-min || 1, pad = 18;
  context.strokeStyle="#e5e9ef"; context.lineWidth=1; for (let i=0;i<4;i++){const y=pad+(canvas.height-pad*2)*i/3; context.beginPath();context.moveTo(pad,y);context.lineTo(canvas.width-pad,y);context.stroke();}
  context.strokeStyle=color; context.lineWidth=2; context.beginPath(); values.forEach((item,index)=>{const x=pad+(canvas.width-pad*2)*index/Math.max(1,values.length-1),y=canvas.height-pad-(item-min)/range*(canvas.height-pad*2); if(index===0)context.moveTo(x,y);else context.lineTo(x,y);}); context.stroke();
}
function drawHistogram(canvas, histogram, color) {
  const context=canvas.getContext("2d");context.clearRect(0,0,canvas.width,canvas.height);if(!histogram?.counts?.length)return;const counts=histogram.counts,max=Math.max(...counts)||1,pad=16,width=(canvas.width-pad*2)/counts.length;context.fillStyle=color;counts.forEach((count,index)=>{const height=(canvas.height-pad*2)*count/max;context.fillRect(pad+index*width,canvas.height-pad-height,Math.max(1,width-1),height);});context.strokeStyle="#d9e0e9";context.beginPath();context.moveTo(pad,canvas.height-pad);context.lineTo(canvas.width-pad,canvas.height-pad);context.stroke();
}

function renderEmpty(){ text("headerSummary","尚未创建批量任务"); document.getElementById("factorRows").replaceChildren(); }
function miniRow(values){const node=element("div","mini-row");values.forEach((item,index)=>node.append(element(index===0?"strong":"span","",String(item))));return node;}
function stat(label,value){const node=element("div");node.append(element("span","",label),element("strong","",String(value)));return node;}
function cell(content){const node=document.createElement("td");node.append(content instanceof Node?content:document.createTextNode(String(content)));return node;}
function element(tag,className="",content=""){const node=document.createElement(tag);if(className)node.className=className;if(content!=="")node.textContent=content;return node;}
function option(value,label){const node=document.createElement("option");node.value=value;node.textContent=label;return node;}
function text(id,value){document.getElementById(id).textContent=value;}
function value(id){return document.getElementById(id).value.trim();}
function formatMetric(value,digits=2){const number=Number(value);return Number.isFinite(number)?number.toFixed(digits):"--";}
function formatPercent(value){const number=Number(value);return Number.isFinite(number)?`${(number*100).toFixed(2)}%`:"--";}
function formatInt(value){const number=Number(value);return Number.isFinite(number)?Math.round(number).toLocaleString():"--";}
function formatCurrency(value){const number=Number(value);return Number.isFinite(number)?`¥${Math.round(number).toLocaleString()}`:"--";}
function formatDuration(seconds){const value=Math.max(0,Number(seconds)||0);if(value<60)return`${value.toFixed(1)}秒`;if(value<3600)return`${Math.floor(value/60)}分${Math.round(value%60)}秒`;return`${Math.floor(value/3600)}时${Math.round((value%3600)/60)}分`;}
function formatDate(value){const parsed=new Date(value);return Number.isNaN(parsed.valueOf())?"--":parsed.toLocaleString("zh-CN",{hour12:false});}
function statusLabel(value){return({READY:"等待启动",PAUSED:"已暂停，可断点续跑",COMPLETED:"全部完成",FAILED:"任务失败"})[value]||value;}
function phaseLabel(value){return({WAITING:"等待任务启动",DATA_LOADING:"加载共享行情面板",FACTOR_EVALUATION:"全因子向量回测与蒙特卡洛",MULTIPLE_TESTING:"全库多重检验修正",CHECKPOINTED:"断点已完整保存",COMPLETED:"全量结果已持久化",FAILED:"批量任务失败"})[value]||value;}
async function api(path,options={}){const response=await fetch(path,{...options,headers:{"Content-Type":"application/json",...(options.headers||{})}});if(!response.ok){const body=await response.json().catch(()=>({}));throw new Error(body.detail||`HTTP ${response.status}`);}return response.json();}
let toastTimer;function toast(message,error=false){const node=document.getElementById("toast");node.textContent=message;node.style.background=error?"#a93430":"#202a3b";node.classList.add("show");clearTimeout(toastTimer);toastTimer=setTimeout(()=>node.classList.remove("show"),2800);}
