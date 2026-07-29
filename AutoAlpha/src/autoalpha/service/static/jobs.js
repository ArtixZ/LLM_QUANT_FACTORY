const jobState = { snapshot: null, focusJobId: null };
const $ = id => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  const params = new URLSearchParams(window.location.search);
  if (params.get("queue")) $("queueFilter").value = params.get("queue");
  if (params.get("status")) $("statusFilter").value = params.get("status");
  if (params.get("job")) jobState.focusJobId = params.get("job");
  $("refreshJobs").onclick = () => loadJobs();
  $("recoverJobs").onclick = () => recoverJobs();
  $("runNextJob").onclick = () => runNextJob();
  $("closeJobLog").onclick = () => {
    $("jobLogPanel").hidden = true;
  };
  $("statusFilter").onchange = () => loadJobs();
  $("queueFilter").onchange = () => loadJobs();
  $("jobForm").onsubmit = event => {
    event.preventDefault();
    enqueueJob();
  };
  loadJobs();
  setInterval(loadJobs, 15000);
});

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json();
}

async function loadJobs() {
  try {
    const params = new URLSearchParams();
    const status = $("statusFilter").value;
    const queue = $("queueFilter").value.trim();
    if (status) params.set("status", status);
    if (queue) params.set("queue", queue);
    const snapshot = await api(`/api/jobs${params.toString() ? `?${params}` : ""}`);
    jobState.snapshot = snapshot;
    renderSnapshot(snapshot);
  } catch (error) {
    toast(error.message, true);
  }
}

function renderSnapshot(snapshot) {
  const scheduler = snapshot.scheduler || {};
  $("schedulerState").textContent = scheduler.status || "--";
  $("schedulerState").className = `state-pill small ${scheduler.status === "running" ? "ok" : ""}`;
  $("schedulerStatus").textContent = scheduler.status || "--";
  $("schedulerDetail").textContent = scheduler.enabled ? `queue=${scheduler.queue || "system"} · poll=${scheduler.poll_seconds?.idle || "--"}s` : "disabled";
  const counts = summarize(snapshot.jobs || []);
  $("queuedJobs").textContent = counts.QUEUED || 0;
  $("runningJobs").textContent = counts.RUNNING || 0;
  $("failedJobs").textContent = (counts.FAILED || 0) + (counts.BLOCKED_UNSUPPORTED || 0);
  $("resourceGroups").textContent = new Set((snapshot.jobs || []).map(job => job.resource_group)).size;
  $("jobSummary").textContent = `${snapshot.jobs.length} 个作业 · ${scheduler.status || "UNKNOWN"} · ${snapshot.resource_policy?.claim_quota || "--"}`;
  $("resourcePolicy").textContent = `${snapshot.resource_policy?.single_writer_group || "--"} · ${snapshot.resource_policy?.retry_policy || "--"}`;
  renderJobTypes(snapshot.resource_policy?.supported_system_job_types || []);
  renderResourceMatrix(snapshot.summary?.resource_utilization || []);
  renderSnapshotPolicy(snapshot.snapshot_policy || {});
  renderJobs(snapshot.jobs || []);
  if (window.lucide) window.lucide.createIcons();
}

function renderJobTypes(types) {
  const select = $("jobType");
  const current = select.value;
  if (select.options.length === types.length && [...select.options].every((option, index) => option.value === types[index])) return;
  select.replaceChildren(...types.map(type => option(type, type)));
  if (types.includes(current)) select.value = current;
}

function renderJobs(jobs) {
  $("jobTableSummary").textContent = `${jobs.length} 个作业 · 按优先级和更新时间展示`;
  const rows = $("jobRows");
  if (!jobs.length) {
    rows.innerHTML = `<tr><td colspan="9" class="empty-table">暂无作业</td></tr>`;
    return;
  }
  rows.innerHTML = jobs.map(job => {
    const progress = progressText(job);
    const lease = job.lease_owner ? `${esc(job.lease_owner)}<br><small>${esc(job.lease_expires_at || "--")}</small>` : "--";
    const result = job.error || compactJson(job.result) || compactJson(job.checkpoint) || "--";
    const logs = (jobState.snapshot?.recent_logs || {})[job.job_id] || [];
    const actions = jobActions(job);
    const focused = jobState.focusJobId && job.job_id === jobState.focusJobId ? " focused-row" : "";
    return `<tr class="${focused}">
      <td><strong>${esc(job.job_id)}</strong><br><small>priority ${job.priority} · attempts ${job.attempts}/${job.max_attempts}</small></td>
      <td>${esc(job.job_type)}</td>
      <td><span class="state ${esc(job.status)}">${esc(job.status)}</span></td>
      <td>${esc(job.queue)}<br><small>${esc(job.resource_group)} · max ${job.max_workers}</small></td>
      <td>${progress}</td>
      <td>${lease}</td>
      <td class="compact-cell" title="${esc(result)}">${esc(result)}${recentLogHtml(logs)}</td>
      <td>${actions}</td>
      <td>${formatTime(job.updated_at)}</td>
    </tr>`;
  }).join("");
  rows.querySelectorAll("[data-job-command]").forEach(button => {
    button.addEventListener("click", () => commandJob(button.dataset.jobId, button.dataset.jobCommand));
  });
  rows.querySelectorAll("[data-job-logs]").forEach(button => {
    button.addEventListener("click", () => showJobLogs(button.dataset.jobId));
  });
}

async function enqueueJob() {
  try {
    const payloadText = $("jobPayload").value.trim() || "{}";
    const body = {
      queue: $("jobQueue").value.trim() || "system",
      job_type: $("jobType").value,
      payload: JSON.parse(payloadText),
      resource_group: $("resourceGroup").value.trim() || "default",
      priority: Number($("jobPriority").value || 100),
      max_workers: Number($("jobMaxWorkers").value || 1),
      max_attempts: Number($("jobMaxAttempts").value || 3),
    };
    await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
    toast("作业已入队");
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  }
}

async function recoverJobs() {
  try {
    const queue = $("queueFilter").value.trim();
    const suffix = queue ? `?queue=${encodeURIComponent(queue)}` : "";
    const result = await api(`/api/jobs/recover${suffix}`, { method: "POST" });
    toast(`已恢复 ${result.recovered_expired_jobs} 个过期租约`);
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  }
}

async function runNextJob() {
  try {
    const queue = $("queueFilter").value.trim() || $("jobQueue").value.trim() || "system";
    const body = { queue };
    const maxQueue = Number($("runMaxQueue").value || 0);
    const maxGlobal = Number($("runMaxGlobal").value || 0);
    if (maxQueue > 0) body.max_queue_running = maxQueue;
    if (maxGlobal > 0) body.max_global_running = maxGlobal;
    const result = await api("/api/jobs/run-next", {
      method: "POST",
      body: JSON.stringify(body),
    });
    toast(result.claimed ? `已执行 ${result.job.job_id}` : "没有可领取作业");
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  }
}

function renderResourceMatrix(resources) {
  const root = $("resourceMatrix");
  const saturated = resources.filter(item => item.saturated);
  $("resourceMatrixSummary").textContent = `${resources.length} 个资源组 · ${saturated.length} 个满载排队`;
  if (!resources.length) {
    root.innerHTML = `<div class="empty-table">暂无资源组</div>`;
    return;
  }
  root.innerHTML = resources.map(resource => {
    const pct = Math.max(0, Math.min(100, Number(resource.utilization || 0) * 100));
    const state = resource.saturated ? "Saturated" : Number(resource.running || 0) ? "Running" : "Idle";
    const stateClass = resource.saturated ? "warn" : Number(resource.running || 0) ? "ok" : "";
    return `<article class="resource-card ${resource.saturated ? "saturated" : ""}">
      <div class="resource-card-head">
        <strong>${esc(resource.resource_group)}</strong>
        <span class="state-pill small ${stateClass}">${state}</span>
      </div>
      <small>${esc(resource.queue)} · capacity ${resource.capacity}</small>
      <div class="mini-progress"><i style="width:${pct}%"></i></div>
      <div class="resource-card-grid">
        <span><b>${resource.running}</b><small>运行</small></span>
        <span><b>${resource.queued}</b><small>排队</small></span>
        <span><b>${resource.idle_capacity}</b><small>空闲</small></span>
        <span><b>${resource.failed}</b><small>失败</small></span>
      </div>
    </article>`;
  }).join("");
}

function renderSnapshotPolicy(policy) {
  const rows = policy.rows || [];
  const missing = (policy.missing_keys || []).length;
  const stale = (policy.stale_keys || []).length;
  $("snapshotPolicySummary").textContent = `${policy.status || "UNKNOWN"} · ${rows.length} 个关键视图 · ${missing} 缺失 · ${stale} 过期`;
  const root = $("snapshotPolicyRows");
  if (!rows.length) {
    root.innerHTML = `<div class="empty-table">暂无快照策略</div>`;
    return;
  }
  root.innerHTML = rows.map(row => {
    const state = row.state || "UNKNOWN";
    const stateClass = state === "FRESH" ? "ok" : state === "STALE" || state === "MISSING" ? "warn" : "";
    const button = row.refresh_job_type
      ? `<button class="button tiny" type="button" data-prefill-job="${esc(row.refresh_job_type)}" data-resource-group="${esc(row.resource_group || "sqlite-writer")}"><i data-lucide="plus"></i>预填作业</button>`
      : `<span class="muted">显式接口</span>`;
    return `<article class="resource-card ${state !== "FRESH" ? "saturated" : ""}">
      <div class="resource-card-head">
        <strong>${esc(row.label || row.key)}</strong>
        <span class="state-pill small ${stateClass}">${esc(state)}</span>
      </div>
      <small>${esc(row.key)} · ${esc(row.source || "--")}</small>
      <div class="resource-card-grid two">
        <span><b>${row.present ? "是" : "否"}</b><small>存在</small></span>
        <span><b>${esc(row.refresh_job_type || "--")}</b><small>作业类型</small></span>
      </div>
      <small>${esc(row.updated_at || "未刷新")} · ${esc(row.expires_at || "无 TTL")}</small>
      <div class="button-row compact-actions">${button}</div>
    </article>`;
  }).join("");
  root.querySelectorAll("[data-prefill-job]").forEach(button => {
    button.addEventListener("click", () => {
      $("jobType").value = button.dataset.prefillJob;
      $("resourceGroup").value = button.dataset.resourceGroup || "sqlite-writer";
      $("jobQueue").value = "system";
      $("jobPayload").value = "{}";
      $("jobPriority").value = "40";
      toast(`已预填 ${button.dataset.prefillJob}`);
    });
  });
}

function jobActions(job) {
  const status = String(job.status || "");
  const buttons = [];
  if (status === "QUEUED") {
    buttons.push(actionButton(job.job_id, "pause", "暂停", "pause"));
    buttons.push(actionButton(job.job_id, "cancel", "取消", "x"));
  } else if (status === "PAUSED") {
    buttons.push(actionButton(job.job_id, "resume", "恢复", "play"));
    buttons.push(actionButton(job.job_id, "cancel", "取消", "x"));
  } else if (status === "RUNNING") {
    buttons.push(actionButton(job.job_id, "pause", "请求暂停", "pause"));
    buttons.push(actionButton(job.job_id, "cancel", "请求取消", "x"));
  } else if (status === "PAUSE_REQUESTED") {
    buttons.push(actionButton(job.job_id, "cancel", "改为取消", "x"));
  }
  buttons.push(`<button class="button tiny" type="button" data-job-logs="${esc(job.job_id)}"><i data-lucide="list"></i>日志</button>`);
  if (!buttons.length) return `<span class="muted">--</span>`;
  return `<div class="button-row compact-actions">${buttons.join("")}</div>`;
}

function actionButton(jobId, command, label, icon) {
  return `<button class="button tiny" type="button" data-job-id="${esc(jobId)}" data-job-command="${esc(command)}"><i data-lucide="${esc(icon)}"></i>${esc(label)}</button>`;
}

async function commandJob(jobId, command) {
  try {
    const result = await api(`/api/jobs/${encodeURIComponent(jobId)}/${command}`, {
      method: "POST",
      body: JSON.stringify({
        actor: "local-operator",
        reason: `ui ${command}`,
      }),
    });
    toast(`${result.job_id} -> ${result.status}`);
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  }
}

async function showJobLogs(jobId) {
  try {
    const result = await api(`/api/jobs/${encodeURIComponent(jobId)}/logs?limit=80`);
    $("jobLogTitle").textContent = `${result.job.job_id} 日志`;
    $("jobLogSubtitle").textContent = `${result.job.job_type} · ${result.job.status} · ${result.logs.length} 条`;
    $("jobLogRows").innerHTML = result.logs.length
      ? result.logs.map(log => jobLogRow(log)).join("")
      : `<div class="empty-table">暂无日志</div>`;
    $("jobLogPanel").hidden = false;
    if (window.lucide) window.lucide.createIcons();
  } catch (error) {
    toast(error.message, true);
  }
}

function recentLogHtml(logs) {
  if (!logs.length) return "";
  const latest = logs[0];
  return `<div class="recent-job-log">${esc(latest.event)} · ${esc(latest.message)}</div>`;
}

function jobLogRow(log) {
  return `<div class="job-log-row">
    <span class="state ${esc(log.level)}">${esc(log.level)}</span>
    <strong>${esc(log.event)}</strong>
    <span>${esc(log.message)}</span>
    <small>${formatTime(log.timestamp_utc)}</small>
    <code>${esc(compactJson(log.payload) || "{}")}</code>
  </div>`;
}

function summarize(jobs) {
  return jobs.reduce((acc, job) => {
    acc[job.status] = (acc[job.status] || 0) + 1;
    return acc;
  }, {});
}

function progressText(job) {
  const total = Number(job.progress_total || 0);
  const current = Number(job.progress_current || 0);
  if (!total) return current ? String(current) : "--";
  const pct = Math.max(0, Math.min(100, current / total * 100));
  return `<div class="mini-progress"><i style="width:${pct}%"></i></div><small>${current}/${total}</small>`;
}

function compactJson(value) {
  if (!value || Object.keys(value).length === 0) return "";
  return JSON.stringify(value).slice(0, 180);
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

function formatTime(value) {
  if (!value) return "--";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = error ? "error show" : "show";
  setTimeout(() => node.classList.remove("show"), 2600);
}
