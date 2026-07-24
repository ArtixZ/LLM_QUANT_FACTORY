const screenerState = { library: null, selected: new Map(), query: null, result: null, presets: [] };

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  document.getElementById("screenerForm").addEventListener("submit", runScreen);
  document.getElementById("factorSearch").addEventListener("input", event => {
    screenerState.query = event.target.value.trim().toLowerCase();
    renderPicker();
  });
  document.getElementById("equalWeights").onclick = () => {
    screenerState.selected.forEach((_, factorId) => screenerState.selected.set(factorId, 1));
    renderWeights();
  };
  document.getElementById("saveScreenPreset").onclick = savePreset;
  document.getElementById("deleteScreenPreset").onclick = deletePreset;
  document.getElementById("savedScreenPreset").onchange = applyPreset;
  await loadLibrary();
});

async function loadLibrary() {
  try {
    screenerState.library = await api("/api/factors");
    const data = screenerState.library.data;
    const input = document.getElementById("asOfDate");
    input.min = data.first_trade_date;
    input.max = data.last_trade_date;
    input.value = data.last_trade_date;
    text("screenerDataRange", `${data.first_trade_date} - ${data.last_trade_date}`);
    text("screenerSummary", `${screenerState.library.summary.factor_count} 个因子 · 收盘后横截面评分`);
    const queryParameters = new URLSearchParams(window.location.search);
    const queryFactors = queryParameters.get("factors");
    if (queryFactors) {
      const available = new Set(screenerState.library.factors.map(factor => factor.factor_id));
      const queryWeights = (queryParameters.get("weights") || "").split(",").map(Number);
      queryFactors.split(",").slice(0, 12).forEach((factorId, index) => {
        if (!available.has(factorId)) return;
        const weight = queryWeights[index];
        screenerState.selected.set(factorId, Number.isFinite(weight) && weight > 0 ? weight : 1);
      });
    }
    const requestedDate = queryParameters.get("as_of_date");
    if (requestedDate && requestedDate >= input.min && requestedDate <= input.max) input.value = requestedDate;
    const requestedCount = Number(queryParameters.get("selection_count"));
    if (Number.isInteger(requestedCount) && requestedCount >= 1 && requestedCount <= 500) {
      document.getElementById("selectionCount").value = requestedCount;
    }
    const requestedSide = queryParameters.get("selection_side");
    if (["TOP", "BOTTOM"].includes(requestedSide)) document.getElementById("selectionSide").value = requestedSide;
    await loadPresets();
    renderPicker();
    if (queryParameters.get("run") === "1" && screenerState.selected.size) {
      await runScreen({ preventDefault() {} });
    }
  } catch (error) {
    toast(error.message, true);
  }
}

async function loadPresets(selectedId = "") {
  const data = await api("/api/favorites?entity_type=screener_preset");
  screenerState.presets = data.favorites || [];
  const select = document.getElementById("savedScreenPreset");
  const current = selectedId || select.value;
  select.replaceChildren(option("", "选择收藏方案"), ...screenerState.presets.map(item => option(item.entity_id, item.label || "未命名选股方案")));
  select.value = screenerState.presets.some(item => item.entity_id === current) ? current : "";
  document.getElementById("deleteScreenPreset").disabled = !select.value;
}

async function savePreset() {
  const factors = selectedFactors();
  if (!factors.length) { toast("请先选择至少一个因子", true); return; }
  const name = document.getElementById("screenPresetName").value.trim() || factors.slice(0, 2).map(item => item.name).join(" + ");
  const entityId = document.getElementById("savedScreenPreset").value || (crypto.randomUUID ? crypto.randomUUID() : `screen-${Date.now()}`);
  const context = {
    factor_ids: factors.map(item => item.factor_id),
    weights: factors.map(item => Number(screenerState.selected.get(item.factor_id))),
    as_of_date: document.getElementById("asOfDate").value,
    selection_count: Number(document.getElementById("selectionCount").value),
    selection_side: document.getElementById("selectionSide").value,
  };
  try {
    await api(`/api/favorites/screener_preset/${encodeURIComponent(entityId)}`, { method: "PUT", body: JSON.stringify({ favorite: true, label: name, context }) });
    document.getElementById("screenPresetName").value = name;
    await loadPresets(entityId);
    toast("选股方案已收藏");
  } catch (error) { toast(error.message, true); }
}

function applyPreset() {
  const id = document.getElementById("savedScreenPreset").value;
  document.getElementById("deleteScreenPreset").disabled = !id;
  const preset = screenerState.presets.find(item => item.entity_id === id);
  if (!preset) return;
  const context = preset.context || {};
  const available = new Set((screenerState.library?.factors || []).map(item => item.factor_id));
  screenerState.selected.clear();
  (context.factor_ids || []).forEach((factorId, index) => {
    if (available.has(factorId)) screenerState.selected.set(factorId, Number(context.weights?.[index]) || 1);
  });
  if (context.as_of_date) document.getElementById("asOfDate").value = context.as_of_date;
  if (context.selection_count) document.getElementById("selectionCount").value = context.selection_count;
  if (context.selection_side) document.getElementById("selectionSide").value = context.selection_side;
  document.getElementById("screenPresetName").value = preset.label || "";
  renderPicker();
  toast(`已载入“${preset.label || "收藏方案"}”`);
}

async function deletePreset() {
  const id = document.getElementById("savedScreenPreset").value;
  const preset = screenerState.presets.find(item => item.entity_id === id);
  if (!preset) return;
  try {
    await api(`/api/favorites/screener_preset/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify({ favorite: false, label: preset.label || "" }) });
    document.getElementById("screenPresetName").value = "";
    await loadPresets();
    toast("选股方案已删除");
  } catch (error) { toast(error.message, true); }
}

function filteredFactors() {
  if (!screenerState.library) return [];
  return screenerState.library.factors.filter(factor => {
    if (!screenerState.query) return true;
    return [factor.factor_id, factor.name, factor.family, factor.category]
      .join(" ").toLowerCase().includes(screenerState.query);
  });
}

function renderPicker() {
  const picker = document.getElementById("factorPicker");
  const factors = filteredFactors();
  picker.replaceChildren(...factors.map(factor => {
    const row = element("label", "picker-row");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = screenerState.selected.has(factor.factor_id);
    checkbox.onchange = () => {
      if (checkbox.checked) {
        if (screenerState.selected.size >= 12) {
          checkbox.checked = false;
          toast("最多选择 12 个因子", true);
          return;
        }
        screenerState.selected.set(factor.factor_id, 1);
      } else {
        screenerState.selected.delete(factor.factor_id);
      }
      renderPicker();
      renderWeights();
    };
    const identity = element("span", "picker-identity");
    identity.append(element("strong", "", factor.name), element("small", "", `${factor.category} · ${factor.factor_id}`));
    row.append(checkbox, identity, element("span", "picker-score", `#${factor.rank}`));
    return row;
  }));
  text("selectedCount", `${screenerState.selected.size} / 12`);
  renderWeights();
}

function renderWeights() {
  const container = document.getElementById("selectedFactors");
  const editor = document.getElementById("weightEditor");
  const factors = selectedFactors();
  container.hidden = factors.length === 0;
  editor.replaceChildren(...factors.map(factor => {
    const row = element("div", "screener-weight-row");
    const name = element("span", "", factor.name);
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0.0001";
    input.step = "0.1";
    input.value = screenerState.selected.get(factor.factor_id);
    input.onchange = () => {
      const value = Number(input.value);
      if (!Number.isFinite(value) || value <= 0) {
        input.value = screenerState.selected.get(factor.factor_id);
        toast("权重必须为正数", true);
        return;
      }
      screenerState.selected.set(factor.factor_id, value);
    };
    row.append(name, input);
    return row;
  }));
}

function selectedFactors() {
  return screenerState.library?.factors.filter(factor => screenerState.selected.has(factor.factor_id)) || [];
}

async function runScreen(event) {
  event.preventDefault();
  const factors = selectedFactors();
  if (!factors.length) {
    toast("请至少选择一个因子", true);
    return;
  }
  const button = document.getElementById("runScreener");
  button.disabled = true;
  button.innerHTML = '<i data-lucide="loader-circle"></i>正在计算';
  if (window.lucide) window.lucide.createIcons();
  text("resultStatus", "RUNNING");
  try {
    const result = await api("/api/screener", {
      method: "POST",
      body: JSON.stringify({
        factor_ids: factors.map(factor => factor.factor_id),
        weights: factors.map(factor => Number(screenerState.selected.get(factor.factor_id))),
        as_of_date: document.getElementById("asOfDate").value,
        selection_count: Number(document.getElementById("selectionCount").value),
        selection_side: document.getElementById("selectionSide").value,
      }),
    });
    screenerState.result = result;
    renderResult();
    toast("选股快照已生成");
  } catch (error) {
    text("resultStatus", "FAILED");
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.innerHTML = '<i data-lucide="scan-search"></i>快速选股';
    if (window.lucide) window.lucide.createIcons();
  }
}

function renderResult() {
  const result = screenerState.result;
  text("resultStatus", "COMPLETED");
  text("signalDate", result.as_of_date);
  text("evaluatedFactors", result.evaluated_factors.length);
  text("universeSize", result.universe_size.toLocaleString());
  text("resultCount", result.rows.length);
  text("resultSummary", `${result.requested_as_of_date === result.as_of_date ? "指定交易日" : `回退至最近交易日 ${result.as_of_date}`} · ${result.selection_side === "TOP" ? "综合分最高" : "综合分最低"}`);
  const summary = document.getElementById("factorSummary");
  summary.replaceChildren(...result.evaluated_factors.map(item => element("span", "tag", `${item.name} ${percent(item.normalized_weight)}`)));
  const head = document.getElementById("resultTableHead");
  const body = document.getElementById("resultTableBody");
  const columns = ["排名", "代码", "名称", "综合分", "分位", "未复权收盘", "研究收盘", "成交额（元）", ...result.evaluated_factors.map(item => item.name)];
  const headerRow = document.createElement("tr");
  columns.forEach(label => headerRow.append(element("th", "", label)));
  head.replaceChildren(headerRow);
  body.replaceChildren(...result.rows.map(row => {
    const tr = document.createElement("tr");
    [row.rank, row.ts_code, row.name, number(row.composite_score), percent(row.score_percentile), price(row.raw_close), price(row.research_close), currency(row.amount_cny), ...result.evaluated_factors.map(item => number(row.factor_scores[item.factor_id]))]
      .forEach(value => tr.append(element("td", "", value)));
    return tr;
  }));
  document.getElementById("resultEmpty").hidden = result.rows.length > 0;
  const warnings = document.getElementById("screenWarnings");
  warnings.hidden = result.skipped_factors.length === 0;
  warnings.replaceChildren(...result.skipped_factors.map(item => element("p", "", `${item.name} 未纳入：${item.reason}`)));
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, credentials: "same-origin", headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function element(tagName, className = "", content = "") { const node = document.createElement(tagName); if (className) node.className = className; if (content !== "") node.textContent = content; return node; }
function option(value, label) { const node = document.createElement("option"); node.value = value; node.textContent = label; return node; }
function text(id, value) { document.getElementById(id).textContent = value; }
function number(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed.toFixed(3) : "--"; }
function percent(value) { const parsed = Number(value); return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(1)}%` : "--"; }
function price(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed.toFixed(2) : "--"; }
function currency(value) { const parsed = Number(value); return Number.isFinite(parsed) ? Math.round(parsed).toLocaleString() : "--"; }

let toastTimer;
function toast(message, isError = false) {
  const node = document.getElementById("toast");
  node.textContent = message;
  node.style.background = isError ? "#a93430" : "#202a3b";
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 2400);
}
