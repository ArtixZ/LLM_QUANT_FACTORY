(function installPlatformHeader() {
  const header = document.querySelector(".app-header, .batch-header");
  if (!header || header.dataset.platformHeaderReady === "true") return;

  const alphaOrigin = `${window.location.protocol}//${window.location.hostname}:8788`;
  const combineOrigin = `${window.location.protocol}//${window.location.hostname}:8888`;
  const quantOrigin = `${window.location.protocol}//${window.location.hostname}:8889`;
  const strategyOrigin = window.location.port === "8889" ? quantOrigin : combineOrigin;
  const items = [
    ["research", `${alphaOrigin}/`, "activity", "自动研究"],
    ["tasks", `${alphaOrigin}/research-tasks`, "list-tree", "任务总表"],
    ["combine", `${combineOrigin}/`, "network", "组合任务"],
    ["quantcombine", `${quantOrigin}/`, "binary", "统计组合"],
    ["strategies", `${strategyOrigin}/strategies`, "archive", "策略库"],
    ["llm", `${alphaOrigin}/llm-team`, "brain-circuit", "LLM 团队"],
    ["factors", `${alphaOrigin}/factors`, "library", "因子库"],
    ["screener", `${alphaOrigin}/screener`, "list-filter", "选股器"],
    ["paper", `${alphaOrigin}/paper-trading`, "wallet-cards", "模拟交易"],
    ["backtest", `${alphaOrigin}/backtest`, "chart-no-axes-combined", "手动回测"],
    ["data", `${alphaOrigin}/data`, "database-zap", "数据中心"],
    ["guide", `${alphaOrigin}/guide`, "map", "系统导览"],
    ["settings", `${alphaOrigin}/settings`, "settings-2", "系统设置"],
  ];

  const identity = header.querySelector(":scope > .identity") || document.createElement("div");
  const oldActions = header.querySelector(":scope > .header-actions");
  const existingNav = header.querySelector(".app-nav");
  const tools = document.createElement("div");
  const navSlot = document.createElement("div");
  const nav = existingNav || document.createElement("nav");

  identity.classList.add("platform-header-identity");
  tools.className = "platform-header-tools";
  navSlot.className = "platform-header-nav-slot";
  nav.className = "app-nav platform-nav";
  nav.setAttribute("aria-label", "全局功能导航");
  nav.innerHTML = items.map(([key, href, icon, label]) => `
    <a data-platform-route="${key}" href="${href}" title="${label}">
      <i data-lucide="${icon}"></i><span>${label}</span>
    </a>`).join("");
  nav.querySelector('[data-platform-route="research"]').id = "autoResearchNav";
  navSlot.append(nav);

  const actionNodes = oldActions
    ? [...oldActions.children].filter(node => node !== existingNav)
    : [...header.children].filter(node => node !== identity && node !== existingNav);
  actionNodes.forEach(node => tools.append(node));
  if (oldActions) oldActions.remove();

  header.replaceChildren(identity, navSlot, tools);
  header.classList.add("platform-header");
  header.dataset.platformHeaderReady = "true";
  activateRoute(nav);
  refreshIcons();
  window.addEventListener("load", refreshIcons, { once: true });

  function activateRoute(container) {
    const active = currentRoute();
    container.querySelectorAll("[data-platform-route]").forEach(link => {
      const selected = link.dataset.platformRoute === active;
      link.classList.toggle("active", selected);
      if (selected) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  }

  function currentRoute() {
    const port = window.location.port;
    const path = window.location.pathname;
    if (port === "8888") return path.startsWith("/strategies") ? "strategies" : "combine";
    if (port === "8889") return path.startsWith("/strategies") ? "strategies" : "quantcombine";
    if (port === "8789" || port === "8790") return "backtest";
    if (path.startsWith("/research-tasks")) return "tasks";
    if (path.startsWith("/llm-team")) return "llm";
    if (path.startsWith("/factors")) return "factors";
    if (path.startsWith("/screener")) return "screener";
    if (path.startsWith("/paper-trading")) return "paper";
    if (path.startsWith("/backtest")) return "backtest";
    if (path.startsWith("/data")) return "data";
    if (path.startsWith("/guide")) return "guide";
    if (path.startsWith("/settings")) return "settings";
    return "research";
  }

  function refreshIcons() {
    if (window.lucide) window.lucide.createIcons();
  }
})();
