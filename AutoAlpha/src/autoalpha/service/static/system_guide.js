const FLOW_STAGES = [
  {
    owner: "DATA CONTROL",
    title: "数据准入与快照",
    icon: "database",
    description: "解析数据工作区、质量报告、字段单位、日期覆盖和数据产品状态，为研究任务形成可复算的数据事实。",
    input: "源表、研究面板、catalog、质量报告与元数据。",
    gate: "完整性、时间语义、单位、PIT 状态和资本账本适用性检查。",
    output: "数据指纹、可用字段白名单、阻断项和任务上下文。",
  },
  {
    owner: "RESEARCH PROTOCOL",
    title: "冻结研究任务",
    icon: "file-lock-2",
    description: "为一个研究实例绑定市场、数据路径、公开可见时间、隐藏边界、预算和研究世代。",
    input: "市场、时间切分、数据路径、任务目标与资源预算。",
    gate: "区间顺序、最小样本、协议可行性和数据范围校验。",
    output: "任务 ID、协议指纹、研究世代和独立连续记忆。",
  },
  {
    owner: "LLM RESEARCHER",
    title: "提出经济假设与候选表达式",
    icon: "lightbulb",
    description: "LLM 基于公开数据能力、因子库空白和历史失败提出可解释机制，并生成受限表达式。",
    input: "字段白名单、机制知识、相似因子、公开失败分类与方向预算。",
    gate: "DSL 算子、字段权限、复杂度、方向和未来函数静态检查。",
    output: "候选假设、类型化表达式、预期方向和机制标签。",
  },
  {
    owner: "LLM REVIEW TEAM",
    title: "独立审查与预注册证伪",
    icon: "shield-question",
    description: "在看到绩效前检查公式是否表达假设，并冻结用于推翻该机制的测试与终止条件。",
    input: "候选、数据契约、非绩效因子库上下文和时点约定。",
    gate: "结构契约校验；审查意见仅咨询，不代替确定性静态门禁。",
    output: "独立审查、证伪计划、机制分类和不可变角色制品。",
  },
  {
    owner: "DETERMINISTIC EVALUATOR",
    title: "公开区因果评价",
    icon: "calculator",
    description: "按收盘信号、下一开盘执行的统一约定计算纯多、预测诊断、成本和滚动样本外证据。",
    input: "编译后的因子值、公开数据、产品模板与冻结评价协议。",
    gate: "覆盖、因果时点、统计可靠性、Walk-forward、成本和交易约束。",
    output: "统一指标、逐折证据、失败门禁、净值路径与评价哈希。",
  },
  {
    owner: "FACTOR REGISTRY",
    title: "因子资产入库",
    icon: "library",
    description: "完成评价的唯一候选进入分类因子库，未成为冠军只改变研究状态，不删除资产。",
    input: "候选定义、来源任务、评价证据、角色制品和污染台账。",
    gate: "唯一性、表达式哈希、来源完整性和证据版本一致性。",
    output: "因子 ID、机制簇、相似关系、统一榜单和生命周期记录。",
  },
  {
    owner: "PORTFOLIO RESEARCH",
    title: "相对 control 的边际实验",
    icon: "git-compare-arrows",
    description: "候选通过 ADD、REMOVE、REPLACE 与当前组合配对比较，回答其是否改善最终投资组合。",
    input: "候选、当前 control、冻结权重规则、风险和成本模型。",
    gate: "增量净 IR、年化、回撤、最差折、相关性、换手和绝对门禁。",
    output: "组合动作结论、边际贡献、冻结候选或公开可行性恢复版本。",
  },
  {
    owner: "ROBUSTNESS GATES",
    title: "统计与交易稳健门禁",
    icon: "scan-search",
    description: "检查多个时间折、状态、参数邻域、延迟、随机扰动、成本压力和候选数量惩罚。",
    input: "公开组合增量序列、候选族规模、交易与风险诊断。",
    gate: "FDR、DSR、PBO、正折比例、最差折、回撤、容量和执行压力。",
    output: "门禁矩阵、Pareto 层级和是否具备盲测资格。",
  },
  {
    owner: "BLIND EVALUATOR",
    title: "一次性隔离盲测",
    icon: "eye-off",
    description: "冻结候选在隐藏边界内低频评价，研究模型只获得通过/不通过和有限失败分类。",
    input: "候选绑定、世代预算、污染台账和隐藏数据句柄。",
    gate: "人工暴露、访问次数、同世代污染和公开门禁完整性。",
    output: "分类 verdict、证据哈希和已消费的盲测预算；不输出隐藏精确指标。",
  },
  {
    owner: "AUTOCOMBINE",
    title: "冻结范围内组合与权重搜索",
    icon: "network",
    description: "在因子快照、非负权重、数量和目标约束内联合搜索子集与稳健权重。",
    input: "冻结因子快照、必选范围、约束、目标预设和实验预算。",
    gate: "可行权重、机制分散、相关性、成本、最差折、回撤和组合盲测。",
    output: "实验账本、冠军组合、Pareto 前沿和不可变 StrategySpec。",
  },
  {
    owner: "CAPITAL SIMULATION",
    title: "US 股票资金账本仿真",
    icon: "receipt-text",
    description: "在未复权价格、现金、整数手数、费用和市场状态约束下重放策略交易。",
    input: "StrategySpec、未复权行情、可交易状态、资金规模和执行参数。",
    gate: "停牌与开盘资格、最低佣金、SEC/FINRA 费用、ADV 参与率、现金和残余持仓。",
    output: "损益曲线、持仓、订单、成交、交割单、TCA 与容量证据。",
  },
  {
    owner: "DELIVERY & HUMAN RISK",
    title: "策略交付与生命周期",
    icon: "package-check",
    description: "把通过证据边界的策略封装为版本化交付物，进入 shadow、paper 与独立人类风险审批。",
    input: "策略规范、盲测分类、资金仿真、风险和审计制品。",
    gate: "模拟观察期、TCA 一致性、容量、告警、回滚和人类签核。",
    output: "策略版本、交付日志、发布状态、监控基线与回滚指针。",
  },
];

const serviceOrigins = {
  combine: `${window.location.protocol}//${window.location.hostname}:8888/`,
  quantcombine: `${window.location.protocol}//${window.location.hostname}:8889/`,
  strategies: `${window.location.protocol}//${window.location.hostname}:8888/strategies`,
  batch: `${window.location.protocol}//${window.location.hostname}:8790/`,
};
document.querySelectorAll("[data-service-link]").forEach(link => {
  link.href = serviceOrigins[link.dataset.serviceLink];
});

document.querySelectorAll("#researchFlow button").forEach(button => {
  button.addEventListener("click", () => renderFlowStage(Number(button.dataset.stage)));
});

document.getElementById("printGuide").addEventListener("click", () => window.print());

const sectionLinks = [...document.querySelectorAll("#guideNavigation a")];
const sections = sectionLinks.map(link => document.querySelector(link.getAttribute("href"))).filter(Boolean);
const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!visible) return;
  sectionLinks.forEach(link => link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`));
}, { rootMargin: "-12% 0px -72% 0px", threshold: [0, 0.2, 0.5] });
sections.forEach(section => observer.observe(section));

renderFlowStage(0);

function renderFlowStage(index) {
  const stage = FLOW_STAGES[index];
  if (!stage) return;
  document.querySelectorAll("#researchFlow button").forEach((button, position) => button.classList.toggle("active", position === index));
  document.getElementById("flowStageTag").textContent = `${String(index + 1).padStart(2, "0")} / ${FLOW_STAGES.length}`;
  document.getElementById("flowDetailIndex").textContent = String(index + 1).padStart(2, "0");
  document.getElementById("flowDetailOwner").textContent = stage.owner;
  document.getElementById("flowDetailTitle").textContent = stage.title;
  document.getElementById("flowDetailDescription").textContent = stage.description;
  document.getElementById("flowDetailInput").textContent = stage.input;
  document.getElementById("flowDetailGate").textContent = stage.gate;
  document.getElementById("flowDetailOutput").textContent = stage.output;
  const icon = document.getElementById("flowDetailIcon");
  icon.setAttribute("data-lucide", stage.icon);
  if (window.lucide) window.lucide.createIcons();
}
