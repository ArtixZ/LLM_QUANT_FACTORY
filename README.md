# AutoAlpha

Production research controls for product templates, executable capital ledgers, manual-test
contamination, constrained index enhancement, and factor lifecycle governance are documented in
[`docs/INSTITUTIONAL_RESEARCH_CONTROLS.md`](docs/INSTITUTIONAL_RESEARCH_CONTROLS.md).

AutoAlpha 是面向 A 股多因子截面研究的生产级、可审计 LLM 辅助因子挖掘平台。LLM
只负责提出研究假设、生成受控因子表达式和解释证据；数据准入、回测、统计检验、风险、
交易约束及发布结论全部由确定性组件执行。

## Production invariants

- 生产研究只使用带来源、版本、`knowledge_time` 和修订历史的 point-in-time 快照。
- 因子通过类型化 DSL 表达并经过时间语义校验；任意 Python 候选在隔离进程和最小权限策略下运行。
- 候选与现有 control 组合做配对比较，以成本后组合增量为核心证据。
- 不存在可用于生产准入的连续总分；任何硬门禁失败都不能由 IC 或其他指标补偿。
- IC 仅用于预测诊断。准入依次检查数据、统计可靠性、稳健性、组合增量、风险、交易与容量。
- holdout、模拟盘、风险审批和生产发布彼此独立，所有结论写入不可变制品和哈希链审计日志。
- 公开研究采用 5 年训练接 1 年验证的年度 walk-forward；固定窗口反复反馈不再作为晋级证据。
- 日终量价信号最早按下一交易日开盘执行，评价收益统一为下一开盘到再下一开盘。
- 隔离 holdout 只返回分类结论与证据哈希，LLM 无法读取隐藏夏普、收益、回撤或逐期表现。
- 研究候选、因子族和 holdout 访问都受世代预算约束，并计入 DSR、FDR 和 PBO。
- 公开证据自检会冻结最多三次的优化方向微战役，连续失败提前停止并强制方向冷却。
- 所有完成评价的唯一候选都保留在分类因子库中，未进入冠军只改变研究状态，不删除研究资产。
- 人工组合回测可自选因子、权重和日期，并与冠军、LLM 记忆及 holdout 预算严格隔离。
- 生产故障采用 no-trade、暂停、回滚和退役状态，不允许 LLM 自行扩大风险。

## Repository

```text
src/autoalpha/
  agents/       Researcher / Reviewer / Executor 编排和权限边界
  data/         PIT 合约、动态股票池、快照和数据准入
  dsl/          因子表达式、时间语义和编译器
  research/     切分、统计、证据矩阵、硬门禁和 Pareto 排序
  portfolio/    风险模型、约束优化和归因
  execution/    A 股成交、费用、冲击、容量和 TCA
  governance/   审计、holdout、发布和回滚
  operations/   不可变制品、幂等任务和生产监控
  service/      连续研究、持久因子池、多因子增删循环和控制台
config/         版本化研究协议与门槛
docs/           数据准入、验收矩阵和生产运行手册
tests/          单元、对抗、会计、优化和端到端测试
```

旧的单文件评分循环已从仓库删除，不属于兼容接口，也不应出现在生产任务中。

## Environment

要求 Python 3.12 和 `uv`：

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run pytest --cov=autoalpha --cov-report=term-missing
```

研究配置必须先生成协议指纹，并绑定实际数据校验和：

```bash
uv run autoalpha fingerprint \
  --config config/research.toml \
  --checksum daily_panel=<sha256>
```

生产执行前必须运行数据准入检查：

```bash
uv run autoalpha inspect-data ../data/processed/daily_panel
```

当前本地面板只允许延迟后的价格/成交量探索研究，尚未通过机构级 PIT 数据门禁。缺失项和
禁止的补全方式见 [`docs/DATA_READINESS.md`](docs/DATA_READINESS.md)。
该面板使用前复权 OHLC、手口径成交量和千元口径成交额，因此现金资本账本、容量和模拟交割单
会 fail-closed；必须另行提供未复权价格、股数成交量和人民币成交额才能启用。

## Decision contract

`ResearchOrchestrator` 的执行器必须返回完整 `CandidateEvidence`。平台生成
`EvaluationMatrix`，并只输出以下研究准入结论：

- `REJECTED`
- `RESEARCH`
- `APPROVED_FOR_PAPER`
- `APPROVED_FOR_PRODUCTION`

批量候选只能在通过相同硬门禁后，使用 `pareto_rank` 按净 IR、增量收益、容量、换手、
回撤和稳健性进行非支配排序。最终生产发布仍需人类风险审批。

## Operations

生产职责、日切、告警、降级和回滚流程见
[`docs/PRODUCTION_RUNBOOK.md`](docs/PRODUCTION_RUNBOOK.md)。平台能力和当前数据阻断状态见
[`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md)。评价宪法见
[`evaluation.md`](evaluation.md)，Agent 行为规约见 [`program.md`](program.md)。

## Online service

启动持续研究控制台：

```bash
export AUTOALPHA_SERVICE_TOKEN='<strong-random-admin-token>'
uv run autoalpha-service
```

打开 `http://127.0.0.1:8787`，在设置中填写：

- OpenAI-compatible Base URL，例如 `https://api.openai.com/v1`；
- API Key；
- 模型名称；
- 研究数据目录和迭代间隔。可直接填写项目的 `data/` 根目录，服务会自动解析
  `processed/daily_panel`、`catalog/data_quality.json`、源数据目录和 catalog。
- 最大活跃因子数。组合权重由确定性引擎计算，LLM不能直接指定权重。

`/research-tasks` 是独立的自动研究任务控制台。每个任务冻结市场、数据目录，以及独立的
探索区、公开验证区和隐藏测试区；任务停止后可热调整这些切分，保存时生成新的协议指纹和研究世代。
LLM 只看到探索与公开验证范围，隐藏测试仍只返回分类结论。每个任务拥有独立的运行状态、
停止信号、连续记忆、指标历史、组合版本和研究世代；
统一因子库通过 `source_task_id` 保留来源。任务详情提供启动、停止、协议阻塞原因和实时活动流。
默认最多同时执行两个重型研究迭代，可通过 `AUTOALPHA_MAX_CONCURRENT_RESEARCH=1..8`
调整。自动研究与人工回测可以并行读取不可变面板，数据增量更新仍保持独占，避免读取半成品。

其他入口包括：`/` 自动研究控制面、`/factors` 分类因子库、`/backtest` 人工组合回测、
`/screener` 截面选股器、`/paper-trading` 模拟交易和 `/data` 数据中心。人工回测净值制品存入
`runtime/artifacts/manual-backtests/`，不会成为自动研究证据。

API Key 保存在系统 Keychain，也可通过 `AUTOALPHA_API_KEY` 注入；不会写入 SQLite、日志或
研究制品。`AUTOALPHA_SERVICE_TOKEN` 启用控制面访问认证，公网部署时必须设置，并应由反向
代理提供 TLS。运行状态、连续记忆、因子池、组合版本、指标历史及审计/行动/研究/交付日志保存在
`runtime/autoalpha.sqlite3`，交付制品保存在 `runtime/artifacts/`。

评价协议迁移后，可在自动循环停止时批量重评完整因子库：

```bash
uv run python scripts/batch_reevaluate_factor_library.py --apply
```

脚本默认使用本机全部逻辑核心的并行进程，行情矩阵以写时复制方式共享；它支持逐因子检查点恢复，
完成后统一计算全候选族 FDR/PBO，先备份 SQLite，再以单事务写回。批处理只读取公开训练与验证区，
不会访问隐藏测试期。

工作器会持续迭代并在 API 或评估错误后自动退避重试，只有控制台的显式停止请求会结束循环。
当前价量后端产生真实探索指标，但因数据门禁未通过，结论固定为
`RESEARCH_ONLY_DATA_BLOCKED`，不能进入模拟盘或生产发布。

在线循环采用四层协议：公开探索、年度滚动验证、隔离盲测和最终资金仿真。因子循环先持续生成和
筛选单因子，再对当前冠军组合枚举
`HOLD / ADD / REMOVE / REPLACE`。组合动作通过收益增厚或风险调整改善双通道门禁后才会生成
冻结候选；冻结候选只有在一次性盲测和资金仿真均通过后才能生成新的不可变冠军版本。详细协议见
[`docs/MULTIFACTOR_RESEARCH.md`](docs/MULTIFACTOR_RESEARCH.md)。

容器运行时挂载配置、数据与持久化目录：

```bash
docker run --rm -p 8787:8787 \
  -e AUTOALPHA_SERVICE_TOKEN='<token>' \
  -e AUTOALPHA_API_KEY='<key>' \
  -v "$PWD/config:/workspace/config:ro" \
  -v "$PWD/runtime:/workspace/runtime" \
  -v "/absolute/data:/data:ro" \
  -e AUTOALPHA_DATA_PATH=/data \
  autoalpha:latest
```

每轮研究会绑定数据工作区指纹，并把行数、证券数、日期范围、可用字段、质量报告状态和 PIT
阻断项写入 Agent 上下文、审计日志及不可变研究制品。质量报告失败或价量字段不完整时，工作器
不会执行回测。
