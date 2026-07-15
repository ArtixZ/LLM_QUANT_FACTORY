# AutoAlpha — 评估宪法（Evaluation Constitution）· demo v1

> **本文件是人类维护的"评估锁定"** — agent 不可修改。
> 所有"主分如何计算 / 标签如何定义 / 回测如何模拟"的真理来源在 `prepare.py`，本文件是其规约说明。
> 当且仅当人类完成一轮 `judge.py` 测试集复核后，可主动迭代本文件。

---

## 1. 主分（Primary Score）· demo v1 教学版 — runner 的 accept / revert 唯一依据

**教学版设计动机**：AutoAlpha 讲习课的目标是让学员看清 Agent 的自我优化循环。
复杂的多分量公式（如加入 Sharpe / MDD / 换手率 / 时间加权等）表达能力强但结构复杂，
不利于讲清 "Agent 到底在优化什么"。demo v1 主分**只考虑 IC 类指标**：

```
score = 1.0  * rank_ic_ir          # IC 稳定性（主项）
      + 10.0 * pearson_ic_mean     # 截面强度（辅项）
      + 10.0 * rank_ic_mean        # 截面单调（辅项）
```

三项含义：
- **`rank_ic_ir`**：每日 rank IC 的 mean/std × sqrt(245/H) —— 反映信号预测能力的**稳定性**。
  IC 均值 0.05、std 0.10 → IC_IR ≈ 0.5×sqrt(49)=3.5 是很好的水平。
- **`pearson_ic_mean`**：每日 Pearson IC（原始值相关性）的均值 —— 反映信号与真实收益的
  **强度相关性**；对极端值敏感，做辅项。
- **`rank_ic_mean`**：每日 rank IC 的均值 —— 反映**排序相关性的平均水平**；对极端值稳健。

10 倍权重的原因：`pearson_ic_mean` 和 `rank_ic_mean` 的典型量级在 0.01-0.05，
`rank_ic_ir` 的典型量级在 1-5，为了让三项在 score 里贡献量级相当所以放大 10 倍。

**Sharpe / 年化 / 回撤 / 单调性 / 换手率**仍会由 `backtest` 计算并写入 `ScoreReport`，
但**不进 score**。它们出现在 `runs.jsonl`、因子卡、6 联图里，供人类和学员观察。

**学员练习**：把这些展示指标加进公式，观察分数如何随之变化。
建议依序尝试：先加 Sharpe（`+ 0.5 * clip(sharpe, -2, 2)`）→ 再加 MDD → 再加换手率惩罚。

实现位置：`prepare.primary_score()` — **唯一来源**。
agent 不得镜像、重写、绕过；`runner.py` 永远只读 `score` 这一个数字判断 accept / revert。

### 三项如何解读

| 场景                                | rank_ic_ir | pearson_ic_mean | rank_ic_mean | score |
|-------------------------------------|-----------:|----------------:|-------------:|------:|
| 无信号（IC ≈ 0）                     |       0.0  |          0.000  |       0.000  |  0.00 |
| 弱信号（IC=0.02、IR=1.0）            |       1.0  |          0.020  |       0.020  |  1.40 |
| 中等信号（IC=0.04、IR=2.0）          |       2.0  |          0.035  |       0.040  |  2.75 |
| 强信号（IC=0.06、IR=3.0）            |       3.0  |          0.050  |       0.060  |  4.10 |
| 优秀信号（IC=0.08、IR=4.0）          |       4.0  |          0.065  |       0.080  |  5.45 |


---

## 2. HORIZON（持有期）— 单一真理来源

允许值：`(1, 3, 5, 10, 20)` 交易日。`alpha.HORIZON` 一旦设定，
`prepare.make_labels` 与 `prepare.backtest` 必须用同一个 H。
错配（label 用 H=5、回测用 H=10）一律视为作弊，runner 校验拒绝。

---

## 3. LABEL_KIND（标签口味菜单）

| kind | 公式 | 用途 |
|---|---|---|
| `raw` | `open[T+1+H] / open[T+1] - 1` | 看绝对收益方向 |
| `market_neutral` | `raw - mean(raw, axis=1)` | **推荐**：扣除当日截面等权市场收益 |
| `vol_adjusted` | `raw / std_20(daily_return)` | 高波动股惩罚，更稳健 |
| `rank` | 截面分位 ∈ [0, 1] | 用 rank 作回归目标 |
| `zscore` | 截面 z-score | 与 rank 类似，保留尾部信息 |

实现位置：`prepare.make_labels(panel, horizon, kind=...)`。
所有标签口味**只影响 IC 计算**，**不影响回测的真金白银模拟**（回测永远是 T+1 开盘买入、Top 10% 等权、双边 15bp、涨停板/停牌剔除）。

---

## 4. 双 IC 指标

`prepare.compute_rank_ic` 与 `prepare.compute_pearson_ic` **同时计算并暴露**：

| 指标 | 算法 | 解读 |
|---|---|---|
| **rank IC** | 截面 rank 后 Pearson | 对极端值稳健；A 股研究主用；进入主分 |
| **Pearson IC** | 截面原始值 Pearson | 捕捉强度信息；用于诊断 |

**对照看的研究信号**：
- rank IC 高、Pearson IC 低 → 因子**排序对，量级失真**（如有少数极端值主导）
- Pearson IC 高、rank IC 低 → 因子**被极端值主导**，整体相关性弱

**主分中只用 rank_ic_ir**（A 股横截面噪声大，rank 更稳健）。
Pearson IC 仅作诊断，不进主分，但写入 `ScoreReport` 与因子卡，agent 可在 `metrics.py` 自由扩展使用。

---

## 5. 诊断指标（agent 可在 `metrics.py` 自由扩展）

诊断指标**不进入主分**，**不影响 accept / revert**，仅供：
- 写入 `journal/runs.jsonl` + 因子卡，给人类回看
- 给 agent 自己在下一轮迭代时参考

**白名单方向**：rank IC、Pearson IC、IC_IR、IC decay（不同 H）、分位数收益、
因子相关性、按市值/波动分组的子集 IC、换手分解、Top 篮子 Jaccard。

**禁止方向**：偷偷加权再 return；偷看 test 段（`AUTOALPHA_TEST_LOCKED` 锁会拦）。

---

## 6. 信号格式硬约束（违者 runner 直接判失败）

`alpha.run` 返回的 `(signal_train, signal_val)` 必须满足：

| 约束 | 阈值 |
|---|---|
| shape | `[date × symbol]` 二维 DataFrame |
| index | `pd.DatetimeIndex` |
| dtype | float |
| 截面规模 | 每日列数 ≥ 30 |
| 截面均值 | `mean(\|daily_mean\|) < 0.05` |
| 截面标准差 | `0.5 < mean(daily_std) < 2.0` |
| 全 NaN 行占比 | < 50% |

实现位置：`prepare.validate_signal`。
这是**截面研究的硬门禁**：agent 必须每天对信号做 z-score 或 rank 后才能交。

---

## 7. 多头回测定义（人类锁定）

| 项 | 规则 |
|---|---|
| 选股 | 每日按 signal 截面排名取 Top 10% |
| 入场 | T+1 开盘 |
| 出场 | T+1+H 开盘 |
| 持仓 | 等权 |
| 重叠持仓 | 每日 1/H 资金被换仓 |
| 不可买 | `trade_status==1` 或 `close >= limit_up*0.99` |
| 不可卖 | `trade_status==1` 或 `close <= limit_down*1.01` |
| 手续费 | 双边 15 bp |

### 7.1 Benchmark 与超额指标

**数据**：`stock_data/benchmark_852_all.parquet` — 中证 1000 指数日 close。
2016-01-04 → 2026-06-04，2529 个交易日，由 `prepare.load_benchmark_series()` 公开。

**对齐**：所有 prepare 内部的 `_pivot` 现在自动 reindex 到 `trading_Calendar.parquet`
中 `is_trade==1` 区间内的交易日。这意味着 `shift(-1)` 严格按"下一个交易日"语义工作，
不会因 panel 个别日期缺失而跨假期错位。

**超额定义（加性）**：
```
excess_daily = portfolio_daily − benchmark_daily
excess_nav   = (1 + excess_daily).cumprod()
excess_dd    = excess_nav / excess_nav.cummax() − 1.0
excess_sharpe        = excess_daily.mean() / excess_daily.std() * √245
excess_annual_return = excess_nav[-1]^(245/n_days) − 1
excess_max_drawdown  = excess_dd.min()
```

**ScoreReport 新字段**（仅展示，**不进 score 公式**）：
- `excess_annual_return`
- `excess_sharpe`
- `excess_max_drawdown`

**runner / judge / factor_library 联动**：
- runner 的 `runs.jsonl` 与 `journal/notes/{iter}.md` 都会带这三个字段
- factor_library 的 `chart_overview.png` 升级为 6 联图：
  ①多头净值 ②累计 IC ③10 分组日均 ④多头回撤 ⑤超额收益（多头/基准/超额三线）⑥超额回撤
- judge.py 现在在 `journal/test_charts/<ts>_<F00xx>_<name>/` 下生成 test 与 val 两张同款 6 联图



---

## 8. 数据切分（永久冻结）

```
train: 2016-01-04 → 2021-12-03    (~5.9 年)
val:   2021-12-04 → 2024-12-03    (3.0 年)
test:  2024-12-04 → 2026-06-04    (1.5 年)
```

任何对 splits.json 的人为修改、对 parquet 文件的替换都会被 checksum 校验抓出。

---

## 9. 教学版说明

**版本**：demo_v1

**主分公式**：`score = 1.0 * rank_ic_ir + 10.0 * pearson_ic_mean + 10.0 * rank_ic_mean`（详见 §1）

**评估方式**：固定 train / val / test 三段切分（详见 §8）。训练在 train 段，
Agent accept/revert 依据 primary_score 在 val 段的分数。test 段由 `judge.py`
在人类主动触发时才解锁。

**学员可扩展方向**：
- 在 `prepare.primary_score` 里加入 Sharpe / MDD / 换手率等分量，观察分数变化
- 在 `alpha.py` 里试不同 HORIZON（1 / 3 / 5 / 10 / 20）与 LABEL_KIND（5 种口味）
- 在 `combine_*` 里把 “等权” 换成 “IC_IR 加权”，观察分数变化

**Score 公式的合理性由 §13 异常熔断机制兜底**（详见 `program.md` §13）：
若 score REJECTED 但底层指标显著改善（Sharpe > +30% / MDD 改善 > 20% / 年化 > +30%），
runner 会打出 `⚠ SCORE_ANOMALY` 标记，提示人类介入检查公式设计。
