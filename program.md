# AutoAlpha — 研究议程（Program）

> **本文件由人类维护，agent 必读且必须遵守。**
> 它定义了 agent 在 `alpha.py` 里"想做什么、能做什么、不能做什么"。
> 评估宪法见 `evaluation.md`，那是衡量你的尺子；本文件是你前进的方向。

---

## 1. 目标（Mission）

最大化 **`prepare.primary_score()` 在验证段（val）上的得分**，同时保持
代码简洁、结果可解释、研究可复用。

**教学版主分公式（demo v1）**：只考虑 IC 三项指标，让学员看清 “Agent 是在优化预测信号本身”。
Sharpe / 年化 / 回撤 / 单调性 / 换手率仍会计算并展示，但**不进 score**。
学员可以自己动手把这些指标加进 `prepare.primary_score` 里，观察分数如何变化。

```
score = 1.0  * rank_ic_ir          # IC 稳定性（IC 均值 / IC 标准差 · 年化因子）· 主项
      + 10.0 * pearson_ic_mean     # 截面强度（原始值相关性均值）· 辅项
      + 10.0 * rank_ic_mean        # 截面单调（排序相关性均值）· 辅项
```

含义：
- `rank_ic_ir` 反映 “预测能力是否稳定” —— IC 均值高但方差也高等于运气；主项
- `pearson_ic_mean` 反映 “原始预测值和真实收益的相关性” —— 会受极端值影响，做辅助
- `rank_ic_mean` 反映 “排名相关性的平均水平” —— 对极端值稳健

**学员练习方向**（在 `prepare.primary_score` 里自己动手）：
- 加入 Sharpe 项：`+ 0.5 * clip(sharpe, -2, 2)`
- 加入 MDD 项：`+ 0.3 * clip(1 + max_drawdown / 0.30, -2, 1)`
- 加入换手率惩罚：`- 0.1 * clip(annual_turnover/30 - 1, 0, 3)`

不要试图绕过、镜像、近似这条公式 —— runner 永远只读 `prepare.primary_score().score`。

---

## 2. 范式（Paradigm）

**仅做截面研究（pure cross-section）**：

- 每个 T 日，对所有股票按因子值横截面排序，预测**截面相对收益排名**
- 因子可以用滚动窗口算"当日一个数"（如 20 日动量），但**禁止**对**单只股票的序列**单独建模
- 模型必须是 **pooled cross-section**：所有股票的样本混合训练

**训练与切分（教学版）**：
- 固定切分：train / val / test 三段（见 §14）；训练在 train 段，评估在 val 段
- 教学版 baseline 使用线性等权组合，不需要 fit ——`alpha.run(train, val)` 直接返回两段独立的信号
- 如学员探索 ML 模型（Ridge / LightGBM 等），可在 `combine_*` 里加 fit 流程，接口签名不变

**禁止清单**（违者一律视为越权，runner 也会通过信号校验拦截）：
- ❌ LSTM / GRU / RNN / Transformer 等时序结构
- ❌ 按 `symbol` 分组各自拟合一个模型
- ❌ 任何形式触碰 `prepare._*` 私有函数 / 解锁 `AUTOALPHA_TEST_LOCKED`
- ❌ 引入 `prepare.py` / `stock_data/*.parquet` 之外的数据源
- ❌ 镜像、重写或绕过 `prepare.primary_score`

---

## 3. 研究方向（按价值排序）

1. **量价因子族**（首要探索）
   - 反转 / 动量：1d, 5d, 20d, 60d
   - 波动率 / 流动性：vol_20, amihud, turnover
   - 形态：跳空、振幅、阶段高低、价量背离
   - 量能分布：上下影线占比、量价共振、放量缩量
2. **因子组合方法升级**（在因子库稳定后做）
   - 等权 → IC_IR 加权 → 截面 Ridge → LightGBM Pooled → 简单 MLP
3. **标签口味实验**
   - 默认 `market_neutral`，可尝试 `vol_adjusted`（更稳健）和 `rank`（更鲁棒）
4. **HORIZON 探索**
   - 默认 5；可尝试 1（高频反转）、10、20（中长周期动量）

---

## 4. 信噪比与标准化（重要，研究质量基础）

A 股横截面噪声极大，**任何裸因子值都不要直接相加 / 喂模型**。
固定流程（已在 `alpha.py:cs_winsorize_zscore()` 实现）：

1. **截面 MAD winsorize**：用 1.4826·MAD 当 σ，把超过 ±3σ 的截面外点拉回边界
2. **截面 z-score**：减截面均值，除以截面标准差
3. **不要做时序标准化**（违反"纯截面"）

每个**单因子**进入组合前必须做这一对操作；最终组合信号在返回前**再做一次**，
保证 `prepare.validate_signal` 通过（截面 mean≈0、std≈1）。

---

## 5. 风格偏好（Style Preferences）

| 偏好 | 反偏好 |
|---|---|
| 物理含义清晰的因子 | 纯统计搜索的"幸存者" |
| 简单可解释（5–20 行能写完） | 50 行复杂特征工程换 0.005 IC |
| 因子族多样（动量/反转/波动/量能…） | 单族堆叠 10 个高度相关因子 |
| 多因子两两 IC 相关 < 0.6 | 互相关 > 0.8 还都保留 |
| 横截面信号稳健（Top/Bottom 分组都单调） | 只在某一极端表现好 |

---

## 6. 硬约束（Hard Constraints）

| 约束 | 阈值 |
|---|---|
| `alpha.py` 总行数 | ≤ 1600 |
| 单次 `alpha.run(train, val)` 总耗时 | ≤ 48 分钟 |
| 单因子计算复杂度 | O(N·T) — 不许 O(N²) 全配对计算 |
| `HORIZON` | 必须 ∈ `prepare.ALLOWED_HORIZONS = (1, 3, 5, 10, 20)` |
| `LABEL_KIND` | 必须 ∈ `prepare.ALLOWED_LABEL_KINDS` |
| 允许 import | `numpy / pandas / scipy / sklearn / lightgbm / torch / joblib` + 本项目模块 `prepare / metrics` |
| 输出形态 | `(signal_train, signal_val)` 都是 `[date×symbol]` 浮点 DataFrame |
| 截面标准化 | 信号必须满足 \|daily_mean\|<0.05 且 0.5<daily_std<2.0 |

### 6.1 防数据泄露禁词清单（runner 静态扫描即时拦截）

`alpha.py` 源码中**严禁出现**以下任一字符串。出现即视为越权，runner 在
import 前就会判 crash + revert，不会真正执行你的代码：

```
_load_test_panel        # prepare 私有函数
AUTOALPHA_TEST_LOCKED   # 测试段环境变量锁
factor_library          # 归档目录/模块（含每张卡的 _test_metrics.json）
_test_metrics           # 上一行的私有产物
journal                 # journal/ 目录（含 best.json / runs.jsonl / test_eval.jsonl）
test_eval               # judge.py 产出
judge                   # judge.py 模块
splits.json             # 切分定义（含 test 段日期）
```

为什么这些必须禁：
- 看到了 test 段数据就**已经污染了"未见性"**，三段切分立刻作废
- 读了别人的 `card.json` 或 `_test_metrics.json` 等于偷看真未来
- 读了 `journal/best.json` 等于知道"哪条路径分数高"，会引导 agent 走捷径
- 即使是无意识的 `import factor_library` 也会触发拦截 —— 这是**有意为之**

如果你想要某个分析能力（例如查看自己上次的 IC decay），在
`metrics.py` 里用 in-process 数据计算，不要去文件系统翻历史。

---

## 7. agent 工作循环（与 runner 配合）

每次实验一轮：

1. 读 `program.md` + `evaluation.md`（这两份是宪法，必读）
2. 读当前 `alpha.py` 全文（≤1600 行，能一口气读完）
3. 形成一个**单点假设**（"加一个 amihud 流动性因子" / "把组合从等权换成 IC_IR 加权"）
4. 修改 `alpha.py`，**包括 ITER_NOTE**（见 §7.1）
5. 由 runner 调 `alpha.run()` → `prepare.primary_score()` 给分
6. 分数提升：runner 自动 snapshot；分数下降或崩溃：runner 自动回滚到上一最优
7. 全部记录在 `journal/runs.jsonl` + `journal/notes/{iter}.md`，最优代码快照在 `journal/snapshots/`

**单点假设原则**：每轮只改**一件事**。把"换标签 + 换模型 + 加因子"打包改的实验
即使提分也不知道是哪个起作用 —— 后续无法复用。

### 7.1 ITER_NOTE 协议（每次实验强制）

每次改 `alpha.py` 都必须同步更新顶层 `ITER_NOTE: dict`，否则 runner 拒绝执行。
模板（**所有键值都必须有内容**）：

```python
ITER_NOTE: dict = {
    # 必需字段：
    "op_type":   "add_factor",        # 见 §11 op_type 分类
    "hypothesis": "新增 amihud 流动性因子，与现有低波家族相关性低，应能补充信息。",
    "change":     "在 FACTORS 末尾加 f_amihud_20；其它不动。",
    "expected":   "score +3.5 → +3.7 左右；turnover 略升。",

    # 推荐字段：
    "parent_iter": 7,                 # 当前 best 来自第几次实验
    "reasoning":   "F0004 单调性 0.93 但 turnover=33 偏高；amihud 倾向稳健大票，应能压换手。",

    # add_factor 特定字段（强烈推荐）：
    "new_factor":  "f_amihud_20",     # 新因子名；不写时 runner 默认取 FACTORS 最后一项
}
```

写完后 runner 会：
- 读取并校验完整性
- 把 note + 实际跑分结果落到 `journal/notes/{iter}.md`
- ACCEPTED 时同步进 `card.json` 的 `note` 字段（永久档案）

---

## 8. 想申请新口味 / 新指标？

如需 `prepare.py` 加新东西（例如新标签口味、把行业字段加入数据），
请在 `journal/runs.jsonl` 的 `note` 里留言。**人类**会评估并决定。
**禁止**自己绕过 `prepare.py` 实现等价逻辑。

---

## 9. 当前阶段（Phase 1）

- 数据：仅量价（OHLCV + 涨跌停 + 停牌状态 + name）
- 股票池：中证 1000 历史成分（每日动态）
- 切分：永久冻结（见 `cache/splits.json`）
- 财务因子、行业字段：留待 Phase 2

---

## 10. 当前 baseline

`alpha.py` 初始版本（人类提供）：
- 5 个经典量价因子（reversal_5, momentum_20, vol_20, amihud_20, hl_range_20）
- 截面 winsorize + z-score
- 等权组合
- HORIZON = 5, LABEL_KIND = "market_neutral"

agent 应该先理解这个 baseline，再做改动。第一轮可尝试的最小改动：
- 把组合从等权换成 IC_IR 加权
- 或加一个新因子族（如 turnover 流动性）

---

## 11. op_type 分类（ITER_NOTE 必填）

| op_type | 含义 | 触发相关性门控 | 典型分数变化 |
|---|---|---|---|
| `add_factor` | 加一个新因子（FACTORS 列表 +1） | ✅ 强制 | +0.05 ~ +0.5 |
| `modify_factor` | 改已有因子的窗口/参数/实现 | ✅ 与原版本算 ρ | ±0.05 ~ ±0.2 |
| `delete_factor` | 删一个因子 | ❌ | 通常 ≤0 |
| `combine_method` | 改组合方法（等权 / IC_IR 加权 / 其它） | ❌ | ±0.1 ~ ±0.5 |
| `label_kind` | 改 LABEL_KIND（5 种菜单切换） | ❌ | ±0.05 ~ ±0.2 |
| `horizon` | 改 HORIZON（1/3/5/10/20） | ❌ | ±0.1 ~ ±1.0 |
| `preprocess` | 改预处理（winsorize 阈值 / zscore 方式） | ❌ | ±0.05 ~ ±0.1 |
| `other` | 重构 / 兼容性变更，逻辑不变 | ❌ | ≈ 0 |

如果改动跨越多个 op_type（例如同时加因子+换组合方法），违反单点假设原则，
请拆成两次实验。

---

## 12. 因子相关性门控

`add_factor` 和 `modify_factor` 时，runner 在跑 alpha.run 之前，
**先单独算所有因子两两的截面 spearman 相关性**，对新因子做检查：

| \|ρ\|（与任一旧因子） | 行为 |
|---|---|
| ≥ **0.85** | **直接 PermissionError**，CRASH 后自动 revert，写 `journal/last_failed/correlation_{iter}.txt` |
| 0.60 ~ 0.85 | 警告打印，但允许继续；跑完按主分决定 ACCEPT/REJECT |
| < 0.60 | 静默通过 |

**为什么要这条**：A 股量价因子族很容易"换皮重复"（5 日反转 vs 1 日反转、波动率 vs 振幅 vs MAX）。
高相关性的因子加进 ridge / IC_IR 加权时只会带来共线性而非新信息。
门控帮你识别"已有因子族的同质化扩展"。

**绕过方式**：如果你确信某个高相关因子有独立价值（如做风险因子做对冲），
请改 op_type=`combine_method` 在组合层面专门处理，而不是当 alpha 因子加。

---

## 13. 异常熔断 — score 与底层指标背离时强制停下

**核心原则**：score 是合成衡量器，**不是真目标**。真目标是 sharpe / annual_return / max_drawdown 等底层指标。
score 公式可能与"实战产品质量"脱节。当 agent 发现 runner REJECTED 但底层指标显著改善时，
必须**停下询问人类**，而不是盲目按 score 优化。

### 13.1 触发条件（任一即触发）

- score REJECTED，但 `sharpe` 较 best 提升 > **30%**（例：best=2.0 → 当前 ≥ 2.6）
- score REJECTED，但 `max_drawdown` 较 best 改善 > **20%**（变浅）
- score REJECTED，但 `annual_return` 较 best 提升 > **30%**

runner.py 的 `_detect_score_anomaly()` 会自动检测、写入 `runs.jsonl` 的 `score_anomaly` 字段、控制台高亮 `⚠ SCORE_ANOMALY`。

### 13.2 触发后的动作

1. **不要继续盲跑下一轮**
2. 在 `ITER_NOTE` 加键 `"flag": "score_anomaly"`，并填 `"anomaly_detail"` 字段（说明哪个底层指标背离）
3. 立即 return summary 给主控 agent，明确报告"score 公式可能有缺陷，建议人类介入"
4. 主控 agent 见到此信号必须停止派 batch、汇报人类
5. 人类负责评估 score 公式合理性，必要时更新公式

### 13.3 教学意义

score 公式是由人类设计的合成指标，可能与"实战产品质量"脱节。异常熔断的意义是：**承认评判者本身也可能出错，
留出一条向上申诉的通道**。例如：某次改动让 Sharpe 提升 50%、MDD 改善 20%，但因为 IC 微跌导致 score 反而下降 ——
这种情况就是异常熔断要抓的信号。学员可以在 `prepare.primary_score` 里加入 Sharpe / MDD 权重，观察公式如何影响 accept / revert 决策。

---

## 14. 当前阶段（Phase 1）记忆

- 数据：仅量价（OHLCV + 涨跌停 + 停牌状态 + name + 中证 1000 benchmark）
- 股票池：中证 1000 历史成分（每日动态）
- 切分：永久冻结（train / val / test 三段，定义于 `cache/splits.json`）
- 财务因子、行业字段：留待 Phase 2
