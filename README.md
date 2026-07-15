# AutoAlpha 演示包

> 供 AutoAlpha 讲习课现场演示

## 环境要求

**Python 环境**：
```bash
依赖：`numpy · pandas · scipy · scikit-learn · lightgbm · pyarrow · matplotlib · joblib`
```



## 快速开始

### 1. 检查状态（journal 应为空）
```bash
python runner.py status
```
应该输出：`no journal yet. run 'python runner.py once' first.`

### 2. 运行一轮实验（约 5-6 分钟）
```bash
python runner.py once
```

Runner 会自动执行：
- 读取 `program.md` + `alpha.py`
- 校验 `ITER_NOTE`（4 字段必填）
- 禁词静态扫描
- 因子相关性门控（若 op_type=add_factor）
- 16 fold 滚动评估 + `primary_score`
- 对比 best.json → ACCEPTED / REJECTED / CRASH
- 分数上涨 → 快照到 `journal/snapshots/` + 归档到 `factor_library/`
- 分数下降或崩溃 → 自动回滚 `alpha.py` 到上一 best

### 3. 查看结果
```bash
python runner.py status         # 看当前 best
cat journal/runs.jsonl          # 看历次实验
cat journal/notes/0001.md       # 看第一轮的实验卡
ls factor_library/              # 看归档卡
```

### 4. 重置（如果想重跑）
```bash
python runner.py reset
```

## 文件说明

```
demo/
├── runner.py               实验循环（人类维护，Agent 不改）
├── alpha.py                因子代码（★ Agent 唯一可改文件 ★）
├── prepare.py              数据加载 + 评分公式 primary_score()
├── metrics.py              诊断指标（IC decay / 分位数收益等）
├── factor_library.py       accept 后归档因子卡（含 6 联图）
├── program.md              研究议程宪法（Agent 必读）
├── evaluation.md           评估宪法（人类锁定，不可改）
│
├── cache/
│   └── splits.json         train/val/test 永久冻结切分 + 数据 checksum
│
└── stock_data/
    ├── stock_daily_post_2016_2026_all.parquet  中证 1000 量价（130 MB）
    ├── benchmark_852_all.parquet               中证 1000 指数日 close
    └── trading_Calendar.parquet                交易日历
```

## 演示时的关键脚本

| 命令 | 用途 | 演示时机 |
|---|---|---|
| `python runner.py status` | 查看当前 best.json | 开场 · 证明 journal 是空的 |
| `python runner.py once` | 跑一轮实验 | 现场演示 · 让学员看滚屏 |
| 打开 `alpha.py` | 展示 Agent 待会要改的文件 | 演示前 · 让学员知道基线 |
| 打开 `program.md` | 展示岗位说明书 | 演示前 · 让学员看规则 |
| 打开 `journal/notes/{iter}.md` | 实验卡（可读的假设→变更→结果）| 跑完 · 让学员看 Agent 做了什么 |
| 打开 `factor_library/*/chart_overview.png` | 6 联图 | ACCEPTED 时展示效果 |

## 演示脚本建议（时长 ~10 分钟）

```
0:00-0:30   状态检查：python runner.py status
            → 空 journal，证明从零开始

0:30-1:00   打开 alpha.py，展示当前 baseline（31 因子 + LightGBM）
            打开 program.md，展示宪法

1:00-1:30   （可选）改一下 program.md §3 研究方向，
            让学员看到 “规则杠杆”

1:30-6:30   python runner.py once
            现场滚屏 · 演示过程约 5 分钟

6:30-8:00   查看 journal/runs.jsonl 新增的一行
            查看 journal/notes/0001.md 完整的实验卡
            查看 factor_library/ 新增的一张卡

8:00-10:00  Q&A · 引导学员提出 “它为什么这么做” 的问题
```

## 注意事项

- **test 段永久锁死**：环境变量 `AUTOALPHA_TEST_LOCKED=1`，runner 已在 import prepare 前设置。任何试图读取 test 段的代码都会被拦截。
- **禁词硬扫**：alpha.py 出现 `journal / factor_library / _load_test_panel / ...` 等 8 词之一，runner 直接 CRASH。
- **首轮耗时较长**：如果 `cache/panel_v2.feather` 因为切分变化被删除或损坏，第一次会重新构建（多 ~30 秒）。
- **Windows 中文编码**：runner/judge 已强制 `sys.stdout.reconfigure(encoding='utf-8')`，直接跑不会乱码。

## 出错时

| 问题 | 处理 |
|---|---|
| `KeyError: 'AUTOALPHA_TEST_LOCKED'` | 已在 runner.py 顶部处理，若报此错，检查是否用 python 直接跑了 prepare.py |
| `PermissionError: correlation gate` | Agent 加了跟已有因子相关性 ≥0.85 的因子。属正常护栏行为，看 `journal/last_failed/correlation_*.txt` |
| `ITER_NOTE 字段缺失` | 编辑 alpha.py，检查 ITER_NOTE 是否 4 字段齐全 |
| 数据加载慢 | 首次要构建 panel 缓存，5-10 秒；之后从 `cache/panel_v2.feather` 读，秒级 |

---

**AutoAlpha 演示包 · 教学版 · 5 因子等权 baseline**
