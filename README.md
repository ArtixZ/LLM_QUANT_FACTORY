# A-share Multi-Factor Research Workspace

> **PRIVATE REPOSITORY — All rights reserved.**
> 本仓库仅用于私有云端备份与授权协作，不开源、不授予任何许可，详见 [LICENSE](LICENSE)。
> 仓库不包含任何行情数据、API 密钥或运行期状态；相关内容均被 Git 隔离（见「大文件与敏感数据隔离」）。

A 股截面多因子研究工作区，由两部分组成：

| 组件 | 位置 | 职责 |
|---|---|---|
| **数据管线** `multifactor_ashare` | `src/multifactor_ashare/` | 把不可变的源行情快照审计、清洗并构建为按年分区的规范 Parquet 面板 |
| **研究平台 AutoAlpha** | `AutoAlpha/`（本仓库子目录，历史已完整合并） | 生产级、可审计的 LLM 辅助因子挖掘、组合治理与三套研究控制台 |

## Repository layout

```text
MultiFactorAshare/            ← 单一 Git 仓库（monorepo）
  src/multifactor_ashare/     数据审计与面板构建（mf-data CLI）
  tests/                      数据管线测试
  data/                       行情数据与生成面板（≈1.6 GB，被 ignore，不入 Git）
    mainboard_non_st_qfq_20100101_20260715/   不可变源快照
    catalog/                  生成的目录与质量报告
    processed/daily_panel/    规范面板（trade_year= 分区 Parquet）
  AutoAlpha/                  研究平台子目录（工作区 ≈2.2 GB，仅 ~5 MB 源码入 Git）
```

## 云端推送

单仓库包含数据管线与 AutoAlpha 平台的全部历史（原嵌套仓库历史已通过 subtree 合并完整保留，
可用 `git log -- AutoAlpha/...` 追溯），推送到**私有**远端即可：

```bash
git remote add origin <private-remote-url>
git push -u origin main
```

Git 历史已审计：最大历史 blob < 400 KB（`uv.lock`），无泄露密钥、无二进制行情数据，
**不需要 Git LFS**。合并前的嵌套 `.git` 目录备份在 `.git-autoalpha-backup/`（被 ignore，
确认无误后可删除）。

## 大文件与敏感数据隔离

以下内容一律不进入 Git（两层 `.gitignore` 均含兜底模式 `*.sqlite3*`、`*.log`、`.env*` 等）：

- **行情数据**：`data/` 全部（源快照、catalog、面板）。换机后从源快照按下文管线重建。
- **运行期状态**：`AutoAlpha/runtime*/`（SQLite 研究库、因子池、审计链、日志、PID、制品）。
- **研究产出**：`AutoAlpha/output/`、`cache/`、`tmp/`（回测报告、PDF、复评结果）。
- **密钥**：LLM API Key 存于系统 Keychain 或经 `AUTOALPHA_API_KEY` 注入；控制台管理令牌用
  `AUTOALPHA_SERVICE_TOKEN` 注入。密钥不写入 SQLite、日志、制品或 Git。

## Dataset layout

云端数据包（可直接下载并解压到本仓库根目录）：

- Google Drive: https://drive.google.com/file/d/1rchy8L-dMXHfaqxd7fx0T7SrhgaZBvgP/view?usp=sharing

```text
data/
  mainboard_non_st_qfq_20100101_20260715/  # immutable source snapshot
  catalog/                                  # generated catalog and quality report
  processed/daily_panel/                    # canonical year-partitioned Parquet panel
```

The source snapshot contains forward-adjusted daily OHLCV observations. It is a sparse
observation table, not a point-in-time security master: a missing row cannot by itself tell
whether a stock was suspended, not yet listed, or already delisted. Daily ST status and
historical universe membership are also unavailable. Research code must not treat the folder
name `non_st` as a historical universe definition.

## Setup and pipeline

Requires Python 3.12+ and `uv`:

```bash
uv sync
uv run mf-data all        # audit + build in one step
uv run pytest
```

Useful individual commands:

```bash
uv run mf-data audit
uv run mf-data build
uv run mf-data build --overwrite
```

`audit` writes `data/catalog/daily_catalog.csv` with paths relative to the source snapshot and
`data/catalog/data_quality.json` with structural and row-level checks. `build` writes a canonical
panel partitioned by `trade_year`; it refuses to replace an existing output unless `--overwrite`
is supplied.

## Canonical panel semantics

- `trade_date` is a proper date and `(trade_date, ts_code)` is the intended primary key.
- Prices are the supplied forward-adjusted prices; `close` is also exposed as `adj_close`.
- `ret_1d` is `close / pre_close - 1`, while `close_to_close_ret` uses the prior available
  observation. Their distinction matters after missing trading days.
- `history_observations` counts available observations, not exchange-listed calendar days.
- `is_tradable_observation` only confirms valid OHLC and positive volume/amount on an existing
  row. It does not reconstruct suspension or price-limit eligibility.
- No forward return label is materialized here. Labels should be built later against an explicit
  market calendar and execution convention to avoid accidental look-ahead.

## AutoAlpha research platform

因子挖掘循环、协议冻结与盲测治理、三套控制台（AutoAlpha 8787 / AutoCombine 8888 /
QuantCombine 8889）、起停脚本与容器部署见 [AutoAlpha/README.md](AutoAlpha/README.md)。

## License

Proprietary — All rights reserved. See [LICENSE](LICENSE).
