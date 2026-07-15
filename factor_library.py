"""
factor_library.py — 因子档案库（持久化交付物）

每次 runner 判定 ACCEPTED 时调用 archive()，在 factor_library/ 下生成：

    20260605_164555_F0001_baseline/
        card.json              完整指标 + 元数据 + parent 链
        alpha.py               当时的代码快照
        chart_overview.png     四联图：净值 / 累计 IC / 分组收益 / 回撤
        data_nav.csv           每日净值序列
        data_ic.csv            每日截面 IC 序列
        data_quantiles.csv     10 分组日均收益

同时维护 factor_library/INDEX.md 与 INDEX.json，方便人类回看与机器检索。

设计原则：
    - 只读 prepare.py 的公开 API，不改锁定区
    - 不影响 runner 的 accept/revert 决策——只做归档
    - 失败不抛异常向外传（图表生成挂了也不应让一次合格因子丢分）
"""
from __future__ import annotations

import json
import shutil
import hashlib
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 服务端无显示场景安全
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体（finance 环境已有 SimHei / Microsoft YaHei）
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

import prepare

ROOT = Path(__file__).resolve().parent
LIB_DIR = ROOT / "factor_library"
INDEX_MD = LIB_DIR / "INDEX.md"
INDEX_JSON = LIB_DIR / "INDEX.json"


# =============================================================================
# 索引读写
# =============================================================================
def _ensure_lib() -> None:
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_JSON.exists():
        INDEX_JSON.write_text(json.dumps({"factors": []}, indent=2), encoding="utf-8")
    if not INDEX_MD.exists():
        INDEX_MD.write_text(_render_index_md([]), encoding="utf-8")


def _read_index() -> dict[str, Any]:
    return json.loads(INDEX_JSON.read_text(encoding="utf-8"))


def _write_index(idx: dict[str, Any]) -> None:
    INDEX_JSON.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
    INDEX_MD.write_text(_render_index_md(idx["factors"]), encoding="utf-8")


def _render_index_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AutoAlpha 因子库（INDEX）",
        "",
        "> 每条记录对应一次 runner ACCEPTED 的合格因子；按时间倒序。",
        "> `test` 列由 judge.py 写入；空值意味着该因子尚未做 test 复核。",
        "> 教学版 score 公式：1.0 * rank_ic_ir + 10.0 * pearson_ic_mean + 10.0 * rank_ic_mean。",
        "> 三列 excess_* 显示相对中证 1000 的超额（仅展示，不进 score）。",
        "",
        "| ID | 时间 | 名称 | val_score | rankIC_IR | pearsonIC_IR | 单调 | 年化 | Sharpe | MDD | 换手 | excess_ret | excess_Sharpe | excess_MDD | parent | test_score | val−test |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reversed(rows):  # 最新在前
        m = r["metrics"]
        tm = r.get("test_metrics")
        if tm is None:
            test_cell = "—"
            gap_cell = "—"
        else:
            test_cell = f"**{tm['score']:+.4f}**"
            gap = r.get("val_minus_test_score")
            gap_cell = f"{gap:+.2f}" if gap is not None else "—"
        rk_ir = m.get("rank_ic_ir", m.get("ic_ir", float("nan")))
        ps_ir = m.get("pearson_ic_ir", float("nan"))
        ar = m.get("annual_return", float("nan"))
        ex_ret = m.get("excess_annual_return", float("nan"))
        ex_sh = m.get("excess_sharpe", float("nan"))
        ex_mdd = m.get("excess_max_drawdown", float("nan"))
        # 老卡片缺 excess_* 时显示 "—"，新卡片显示数值
        def _fmt_pct(v): return f"{v:+.2%}" if pd.notna(v) else "—"
        def _fmt_f(v): return f"{v:+.3f}" if pd.notna(v) else "—"
        lines.append(
            f"| `{r['factor_id']}` "
            f"| {r['ts'][:19]} "
            f"| {r['name']} "
            f"| **{m['primary_score']:+.4f}** "
            f"| {rk_ir:+.3f} "
            f"| {ps_ir:+.3f} "
            f"| {m['monotonicity']:+.3f} "
            f"| {ar:+.2%} "
            f"| {m['sharpe']:+.3f} "
            f"| {m['max_drawdown']:+.2%} "
            f"| {m['annual_turnover']:.1f} "
            f"| {_fmt_pct(ex_ret)} "
            f"| {_fmt_f(ex_sh)} "
            f"| {_fmt_pct(ex_mdd)} "
            f"| {r.get('parent_factor_id') or '—'} "
            f"| {test_cell} "
            f"| {gap_cell} |"
        )
    if not rows:
        lines.append("| _empty_ | | | | | | | | | | | | | | | | |")
    lines.append("")
    return "\n".join(lines)


def _next_seq(idx: dict[str, Any]) -> int:
    if not idx["factors"]:
        return 1
    last = idx["factors"][-1]["factor_id"]  # 形如 F0007
    return int(last.lstrip("F")) + 1


# =============================================================================
# 主入口
# =============================================================================
@dataclass
class ArchiveResult:
    factor_id: str
    dirname: str
    chart_path: str


def archive(
    *,
    iter_id: int,
    score_report: prepare.ScoreReport,
    diag: dict[str, Any],
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    alpha_path: Path,
    factor_name: str,
    factor_function_names: list[str],
    parent_card: dict[str, Any] | None,
    note: dict[str, Any] | None = None,
) -> ArchiveResult | None:
    """
    把当前合格因子归档到 factor_library/。失败时返回 None（不阻塞 runner）。
    """
    try:
        _ensure_lib()
        idx = _read_index()
        seq = _next_seq(idx)
        ts = datetime.now()
        factor_id = f"F{seq:04d}"
        dirname = f"{ts.strftime('%Y%m%d_%H%M%S')}_{factor_id}_{factor_name}"
        # 防御非法目录字符
        dirname = "".join(c if c.isalnum() or c in "._-" else "_" for c in dirname)
        out = LIB_DIR / dirname
        out.mkdir(parents=True, exist_ok=False)

        # ---- 1. 复制代码快照 ----
        shutil.copy2(alpha_path, out / "alpha.py")

        # ---- 2. 重新计算需要画图的数据（IC 序列 + backtest）----
        # 这些算量都小（< 5s），换来 prepare.py 不必改
        labels = prepare.make_labels(panel, score_report.horizon, score_report.label_kind)
        ic_series = prepare.compute_rank_ic(signal, labels)
        bt = prepare.backtest(signal, panel, horizon=score_report.horizon)
        # IC 单调性仍用配置的 LABEL_KIND（与主分一致）
        mono, _ = prepare.compute_group_monotonicity(signal, labels, n_groups=10)
        # 但 10 分组**收益柱状图**永远画"真实 H 日收益"——
        # LABEL_KIND=rank/zscore 时 group_means 会被归一到 0~1，柱子肉眼平，看不出区分。
        # 这里强制用 raw 标签（绝对收益）来画分组图，单调性数字仍来自配置标签。
        raw_labels = prepare.make_labels(panel, score_report.horizon, kind="raw")
        _, group_means = prepare.compute_group_monotonicity(signal, raw_labels, n_groups=10)
        # 10 分组累计净值曲线（与 backtest 同口径，每组 T+1 等权买入）
        quantile_navs = _quantile_navs(signal, panel, score_report.horizon, n_groups=10)

        # ---- 3. 落 csv ----
        bt.nav.rename("nav").to_csv(out / "data_nav.csv", header=True)
        ic_series.rename("rank_ic").to_csv(out / "data_ic.csv", header=True)
        pd.Series(group_means, name="group_mean_return").to_csv(
            out / "data_quantiles.csv", header=True
        )
        # 10 条分位净值曲线
        quantile_navs.to_csv(out / "data_quantile_navs.csv", header=True)
        # benchmark / excess 净值 csv
        if len(getattr(bt, "benchmark_nav", pd.Series(dtype=float))):
            bt.benchmark_nav.rename("benchmark_nav").to_csv(
                out / "data_benchmark_nav.csv", header=True
            )
        if len(getattr(bt, "excess_nav", pd.Series(dtype=float))):
            bt.excess_nav.rename("excess_nav").to_csv(
                out / "data_excess_nav.csv", header=True
            )

        # ---- 4. 画六联图（含 benchmark / excess / 10 分组净值）----
        chart_path = out / "chart_overview.png"
        _plot_overview(
            nav=bt.nav,
            daily_ret=bt.daily_ret,
            ic_series=ic_series,
            group_means=group_means,
            score_report=score_report,
            factor_id=factor_id,
            name=factor_name,
            chart_path=chart_path,
            benchmark_nav=getattr(bt, "benchmark_nav", None),
            excess_nav=getattr(bt, "excess_nav", None),
            quantile_navs=quantile_navs,
        )

        # ---- 5. 写 card.json ----
        improvement = None
        if parent_card is not None:
            improvement = float(score_report.score) - float(parent_card["metrics"]["primary_score"])

        # ---- 计算"局部段"实际指标（与 nav.csv / chart 同口径）----
        # metrics       — score_report 提供（固定 val 段）
        # metrics_val_only — bt 提供（nav.csv 同段）
        # 教学版两者口径相同。

        # 算 nav.csv 同段的 IC / 单调性
        local_rank_ic = ic_series  # 已用 panel 段算
        if len(local_rank_ic):
            li_mean = float(local_rank_ic.mean())
            li_std = float(local_rank_ic.std())
            n_per_year = 245.0 / float(score_report.horizon)
            li_ir = float(li_mean / li_std * (n_per_year ** 0.5)) if li_std > 0 else float("nan")
        else:
            li_mean = li_std = li_ir = float("nan")

        metrics_val_only = {
            "annual_return": float(bt.annual_return),
            "sharpe": float(bt.sharpe),
            "max_drawdown": float(bt.max_drawdown),
            "annual_turnover": float(bt.annual_turnover),
            "n_days": int(bt.n_days),
            "rank_ic_mean": li_mean,
            "rank_ic_ir": li_ir,
            "monotonicity": float(mono),
            "excess_annual_return": float(getattr(bt, "excess_annual_return", float("nan"))),
            "excess_sharpe":         float(getattr(bt, "excess_sharpe", float("nan"))),
            "excess_max_drawdown":   float(getattr(bt, "excess_max_drawdown", float("nan"))),
            "_segment": "panel-passed-to-archive (= sig_for_diag 段)",
        }

        card: dict[str, Any] = {
            "factor_id": factor_id,
            "name": factor_name,
            "dirname": dirname,
            "iter_id": iter_id,
            "ts": ts.isoformat(timespec="seconds"),
            "horizon": score_report.horizon,
            "label_kind": score_report.label_kind,
            "score_version": getattr(score_report, "score_version", "demo_v1"),
            "metrics": {
                "primary_score": float(score_report.score),
                "_segment": "fixed val segment",
                # 双 IC 系列（v2 新增）
                "rank_ic_mean": float(getattr(score_report, "rank_ic_mean", score_report.ic_mean)),
                "rank_ic_std":  float(getattr(score_report, "rank_ic_std",  score_report.ic_std)),
                "rank_ic_ir":   float(getattr(score_report, "rank_ic_ir",   score_report.ic_ir)),
                "pearson_ic_mean": float(getattr(score_report, "pearson_ic_mean", float("nan"))),
                "pearson_ic_std":  float(getattr(score_report, "pearson_ic_std",  float("nan"))),
                "pearson_ic_ir":   float(getattr(score_report, "pearson_ic_ir",   float("nan"))),
                # 别名（向下兼容）
                "ic_mean": float(score_report.ic_mean),
                "ic_std": float(score_report.ic_std),
                "ic_ir": float(score_report.ic_ir),
                # 组合 / 回测
                "monotonicity": float(score_report.monotonicity),
                "sharpe": float(score_report.sharpe),
                "annual_return": float(score_report.annual_return),
                "max_drawdown": float(score_report.max_drawdown),
                "annual_turnover": float(score_report.annual_turnover),
                "n_days": int(score_report.n_days),
                # 超额（vs 中证 1000）
                "excess_annual_return": float(getattr(score_report, "excess_annual_return", float("nan"))),
                "excess_sharpe":         float(getattr(score_report, "excess_sharpe", float("nan"))),
                "excess_max_drawdown":   float(getattr(score_report, "excess_max_drawdown", float("nan"))),
            },
            "metrics_val_only": metrics_val_only,
            "score_breakdown": getattr(score_report, "score_breakdown", None),
            "diag": diag,
            "factor_function_names": factor_function_names,
            "parent_factor_id": parent_card["factor_id"] if parent_card else None,
            "improvement_over_parent": improvement,
            "alpha_lines": _count_alpha_lines(alpha_path),
            "note": note,
        }
        (out / "card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # ---- 6. 更新索引 ----
        idx["factors"].append(card)
        _write_index(idx)

        return ArchiveResult(factor_id=factor_id, dirname=dirname, chart_path=str(chart_path))
    except Exception as e:
        # 归档失败不影响 runner 主流程
        err_path = LIB_DIR / "_archive_errors.log"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with err_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat()}] iter#{iter_id}: {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc(limit=8))
        return None


def get_best_card() -> dict[str, Any] | None:
    """返回索引中 score 最高的卡片（runner 用作 parent）。"""
    if not INDEX_JSON.exists():
        return None
    idx = _read_index()
    if not idx["factors"]:
        return None
    return max(idx["factors"], key=lambda c: c["metrics"]["primary_score"])


# =============================================================================
# Test 指标的私有写入（仅 judge.py 调用，agent 永不可见）
# =============================================================================
def _alpha_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def find_card_by_alpha_hash(alpha_path: Path) -> dict[str, Any] | None:
    """
    按 alpha.py 内容哈希精确匹配归档目录中的快照。
    judge.py 用：拿当前/best snapshot 的内容哈希去 factor_library 里找对应卡。
    """
    target = _alpha_sha1(alpha_path)
    if not LIB_DIR.exists():
        return None
    for sub in sorted(LIB_DIR.iterdir()):
        if not sub.is_dir():
            continue
        snap = sub / "alpha.py"
        if snap.exists() and _alpha_sha1(snap) == target:
            card_path = sub / "card.json"
            if card_path.exists():
                card = json.loads(card_path.read_text(encoding="utf-8"))
                card["_dir"] = str(sub)  # 临时附带，便于写入 _test_metrics.json
                return card
    return None


def write_test_metrics(
    card_dir: Path,
    test_report,
    val_for_reference: dict[str, float] | None = None,
) -> Path:
    """
    把 test 指标写到 card 目录的 _test_metrics.json（下划线前缀 = 私有约定）。

    ⚠️ agent 永远不应读这个文件；program.md 明令禁止访问 factor_library/。
    runner 也通过 alpha.py 静态扫描拒绝任何带 'factor_library' 字样的源码。
    """
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "horizon": int(test_report.horizon),
        "label_kind": test_report.label_kind,
        "test": {
            "score": float(test_report.score),
            "ic_mean": float(test_report.ic_mean),
            "ic_ir": float(test_report.ic_ir),
            "monotonicity": float(test_report.monotonicity),
            "sharpe": float(test_report.sharpe),
            "annual_return": float(test_report.annual_return),
            "max_drawdown": float(test_report.max_drawdown),
            "annual_turnover": float(test_report.annual_turnover),
            "n_days": int(test_report.n_days),
        },
    }
    if val_for_reference is not None:
        payload["val_for_reference"] = val_for_reference
        # 简单的过拟合诊断
        gap = float(val_for_reference.get("score", float("nan"))) - float(test_report.score)
        payload["val_minus_test_score"] = gap
    out = card_dir / "_test_metrics.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# =============================================================================
# 工具
# =============================================================================
def _count_alpha_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8"))
    except Exception:
        return -1


def _quantile_navs(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int,
    n_groups: int = 10,
) -> pd.DataFrame:
    """计算 n 个分位组各自的累计净值曲线。

    与 prepare.backtest 同口径：T+1 开盘买入 → T+1+H 开盘卖出，重叠持仓 1/H 资金，
    扣双边 15bp 手续费；但 selected 这里是按截面 rank 等分到 n_groups 组，
    每组等权持有所有命中股票。

    返回 DataFrame: index=date, columns=["Q01"..f"Q{n}"], values=累计净值（起点 1.0）。
    """
    open_w = panel.pivot(index="date", columns="symbol", values="open").sort_index()
    close_w = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    status = panel.pivot(index="date", columns="symbol", values="trade_status").sort_index()
    lim_up = panel.pivot(index="date", columns="symbol", values="limit_up").sort_index()
    lim_dn = panel.pivot(index="date", columns="symbol", values="limit_down").sort_index()

    idx = signal.index.intersection(open_w.index)
    sig = signal.loc[idx]
    open_w = open_w.loc[idx].reindex(columns=sig.columns)
    close_w = close_w.loc[idx].reindex(columns=sig.columns)
    status = status.loc[idx].reindex(columns=sig.columns)
    lim_up = lim_up.loc[idx].reindex(columns=sig.columns)
    lim_dn = lim_dn.loc[idx].reindex(columns=sig.columns)

    next_status = status.shift(-1)
    next_close = close_w.shift(-1)
    next_lim_up = lim_up.shift(-1)
    can_buy = (next_status == 0) & (next_close < next_lim_up * 0.99)

    buy_open = open_w.shift(-1)
    sell_open = open_w.shift(-1 - horizon)
    holding_ret = sell_open / buy_open - 1.0
    sell_status = status.shift(-1 - horizon)
    sell_close = close_w.shift(-1 - horizon)
    sell_lim_dn = lim_dn.shift(-1 - horizon)
    can_sell = (sell_status == 0) & (sell_close > sell_lim_dn * 1.01)
    holding_ret = holding_ret.where(can_sell)

    masked_sig = sig.where(can_buy)
    q_pct = masked_sig.rank(axis=1, pct=True)  # [0, 1]
    # 把 [0,1] 切成 n_groups 份；q_pct=1.0 也归到第 n-1 组
    q_idx = (q_pct * n_groups).clip(upper=n_groups - 1e-9)

    cost = 2 * 15.0 / 1e4  # 双边 15bp

    out_navs = {}
    for k in range(n_groups):
        # 第 k 组：q_pct ∈ (k/n, (k+1)/n]
        in_group = (q_idx >= k) & (q_idx < k + 1)
        basket_total_ret = holding_ret.where(in_group).mean(axis=1) - cost
        # 重叠持仓：H 日总收益均化为日收益
        daily_each = (1.0 + basket_total_ret) ** (1.0 / horizon) - 1.0
        portfolio_daily = pd.Series(0.0, index=daily_each.index)
        counts = pd.Series(0, index=daily_each.index)
        arr = daily_each.values
        for offset in range(1, horizon + 1):
            shifted = pd.Series(arr, index=daily_each.index).shift(offset)
            portfolio_daily = portfolio_daily.add(shifted.fillna(0.0), fill_value=0.0)
            counts = counts.add(shifted.notna().astype(int), fill_value=0)
        portfolio_daily = portfolio_daily / counts.replace(0, np.nan)
        portfolio_daily = portfolio_daily.dropna()
        if len(portfolio_daily):
            nav = (1.0 + portfolio_daily).cumprod()
        else:
            nav = pd.Series(dtype=float)
        out_navs[f"Q{k+1:02d}"] = nav

    df = pd.DataFrame(out_navs)
    return df


def _plot_overview(
    *,
    nav: pd.Series,
    daily_ret: pd.Series,
    ic_series: pd.Series,
    group_means: np.ndarray,
    score_report: prepare.ScoreReport,
    factor_id: str,
    name: str,
    chart_path: Path,
    benchmark_nav: pd.Series | None = None,
    excess_nav: pd.Series | None = None,
    quantile_navs: pd.DataFrame | None = None,
) -> None:
    """6 联图 (2×3)：
        ① 多头净值          ② 累计 IC          ③ 10 分组累计净值（10 条曲线）
        ④ 多头回撤          ⑤ 超额收益（多头/基准/超额三线）  ⑥ 超额回撤

    benchmark_nav / excess_nav / quantile_navs 为 None 时退化展示空面板（向下兼容老 caller）。
    所有时间轴子图的 x 轴标签自动旋转 45°，避免日期标签重叠。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    ex_ret = getattr(score_report, "excess_annual_return", float("nan"))
    ex_sharpe = getattr(score_report, "excess_sharpe", float("nan"))
    ex_mdd = getattr(score_report, "excess_max_drawdown", float("nan"))

    # nav 段标识（图表指标都来自 nav 这段时间）
    if len(nav):
        nav_start = nav.index.min().date()
        nav_end = nav.index.max().date()
        nav_yrs = (nav.index.max() - nav.index.min()).days / 365.25
        nav_label = f"{nav_start} → {nav_end} ({nav_yrs:.1f}y)"
    else:
        nav_label = "no nav"

    score_seg = "score 与图同段（固定 val）"

    fig.suptitle(
        f"{factor_id} · {name}   "
        f"score={score_report.score:+.4f}   "
        f"H={score_report.horizon}   "
        f"label={score_report.label_kind}\n"
        f"图段: {nav_label}   |   {score_seg}",
        fontsize=11, fontweight="bold",
    )

    def _rotate_xticks(ax, degrees: int = 45) -> None:
        """所有时间轴子图都用：旋转 + 右对齐，避免重叠。"""
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(degrees)
            lbl.set_ha("right")

    # ① 多头净值曲线
    ax = axes[0, 0]
    if len(nav):
        # 直接从 nav 算"图段"的真实指标（与 score 可能不同段）
        nav_daily = nav.pct_change().dropna()
        if len(nav_daily) > 0 and nav.iloc[-1] > 0:
            local_ann = nav.iloc[-1] ** (245.0 / max(len(nav_daily), 1)) - 1.0
            local_std = nav_daily.std()
            local_sh = float(nav_daily.mean() / local_std * (245.0 ** 0.5)) if local_std > 0 else float("nan")
        else:
            local_ann = local_sh = float("nan")
        ax.plot(nav.index, nav.values, color="#1f6feb", linewidth=1.4, label="多头 Top10% 净值")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"① 多头组合净值（图段年化 {local_ann:+.2%}, Sharpe {local_sh:+.2f}）")
        ax.set_ylabel("净值")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        _rotate_xticks(ax)
    else:
        ax.text(0.5, 0.5, "无可用净值数据", ha="center", va="center", transform=ax.transAxes)

    # ② 累计 IC 曲线
    ax = axes[0, 1]
    if len(ic_series):
        cum_ic = ic_series.cumsum()
        local_ic_mean = float(ic_series.mean())
        local_ic_std = float(ic_series.std())
        n_per_year = 245.0 / float(score_report.horizon)
        local_ic_ir = float(local_ic_mean / local_ic_std * (n_per_year ** 0.5)) if local_ic_std > 0 else float("nan")
        ax.plot(cum_ic.index, cum_ic.values, color="#1a7f37", linewidth=1.4, label="cumulative rank IC")
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"② 累计 IC（图段 IC_mean {local_ic_mean:+.4f}, IC_IR(年化) {local_ic_ir:+.2f}）")
        ax.set_ylabel("Σ rank IC")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        _rotate_xticks(ax)
    else:
        ax.text(0.5, 0.5, "无可用 IC 数据", ha="center", va="center", transform=ax.transAxes)

    # ③ 10 分组累计净值曲线（10 条线）
    ax = axes[0, 2]
    if quantile_navs is not None and len(quantile_navs.columns):
        # 颜色：Q1（红 → 信号最低）渐变到 Q10（绿 → 信号最高）
        cmap = plt.get_cmap("RdYlGn")
        n_q = len(quantile_navs.columns)
        for i, col in enumerate(quantile_navs.columns):
            s = quantile_navs[col].dropna()
            if len(s):
                color = cmap(i / max(1, n_q - 1))
                lw = 1.6 if (i == 0 or i == n_q - 1) else 1.0  # Q1/Q10 加粗
                ax.plot(s.index, s.values, color=color, linewidth=lw, label=col, alpha=0.9)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        # Top-Bottom 终值差
        try:
            top = quantile_navs.iloc[:, -1].dropna().iloc[-1]
            bot = quantile_navs.iloc[:, 0].dropna().iloc[-1]
            ax.set_title(f"③ 10 分组累计净值（单调性 {score_report.monotonicity:+.3f}, "
                         f"Q10/Q1 末值 {top:.2f}/{bot:.2f}）")
        except Exception:
            ax.set_title(f"③ 10 分组累计净值（单调性 {score_report.monotonicity:+.3f}）")
        ax.set_ylabel("净值")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
        _rotate_xticks(ax)
    else:
        ax.text(0.5, 0.5, "无 quantile 净值数据", ha="center", va="center", transform=ax.transAxes)

    # ④ 多头回撤曲线
    ax = axes[1, 0]
    if len(nav):
        dd = nav / nav.cummax() - 1.0
        local_mdd = float(dd.min())
        ax.fill_between(dd.index, dd.values, 0.0, color="#cf222e", alpha=0.35)
        ax.plot(dd.index, dd.values, color="#cf222e", linewidth=1.0)
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"④ 多头回撤（图段 MDD {local_mdd:+.2%}, "
                     f"换手 {score_report.annual_turnover:.1f}/年）")
        ax.set_ylabel("drawdown")
        ax.grid(alpha=0.3)
        _rotate_xticks(ax)
    else:
        ax.text(0.5, 0.5, "无可用回撤数据", ha="center", va="center", transform=ax.transAxes)

    # ⑤ 超额收益（多头 / 基准 / 超额 三条净值线）
    ax = axes[1, 1]
    if benchmark_nav is not None and len(benchmark_nav) and len(nav):
        # 从 nav/benchmark_nav 直接算图段超额指标（不用 score_report）
        nav_d = nav.pct_change().dropna()
        bench_d = benchmark_nav.pct_change().reindex(nav_d.index).fillna(0.0)
        ex_d = nav_d.sub(bench_d, fill_value=0.0)
        ex_nav = (1.0 + ex_d).cumprod()
        if len(ex_d) > 0:
            local_ex_ret = float(ex_nav.iloc[-1] ** (245.0 / len(ex_d)) - 1.0)
            ex_std = ex_d.std()
            local_ex_sh = float(ex_d.mean() / ex_std * (245.0 ** 0.5)) if ex_std > 0 else float("nan")
        else:
            local_ex_ret = local_ex_sh = float("nan")

        ax.plot(nav.index, nav.values, color="#1f6feb", linewidth=1.2,
                label="多头净值")
        ax.plot(benchmark_nav.index, benchmark_nav.values, color="#8957e5",
                linewidth=1.2, linestyle="--", label="中证1000 基准")
        if excess_nav is not None and len(excess_nav):
            ax.plot(excess_nav.index, excess_nav.values, color="#bf8700",
                    linewidth=1.6, label="超额净值")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_title(f"⑤ 超额收益（图段 excess_ret {local_ex_ret:+.2%}, "
                     f"excess_Sharpe {local_ex_sh:+.2f}）")
        ax.set_ylabel("净值")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        _rotate_xticks(ax)
    else:
        ax.text(0.5, 0.5, "无 benchmark 数据", ha="center", va="center", transform=ax.transAxes)

    # ⑥ 超额回撤（用户口径：模型回撤 − 基准回撤）
    # 正值（绿）= 同时刻模型比基准抗跌；负值（红）= 模型在基准强势期跑输
    ax = axes[1, 2]
    if len(nav) and benchmark_nav is not None and len(benchmark_nav):
        nav_dd = nav / nav.cummax() - 1.0
        bench_dd = benchmark_nav / benchmark_nav.cummax() - 1.0
        ex_dd = nav_dd.sub(bench_dd, fill_value=0.0)
        local_ex_mdd = float(ex_dd.min()) if len(ex_dd) else float("nan")
        ax.fill_between(ex_dd.index, ex_dd.values, 0.0,
                        where=(ex_dd.values >= 0), color="#1a7f37", alpha=0.35,
                        interpolate=True, label="模型抗跌（+）")
        ax.fill_between(ex_dd.index, ex_dd.values, 0.0,
                        where=(ex_dd.values < 0), color="#cf222e", alpha=0.35,
                        interpolate=True, label="模型跑输（−）")
        ax.plot(ex_dd.index, ex_dd.values, color="#24292f", linewidth=0.8)
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"⑥ 超额回撤 = 多头回撤 − 基准回撤（图段最深 {local_ex_mdd:+.2%}）")
        ax.set_ylabel("model_dd − bench_dd")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        _rotate_xticks(ax)
    else:
        ax.text(0.5, 0.5, "无超额回撤数据", ha="center", va="center", transform=ax.transAxes)

    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
