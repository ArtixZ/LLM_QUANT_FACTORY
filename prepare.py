"""
prepare.py — AutoAlpha 锁定区（人类维护，agent 永不可改）

职责（Karpathy 范式中的 "evaluation lock"）：
    1. 唯一的数据加载入口（train / val / test 物理隔离）
    2. 切分一次性冻结（splits.json + 数据 checksum）
    3. 标签生成（5 种菜单口味，全部禁未来函数）
    4. 多头回测（T+1 开盘成交、涨停板/停牌过滤、双边 15bp 手续费）
    5. 主分（primary_score）—— runner 用它做 accept/revert 的唯一依据
    6. 信号校验器（强制截面 z-score / rank 形态）

模块对外只暴露 6 个公开 API + 3 个常量；其它一律以 _ 开头，
agent 不应触碰。运行时通过环境变量 AUTOALPHA_TEST_LOCKED 锁定 test 段，
仅 judge.py 在最终复核时才解锁。

设计原则（与 evaluation.md 一致）：
    - HORIZON 单一真理来源：label 与 backtest 必须用同一 H
    - 标签变体只影响 IC 计算，不影响回测的真金白银模拟
    - 任何对 _* 函数的调用、对 _FROZEN_SPLITS 的修改都视为越权
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# =============================================================================
# 0. 常量与路径（人类锁定，禁止修改）
# =============================================================================
_ROOT = Path(__file__).resolve().parent
_PARQUET_PATH = _ROOT / "stock_data" / "stock_daily_post_2016_2026_all.parquet"
_BENCHMARK_PATH = _ROOT / "stock_data" / "benchmark_852_all.parquet"
_CALENDAR_PATH = _ROOT / "stock_data" / "trading_Calendar.parquet"
_CACHE_DIR = _ROOT / "cache"
_SPLITS_PATH = _CACHE_DIR / "splits.json"
# 面板缓存（calendar 对齐版）
_PANEL_CACHE_VERSION = "v2"
_PANEL_CACHE = _CACHE_DIR / f"panel_{_PANEL_CACHE_VERSION}.feather"

# 标签口味菜单（agent 选用，不能定义新口味）
LabelKind = Literal["raw", "market_neutral", "vol_adjusted", "rank", "zscore"]
ALLOWED_LABEL_KINDS: tuple[str, ...] = (
    "raw", "market_neutral", "vol_adjusted", "rank", "zscore",
)

# Horizon 白名单（与 evaluation.md 第 2 节一致）
ALLOWED_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)

# 回测常量（人类锁定）
_COST_BPS = 15.0          # 双边手续费 15bp（含印花税与冲击成本）
_TOP_PCT = 0.10           # 多头组合：每日 Top 10%
_TRADING_DAYS_PER_YEAR = 245
_MIN_CROSSSECTION_NAMES = 30  # 截面股票数下限，过少则当日不入回测/IC

# Test 段锁
_TEST_LOCK_ENV = "AUTOALPHA_TEST_LOCKED"


# =============================================================================
# 1. 切分定义（首次运行写入 splits.json，永久冻结）
# =============================================================================
# 数据实际范围：2016-01-04 → 2026-06-04（10.4 年）
# 切分比 train : val : test ≈ 5.9 : 3.0 : 1.5（年）
_DEFAULT_SPLITS = {
    "train": ["2016-01-04", "2021-12-03"],
    "val":   ["2021-12-04", "2024-12-03"],
    "test":  ["2024-12-04", "2026-06-04"],
}


def _data_checksum(path: Path) -> str:
    """对 parquet 文件做 md5（首次运行登记，后续校验防篡改）"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_splits() -> dict:
    """首次运行写 splits.json + checksum；后续运行只校验，不改动。

    计算并校验 calendar / benchmark 两份 checksum。
    缺这两个键就追加并写回（不强制重建），但 panel parquet 的 checksum 仍要严格匹配。
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cur_panel = _data_checksum(_PARQUET_PATH)
    cur_cal = _data_checksum(_CALENDAR_PATH)
    cur_bench = _data_checksum(_BENCHMARK_PATH)

    if _SPLITS_PATH.exists():
        meta = json.loads(_SPLITS_PATH.read_text(encoding="utf-8"))
        # 校验 panel checksum（严格）
        if meta.get("checksum") != cur_panel:
            raise RuntimeError(
                f"[prepare] parquet checksum mismatch.\n"
                f"  splits.json 记录: {meta.get('checksum')}\n"
                f"  当前文件:        {cur_panel}\n"
                f"数据已变更，可能影响所有历史实验的可比性。"
                f"如确实要重新冻结，请手工删除 {_SPLITS_PATH}。"
            )
        # 校验 calendar / benchmark checksum：缺则补写，不一致则报错
        dirty = False
        if meta.get("calendar_checksum") is None:
            meta["calendar_checksum"] = cur_cal
            dirty = True
        elif meta["calendar_checksum"] != cur_cal:
            raise RuntimeError(
                f"[prepare] trading_Calendar checksum mismatch.\n"
                f"  splits.json 记录: {meta['calendar_checksum']}\n"
                f"  当前文件:        {cur_cal}\n"
                f"如确实要重新冻结，请手工删除 {_SPLITS_PATH}。"
            )
        if meta.get("benchmark_checksum") is None:
            meta["benchmark_checksum"] = cur_bench
            dirty = True
        elif meta["benchmark_checksum"] != cur_bench:
            raise RuntimeError(
                f"[prepare] benchmark checksum mismatch.\n"
                f"  splits.json 记录: {meta['benchmark_checksum']}\n"
                f"  当前文件:        {cur_bench}\n"
                f"如确实要重新冻结，请手工删除 {_SPLITS_PATH}。"
            )
        if dirty:
            _SPLITS_PATH.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return meta

    meta = {
        "splits": _DEFAULT_SPLITS,
        "checksum": cur_panel,
        "calendar_checksum": cur_cal,
        "benchmark_checksum": cur_bench,
        "frozen_at_utc_date": "2026-06-05",
        "parquet": str(_PARQUET_PATH.name),
        "calendar": str(_CALENDAR_PATH.name),
        "benchmark": str(_BENCHMARK_PATH.name),
        "note": "Frozen by prepare.py on first run. Do NOT edit by hand.",
    }
    _SPLITS_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


# =============================================================================
# 2. 数据加载（带 feather 缓存）
# =============================================================================
# 模块级缓存：交易日历 / benchmark close（首次访问时填充）
_TRADING_CAL: pd.DatetimeIndex | None = None
_BENCHMARK_CLOSE: pd.Series | None = None


def _load_trading_calendar() -> pd.DatetimeIndex:
    """加载交易日历（is_trade==1 的 nature_date）。结果缓存到模块全局。

    返回排序去重的 pd.DatetimeIndex。这是"下一个交易日"语义的唯一真理来源。
    """
    global _TRADING_CAL
    if _TRADING_CAL is not None:
        return _TRADING_CAL
    cal = pd.read_parquet(_CALENDAR_PATH)
    cal = cal[cal["is_trade"] == 1]
    dates = pd.to_datetime(cal["nature_date"].astype(str), format="%Y%m%d")
    _TRADING_CAL = pd.DatetimeIndex(sorted(dates.drop_duplicates()))
    return _TRADING_CAL


def _load_full_panel() -> pd.DataFrame:
    """加载全量行情，做基础清洗与索引。结果缓存到 feather。

    加 calendar 防御过滤（剔除任何非交易日条目）。当前 panel 内所有日期
    都是合法交易日，但保留 isin(cal) 做兜底。
    """
    if _PANEL_CACHE.exists():
        return pd.read_feather(_PANEL_CACHE)

    df = pd.read_parquet(_PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    # 排除明显异常：开盘价/收盘价非正
    df = df[(df["open"] > 0) & (df["close"] > 0)].reset_index(drop=True)
    # calendar 防御过滤
    cal = _load_trading_calendar()
    df = df[df["date"].isin(cal)].reset_index(drop=True)
    df.to_feather(_PANEL_CACHE)
    return df


def load_benchmark_series() -> pd.Series:
    """中证 1000 指数 close 序列（已对齐到交易日历）。

    公开 API：可被回测、归档、judge 引用，但 alpha.py 不应直接使用
    （benchmark 是评估器的事；agent 写信号不需要看 benchmark）。
    """
    global _BENCHMARK_CLOSE
    if _BENCHMARK_CLOSE is not None:
        return _BENCHMARK_CLOSE.copy()
    bench = pd.read_parquet(_BENCHMARK_PATH)
    bench["date"] = pd.to_datetime(bench["date"])
    s = bench.set_index("date")["close"].sort_index()
    # 防御：reindex 到完整 calendar（benchmark 已经天然对齐，但写明意图）
    cal = _load_trading_calendar()
    s = s.reindex(cal).rename("benchmark_close")
    _BENCHMARK_CLOSE = s
    return s.copy()


def _slice_by_dates(panel: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
    mask = (panel["date"] >= lo_ts) & (panel["date"] <= hi_ts)
    return panel.loc[mask].reset_index(drop=True)


# 标签 / 回测都需要"未来开盘价"做严谨的 T+1 模拟，但 agent 拿不到 future。
# 为此 train/val 段在加载时附带一个 "buffer"：把切分窗口右端再向后延伸
# 足够多的交易日，仅供内部 _make_labels 与 backtest 使用，
# 公开返回给 agent 的 panel **不会包含 buffer 段**。
_BUFFER_DAYS = 30  # 覆盖 H<=20 的最长 horizon 加余量


def _load_panel_with_buffer(seg: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (panel_visible, panel_with_buffer)
    panel_visible: agent 可见区间（严格 [lo, hi]）
    panel_with_buffer: 内部用，[lo, hi + buffer]，用于安全地构造 T+1 未来收益
    """
    if seg == "test" and os.environ.get(_TEST_LOCK_ENV, "1") == "1":
        raise PermissionError(
            "[prepare] Test 段被锁定。runner.py 默认锁定，仅 judge.py 解锁。"
            f"如需手工访问，请先在当前进程设置 {_TEST_LOCK_ENV}=0。"
        )
    meta = _ensure_splits()
    lo, hi = meta["splits"][seg]
    full = _load_full_panel()
    visible = _slice_by_dates(full, lo, hi)

    # buffer：把 hi 向后延伸 _BUFFER_DAYS 个**自然日**，足以覆盖 H<=20
    hi_ts = pd.Timestamp(hi)
    buf_hi = hi_ts + pd.Timedelta(days=_BUFFER_DAYS * 2)  # 多给一倍裕度
    with_buffer = _slice_by_dates(full, lo, buf_hi.strftime("%Y-%m-%d"))
    return visible, with_buffer


# =============================================================================
# 3. 公开 API：数据加载
# =============================================================================
def load_train_panel() -> pd.DataFrame:
    """返回训练段行情（agent 可见）。columns 与原 parquet 一致。"""
    visible, _ = _load_panel_with_buffer("train")
    return visible


def load_val_panel() -> pd.DataFrame:
    """返回验证段行情（agent 可见）。"""
    visible, _ = _load_panel_with_buffer("val")
    return visible


def _load_test_panel() -> pd.DataFrame:
    """⚠️ 内部使用，仅 judge.py 解锁后才能调用。agent 永不可调。"""
    visible, _ = _load_panel_with_buffer("test")
    return visible


# =============================================================================
# 4. 标签生成（HORIZON + LabelKind 双参数；防未来函数）
# =============================================================================
def _pivot(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """long → wide：index=date, columns=symbol。

    自动 reindex 到 panel 区间内的完整交易日历，使 shift(-1) 严格按
    "下一个交易日"语义工作（不会因 panel 缺数据而跳过假期相邻的真交易日）。
    """
    wide = panel.pivot(index="date", columns="symbol", values=col).sort_index()
    if len(wide) == 0:
        return wide
    cal = _load_trading_calendar()
    full_idx = cal[(cal >= wide.index.min()) & (cal <= wide.index.max())]
    return wide.reindex(full_idx)


def make_labels(
    panel: pd.DataFrame,
    horizon: int,
    kind: LabelKind = "market_neutral",
) -> pd.DataFrame:
    """
    构造 [date × symbol] 的未来 H 日截面标签。

    构造规则（防未来函数）：
        - 信号在 T 日生成
        - T+1 开盘买入，持有 H 个交易日，T+1+H 开盘卖出
        - 因此 label_T = open[T+1+H] / open[T+1] - 1
        - 调用方传入的 panel 必须包含 [start, end + buffer] 区间，
          但本函数只在 [start, end] 这一段返回 label；超出 panel 末端
          无法计算的日期一律 NaN（不向 agent 泄露未来）

    label kind:
        raw            原始未来 H 日收益
        market_neutral 减去当日截面等权收益（中证 1000 内）
        vol_adjusted   raw / 过去 20 日日收益标准差（截面对齐）
        rank           截面分位数 ∈ [0, 1]
        zscore         截面 z-score（截面去均值 / 标准差）
    """
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError(f"horizon {horizon} 不在白名单 {ALLOWED_HORIZONS}")
    if kind not in ALLOWED_LABEL_KINDS:
        raise ValueError(f"label kind {kind!r} 不在菜单 {ALLOWED_LABEL_KINDS}")

    open_w = _pivot(panel, "open").sort_index()

    # 严格 T+1 → T+1+H：用 shift(-1) 拿明日开盘，shift(-1-H) 拿持有期末开盘
    buy_open = open_w.shift(-1)
    sell_open = open_w.shift(-1 - horizon)
    raw = sell_open / buy_open - 1.0  # [date × symbol]

    if kind == "raw":
        out = raw
    elif kind == "market_neutral":
        # 减去当日截面平均（每行均值，作为"市场基准"）
        out = raw.sub(raw.mean(axis=1), axis=0)
    elif kind == "vol_adjusted":
        close_w = _pivot(panel, "close").sort_index()
        ret = close_w.pct_change(fill_method=None)
        # 过去 20 日波动率（不含 T 日；用 shift(1) 防止包含 T 日的成分泄露）
        vol20 = ret.shift(1).rolling(20, min_periods=10).std()
        out = raw / vol20.replace(0, np.nan)
    elif kind == "rank":
        # 横截面分位 ∈ [0, 1]
        out = raw.rank(axis=1, pct=True)
    elif kind == "zscore":
        mu = raw.mean(axis=1)
        sd = raw.std(axis=1).replace(0, np.nan)
        out = raw.sub(mu, axis=0).div(sd, axis=0)
    else:
        raise AssertionError("unreachable")

    # 把"超出 visible 段"的日期裁掉——不向 agent 泄露未来
    # 使用 panel 的 [min, max] 区间内的交易日历做切片（与 _pivot 同口径），
    # 防止 panel 缺数据日期被误剔（panel 可能短缺 calendar 中的某几天，
    # 但 reindex 后的 out 索引是完整 calendar，应保留这些行）
    cal = _load_trading_calendar()
    if len(panel):
        lo, hi = panel["date"].min(), panel["date"].max()
        visible = cal[(cal >= lo) & (cal <= hi)]
        out = out.loc[out.index.isin(visible)]
    return out


# =============================================================================
# 5. 信号校验器（截面研究的硬门禁）
# =============================================================================
def validate_signal(signal: pd.DataFrame, name: str = "signal") -> None:
    """
    校验信号符合"截面研究"的硬约束。runner 在每次实验后强制调用。
    违反任一条都视为这次实验失败（runner 会回滚）。

    硬约束：
        1. shape: index=date(DatetimeIndex), columns=symbol
        2. dtype: 浮点
        3. 每日截面 |mean| < 0.05 且 std ∈ [0.5, 2.0]
           （即 agent 必须做横截面 z-score 或 rank 标准化）
        4. 全 NaN 行不超过 50%
    """
    if not isinstance(signal, pd.DataFrame):
        raise TypeError(f"[{name}] 必须是 DataFrame，收到 {type(signal)}")
    if not isinstance(signal.index, pd.DatetimeIndex):
        raise TypeError(f"[{name}] index 必须是 DatetimeIndex（date）")
    if signal.shape[1] < _MIN_CROSSSECTION_NAMES:
        raise ValueError(f"[{name}] 横截面股票数过少: {signal.shape[1]}")
    if not np.issubdtype(signal.values.dtype, np.floating):
        raise TypeError(f"[{name}] dtype 必须是 float，收到 {signal.values.dtype}")

    all_nan_rows = signal.isna().all(axis=1).mean()
    if all_nan_rows > 0.5:
        raise ValueError(f"[{name}] {all_nan_rows:.1%} 的日期信号全 NaN，可能未对齐")

    daily_mean_abs = signal.mean(axis=1).abs().mean()
    daily_std = signal.std(axis=1).mean()
    if daily_mean_abs > 0.05 or not (0.5 < daily_std < 2.0):
        raise ValueError(
            f"[{name}] 信号未做横截面标准化: |daily_mean|={daily_mean_abs:.3f}, "
            f"daily_std={daily_std:.3f}。截面研究要求每天 z-score 或 rank（均值≈0, 标准差≈1）。"
        )


# =============================================================================
# 6. 多头回测（纯多头 Top 10%，T+1 开盘成交，涨停/停牌剔除）
# =============================================================================
@dataclass
class BacktestResult:
    nav: pd.Series          # 净值曲线
    daily_ret: pd.Series    # 组合日收益
    annual_return: float
    sharpe: float
    max_drawdown: float     # 负数
    annual_turnover: float  # 单边换手 / 年（×2 即双边）
    n_days: int
    # —— benchmark / excess（仅展示，不进 score）——
    benchmark_nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_daily_ret: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    excess_nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    excess_daily_ret: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    excess_annual_return: float = float("nan")
    excess_sharpe: float = float("nan")
    excess_max_drawdown: float = float("nan")


def backtest(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int,
) -> BacktestResult:
    """
    纯多头组合回测：
        - 每个交易日按 signal 的截面排名选 Top 10%
        - 等权买入，T+1 开盘以 open 价格成交
        - 持有 horizon 个交易日，到期 T+1+H 开盘卖出
        - 不可买：trade_status==1（停牌）或 close >= limit_up*0.99（涨停封板）
        - 不可卖：trade_status==1 或 close <= limit_down*1.01（跌停封板）
        - 双边手续费 15bp 一次性扣除（买入 + 持有期末卖出）

    简化处理：用"重叠持仓平均"近似——每天有 1/H 的资金正在被换仓。
    具体：每日组合收益 ≈ 过去 H 日"今日入场篮子"的等权平均收益的 1/H。
    这是业界标准做法（避免 H 重复回测带来的样本依赖）。
    """
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError(f"horizon {horizon} 不在白名单 {ALLOWED_HORIZONS}")
    validate_signal(signal, "backtest.signal")

    # ---------- 准备宽表 ----------
    open_w = _pivot(panel, "open").sort_index()
    close_w = _pivot(panel, "close").sort_index()
    lim_up = _pivot(panel, "limit_up").sort_index()
    lim_dn = _pivot(panel, "limit_down").sort_index()
    status = _pivot(panel, "trade_status").sort_index()

    # 对齐到 signal.index
    idx = signal.index.intersection(open_w.index)
    signal = signal.loc[idx]
    open_w = open_w.loc[idx]
    close_w = close_w.loc[idx]
    lim_up = lim_up.loc[idx]
    lim_dn = lim_dn.loc[idx]
    status = status.loc[idx]

    cols = signal.columns
    open_w = open_w.reindex(columns=cols)
    close_w = close_w.reindex(columns=cols)
    lim_up = lim_up.reindex(columns=cols)
    lim_dn = lim_dn.reindex(columns=cols)
    status = status.reindex(columns=cols)

    # ---------- 可交易性掩码 ----------
    # 信号是 T 日产生 → T+1 开盘买入；T+1 是否可买取决于 T+1 的状态
    # 用"次日"快照：买入约束用 shift(-1) 后的 panel
    next_status = status.shift(-1)
    next_close = close_w.shift(-1)
    next_lim_up = lim_up.shift(-1)
    can_buy = (next_status == 0) & (next_close < next_lim_up * 0.99)

    # ---------- 选股：每日 Top 10% ----------
    masked = signal.where(can_buy)  # 不可买的票直接置 NaN
    # 截面分位
    q = masked.rank(axis=1, pct=True)
    selected = (q >= 1.0 - _TOP_PCT)  # bool [date × symbol]

    # ---------- 单批次（T 日选出篮子）的收益序列 ----------
    # 每个 T 日入场篮子的"H 日总收益"= open[T+1+H] / open[T+1] - 1
    buy_open = open_w.shift(-1)
    sell_open = open_w.shift(-1 - horizon)
    holding_ret = sell_open / buy_open - 1.0   # 单只股票 T+1 → T+1+H

    # 卖出端不可卖的（停牌/跌停封板）回退到下一个可卖日：简化为剔除该笔，
    # 由"等权平均"自动稀释（更严格的做法是延后卖出；此处保持简化与一致）
    sell_status = status.shift(-1 - horizon)
    sell_close = close_w.shift(-1 - horizon)
    sell_lim_dn = lim_dn.shift(-1 - horizon)
    can_sell = (sell_status == 0) & (sell_close > sell_lim_dn * 1.01)
    holding_ret = holding_ret.where(can_sell)

    # 每日篮子等权收益（H 日总收益）
    basket_total_ret = holding_ret.where(selected).mean(axis=1)
    # 扣双边手续费（买 + 卖各 0.5 倍 _COST_BPS）
    basket_total_ret = basket_total_ret - 2 * _COST_BPS / 1e4

    # ---------- 重叠持仓平均：把 H 日总收益拆为日均 ----------
    # 严谨展开：每日组合 = 过去 H 个 T 日篮子的"日化后收益"等权平均
    # 这里按几何均化：(1+R)^(1/H) - 1
    daily_each = (1.0 + basket_total_ret) ** (1.0 / horizon) - 1.0
    # 第 i 个 T 日篮子在第 i+1, ..., i+H 日各贡献一份
    portfolio_daily = pd.Series(0.0, index=daily_each.index)
    counts = pd.Series(0, index=daily_each.index)
    arr = daily_each.values
    for offset in range(1, horizon + 1):
        shifted = pd.Series(arr, index=daily_each.index).shift(offset)
        portfolio_daily = portfolio_daily.add(shifted.fillna(0.0), fill_value=0.0)
        counts = counts.add(shifted.notna().astype(int), fill_value=0)
    portfolio_daily = portfolio_daily / counts.replace(0, np.nan)
    portfolio_daily = portfolio_daily.dropna()

    if portfolio_daily.empty:
        return BacktestResult(
            nav=pd.Series(dtype=float), daily_ret=pd.Series(dtype=float),
            annual_return=np.nan, sharpe=np.nan, max_drawdown=np.nan,
            annual_turnover=np.nan, n_days=0,
        )

    # ---------- 净值与指标（全期等权口径）----------
    nav = (1.0 + portfolio_daily).cumprod()
    n_days = portfolio_daily.size

    annual_ret = nav.iloc[-1] ** (_TRADING_DAYS_PER_YEAR / n_days) - 1.0
    daily_std = portfolio_daily.std()
    sharpe = (
        portfolio_daily.mean() / daily_std * np.sqrt(_TRADING_DAYS_PER_YEAR)
        if daily_std and not np.isnan(daily_std) and daily_std > 0 else np.nan
    )
    full_drawdown = nav / nav.cummax() - 1.0
    mdd = float(full_drawdown.min())

    # 换手：相邻两日"持仓股票集合"差异占当前持仓比例
    turnover_daily = (selected.astype(int).diff().abs().sum(axis=1)
                      / selected.sum(axis=1).replace(0, np.nan)).fillna(0.0) * 0.5
    turnover_daily = turnover_daily.loc[portfolio_daily.index.intersection(turnover_daily.index)]
    annual_turnover = float(turnover_daily.mean() * _TRADING_DAYS_PER_YEAR)

    # ---------- Benchmark 对齐 + 加性 Excess ----------
    bench_close = load_benchmark_series().reindex(portfolio_daily.index).ffill()
    bench_daily = bench_close.pct_change().fillna(0.0)
    bench_nav = (1.0 + bench_daily).cumprod()

    excess_daily = portfolio_daily.sub(bench_daily, fill_value=0.0)
    excess_nav = (1.0 + excess_daily).cumprod()

    # —— 超额回撤的定义 ——
    # 用户口径：超额回撤 = 模型回撤 − 基准回撤（每个时点上的相对回撤差）
    #   正值：模型同时刻比基准抗跌（如模型 -10%、基准 -20% → +10%）
    #   负值：模型在基准平稳期 / 牛市跑输（如模型 -30%、基准 0% → -30%）
    # 这与"超额收益曲线自身的回撤"（即 excess_nav 的水底曲线）是两个不同概念。
    nav_dd = nav / nav.cummax() - 1.0
    bench_dd = bench_nav / bench_nav.cummax() - 1.0
    excess_dd = nav_dd.sub(bench_dd, fill_value=0.0)

    ex_std = excess_daily.std()
    excess_sharpe = (
        excess_daily.mean() / ex_std * np.sqrt(_TRADING_DAYS_PER_YEAR)
        if ex_std and not np.isnan(ex_std) and ex_std > 0 else float("nan")
    )
    excess_annual_return = (
        float(excess_nav.iloc[-1] ** (_TRADING_DAYS_PER_YEAR / n_days) - 1.0)
        if n_days else float("nan")
    )
    # 在用户口径下，excess_max_drawdown = min(excess_dd) = "模型相对基准最大额外下挫"
    excess_max_drawdown = float(excess_dd.min()) if len(excess_dd) else float("nan")

    return BacktestResult(
        nav=nav, daily_ret=portfolio_daily,
        annual_return=float(annual_ret), sharpe=float(sharpe),
        max_drawdown=mdd, annual_turnover=annual_turnover,
        n_days=int(n_days),
        benchmark_nav=bench_nav, benchmark_daily_ret=bench_daily,
        excess_nav=excess_nav, excess_daily_ret=excess_daily,
        excess_annual_return=excess_annual_return,
        excess_sharpe=float(excess_sharpe) if not np.isnan(excess_sharpe) else float("nan"),
        excess_max_drawdown=excess_max_drawdown,
    )


# =============================================================================
# 7. IC / 分组单调性（主分组件）
# =============================================================================
def compute_rank_ic(signal: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """每日横截面 spearman rank IC（用 rank 算 Pearson，对极端值稳健）。"""
    idx = signal.index.intersection(labels.index)
    s = signal.loc[idx]
    y = labels.loc[idx]
    cols = s.columns.intersection(y.columns)
    s = s[cols]; y = y[cols]
    # 截面 spearman = 截面 rank 后的 pearson
    sr = s.rank(axis=1)
    yr = y.rank(axis=1)
    sr_c = sr.sub(sr.mean(axis=1), axis=0)
    yr_c = yr.sub(yr.mean(axis=1), axis=0)
    num = (sr_c * yr_c).sum(axis=1)
    den = np.sqrt((sr_c ** 2).sum(axis=1) * (yr_c ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    valid = (s.notna() & y.notna()).sum(axis=1) >= _MIN_CROSSSECTION_NAMES
    ic = ic.where(valid)
    return ic.dropna()


def compute_pearson_ic(signal: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """每日横截面 Pearson IC（用原始数值算线性相关，捕捉强度信息）。
    与 rank IC 对照看：rank IC 高但 Pearson IC 低 → 因子排序对但量级失真；
    Pearson IC 高但 rank IC 低 → 因子被极端值主导。"""
    idx = signal.index.intersection(labels.index)
    s = signal.loc[idx]
    y = labels.loc[idx]
    cols = s.columns.intersection(y.columns)
    s = s[cols]; y = y[cols]
    # 直接用原始值算 Pearson
    s_c = s.sub(s.mean(axis=1), axis=0)
    y_c = y.sub(y.mean(axis=1), axis=0)
    num = (s_c * y_c).sum(axis=1)
    den = np.sqrt((s_c ** 2).sum(axis=1) * (y_c ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    valid = (s.notna() & y.notna()).sum(axis=1) >= _MIN_CROSSSECTION_NAMES
    ic = ic.where(valid)
    return ic.dropna()


def compute_factor_correlation(
    f1: pd.DataFrame, f2: pd.DataFrame
) -> float:
    """
    两个因子面板（[date × symbol]）的截面 spearman 相关：
        每日各算一次截面 rank-corr，再对所有日期取均值。
    返回 ∈ [-1, 1] 的标量；若两因子完全不重叠返回 NaN。
    """
    idx = f1.index.intersection(f2.index)
    cols = f1.columns.intersection(f2.columns)
    if len(idx) == 0 or len(cols) == 0:
        return float("nan")
    a = f1.loc[idx, cols]
    b = f2.loc[idx, cols]
    ar = a.rank(axis=1)
    br = b.rank(axis=1)
    ac = ar.sub(ar.mean(axis=1), axis=0)
    bc = br.sub(br.mean(axis=1), axis=0)
    num = (ac * bc).sum(axis=1)
    den = np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
    daily = num / den.replace(0, np.nan)
    valid = (a.notna() & b.notna()).sum(axis=1) >= _MIN_CROSSSECTION_NAMES
    daily = daily.where(valid)
    return float(daily.mean()) if daily.notna().any() else float("nan")


def compute_factor_correlation_matrix(
    factors: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    一组已计算的因子面板，返回 [n × n] 截面 spearman 平均相关性矩阵。
    factors: {factor_name: factor_panel}
    """
    names = list(factors.keys())
    n = len(names)
    out = pd.DataFrame(np.eye(n), index=names, columns=names, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            r = compute_factor_correlation(factors[names[i]], factors[names[j]])
            out.iloc[i, j] = out.iloc[j, i] = r
    return out


def compute_group_monotonicity(
    signal: pd.DataFrame, labels: pd.DataFrame, n_groups: int = 10
) -> tuple[float, np.ndarray]:
    """
    分 n_groups 组（按 signal 横截面 rank），返回:
        monotonicity ∈ [-1, 1]：组号与平均收益的 spearman 相关
        group_returns: shape (n_groups,) 各组日均标签
    """
    idx = signal.index.intersection(labels.index)
    s = signal.loc[idx]
    y = labels.loc[idx]
    cols = s.columns.intersection(y.columns)
    s = s[cols]; y = y[cols]

    g = (s.rank(axis=1, pct=True) * n_groups).clip(upper=n_groups - 1e-9)
    g = g.fillna(-1).astype(int)  # -1 = 缺失
    means = np.full(n_groups, np.nan)
    for k in range(n_groups):
        mask = (g == k)
        vals = y.where(mask).stack()
        means[k] = vals.mean() if not vals.empty else np.nan
    if np.isnan(means).any():
        return float("nan"), means
    # 单调性 = spearman(组号, 组均值)
    rk_x = pd.Series(np.arange(n_groups)).rank()
    rk_y = pd.Series(means).rank()
    mono = float(rk_x.corr(rk_y))
    return mono, means


# =============================================================================
# 8. 主分（runner 唯一引用）
#
#   设计动机：
#   v1 公式权重 1.0*IC_IR 让单一指标主导（典型 ~85% 的总分由 IC_IR 贡献），
#   导致"刷 IC 不顾 Sharpe / MDD / 年化收益"的因子也能赢。
#   教学版仅使用 IC 三项，学员可自行扩展加入 Sharpe / MDD 等。
#   每个分量先归一到 [-2, +2] 量级再加权 —— 权重数字本身就是"重要性比例"。
# =============================================================================
SCORE_VERSION: str = "demo_v1"


@dataclass
class ScoreReport:
    score: float
    # rank IC 系列（A 股研究主用，对极端值稳健）
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float
    # pearson IC 系列（捕捉强度信息，做对照）
    pearson_ic_mean: float
    pearson_ic_std: float
    pearson_ic_ir: float
    # 别名（向下兼容老代码 / 报表）
    ic_mean: float       # = rank_ic_mean
    ic_std: float        # = rank_ic_std
    ic_ir: float         # = rank_ic_ir
    # 组合 / 回测指标
    monotonicity: float
    sharpe: float
    annual_return: float
    max_drawdown: float
    annual_turnover: float
    n_days: int
    horizon: int
    label_kind: str
    # 公式各分量贡献（便于诊断哪一项在拉分）
    score_breakdown: dict
    score_version: str
    # 超额指标（vs 中证 1000，仅展示，不进 score）
    excess_annual_return: float = float("nan")
    excess_sharpe: float = float("nan")
    excess_max_drawdown: float = float("nan")


def primary_score(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int,
    label_kind: LabelKind = "market_neutral",
) -> ScoreReport:
    """
    主分 · 教学版（demo v1 · IC 三项组合）
    ================================================================
    专注 IC 类指标，让学员看清 “Agent 是在优化预测信号本身”。
    Sharpe / 年化 / 回撤 / 单调性 / 换手率仍会计算并展示，但不进 score。
    学员之后可以自己把这些指标加进公式，观察分数如何变化。

    score = 1.0 * rank_ic_ir  +  10.0 * pearson_ic_mean  +  10.0 * rank_ic_mean

    含义：
      - rank_ic_ir       稳定性（IC 均值 / 标准差 · 年化因子）· 主项
      - pearson_ic_mean  截面强度（原始值相关性均值）· 辅项
      - rank_ic_mean     截面单调（排序相关性均值）· 辅项
    """
    validate_signal(signal, "primary_score.signal")

    labels = make_labels(panel, horizon=horizon, kind=label_kind)

    # ---- 双 IC 计算（全期等权，教学口径清晰）----
    rank_ic = compute_rank_ic(signal, labels)
    rank_ic_mean = float(rank_ic.mean()) if len(rank_ic) else float("nan")
    rank_ic_std = float(rank_ic.std()) if len(rank_ic) else float("nan")
    rank_ic_ir = (
        rank_ic_mean / rank_ic_std * np.sqrt(_TRADING_DAYS_PER_YEAR / horizon)
        if rank_ic_std and rank_ic_std > 0 else float("nan")
    )

    pearson_ic = compute_pearson_ic(signal, labels)
    pearson_ic_mean = float(pearson_ic.mean()) if len(pearson_ic) else float("nan")
    pearson_ic_std = float(pearson_ic.std()) if len(pearson_ic) else float("nan")
    pearson_ic_ir = (
        pearson_ic_mean / pearson_ic_std * np.sqrt(_TRADING_DAYS_PER_YEAR / horizon)
        if pearson_ic_std and pearson_ic_std > 0 else float("nan")
    )

    # ---- 展示用指标（不进 score，供 runs.jsonl / 因子卡 / 学员自己扩展公式时使用）----
    mono, _ = compute_group_monotonicity(signal, labels, n_groups=10)
    bt = backtest(signal, panel, horizon=horizon)

    # ---- 安全取数（NaN 视为 0）----
    def _s(x):
        return 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

    rank_ir_v = _s(rank_ic_ir)
    rank_ic_mean_v = _s(rank_ic_mean)
    pearson_ic_mean_v = _s(pearson_ic_mean)

    # ---- 教学版分数公式 ----
    parts = {
        "rank_ic_ir_term":       1.0  * rank_ir_v,
        "pearson_ic_mean_term":  10.0 * pearson_ic_mean_v,
        "rank_ic_mean_term":     10.0 * rank_ic_mean_v,
    }
    score = sum(parts.values())

    breakdown = {
        "raw": {
            "rank_ic_ir":       rank_ir_v,
            "rank_ic_mean":     rank_ic_mean_v,
            "pearson_ic_mean":  pearson_ic_mean_v,
            # 下列指标不进 score，仅展示（学员可自行扩展公式使用）：
            "sharpe":           _s(bt.sharpe),
            "annual_return":    _s(bt.annual_return),
            "max_drawdown":     _s(bt.max_drawdown),
            "monotonicity":     _s(mono),
            "annual_turnover":  _s(bt.annual_turnover),
        },
        "weighted": parts,
        "weights": {
            "rank_ic_ir":       1.0,
            "pearson_ic_mean":  10.0,
            "rank_ic_mean":     10.0,
        },
        "note": (
            "demo v1 教学版：score = 1.0*rank_ic_ir + 10.0*pearson_ic_mean + 10.0*rank_ic_mean。"
            "Sharpe / 年化 / 回撤 / 单调性 / 换手率仅展示，不进 score。"
            "学员可在 prepare.primary_score 里自行扩展公式。"
        ),
    }

    return ScoreReport(
        score=float(score),
        rank_ic_mean=rank_ic_mean, rank_ic_std=rank_ic_std, rank_ic_ir=float(rank_ic_ir) if not np.isnan(rank_ic_ir) else float("nan"),
        pearson_ic_mean=pearson_ic_mean, pearson_ic_std=pearson_ic_std,
        pearson_ic_ir=float(pearson_ic_ir) if not np.isnan(pearson_ic_ir) else float("nan"),
        # 别名（rank IC 系列）
        ic_mean=rank_ic_mean, ic_std=rank_ic_std,
        ic_ir=float(rank_ic_ir) if not np.isnan(rank_ic_ir) else float("nan"),
        monotonicity=float(mono) if not np.isnan(mono) else float("nan"),
        sharpe=bt.sharpe, annual_return=bt.annual_return, max_drawdown=bt.max_drawdown,
        annual_turnover=bt.annual_turnover, n_days=bt.n_days,
        horizon=horizon, label_kind=label_kind,
        score_breakdown=breakdown,
        score_version=SCORE_VERSION,
        # 超额指标（来自扩展后的 BacktestResult）
        excess_annual_return=float(getattr(bt, "excess_annual_return", float("nan"))),
        excess_sharpe=float(getattr(bt, "excess_sharpe", float("nan"))),
        excess_max_drawdown=float(getattr(bt, "excess_max_drawdown", float("nan"))),
    )


# =============================================================================
# 9. CLI smoke test：python prepare.py
# =============================================================================
if __name__ == "__main__":
    print("[prepare] ensuring splits.json + parquet checksum ...")
    meta = _ensure_splits()
    for seg, (lo, hi) in meta["splits"].items():
        print(f"  {seg:5s}: {lo} → {hi}")
    print(f"  checksum:           {meta['checksum'][:16]}...")

    cal = _load_trading_calendar()
    print(f"\n[prepare] trading calendar loaded: {len(cal)} days "
          f"({cal[0].date()} → {cal[-1].date()})")

    print(f"\n[prepare] score formula (demo_v1): "
          f"1.0 * rank_ic_ir + 10.0 * pearson_ic_mean + 10.0 * rank_ic_mean")

    bench = load_benchmark_series()
    print(f"[prepare] benchmark loaded: {bench.notna().sum()}/{len(bench)} non-NaN days "
          f"(000852.SH close)")

    print("\n[prepare] loading train panel ...")
    train = load_train_panel()
    print(f"  train shape: {train.shape}, symbols: {train['symbol'].nunique()}, "
          f"dates: {train['date'].min().date()} → {train['date'].max().date()}")

    print("\n[prepare] making labels (horizon=5, kind=market_neutral) ...")
    y = make_labels(train, horizon=5, kind="market_neutral")
    print(f"  labels shape: {y.shape}, non-NaN cells: {y.notna().sum().sum()}")

    print("\n[prepare] sanity-check: a tiny baseline signal (-volatility) ...")
    close_w = _pivot(train, "close").sort_index()
    ret = close_w.pct_change(fill_method=None)
    vol20 = ret.rolling(20, min_periods=10).std()
    sig = (-vol20).sub((-vol20).mean(axis=1), axis=0).div(
        (-vol20).std(axis=1).replace(0, np.nan), axis=0
    )
    sig = sig.dropna(how="all")
    validate_signal(sig, "smoke.sig")
    print(f"  signal shape: {sig.shape}, daily |mean|≈"
          f"{sig.mean(axis=1).abs().mean():.4f}, daily std≈{sig.std(axis=1).mean():.3f}")

    print("\n[prepare] computing primary_score on train (smoke only) ...")
    rpt = primary_score(sig, train, horizon=5, label_kind="market_neutral")
    print(f"  score            : {rpt.score:+.4f}")
    print(f"  rank_ic_ir       : {rpt.rank_ic_ir:+.4f}    (rank_ic_mean = {rpt.rank_ic_mean:+.4f})")
    print(f"  pearson_ic_ir    : {rpt.pearson_ic_ir:+.4f}    (pearson_ic_mean = {rpt.pearson_ic_mean:+.4f})")
    print(f"  monotonicity     : {rpt.monotonicity:+.4f}")
    print(f"  sharpe           : {rpt.sharpe:+.4f}")
    print(f"  annual_ret       : {rpt.annual_return:+.4%}")
    print(f"  max_drawdown     : {rpt.max_drawdown:+.4%}")
    print(f"  ann_turnover     : {rpt.annual_turnover:.2f}")
    print(f"  n_days           : {rpt.n_days}")
    print(f"  ---- 超额（vs 中证 1000，仅展示不进 score）----")
    print(f"  excess_ret       : {rpt.excess_annual_return:+.4%}")
    print(f"  excess_sharpe    : {rpt.excess_sharpe:+.4f}")
    print(f"  excess_mdd       : {rpt.excess_max_drawdown:+.4%}")
    print(f"  ---- 各分量 (weighted contribution to score) ----")
    for k, v in rpt.score_breakdown["weighted"].items():
        print(f"    {k:<24}: {v:+.4f}")

    print("\n[prepare] OK.")
