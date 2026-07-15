"""
metrics.py — agent 可扩展的诊断指标（不进入主分）

约束（详见 evaluation.md §4）：
    - 这些指标 *不影响* runner 的 accept/revert（主分由 prepare.primary_score 唯一产出）
    - 只用于写入 journal/runs.jsonl 给人类回看，或给 agent 自己做下一轮决策参考
    - 严禁在这里偷偷加权再 return；严禁触碰 prepare 私有成员

agent 想加新角度（如行业 IC、市值分组 IC）尽情加，规则：
    - 函数签名风格一致（吃 signal/labels/panel，吐数字或 dict）
    - 返回值最终汇入 diagnose() 的字典
    - 命名以 d_ 开头便于人类一眼识别
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# 基础诊断（baseline 提供）
# =============================================================================
def d_ic_decay(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """
    IC 在不同 horizon 下的衰减曲线。理想因子应在中短期保持 IC 单调下降。
    """
    open_w = panel.pivot(index="date", columns="symbol", values="open").sort_index()
    out = {}
    for h in horizons:
        buy = open_w.shift(-1)
        sell = open_w.shift(-1 - h)
        y = sell / buy - 1.0
        # 截面 spearman
        ic = _cs_spearman(signal, y).mean()
        out[f"ic_h{h}"] = float(ic) if pd.notna(ic) else float("nan")
    return out


def d_quantile_returns(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int,
    n_groups: int = 10,
) -> list[float]:
    """
    n 分组组均收益（H 日总收益的截面平均）。Top - Bottom 是常见 alpha 衡量。
    """
    open_w = panel.pivot(index="date", columns="symbol", values="open").sort_index()
    buy = open_w.shift(-1)
    sell = open_w.shift(-1 - horizon)
    y = sell / buy - 1.0

    cols = signal.columns.intersection(y.columns)
    s = signal[cols]; y = y.reindex(columns=cols).loc[s.index.intersection(y.index)]
    s = s.loc[y.index]

    g = (s.rank(axis=1, pct=True) * n_groups).clip(upper=n_groups - 1e-9)
    g = g.fillna(-1).astype(int)

    means = []
    for k in range(n_groups):
        mask = (g == k)
        v = y.where(mask).stack()
        means.append(float(v.mean()) if not v.empty else float("nan"))
    return means


def d_factor_correlations(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    因子两两相关性（截面平均的 spearman）。理想：< 0.6。
    factors: {name: signal_df}
    """
    names = list(factors.keys())
    n = len(names)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for i in range(n):
        for j in range(i, n):
            r = _cs_spearman(factors[names[i]], factors[names[j]]).mean()
            out.iloc[i, j] = out.iloc[j, i] = float(r) if pd.notna(r) else float("nan")
    return out


def d_top_basket_stability(signal: pd.DataFrame, top_pct: float = 0.10) -> float:
    """
    Top 10% 篮子相邻日重合度（Jaccard）。越高越稳定，意味着换手低。
    """
    q = signal.rank(axis=1, pct=True)
    top = q >= (1.0 - top_pct)
    inter = (top & top.shift(1)).sum(axis=1)
    union = (top | top.shift(1)).sum(axis=1).replace(0, np.nan)
    return float((inter / union).mean())


# =============================================================================
# 工具
# =============================================================================
def _cs_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """逐行（截面）spearman 相关。返回 pd.Series，index=date。"""
    idx = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    a = a.loc[idx, cols]; b = b.loc[idx, cols]
    ar = a.rank(axis=1)
    br = b.rank(axis=1)
    ac = ar.sub(ar.mean(axis=1), axis=0)
    bc = br.sub(br.mean(axis=1), axis=0)
    num = (ac * bc).sum(axis=1)
    den = np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
    return num / den.replace(0, np.nan)


# =============================================================================
# 集成入口（runner 调用）
# =============================================================================
def diagnose(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    """
    runner 在 primary_score 之后调用。返回 JSON 友好的诊断字典。
    扩展方法：在此函数里 append 新键即可，不要改 prepare.primary_score。
    """
    return {
        "ic_decay": d_ic_decay(signal, panel),
        "quantile_returns": d_quantile_returns(signal, panel, horizon),
        "top_basket_jaccard": d_top_basket_stability(signal),
    }
