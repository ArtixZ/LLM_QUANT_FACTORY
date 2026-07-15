"""
runner.py — AutoAlpha 实验循环（autoresearch 范式的本地版本）

职责：
    1. 锁定 test 段（设置环境变量）
    2. 调用 alpha.run() 跑一次完整 pipeline
    3. 信号校验 + 主分计算 + 诊断指标
    4. 本地 journal 管理（替代 git）：
        - journal/runs.jsonl    所有实验记录（成功+失败）append-only
        - journal/best.json     当前最优 score 与对应快照路径
        - journal/snapshots/    每次 accept 时的 alpha.py 快照
        - journal/last_failed/  上一次失败的 alpha.py（便于 agent 调试）
    5. accept/revert：分数提升 → 保留并 snapshot；下降或崩溃 → 回滚到 best

用法：
    python runner.py once      # 单次评估当前 alpha.py
    python runner.py status    # 查看当前 best
    python runner.py reset     # 删除 journal 重新开始（慎用）
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Windows GBK locale 下 print(⚠/汉字/箭头) 会 UnicodeEncodeError，
# 进而吞掉 anomaly 详情、阻断 alpha.py 自动 revert。强制 stdout/stderr 用 utf-8。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 在导入 prepare 前锁死 test 段
os.environ.setdefault("AUTOALPHA_TEST_LOCKED", "1")

import prepare  # noqa: E402

ROOT = Path(__file__).resolve().parent
ALPHA_PATH = ROOT / "alpha.py"
JOURNAL_DIR = ROOT / "journal"
SNAPSHOTS_DIR = JOURNAL_DIR / "snapshots"
LAST_FAILED_DIR = JOURNAL_DIR / "last_failed"
NOTES_DIR = JOURNAL_DIR / "notes"
BEST_PATH = JOURNAL_DIR / "best.json"
RUNS_PATH = JOURNAL_DIR / "runs.jsonl"

# 允许的 op_type（agent 在 ITER_NOTE 里必须声明其一）
ALLOWED_OP_TYPES: tuple[str, ...] = (
    "add_factor",       # 加新因子（强制相关性门控）
    "modify_factor",    # 改已有因子的实现/参数（与原版本算相关性）
    "delete_factor",    # 删因子
    "combine_method",   # 改组合方法（等权 / IC_IR 加权 / 其它）
    "label_kind",       # 改 LABEL_KIND
    "horizon",          # 改 HORIZON
    "preprocess",       # 改预处理（winsorize / zscore / 标准化）
    "other",            # 其它（如代码重构，分数不应变）
)

# 因子相关性阈值
_CORR_HARD_REJECT = 0.85   # |ρ| ≥ 此值 → 直接 PermissionError
_CORR_WARN = 0.60          # |ρ| ≥ 此值 → 警告


# =============================================================================
# Journal 管理
# =============================================================================
def _ensure_journal() -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    LAST_FAILED_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    if not BEST_PATH.exists():
        BEST_PATH.write_text(json.dumps({
            "score": float("-inf"),
            "snapshot": None,
            "iter_id": 0,
            "ts": None,
        }, indent=2), encoding="utf-8")
    if not RUNS_PATH.exists():
        RUNS_PATH.write_text("", encoding="utf-8")


def _read_best() -> dict[str, Any]:
    return json.loads(BEST_PATH.read_text(encoding="utf-8"))


def _write_best(best: dict[str, Any]) -> None:
    BEST_PATH.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")


def _next_iter_id() -> int:
    if not RUNS_PATH.exists():
        return 1
    n = 0
    with RUNS_PATH.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n + 1


def _append_run(rec: dict[str, Any]) -> None:
    with RUNS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# =============================================================================
# 评估一次
# =============================================================================
# 静态扫描黑名单：alpha.py 源码出现这些字符串即视为越权
# 防止 agent 在循环里偷读 test 段、其它实验的 test 指标、过往最优分数等
_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "_load_test_panel",         # prepare 私有：测试集加载
    "AUTOALPHA_TEST_LOCKED",    # 测试段环境变量锁
    "factor_library",           # 因子库目录/模块（含 _test_metrics.json）
    "_test_metrics",            # test 指标文件名
    "journal",                  # journal/ 目录（含 best.json / runs.jsonl）
    "test_eval",                # judge.py 写入的 test 评估文件
    "judge",                    # judge.py 模块
    "splits.json",              # 切分文件（含 test 段日期）
)


def _scan_alpha_source(path: Path) -> list[str]:
    """返回 alpha.py 源码中命中的禁词；空列表 = 通过。"""
    src = path.read_text(encoding="utf-8")
    hits: list[str] = []
    for tok in _FORBIDDEN_TOKENS:
        if tok in src:
            hits.append(tok)
    return hits


def _validate_iter_note(alpha_mod) -> dict[str, Any]:
    """
    校验 alpha.ITER_NOTE 字典完整性：
        必须有 op_type / hypothesis / change / expected
        op_type 必须 ∈ ALLOWED_OP_TYPES
    返回规范化后的 note dict（带默认值）。缺失即 ValueError。
    """
    note = getattr(alpha_mod, "ITER_NOTE", None)
    if not isinstance(note, dict):
        raise ValueError(
            "alpha.py 缺少 ITER_NOTE: dict —— 每次实验都必须声明本次假设。"
            "示例：见 program.md §7.1。"
        )
    required = ("op_type", "hypothesis", "change", "expected")
    missing = [k for k in required if not note.get(k)]
    if missing:
        raise ValueError(f"ITER_NOTE 缺少必要字段: {missing}")
    if note["op_type"] not in ALLOWED_OP_TYPES:
        raise ValueError(
            f"ITER_NOTE.op_type={note['op_type']!r} 不在白名单 {ALLOWED_OP_TYPES}"
        )
    return {
        "op_type": str(note["op_type"]),
        "hypothesis": str(note["hypothesis"]),
        "change": str(note["change"]),
        "expected": str(note["expected"]),
        "parent_iter": note.get("parent_iter"),
        "reasoning": str(note.get("reasoning", "")),
    }


def _correlation_gate(alpha_mod, train_panel, iter_id: int) -> dict[str, Any]:
    """
    add_factor / modify_factor 时强制做截面相关性检查。
    对 add_factor：新因子 vs 每个旧因子，|ρ|>=0.85 → PermissionError 拒收。
    对 modify_factor：警告同名旧因子被改后相关性 >0.95（"等于没改"）。
    返回相关性矩阵的摘要 dict（写进 note）。
    """
    op_type = alpha_mod.ITER_NOTE["op_type"]
    if op_type not in ("add_factor", "modify_factor"):
        return {}

    factors = getattr(alpha_mod, "FACTORS", [])
    if len(factors) < 2:
        return {}

    # 计算所有因子 + L1 标准化（与 alpha._compute_factors 一致逻辑）
    cs = getattr(alpha_mod, "cs_winsorize_zscore")
    panels = {fn.__name__: cs(fn(train_panel)) for fn in factors}

    corr_mat = prepare.compute_factor_correlation_matrix(panels)
    summary = {
        "names": list(panels.keys()),
        "matrix": corr_mat.round(3).to_dict(),
        "max_offdiag_abs": float(
            corr_mat.where(~np.eye(len(panels), dtype=bool)).abs().max().max()
        ),
    }

    if op_type == "add_factor":
        # 找出"新"因子：notes 里若声明了 new_factor 字段，用它；否则取最后一个
        new_name = alpha_mod.ITER_NOTE.get("new_factor") or list(panels.keys())[-1]
        if new_name not in panels:
            raise ValueError(f"ITER_NOTE.new_factor={new_name!r} 不在 FACTORS 中")
        # 新因子 vs 其它每个旧因子
        offending = []
        warnings_list = []
        for other in panels:
            if other == new_name:
                continue
            r = corr_mat.loc[new_name, other]
            if pd.notna(r) and abs(r) >= _CORR_HARD_REJECT:
                offending.append((other, float(r)))
            elif pd.notna(r) and abs(r) >= _CORR_WARN:
                warnings_list.append((other, float(r)))
        summary["new_factor"] = new_name
        summary["warnings"] = warnings_list
        summary["offending"] = offending
        if offending:
            # 写入 last_failed/correlation_{iter}.txt
            msg = (
                f"add_factor 被相关性门控拒绝。\n"
                f"new_factor = {new_name}\n"
                + "\n".join(f"  vs {n}: ρ={r:+.3f}  (>={_CORR_HARD_REJECT})"
                            for n, r in offending)
                + f"\n阈值: |ρ| ≥ {_CORR_HARD_REJECT}"
            )
            (LAST_FAILED_DIR / f"correlation_{iter_id:04d}.txt").write_text(
                msg + "\n\n完整矩阵:\n" + corr_mat.round(3).to_string(),
                encoding="utf-8",
            )
            raise PermissionError(msg)
        if warnings_list:
            print(f"  [corr warn] {new_name} 与:")
            for n, r in warnings_list:
                print(f"    {n}: ρ={r:+.3f}")
    return summary


def _write_note(iter_id: int, note: dict[str, Any], rec: dict[str, Any]) -> Path:
    """把 ITER_NOTE + 跑分结果写到 journal/notes/{iter}.md。"""
    p = NOTES_DIR / f"{iter_id:04d}.md"
    lines = [
        f"# Iter #{iter_id:04d}  ({rec.get('ts','')})",
        "",
        f"- **op_type**: `{note['op_type']}`",
        f"- **parent_iter**: {note.get('parent_iter') or '—'}",
        "",
        "## 假设 (Hypothesis)",
        note["hypothesis"],
        "",
        "## 变更 (Change)",
        note["change"],
        "",
        "## 预期 (Expected)",
        note["expected"],
        "",
    ]
    if note.get("reasoning"):
        lines += ["## 依据 (Reasoning)", note["reasoning"], ""]
    if note.get("correlation"):
        c = note["correlation"]
        lines += [
            "## 相关性门控",
            f"- 最大非对角 |ρ| = {c.get('max_offdiag_abs', float('nan')):.3f}",
        ]
        if c.get("warnings"):
            lines.append("- 警告 (0.6 ≤ |ρ| < 0.85):")
            for n, r in c["warnings"]:
                lines.append(f"  - vs `{n}`: ρ = {r:+.3f}")
        lines.append("")

    lines += [
        "## 实际结果",
        f"- **status**: `{rec.get('status')}`",
        f"- **score**: `{rec.get('score', float('nan')):+.4f}`",
        f"- **decision**: `{rec.get('decision', '')}`",
        f"- elapsed: {rec.get('elapsed_sec', 0):.1f}s",
    ]
    if rec.get("status") == "ok":
        lines += [
            f"- rank_ic_ir = {rec.get('rank_ic_ir', rec.get('ic_ir', float('nan'))):+.3f}, "
            f"pearson_ic_ir = {rec.get('pearson_ic_ir', float('nan')):+.3f}",
            f"- monotonicity = {rec.get('monotonicity', float('nan')):+.3f}, "
            f"sharpe = {rec.get('sharpe', float('nan')):+.3f}, "
            f"annual_return = {rec.get('annual_return', float('nan')):+.2%}",
            f"- turnover = {rec.get('annual_turnover', float('nan')):.1f}, "
            f"mdd = {rec.get('max_drawdown', float('nan')):+.2%}",
            f"- excess_ret = {rec.get('excess_annual_return', float('nan')):+.2%}, "
            f"excess_sharpe = {rec.get('excess_sharpe', float('nan')):+.3f}, "
            f"excess_mdd = {rec.get('excess_max_drawdown', float('nan')):+.2%}",
        ]
        # v2 分量分解（如果有）
        bd = rec.get("score_breakdown")
        if bd and isinstance(bd, dict) and "weighted" in bd:
            lines.append("")
            lines.append("### v2 分量分解（weighted contribution）")
            for k, v in bd["weighted"].items():
                lines.append(f"- {k}: {v:+.4f}")
    if rec.get("error"):
        lines += ["", "## 错误", "```", str(rec["error"]), "```"]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _reload_alpha():
    """每轮强制 reimport，确保拿到最新文件内容。"""
    if "alpha" in sys.modules:
        importlib.reload(sys.modules["alpha"])
    else:
        importlib.import_module("alpha")
    return sys.modules["alpha"]


def evaluate_once(iter_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    返回 (rec, artifacts):
        rec       : 落 runs.jsonl 用的轻量字典
        artifacts : 重量级对象（signal/panel/score_report/diag/alpha 模块），
                    供 run_once 在 ACCEPTED 时归档到 factor_library
    """
    rec: dict[str, Any] = {
        "iter_id": iter_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "status": "unknown",
        "score": float("-inf"),
        "elapsed_sec": None,
        "error": None,
    }
    artifacts: dict[str, Any] = {}
    note: dict[str, Any] = {}
    t0 = time.time()
    try:
        # ---- 防泄露静态扫描：先于 import alpha 执行 ----
        hits = _scan_alpha_source(ALPHA_PATH)
        if hits:
            raise PermissionError(
                "alpha.py 源码命中防泄露黑名单 "
                f"{hits} —— agent 不得访问 test 段或归档元数据。"
                f"详见 program.md 硬约束清单。"
            )

        alpha = _reload_alpha()

        # 契约校验
        horizon = getattr(alpha, "HORIZON", None)
        label_kind = getattr(alpha, "LABEL_KIND", None)
        if horizon not in prepare.ALLOWED_HORIZONS:
            raise ValueError(f"alpha.HORIZON={horizon} 不在白名单 {prepare.ALLOWED_HORIZONS}")
        if label_kind not in prepare.ALLOWED_LABEL_KINDS:
            raise ValueError(f"alpha.LABEL_KIND={label_kind} 不在菜单 {prepare.ALLOWED_LABEL_KINDS}")

        # ---- ITER_NOTE 校验（每次必须声明假设）----
        note = _validate_iter_note(alpha)

        # 加载数据（test 段已被环境变量锁定）
        train = prepare.load_train_panel()
        val = prepare.load_val_panel()

        # ---- 因子相关性门控（仅 add_factor / modify_factor）----
        corr_summary = _correlation_gate(alpha, train, iter_id)
        if corr_summary:
            note["correlation"] = corr_summary

        # ---- 主评分：固定 train/val 切分 ----
        sig_train, sig_val = alpha.run(train, val)
        prepare.validate_signal(sig_train, "alpha.sig_train")
        prepare.validate_signal(sig_val, "alpha.sig_val")
        rpt = prepare.primary_score(
            sig_val, val, horizon=horizon, label_kind=label_kind,
        )
        sig_for_diag = sig_val
        panel_for_diag = val

        # 诊断指标（不进主分）
        try:
            import metrics
            diag = metrics.diagnose(sig_for_diag, panel_for_diag, horizon=horizon)
        except Exception as e:
            diag = {"_diag_error": f"{type(e).__name__}: {e}"}

        rec.update({
            "status": "ok",
            "score": rpt.score,
            "score_version": getattr(rpt, "score_version", "demo_v1"),
            "horizon": rpt.horizon,
            "label_kind": rpt.label_kind,
            "rank_ic_mean": getattr(rpt, "rank_ic_mean", rpt.ic_mean),
            "rank_ic_ir":   getattr(rpt, "rank_ic_ir", rpt.ic_ir),
            "pearson_ic_mean": getattr(rpt, "pearson_ic_mean", float("nan")),
            "pearson_ic_ir":   getattr(rpt, "pearson_ic_ir", float("nan")),
            # 别名（向下兼容）
            "ic_mean": rpt.ic_mean,
            "ic_ir": rpt.ic_ir,
            "monotonicity": rpt.monotonicity,
            "sharpe": rpt.sharpe,
            "annual_return": rpt.annual_return,
            "max_drawdown": rpt.max_drawdown,
            "annual_turnover": rpt.annual_turnover,
            "n_days": rpt.n_days,
            # 超额（vs 中证 1000；不进 score，仅展示与诊断）
            "excess_annual_return": getattr(rpt, "excess_annual_return", float("nan")),
            "excess_sharpe":         getattr(rpt, "excess_sharpe", float("nan")),
            "excess_max_drawdown":   getattr(rpt, "excess_max_drawdown", float("nan")),
            "score_breakdown": getattr(rpt, "score_breakdown", None),
            "diag": diag,
        })

        # 把"重对象"装进 artifacts，供 ACCEPTED 分支归档使用
        artifacts.update({
            "alpha": alpha,
            "sig_val": sig_for_diag,
            "val_panel": panel_for_diag,
            "rpt": rpt,
            "diag": diag,
        })
    except Exception as e:
        rec["status"] = "crash"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=5)
    finally:
        rec["elapsed_sec"] = round(time.time() - t0, 2)
        # 不论 ok / crash 都把 note 暴露给 run_once
        artifacts["note"] = note

    return rec, artifacts


# =============================================================================
# Accept / Revert
# =============================================================================
def _snapshot_alpha(iter_id: int, score: float) -> Path:
    fname = f"run_{iter_id:04d}_score_{score:+.4f}.py".replace("+", "p").replace("-", "n")
    dst = SNAPSHOTS_DIR / fname
    shutil.copy2(ALPHA_PATH, dst)
    return dst


def _stash_failed(iter_id: int, error: str) -> None:
    """把失败的 alpha.py 移走，避免下一次还在用同一份 broken 代码。"""
    dst = LAST_FAILED_DIR / f"run_{iter_id:04d}_failed.py"
    shutil.copy2(ALPHA_PATH, dst)
    (LAST_FAILED_DIR / f"run_{iter_id:04d}_error.txt").write_text(error, encoding="utf-8")


# 异常熔断阈值（详见 program.md §13）
_ANOMALY_SHARPE_THRESHOLD = 0.30   # sharpe 较 best 提升 >30%
_ANOMALY_MDD_THRESHOLD = 0.20      # mdd 较 best 改善 >20%（变浅 = 数值变大 = 改善）
_ANOMALY_RET_THRESHOLD = 0.30      # annual_return 较 best 提升 >30%


def _detect_score_anomaly(rec: dict[str, Any], best: dict[str, Any]) -> dict[str, Any] | None:
    """在 score REJECTED 但底层指标显著改善时标记 anomaly。

    返回 dict 含 sharpe_improvement / mdd_improvement / ret_improvement / message；
    若无 anomaly 返回 None。
    """
    # best.json 里没记录 sharpe/mdd/ret（旧 schema），跳过检测
    best_sharpe = best.get("sharpe")
    best_mdd = best.get("max_drawdown")
    best_ret = best.get("annual_return")
    if best_sharpe is None or best_mdd is None or best_ret is None:
        return None

    cur_sharpe = rec.get("sharpe", 0.0)
    cur_mdd = rec.get("max_drawdown", 0.0)
    cur_ret = rec.get("annual_return", 0.0)

    def _safe_imp(cur, base, *, lower_better=False):
        denom = abs(base) if abs(base) > 1e-9 else 1.0
        if lower_better:
            return (base - cur) / denom  # mdd: -10% vs -16% → (−16 − (−10))/16 = 6/16 = +0.375 → 改善
        return (cur - base) / denom

    sharpe_imp = _safe_imp(cur_sharpe, best_sharpe)
    mdd_imp = _safe_imp(cur_mdd, best_mdd, lower_better=True)
    ret_imp = _safe_imp(cur_ret, best_ret)

    triggered = []
    if sharpe_imp > _ANOMALY_SHARPE_THRESHOLD:
        triggered.append(f"sharpe +{sharpe_imp:.0%}")
    if mdd_imp > _ANOMALY_MDD_THRESHOLD:
        triggered.append(f"mdd 改善 +{mdd_imp:.0%}")
    if ret_imp > _ANOMALY_RET_THRESHOLD:
        triggered.append(f"ret +{ret_imp:.0%}")

    if not triggered:
        return None

    return {
        "sharpe_improvement": float(sharpe_imp),
        "mdd_improvement": float(mdd_imp),
        "ret_improvement": float(ret_imp),
        "triggers": triggered,
        "message": "score REJECTED 但底层指标显著改善（" + ", ".join(triggered) + "）— 可能 score 公式有缺陷",
    }


def _revert_to_best(best: dict[str, Any]) -> None:
    if best["snapshot"] is None:
        return  # 还没有 best，保留当前 alpha.py 让 agent 自己修
    src = Path(best["snapshot"])
    if src.exists():
        # 拷回前先扫一遍 best snapshot 自己；命中禁词不动 alpha.py，
        # 让 agent / 人类介入清洗，而不是默默把含禁词的旧版恢复到工作区
        legacy_hits = []
        try:
            legacy_src = src.read_text(encoding="utf-8")
            for tok in _FORBIDDEN_TOKENS:
                if tok in legacy_src:
                    legacy_hits.append(tok)
        except Exception:
            pass
        if legacy_hits:
            print(f"  ⚠ best snapshot 含禁词 {legacy_hits}，跳过自动 revert。"
                  f"\n    请人工编辑 {src} 后重试，或重置 journal。")
            return
        shutil.copy2(src, ALPHA_PATH)


def run_once() -> dict[str, Any]:
    _ensure_journal()
    best = _read_best()
    iter_id = _next_iter_id()
    print(f"\n=== run #{iter_id} (best so far: {best['score']:+.4f}) ===")
    rec, artifacts = evaluate_once(iter_id)

    if rec["status"] == "ok" and rec["score"] > best["score"]:
        snap = _snapshot_alpha(iter_id, rec["score"])
        prev_best_score = best["score"]
        best.update({
            "score": rec["score"],
            "snapshot": str(snap.relative_to(ROOT)),
            "iter_id": iter_id,
            "ts": rec["ts"],
            "horizon": rec.get("horizon"),
            "label_kind": rec.get("label_kind"),
            # 写入底层指标，给异常熔断判定用
            "sharpe": rec.get("sharpe"),
            "max_drawdown": rec.get("max_drawdown"),
            "annual_return": rec.get("annual_return"),
            "score_version": rec.get("score_version", "demo_v1"),
        })
        _write_best(best)
        rec["decision"] = "ACCEPTED"
        print(f"[ACCEPTED] score {rec['score']:+.4f} (was {prev_best_score:+.4f}) "
              f"-> {snap.name}")

        # 归档到 factor_library/（不影响 accept 决策；失败仅记日志）
        try:
            import factor_library
            alpha_mod = artifacts["alpha"]
            factor_name = getattr(alpha_mod, "FACTOR_NAME", None) or f"auto_v{iter_id}"
            factor_function_names = [
                fn.__name__ for fn in getattr(alpha_mod, "FACTORS", [])
            ]
            parent_card = factor_library.get_best_card()  # 上一最优卡（None=首次）
            ar = factor_library.archive(
                iter_id=iter_id,
                score_report=artifacts["rpt"],
                diag=artifacts["diag"],
                signal=artifacts["sig_val"],
                panel=artifacts["val_panel"],
                alpha_path=ALPHA_PATH,
                factor_name=factor_name,
                factor_function_names=factor_function_names,
                parent_card=parent_card,
                note=artifacts.get("note"),
            )
            if ar is not None:
                rec["factor_library"] = ar.dirname
                print(f"           archived -> factor_library/{ar.dirname}/")
        except Exception as e:
            print(f"           [archive warn] {type(e).__name__}: {e}")
    elif rec["status"] == "ok":
        rec["decision"] = "REJECTED"
        print(f"[REJECTED] score {rec['score']:+.4f} (best stays {best['score']:+.4f})")

        # ---- 异常熔断：score REJECTED 但底层指标显著改善 → 标记 + 高亮 ----
        # （详见 program.md §13）
        anomaly = _detect_score_anomaly(rec, best)
        if anomaly:
            rec["score_anomaly"] = anomaly
            print(f"  ⚠ SCORE_ANOMALY: {anomaly['message']}")
            for k in ("sharpe_improvement", "mdd_improvement", "ret_improvement"):
                v = anomaly.get(k, 0)
                if abs(v) > 0.01:
                    print(f"    {k}: {v:+.1%}")
            print(f"  ⚠ agent 应按 program.md §13 暂停迭代并报告人类")

        # program.md §7：分数下降也回滚到 best snapshot（与 CRASH 同处理）
        # 这样 agent 下一轮无需手动 cp 恢复 best；REJECTED 后 alpha.py 自动回到上一 best
        _revert_to_best(best)
        if best["snapshot"]:
            print(f"  reverted alpha.py to {best['snapshot']}")
    else:
        _stash_failed(iter_id, rec.get("error", ""))
        _revert_to_best(best)
        rec["decision"] = "REVERTED"
        print(f"[CRASH] {rec['error']}")
        if best["snapshot"]:
            print(f"  reverted alpha.py to {best['snapshot']}")

    # 写 ITER_NOTE + 实验结果到 journal/notes/{iter}.md
    if artifacts.get("note"):
        try:
            np_path = _write_note(iter_id, artifacts["note"], rec)
            rec["note_path"] = str(np_path.relative_to(ROOT))
        except Exception as e:
            print(f"  [note warn] {type(e).__name__}: {e}")

    _append_run(rec)
    return rec


# =============================================================================
# CLI
# =============================================================================
def cmd_status() -> None:
    if not BEST_PATH.exists():
        print("no journal yet. run `python runner.py once` first.")
        return
    best = _read_best()
    print("[best]")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    if RUNS_PATH.exists() and RUNS_PATH.stat().st_size > 0:
        with RUNS_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[runs] total = {len(lines)}")
        # 最近 5 条摘要
        print("[recent]")
        for line in lines[-5:]:
            r = json.loads(line)
            print(f"  #{r['iter_id']:04d} {r['status']:>6s} "
                  f"score={r.get('score', float('nan')):+.4f} "
                  f"({r.get('elapsed_sec', 0):.1f}s) -> {r.get('decision','')}")


def cmd_reset() -> None:
    if JOURNAL_DIR.exists():
        ans = input(f"will remove {JOURNAL_DIR}, continue? [y/N] ").strip().lower()
        if ans == "y":
            shutil.rmtree(JOURNAL_DIR)
            print("journal removed.")
        else:
            print("cancelled.")


def main() -> None:
    args = sys.argv[1:]
    pos_args = [a for a in args if not a.startswith("--")]
    cmd = pos_args[0] if pos_args else "once"
    if cmd == "once":
        run_once()
    elif cmd == "status":
        cmd_status()
    elif cmd == "reset":
        cmd_reset()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
