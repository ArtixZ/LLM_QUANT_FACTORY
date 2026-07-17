from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from autoalpha.data.tushare_catalog import TushareDataProduct, resolve_products

SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_REQUESTS_PER_MINUTE = 170


class RequestPacer:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(1, requests_per_minute)
        self.next_request_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_request_at - now)
            self.next_request_at = max(now, self.next_request_at) + self.interval
        if delay:
            time.sleep(delay)


def run_feature_sync(
    *,
    token: str,
    root: Path,
    dataset_ids: list[str],
    start_date: str,
    end_date: str,
    retries: int = 3,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    workers: int = 4,
) -> dict[str, Any]:
    products = [
        product
        for product in resolve_products(dataset_ids)
        if product.sync_strategy not in {"CORE_PIPELINE", "CATALOG"}
    ]
    unsupported = [
        product.dataset_id
        for product in resolve_products(dataset_ids)
        if product.sync_strategy == "CATALOG"
    ]
    if not products:
        return {"ok": not unsupported, "datasets": [], "unsupported": unsupported}
    pro = _pro_api(token)
    target_date = _latest_complete_trade_date(pro, end_date)
    open_dates = _open_dates(pro, start_date, target_date)
    calendar_dates = _calendar_dates(start_date, target_date)
    pacer = RequestPacer(requests_per_minute)

    def sync_product(product: TushareDataProduct) -> dict[str, Any]:
        dates = calendar_dates if product.sync_strategy == "ANN_DATE" else open_dates
        return _sync_product(
            pro,
            product,
            root=root,
            dates=dates,
            retries=retries,
            pacer=pacer,
        )

    worker_count = min(max(1, workers), len(products))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="tushare-product") as pool:
        results = list(pool.map(sync_product, products))
    return {
        "ok": all(result["ok"] for result in results) and not unsupported,
        "target_date": target_date,
        "datasets": results,
        "unsupported": unsupported,
    }


def _sync_product(
    pro: Any,
    product: TushareDataProduct,
    *,
    root: Path,
    dates: list[str],
    retries: int,
    pacer: RequestPacer,
) -> dict[str, Any]:
    store_root = root / "data" / "downloads" / "a_share_feature_store" / product.dataset_id
    state_path = root / "data" / "state" / f"a_feature_{product.dataset_id}.json"
    log_path = root / "data" / "logs" / f"a_feature_{product.dataset_id}.log"
    store_root.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = _read_json(state_path)
    completed = set(state.get("completed_dates", []))
    failed = dict(state.get("failed_dates", {}))
    if product.sync_strategy == "SNAPSHOT":
        dates = [datetime.now(SHANGHAI).strftime("%Y%m%d")]
        completed = set()
    pending = [value for value in dates if value not in completed]
    state.update(
        {
            "dataset_id": product.dataset_id,
            "api_name": product.api_name,
            "expected_dates": len(dates),
            "target_date": dates[-1] if dates else None,
            "run_started_at": _now(),
        }
    )
    _write_json(state_path, state)
    _log(log_path, f"{product.label} 开始：目标 {len(dates)}，待处理 {len(pending)}。")
    rows_written = 0
    for number, value in enumerate(pending, start=1):
        try:
            frame = _retry(
                lambda current=value: _fetch_product(pro, product, current),
                label=f"{product.dataset_id}:{value}",
                retries=retries,
                pacer=pacer,
            )
            if not frame.empty:
                frame = frame.copy()
                frame["source_batch"] = f"tushare:{product.api_name}:{value}"
                frame["ingested_at"] = _now()
                _write_parquet(frame, store_root / f"{value}.parquet")
                rows_written += len(frame)
            completed.add(value)
            failed.pop(value, None)
            _update_state(state_path, state, completed, failed)
            _log(
                log_path,
                f"[{number}/{len(pending)}] {value} 完成：{len(frame):,} 行。",
            )
        except Exception as error:
            failed[value] = f"{type(error).__name__}: {error}"
            _update_state(state_path, state, completed, failed)
            _log(log_path, f"{value} 失败：{failed[value]}")
    parquet_files = sorted(store_root.glob("*.parquet"))
    manifest = {
        **asdict(product),
        "documentation_url": (f"https://tushare.pro/document/2?doc_id={product.documentation_id}"),
        "storage_root": str(store_root.resolve()),
        "file_count": len(parquet_files),
        "completed_dates": len(completed),
        "failed_dates": len(failed),
        "first_date": min(completed, default=None),
        "last_date": max(completed, default=None),
        "updated_at": _now(),
    }
    _write_json(store_root / "_manifest.json", manifest)
    return {
        "dataset_id": product.dataset_id,
        "ok": not failed,
        "completed": len(completed),
        "failed": len(failed),
        "pending_processed": len(pending),
        "rows_written": rows_written,
        "first_date": manifest["first_date"],
        "last_date": manifest["last_date"],
    }


def _fetch_product(pro: Any, product: TushareDataProduct, value: str) -> pd.DataFrame:
    endpoint = getattr(pro, product.api_name)
    if product.sync_strategy == "SNAPSHOT":
        result = endpoint()
    elif product.date_parameter:
        result = endpoint(**{product.date_parameter: value})
    else:
        raise ValueError(f"No query contract for {product.dataset_id}")
    return pd.DataFrame() if result is None else pd.DataFrame(result)


def _retry(
    request: Callable[[], pd.DataFrame],
    *,
    label: str,
    retries: int,
    pacer: RequestPacer,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            pacer.wait()
            return request()
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(attempt)
    raise RuntimeError(f"{label}: {last_error}")


def _open_dates(pro: Any, start_date: str, end_date: str) -> list[str]:
    frame = pro.trade_cal(exchange="", start_date=start_date, end_date=end_date, is_open="1")
    if frame is None or frame.empty:
        return []
    return sorted(frame["cal_date"].astype(str).unique().tolist())


def _latest_complete_trade_date(pro: Any, requested_end: str) -> str:
    today = datetime.now(SHANGHAI)
    calendar_end = min(requested_end, today.strftime("%Y%m%d"))
    lookback = (today - timedelta(days=20)).strftime("%Y%m%d")
    dates = _open_dates(pro, lookback, calendar_end)
    if not dates:
        return calendar_end
    if dates[-1] == today.strftime("%Y%m%d") and today.hour < 17:
        return dates[-2] if len(dates) > 1 else dates[-1]
    return dates[-1]


def _calendar_dates(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y%m%d").date()
    end = datetime.strptime(end_date, "%Y%m%d").date()
    return [value.strftime("%Y%m%d") for value in pd.date_range(start, end, freq="D")]


def _pro_api(token: str) -> Any:
    if not token:
        raise RuntimeError("Tushare Token is not configured")
    try:
        import tushare as ts
    except ImportError as error:
        raise RuntimeError("The selected downloader Python does not provide tushare") from error
    ts.set_token(token)
    return ts.pro_api(token)


def _update_state(
    path: Path,
    state: dict[str, Any],
    completed: set[str],
    failed: dict[str, str],
) -> None:
    state["completed_dates"] = sorted(completed)
    state["failed_dates"] = failed
    state["updated_at"] = _now()
    _write_json(path, state)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _log(path: Path, message: str) -> None:
    line = f"[{datetime.now(SHANGHAI):%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _now() -> str:
    return datetime.now(SHANGHAI).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Tushare A-share feature-store sync")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--start-date", default="20100101")
    parser.add_argument("--end-date", default="20991231")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--requests-per-minute", type=int, default=DEFAULT_REQUESTS_PER_MINUTE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run_feature_sync(
        token=os.getenv("TUSHARE_TOKEN", ""),
        root=args.root.expanduser().resolve(),
        dataset_ids=[value for value in args.datasets.split(",") if value],
        start_date=args.start_date,
        end_date=args.end_date,
        retries=args.retries,
        requests_per_minute=args.requests_per_minute,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
