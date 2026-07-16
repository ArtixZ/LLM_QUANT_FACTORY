# AutoAlpha Vector Backtest Engine V1

## Purpose

`VectorBacktester` is the reusable matrix engine for cross-sectional factor research. It accepts
an end-of-day signal panel and an open-price panel, then produces positions, target and held
weights, daily returns, turnover, costs, equity, drawdown, and summary metrics.

The engine is for rapid research and model reconciliation. It does not replace the A-share
capital ledger for production simulation because a weight-based engine cannot model integer lots,
minimum commission per order, suspensions, opening price limits, partial fills, cash constraints,
or residual positions exactly.

## Timing Contract

- Signal availability: after close on session `T`.
- Earliest execution: open on session `T+1`.
- Earned return: open `T+1` to open `T+2`.
- Convention id: `EOD_T__OPEN_T1_TO_OPEN_T2`.
- `entry_session` labels P&L by entry date and matches the manual backtest history.
- `signal_session` labels P&L by signal date and matches the automated evaluator history.
- An end-date boundary excludes a return when its exit session falls after the requested end.

## Cost Models

`legacy_half_turnover` exactly reproduces historical AutoAlpha results. It multiplies
`0.5 * sum(abs(delta_weight))` by commission, transfer fee, and half stamp duty. This mode exists
for audit and regression comparison only.

`side_aware` is the corrected research default. It charges commission and transfer fees on buys
and sells separately, stamp duty on sells, and includes initial establishment turnover. Use the
capital ledger when minimum commission and order-level effects matter.

## Real-Data Reconciliation

Command:

```bash
uv run python scripts/reconcile_vector_backtest.py
```

The 2026-07-16 reconciliation selected clean public factor
`F_0f2e2d24ab679a2f` (`Volume_Volatility_Regime_20_60`). Across 3,619 comparable dates, the new
legacy-compatible engine matched the existing evaluator exactly for net return, stressed return,
and turnover. Maximum gross-return difference was `4.34e-19` from floating-point arithmetic.

On the public 2015-01-05 through 2024-11-27 window, correcting the fee model changed:

| Metric | Legacy compatible | Side aware |
| --- | ---: | ---: |
| Simple annual return | 10.65% | 9.85% |
| Compound annual return | 11.10% | 10.22% |
| Sharpe ratio | 2.157 | 1.996 |
| Maximum drawdown | -6.22% | -6.52% |
| Total return | 181.34% | 160.08% |

No hidden-test dates were accessed. The machine-readable report and daily comparison are in
`output/backtests/vector_engine_v1/`.

## API Example

```python
from autoalpha.backtest.vector import VectorBacktestConfig, VectorBacktester

result = VectorBacktester(
    VectorBacktestConfig(
        holding_period_days=5,
        gross_exposure=0.5,
        selection_fraction=0.1,
        maximum_positions_per_side=30,
        cost_model="side_aware",
    )
).run(signal, open_prices, start="2018-01-01", end="2024-11-29")
```

Use `result.path` for daily return reconciliation and `result.metrics` for summary evaluation.
