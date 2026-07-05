#!/usr/bin/env python3
"""Backtesting harness — run all strategies against the current database and report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from earnings_edge.strategies import (
    DataBundle,
    StrategyResult,
    list_strategies,
    get_strategy,
)


def run_all(data: DataBundle, strategies: list[str] | None = None) -> dict[str, StrategyResult]:
    """Run selected (or all) strategies against the data bundle."""
    if strategies is None:
        strategies = list_strategies()

    results: dict[str, StrategyResult] = {}
    for name in strategies:
        try:
            strat = get_strategy(name)
            results[name] = strat.run(data)
        except Exception as e:
            results[name] = StrategyResult(name, [], {"error": str(e)})
    return results


def print_report(results: dict[str, StrategyResult]) -> None:
    """Pretty-print a strategy comparison table."""
    header = f"{'Strategy':<30} {'Trades':>6} {'Taken':>6} {'Avg PnL':>10} {'Win%':>7} {'Total PnL':>12} {'Note'}"
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        s = res.summary
        n_trades = s.get("total", len(res.trades))
        taken = s.get("taken", n_trades)
        pnl_values = [t.pnl for t in res.trades if t.ml_decision in ("TAKE", None) and "SKIP_RISK" not in (t.ml_decision or "")]
        # Recalculate taken list consistency
        actual_taken = [t for t in res.trades if t.ml_decision == "TAKE" or (t.ml_decision != "SKIP" and "SKIP" not in (t.ml_decision or ""))]
        # Summarize from summary dict directly
        avg_pnl = s.get("avg_pnl", s.get("avg_return_pct", 0.0))
        win_rate = s.get("win_rate", 0.0)
        total_pnl = s.get("pnl", s.get("total_pnl_pct", 0.0))
        note = s.get("note", "")

        print(f"{name:<30} {n_trades:>6} {taken:>6} {avg_pnl:>10.2f} {win_rate:>6.1f}% {total_pnl:>12.2f} {note}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Backtest all strategies")
    parser.add_argument("--db", default=None, help="Path to earnings_ml.db")
    parser.add_argument("--strategies", nargs="*", default=None, help="Subset of strategies to run")
    parser.add_argument("--output", default=None, help="Write JSON report to file")
    args = parser.parse_args()

    data = DataBundle.from_db(args.db)
    print(f"Loaded {len(data.snapshots)} snapshots, {len(data.calendar_trades)} calendar trades, "
          f"{len(data.live_candidates)} live candidates, {len(data.scan_outputs)} scan outputs\n")

    results = run_all(data, args.strategies)
    print_report(results)

    if args.output:
        output = {}
        for name, res in results.items():
            output[name] = {
                "summary": res.summary,
                "trades": [t.__dict__ for t in res.trades[:100]],  # cap serialized output
            }
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"\nreport written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
