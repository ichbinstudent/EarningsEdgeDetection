#!/usr/bin/env python3
"""Backtesting harness — run all strategies against the current database and report.

Two strategy families:
  - strategies.py    — calendar-call family (10 strategies)
  - positional_strategies.py — short/long straddle, directional, vol risk premium

Usage:
    ./.venv/bin/python backtest.py                    # calendar family only
    ./.venv/bin/python backtest.py --positional        # add positional family
    ./.venv/bin/python backtest.py --all               # both families
    ./.venv/bin/python backtest.py --strategies short_straddle directional_call
"""
from __future__ import annotations

import argparse
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
from earnings_edge.positional_strategies import (
    run_positional as run_positional_family,
)
from earnings_edge.multi_strike_real import run_multi_strike


def run_all(data: DataBundle, strategies: list[str] | None = None) -> dict[str, StrategyResult]:
    """Run selected (or all calendar) strategies against the data bundle."""
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


def print_report(results: dict[str, StrategyResult], label: str) -> None:
    """Pretty-print a strategy comparison table."""
    print(f"\n=== {label} ===")
    header = f"{'Strategy':<30} {'Trades':>6} {'Taken':>6} {'Avg PnL':>10} {'Win%':>7} {'Total PnL':>12} {'Note'}"
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        s = res.summary
        n_trades = s.get("total", len(res.trades))
        taken = s.get("taken", n_trades)
        avg_pnl = s.get("avg_pnl", s.get("avg_return_pct", 0.0))
        win_rate = s.get("win_rate", 0.0)
        total_pnl = s.get("pnl", s.get("total_pnl", s.get("total_pnl_pct", 0.0)))
        note = s.get("note", "")

        print(f"{name:<30} {n_trades:>6} {taken:>6} {avg_pnl:>10.2f} {win_rate:>6.1f}% {round(total_pnl, 4):>12} {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest all strategies")
    parser.add_argument("--db", default=None, help="Path to earnings_ml.db")
    parser.add_argument("--strategies", nargs="*", default=None, help="Subset of strategies to run")
    parser.add_argument("--positional", action="store_true", help="Include non-calendar positional strategies")
    parser.add_argument("--multi-strike", action="store_true", help="Include multi-strike (iron condor/butterfly/risk reversal)")
    parser.add_argument("--all", action="store_true", help="Run all three strategy families")
    parser.add_argument("--output", default=None, help="Write JSON report to file")
    args = parser.parse_args()

    if args.all:
        args.positional = True
        args.multi_strike = True

    data = DataBundle.from_db(args.db)
    print(f"Loaded {len(data.snapshots)} snapshots, {len(data.calendar_trades)} calendar trades, "
          f"{len(data.live_candidates)} live candidates, {len(data.scan_outputs)} scan outputs, "
          f"{len(data.options_chain)} chain rows\n")

    results = run_all(data, args.strategies)
    print_report(results, "Calendar Call Family")

    if args.positional:
        pos_results = run_positional_family(data, args.strategies)
        print_report(pos_results, "Positional Family")
        for k, v in pos_results.items():
            results[k] = v

    if args.multi_strike:
        ms_results = run_multi_strike(data, args.strategies)
        print_report(ms_results, "Multi-Strike Family")
        for k, v in ms_results.items():
            results[k] = v

    if args.output:
        output = {}
        for name, res in results.items():
            output[name] = {
                "summary": res.summary,
                "trades": [t.__dict__ for t in res.trades[:100]],
            }
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"\nreport written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
