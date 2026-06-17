#!/bin/bash
# EarningsEdgeDetection CLI Scanner Runner
#
# Usage:
#   ./run.sh                     Run with current date (auto-parallel)
#   ./run.sh MM/DD/YYYY          Run with a specific date
#   ./run.sh -l                  Compact ticker-only output
#   ./run.sh -i                  Include iron fly calculations
#   ./run.sh -a TICKER           Analyze a single ticker
#   ./run.sh -c                  Use all data sources (Investing.com + Finnhub + DoltHub)
#
# All scanner flags are passed through directly to scanner.py.

# Auto-detect worker count (half the cores, 2–6)
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS=$(( NUM_CORES / 2 ))
(( WORKERS < 2 )) && WORKERS=2
(( WORKERS > 6 )) && WORKERS=6

exec python3 scanner.py --parallel $WORKERS "$@"
