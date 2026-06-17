#!/usr/bin/env bash
# trading-bot supervisor — auto-restarts on crash, logs to file.
set -euo pipefail

BOT_DIR="$HOME/EarningsEdgeDetection/cli_scanner"
PYTHON="$BOT_DIR/.venv/bin/python3.12"
LOG_DIR="$BOT_DIR/logs"

mkdir -p "$LOG_DIR"

exec >> "$LOG_DIR/bot-supervisor.log" 2>&1
echo "[$(date -Iseconds)] supervisor started (PID $$)"

while true; do
    echo "[$(date -Iseconds)] launching bot..."
    cd "$BOT_DIR"
    "$PYTHON" bot.py
    rc=$?
    echo "[$(date -Iseconds)] bot exited with code $rc, restarting in 10s..."
    sleep 10
done
