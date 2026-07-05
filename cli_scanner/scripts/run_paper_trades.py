#!/usr/bin/env python3
"""Run paper-trading strategies for today's earnings and submit Alpaca orders.

This is the production entry point for automatic execution.

Usage:
    python scripts/run_paper_trades.py --dry-run
    python scripts/run_paper_trades.py --notify-telegram
    python scripts/run_paper_trades.py --strategies short_straddle vol_risk_premium
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the cli_scanner package is importable
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from earnings_edge.alpaca_bridge import run_auto_trade, BEST_STRATEGIES
from earnings_edge.alpaca_trading import AlpacaError

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("paper_trader")


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")


def send_telegram_notification(chat_id: str, text: str) -> bool:
    """Send a Telegram message (if token configured)."""
    if not TELEGRAM_BOT_TOKEN:
        logger.debug("TELEGRAM_BOT_TOKEN not set, skipping notification")
        return False

    try:
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": int(chat_id), "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error("Telegram notification failed: %s", e)
        return False


def format_summary(summary: dict) -> str:
    """Format execution summary for human-readable output / notification."""
    lines = [
        f"📊 *Paper Trade Execution* — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Buying Power: ${summary.get('buying_power', 0):,.2f}",
        f"Dry Run: {'YES' if summary.get('dry_run') else 'LIVE'}",
        "",
    ]

    strat_results = summary.get("strategies", {})
    if not strat_results:
        lines.append("_No strategies processed._")
        return "\n".join(lines)

    lines.append("*Strategy Results:*")
    total_submitted = 0
    for name, result in strat_results.items():
        status = result.get("status", "?")
        trades = result.get("trades", 0)
        submitted = result.get("submitted", 0)
        total_submitted += submitted
        if status == "ok":
            lines.append(f"  • {name}: {trades} signals → {submitted} submitted")
        else:
            lines.append(f"  • {name}: {status} ({trades} signals)")

    orders = summary.get("orders", [])
    if orders:
        lines.append("")
        lines.append("*Orders:*")
        for o in orders[:10]:
            lines.append(f"  • {o['strategy']}: {o['symbol']} ({o['legs']} legs, {o['status']})")
        if len(orders) > 10:
            lines.append(f"  ... and {len(orders) - 10} more")

    lines.append("")
    lines.append(f"Total submitted: {total_submitted}")
    if not summary.get("dry_run") and total_submitted > 0:
        lines.append("⚠️ LIVE ORDERS SUBMITTED — monitor fills")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-trading strategies")
    parser.add_argument("--strategies", nargs="*", default=None, help="Strategies to run (default: BEST_STRATEGIES)")
    parser.add_argument("--dry-run", action="store_true", help="Run without submitting real orders")
    parser.add_argument("--notify", default=None, help="Telegram chat ID to send notification to")
    parser.add_argument("--output", default=None, help="Write JSON output to file")
    parser.add_argument("--db", default=None, help="Path to earnings_ml.db")
    parser.add_argument("--api-key", default=None, help="Alpaca API key (or env APCA_API_KEY_ID)")
    parser.add_argument("--api-secret", default=None, help="Alpaca API secret (or env APCA_API_SECRET_KEY)")
    args = parser.parse_args()

    strategies = args.strategies or BEST_STRATEGIES

    logger.info("Paper trading run: strategies=%s, dry_run=%s", strategies, args.dry_run)

    summary = run_auto_trade(
        strategies=strategies,
        dry_run=args.dry_run,
        db_path=args.db,
        api_key=args.api_key,
        api_secret=args.api_secret,
    )

    # Print formatted summary
    output_text = format_summary(summary)
    print(output_text)

    # Write JSON if requested
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nJSON report written to {args.output}")

    # Telegram notification
    if args.notify:
        send_telegram_notification(args.notify, output_text)

    if summary.get("total_submitted", 0) > 0 and not args.dry_run:
        logger.warning("%d LIVE orders submitted", summary["total_submitted"])
        return 2  # signal: live orders placed

    return 0


if __name__ == "__main__":
    sys.exit(main())
