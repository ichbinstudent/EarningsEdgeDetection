#!/usr/bin/env python3
"""
EarningsEdgeDetection CLI.
Scans for high-probability options plays around upcoming earnings.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from earnings_edge.config import setup_logging
from earnings_edge.scanner import EarningsScanner
from earnings_edge.discord_webhook import send_webhook
from earnings_edge.models import ScanResult, TickerReport


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Scans for recommended options plays based on upcoming earnings.\n"
            "Before 4PM ET: today's post-market + tomorrow's pre-market.\n"
            "After 4PM ET:  tomorrow's post-market + next-day pre-market."
        ),
    )
    p.add_argument("--date", "-d", help="Date in MM/DD/YYYY format")
    p.add_argument("--parallel", "-p", type=int, default=0, help="Worker threads (0 = sequential)")
    p.add_argument("--list", "-l", action="store_true", help="Compact ticker-only output")
    p.add_argument("--iron-fly", "-i", action="store_true", help="Show iron fly strikes")
    p.add_argument("--analyze", "-a", metavar="TICKER", help="Analyze a single ticker")
    p.add_argument("--webhook", "-w", help="Discord webhook URL")
    p.add_argument("--forever", "-fv", type=int, metavar="HOURS", help="Repeat every N hours")
    p.add_argument("--use-finnhub", "-f", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--use-dolthub", "-u", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--all-sources", "-c", action="store_true", help="Use all earnings data sources")
    return p


# ── Single-ticker analysis ───────────────────────────────────────────

def _analyze(args: argparse.Namespace, scanner: EarningsScanner, log: logging.Logger) -> None:
    ticker = args.analyze.strip().upper()
    print(f"\n=== ANALYZING {ticker} ===\n")

    report = scanner.analyze_ticker(ticker)
    if not report.ok:
        print(f"Error: {report.error}")
        return

    m = report.metrics
    print(f"SPY IV/RV: {report.spy_iv_rv:.2f}")
    print(f"Thresholds — pass: {report.iv_rv_pass_threshold:.2f}, "
          f"near-miss: {report.iv_rv_near_miss_threshold:.2f}\n")

    status = "PASS" if report.passed else ("NEAR MISS" if report.near_miss else "FAIL")
    if report.passed and report.tier in (1, 2):
        status += f" TIER {report.tier}"
    print(f"Status: {status}")
    print(f"Reason: {report.reason}\n")

    print("CORE METRICS:")
    print(f"  Price: ${m.price:.2f}")
    print(f"  Volume: {m.volume:,.0f}")
    print(f"  Term Structure: {m.term_structure:.4f}")
    print(f"  IV/RV Ratio: {m.iv_rv_ratio:.2f}")
    if m.win_quarters > 0:
        print(f"  Winrate: {m.win_rate:.1f}% over last {m.win_quarters} earnings")

    extras = {}
    if m.expected_move_dollars:
        extras["expected_move_dollars"] = m.expected_move_dollars
    if m.expected_move_pct:
        extras["expected_move_pct"] = m.expected_move_pct
    if m.open_interest:
        extras["open_interest"] = m.open_interest
    if m.days_to_expiry:
        extras["days_to_expiry"] = m.days_to_expiry
    if extras:
        print("\nADDITIONAL METRICS:")
        for k, v in extras.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.iron_fly:
        _print_iron_fly(scanner, ticker)


# ── Full scan ─────────────────────────────────────────────────────────

def _scan(args: argparse.Namespace, scanner: EarningsScanner, log: logging.Logger) -> None:
    running = True
    while running:
        result = scanner.scan_earnings(
            input_date=args.date,
            workers=args.parallel,
            use_finnhub=args.use_finnhub,
            use_dolthub=args.use_dolthub,
            all_sources=args.all_sources,
        )

        if result.tier1 or result.tier2 or result.near_misses:
            print("\n=== SCAN RESULTS ===")

            if args.list:
                print(f"\nTIER 1: {', '.join(result.tier1) or 'None'}")
                print(f"TIER 2: {', '.join(result.tier2) or 'None'}")
                print(f"NEAR MISSES: {', '.join(nm.ticker for nm in result.near_misses) or 'None'}")
            else:
                _print_recommended("RECOMMENDED TRADES:", result.tier1 + result.tier2, result.reports, args, scanner)
                _print_near_misses(result.near_misses, result.reports)

            if args.webhook:
                _send_webhook(args, scanner, result)
        else:
            log.info("No recommended stocks found")

        if args.forever and args.forever > 0:
            log.info(f"Sleeping {args.forever}h…")
            time.sleep(args.forever * 3600)
        else:
            running = False


# ── Display helpers ───────────────────────────────────────────────────

def _print_recommended(label, tickers, reports, args, scanner):
    print(f"\n{label}")
    if not tickers:
        print("  None")
        return
    for t in tickers:
        r = reports[t]
        m = r.metrics
        print(f"\n  {t}:")
        print(f"    Price: ${m.price:.2f}")
        if m.sigma_baseline_1y is not None:
            print(f"    1Y ATM IV: {m.sigma_baseline_1y:.4f}")
        if m.sigma_short_leg_fair is not None:
            print(f"    Fair IV (Short): {m.sigma_short_leg_fair:.4f}")
        if m.sigma_short_leg is not None:
            print(f"    Actual IV (Short): {m.sigma_short_leg:.4f}")
        if m.actual_to_fair_ratio is not None:
            print(f"    Actual/Fair: {m.actual_to_fair_ratio:.2f}%")
        print(f"    Volume: {m.volume:,.0f}")
        if m.win_quarters > 0:
            print(f"    Winrate: {m.win_rate:.1f}% over last {m.win_quarters} earnings")
        print(f"    IV/RV: {m.iv_rv_ratio:.2f}")
        print(f"    Term Structure: {m.term_structure:.3f}")
        if args.iron_fly:
            _print_iron_fly(scanner, t, indent="    ")


def _print_near_misses(near_misses, reports):
    print("\nNEAR MISSES:")
    if not near_misses:
        print("  None")
        return
    for nm in near_misses:
        r = reports[nm.ticker]
        m = r.metrics
        print(f"\n  {nm.ticker}:")
        print(f"    Failed: {nm.reason}")
        print(f"    Price: ${m.price:.2f}  Volume: {m.volume:,.0f}")
        if m.win_quarters > 0:
            print(f"    Winrate: {m.win_rate:.1f}% ({m.win_quarters} earnings)")
        print(f"    IV/RV: {m.iv_rv_ratio:.2f}  Term: {m.term_structure:.3f}")


def _print_iron_fly(scanner, ticker, indent="    "):
    fly = scanner.calculate_iron_fly_strikes(ticker)
    if not fly.ok:
        print(f"{indent}Iron fly: {fly.error}")
        return
    print(f"{indent}--------------------")
    print(f"{indent}IRON FLY:")
    print(f"{indent}  Exp: {fly.expiration}")
    print(f"{indent}  SHORT: ${fly.short_put_strike}P/${fly.short_call_strike}C "
          f"for ${fly.total_credit} credit")
    print(f"{indent}  LONG:  ${fly.long_put_strike}P/${fly.long_call_strike}C "
          f"for ${fly.total_debit} debit")
    print(f"{indent}  Break-evens: {fly.lower_breakeven}-{fly.upper_breakeven}, "
          f"R/R: 1:{fly.risk_reward_ratio}")


def _send_webhook(args, scanner, result: ScanResult):
    fields = []

    for tier_num, tickers in ((1, result.tier1), (2, result.tier2)):
        for t in tickers:
            r = result.reports[t]
            m = r.metrics
            lines = [
                f"• Price: `${m.price:.2f}`",
                f"• Volume: `{m.volume:,.0f}`",
            ]
            if m.win_quarters > 0:
                lines.append(f"• Winrate: `{m.win_rate:.1f}%` over `{m.win_quarters}` earnings")
            lines += [
                f"• IV/RV: `{m.iv_rv_ratio:.2f}`",
                f"• Term: `{m.term_structure:.3f}`",
                f"• Tier: `{tier_num}`",
            ]
            if args.iron_fly:
                fly = scanner.calculate_iron_fly_strikes(t)
                if fly.ok:
                    lines += [
                        "", "**Iron Fly**:",
                        f"▫️ Exp: `{fly.expiration}`",
                        f"▫️ Short: `{fly.short_put_strike}P/{fly.short_call_strike}C` for `{fly.total_credit}`",
                        f"▫️ Long: `{fly.long_put_strike}P/{fly.long_call_strike}C` for `{fly.total_debit}`",
                        f"▫️ Break-evens: `{fly.lower_breakeven}–{fly.upper_breakeven}`",
                        f"▫️ R/R: `1:{fly.risk_reward_ratio}`",
                    ]
            fields.append({"name": f"Tier {tier_num} — {t}", "value": "\n".join(lines), "inline": False})

    for nm in result.near_misses:
        r = result.reports[nm.ticker]
        m = r.metrics
        lines = [
            f"• Failed: `{nm.reason}`",
            f"• Price: `${m.price:.2f}`",
            f"• Volume: `{m.volume:,.0f}`",
        ]
        if m.win_quarters > 0:
            lines.append(f"• Winrate: `{m.win_rate:.1f}%` over `{m.win_quarters}` earnings")
        lines += [
            f"• IV/RV: `{m.iv_rv_ratio:.2f}`",
            f"• Term: `{m.term_structure:.3f}`",
        ]
        fields.append({"name": f"Near Miss — {nm.ticker}", "value": "\n".join(lines), "inline": False})

    if not fields:
        fields.append({"name": "No recommendations", "value": "None found", "inline": False})

    embed = {
        "title": "Earnings Scanner Results",
        "color": 3066993,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger = logging.getLogger("earnings_edge.cli")
    send_webhook(args.webhook, embed, logger)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()
    setup_logging(log_dir="logs")
    log = logging.getLogger("earnings_edge.cli")

    if args.date:
        try:
            datetime.strptime(args.date, "%m/%d/%Y")
        except ValueError as e:
            log.error(f"Invalid date: {e}")
            sys.exit(1)

    scanner = EarningsScanner()

    if args.analyze:
        _analyze(args, scanner, log)
        return

    try:
        _scan(args, scanner, log)
    except KeyboardInterrupt:
        log.info("Interrupted")
    except ValueError as e:
        log.error(f"Error: {e}")


if __name__ == "__main__":
    main()
