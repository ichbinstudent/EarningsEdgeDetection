#!/usr/bin/env python3
"""
Post-earnings outcome tracker — CLI wrapper around OutcomeService.

Run daily to label historical snapshots with actual outcomes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from earnings_edge.config import setup_logging
from earnings_edge.services.outcome_service import OutcomeService

setup_logging()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Post-earnings outcome tracker")
    p.add_argument("--min-age", type=int, default=2, help="Min days past earnings to check")
    p.add_argument("--limit", type=int, default=0, help="Max outcomes to process (0=all)")
    p.add_argument("--max-retries", type=int, default=2, help="Attempts before marking unavailable")
    p.add_argument("--live-candidates", action="store_true",
                   help="Also process live_calendar_candidates")
    p.add_argument("--live-candidate-limit", type=int, default=0,
                   help="Max live candidates to process (0=all, only with --live-candidates)")
    args = p.parse_args()

    svc = OutcomeService()
    stats = svc.run_outcomes(
        min_age_days=args.min_age,
        limit=args.limit,
        max_retries=args.max_retries,
    )
    print(f"Snapshots: {stats['updated']} updated, {stats['failed']} no data, {stats['processed']} processed")

    if args.live_candidates:
        live_stats = svc.run_live_candidate_outcomes(
            min_age_days=args.min_age,
            limit=args.live_candidate_limit,
            max_retries=args.max_retries,
        )
        print(f"Live candidates: {live_stats['updated']} updated, "
              f"{live_stats['failed']} no data, {live_stats['processed']} processed")
