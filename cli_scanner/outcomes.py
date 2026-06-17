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
    args = p.parse_args()

    svc = OutcomeService()
    stats = svc.run_outcomes(
        min_age_days=args.min_age,
        limit=args.limit,
        max_retries=args.max_retries,
    )
    print(f"Done: {stats['updated']} updated, {stats['failed']} no data")
