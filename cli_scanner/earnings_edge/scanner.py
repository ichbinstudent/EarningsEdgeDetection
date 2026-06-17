"""
EarningsScanner — orchestrator that wires together data fetching,
stock validation, and result collection.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

import pytz
from tqdm import tqdm

from .analyzer import OptionsAnalyzer
from .browser import MarketChameleonBrowser
from .config import get_logger
from .earnings import fetch_earnings, scan_dates
from .iron_fly import calculate_iron_fly
from .models import (
    EarningsCandidate, IronFlyResult, NearMiss,
    ScanResult, TickerReport, ValidationResult,
)
from .validator import StockValidator
from . import yfinance_patch

logger = get_logger("scanner")

# Apply yfinance cookie patch at import time
yfinance_patch.apply()


class EarningsScanner:
    """Top-level scanner: fetch earnings candidates → validate → return results."""

    def __init__(self) -> None:
        self.eastern_tz = pytz.timezone("US/Eastern")
        self.browser = MarketChameleonBrowser()
        self.analyzer = OptionsAnalyzer()
        self.validator = StockValidator(self.analyzer, self.browser)

    def __del__(self) -> None:
        self.browser.close()

    # -- single-ticker analysis ------------------------------------------

    def analyze_ticker(self, ticker: str) -> TickerReport:
        """Analyze a single ticker regardless of earnings calendar."""
        self.validator.adjust_spy_thresholds()
        candidate = EarningsCandidate(ticker=ticker.strip().upper(), timing="Manual Check")
        result = self.validator.validate(candidate)

        spy_iv_rv = 0.0
        try:
            spy = self.analyzer.compute_recommendation("SPY")
            if spy.ok:
                spy_iv_rv = spy.iv30_rv30
        except Exception:
            pass

        return TickerReport(
            ticker=ticker.strip().upper(),
            passed=result.passed,
            tier=result.tier,
            near_miss=result.near_miss,
            reason=result.reason,
            metrics=result.metrics,
            spy_iv_rv=spy_iv_rv,
            iv_rv_pass_threshold=self.validator.iv_rv_pass,
            iv_rv_near_miss_threshold=self.validator.iv_rv_near_miss,
            earnings_date=candidate.earnings_date,
        )

    def calculate_iron_fly_strikes(self, ticker: str) -> IronFlyResult:
        """Delegate to :func:`earnings_edge.iron_fly.calculate_iron_fly`."""
        return calculate_iron_fly(ticker)

    # -- full earnings scan -----------------------------------------------

    def scan_earnings(
        self,
        input_date: Optional[str] = None,
        workers: int = 0,
        use_finnhub: bool = False,
        use_dolthub: bool = False,
        all_sources: bool = False,
    ) -> ScanResult:
        """Main entry point — fetch earnings, validate, return tiered results."""
        self.validator.adjust_spy_thresholds()

        try:
            post_date, pre_date = scan_dates(input_date, self.eastern_tz)
        except Exception as exc:
            logger.error(f"Date error: {exc}")
            return ScanResult()

        # Fetch earnings in parallel
        post_stocks: list[EarningsCandidate] = []
        pre_stocks: list[EarningsCandidate] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_post = pool.submit(fetch_earnings, post_date, use_dolthub, use_finnhub, all_sources)
            f_pre = pool.submit(fetch_earnings, pre_date, use_dolthub, use_finnhub, all_sources)
            try:
                post_stocks = f_post.result(timeout=30)
            except Exception as exc:
                logger.error(f"Post-market fetch failed: {exc}")
            try:
                pre_stocks = f_pre.result(timeout=30)
            except Exception as exc:
                logger.error(f"Pre-market fetch failed: {exc}")

        # Build candidate list — include all timings, not just pre/post
        candidates: list[EarningsCandidate] = []
        for s in post_stocks:
            candidates.append(EarningsCandidate(
                ticker=s.ticker, timing=s.timing, earnings_date=post_date,
            ))
        for s in pre_stocks:
            candidates.append(EarningsCandidate(
                ticker=s.ticker, timing=s.timing, earnings_date=pre_date,
            ))

        logger.info(f"{len(candidates)} candidates")

        reports: dict[str, TickerReport] = {}

        def _process(candidate: EarningsCandidate) -> None:
            result = self.validator.validate(candidate)
            reports[candidate.ticker] = TickerReport(
                ticker=candidate.ticker,
                passed=result.passed,
                tier=result.tier,
                near_miss=result.near_miss,
                reason=result.reason,
                metrics=result.metrics,
                iv_rv_pass_threshold=self.validator.iv_rv_pass,
                iv_rv_near_miss_threshold=self.validator.iv_rv_near_miss,
                earnings_date=candidate.earnings_date,
            )

        if workers > 0:
            n = min(workers, 8)
            logger.info(f"Parallel ({n} workers)")
            with ThreadPoolExecutor(max_workers=n) as pool:
                futs = [pool.submit(_process, s) for s in candidates]
                with tqdm(total=len(candidates), desc="Analyzing") as pbar:
                    for f in futs:
                        try:
                            f.result(timeout=60)
                        except Exception as exc:
                            logger.error(f"Worker error: {exc}")
                        finally:
                            pbar.update(1)
        else:
            batch = 8
            with tqdm(total=len(candidates), desc="Analyzing") as pbar:
                for i in range(0, len(candidates), batch):
                    for s in candidates[i : i + batch]:
                        _process(s)
                        pbar.update(1)
                    if i + batch < len(candidates):
                        time.sleep(5)

        # Separate into tiers
        tier1 = sorted(
            [t for t, r in reports.items() if r.passed and r.tier == 1],
            key=lambda t: reports[t].metrics.actual_to_fair_ratio or 0,
            reverse=True,
        )
        tier2 = sorted(
            [t for t, r in reports.items() if r.passed and r.tier == 2],
            key=lambda t: reports[t].metrics.actual_to_fair_ratio or 0,
            reverse=True,
        )
        near_misses = [
            NearMiss(ticker=t, reason=r.reason)
            for t, r in reports.items()
            if r.near_miss
        ]

        return ScanResult(tier1=tier1, tier2=tier2, near_misses=near_misses, reports=reports)
