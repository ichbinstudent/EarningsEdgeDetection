"""Rate-limited yfinance wrapper with backoff and data validation."""

from __future__ import annotations

import time
import logging
from typing import Optional

import yfinance as yf

from .base import BaseCollector
from ..config import session
from ..settings import get_settings

logger = logging.getLogger("earnings_edge.collectors.market_data")


class MarketDataCollector(BaseCollector):
    """Wraps yfinance calls with rate limiting and retry."""

    def __init__(self):
        s = get_settings()
        super().__init__(
            name="yfinance",
            max_retries=3,
            base_delay=2.0,
            max_delay=30.0,
            circuit_threshold=10,
        )
        self._sleep = s.yfinance_sleep
        self._batch_size = s.yfinance_batch_size
        self._batch_sleep = s.yfinance_batch_sleep
        self._call_count = 0

    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Return a yf.Ticker with the shared session."""
        return yf.Ticker(symbol, session=session)

    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest close price with retry."""
        self._rate_limit()

        def _fetch():
            yt = self.get_ticker(symbol)
            hist = yt.history(period="1d")
            if hist.empty:
                raise ValueError(f"No price data for {symbol}")
            return float(hist["Close"].iloc[-1])

        try:
            return self.with_retry(_fetch)
        except Exception as exc:
            logger.warning("get_price(%s) failed: %s", symbol, exc)
            return None

    def get_options_expiries(self, symbol: str) -> tuple[Optional[object], list[str]]:
        """Get option chain + expiry list with retry.

        Returns (chain, expiries). chain is None on failure.
        """
        self._rate_limit()

        def _fetch():
            yt = self.get_ticker(symbol)
            expiries = yt.options
            if not expiries:
                raise ValueError(f"No options for {symbol}")
            chain = yt.option_chain(expiries[0])
            return chain, list(expiries)

        try:
            return self.with_retry(_fetch)
        except Exception as exc:
            logger.warning("get_options_expiries(%s) failed: %s", symbol, exc)
            return None, []

    def batch_rate_limit(self):
        """Call after processing a batch of tickers."""
        self._call_count += 1
        if self._call_count % self._batch_size == 0:
            logger.info("Batch rate limit: sleeping %.1fs", self._batch_sleep)
            time.sleep(self._batch_sleep)

    def _rate_limit(self):
        time.sleep(self._sleep)
