#!/usr/bin/env python3
"""Alpaca historical options bar / snapshot client.

Wraps the endpoints the user's paper account actually confirmed working:

  - bars:  data.alpaca.markets/v1beta1/options/bars
  - chain: data.alpaca.markets/v1beta1/options/snapshots/{underlying}

Both are free on the Developer/paper tier (no monthly subscription).
"""
from __future__ import annotations

import time
import logging
from typing import Any, Optional

import requests

logger = logging.getLogger("alpaca_options")

DATA_BASE = "https://data.alpaca.markets"


class AlpacaOptionsClient:
    """Historical options bars + chain snapshots client.

    Paper account granted options_trading_level=3: full options access,
    no extra entitlements needed.
    """

    def __init__(self, api_key: str, api_secret: str, timeout: int = 25, max_retries: int = 3):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        )

    def bars(
        self,
        symbols: list[str],
        timeframe: str = "1D",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
        page_token: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Return (bars_list, next_page_token).

        symbols: list of OCC contract symbols (e.g. "AAPL250117C00150000").
        timeframe: Alpaca bar timeframe — "1D", "1Min", etc.
        """
        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "limit": min(limit, 1000),
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if page_token:
            params["page_token"] = page_token
        resp = self._get("/v1beta1/options/bars", params=params)
        bars: list[dict[str, Any]] = []
        next_token: Optional[str] = None
        if resp:
            for sym_contract, rows in resp.get("bars", {}).items():
                for row in rows:
                    row["symbol"] = sym_contract
                    bars.append(row)
            next_token = resp.get("next_page_token")
        return bars, next_token

    def chain_snapshot(
        self,
        underlying: str,
        feed: str = "indicative",
        limit: int = 200,
        page_token: Optional[str] = None,
    ) -> tuple[dict[str, Any], Optional[str]]:
        """Return (snapshot_dict, next_page_token).

        Each value contains latestQuote {ap bp as bs ax bx t}, latestTrade,
        dailyBar {c h l n o t v vw}, minuteBar, prevDailyBar, etc.
        Bars: underlying ticker only (no contract symbol).
        """
        params: dict[str, Any] = {
            "feed": feed,
            "limit": min(limit, 200),
        }
        if page_token:
            params["page_token"] = page_token
        resp = self._get(f"/v1beta1/options/snapshots/{underlying}", params=params)
        if resp:
            next_token = resp.get("next_page_token")
            return resp, next_token
        return {}, None

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(f"{DATA_BASE}{path}", params=params, timeout=self.timeout)
                if r.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning("Alpaca rate-limited on %s; sleeping %ds", path, wait)
                    time.sleep(wait)
                    continue
                if r.status_code == 404:
                    logger.debug("Alpaca 404 on %s", path)
                    return None
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                logger.warning("Alpaca GET %s failed (attempt %d/%d): %s", path, attempt + 1, self.max_retries, exc)
                if attempt + 1 < self.max_retries:
                    time.sleep(5 * (attempt + 1))
        return None
