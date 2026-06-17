"""
Forward Volatility Scanner — Eurex DAX options calendar spread opportunities.

Fetches option contracts via Eurex WebSocket API, computes forward implied
volatility from consecutive expiry pairs, and scores calendar spreads by
FV ratio. Optional Interactive Brokers integration for accurate IVs.
"""

import json
import re
import threading
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import websocket
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

from earnings_edge.base import BaseScanner
from earnings_edge.config import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("forward_volatility")

# ---------------------------------------------------------------------------
# DAX component ISIN → Yahoo Finance ticker
# ---------------------------------------------------------------------------

ISIN_TO_TICKER: Dict[str, str] = {
    "DE000A1EWWW0": "ADS.DE",   # Adidas
    "NL0000235190": "AIR.PA",   # Airbus (Paris)
    "DE0008404005": "ALV.DE",   # Allianz
    "DE000BASF111": "BAS.DE",   # BASF
    "DE000BAY0017": "BAYN.DE",  # Bayer
    "DE0005200000": "BEI.DE",   # Beiersdorf
    "DE0005190003": "BMW.DE",   # BMW
    "DE000A1DAHH0": "BNR.DE",   # Brenntag
    "DE0005439004": "CON.DE",   # Continental
    "DE000CBK1001": "CBK.DE",   # Commerzbank
    "DE000DTR0CK8": "DTG.DE",   # Daimler Truck
    "DE0005140008": "DBK.DE",   # Deutsche Bank
    "DE0005810055": "DB1.DE",   # Deutsche Börse
    "DE0005552004": "DHL.DE",   # Deutsche Post
    "DE0005557508": "DTE.DE",   # Deutsche Telekom
    "DE000ENAG999": "EOAN.DE",  # E.ON
    "DE0005785802": "FME.DE",   # Fresenius Medical
    "DE0005785604": "FRE.DE",   # Fresenius
    "DE0008402215": "HNR1.DE",  # Hannover Rück
    "DE0006047004": "HEI.DE",   # Heidelberg Materials
    "DE0006048432": "HEN3.DE",  # Henkel
    "DE0006231004": "IFX.DE",   # Infineon
    "DE0007100000": "MBG.DE",   # Mercedes-Benz
    "DE0006599905": "MRK.DE",   # Merck
    "DE000A0D9PT0": "MTX.DE",   # MTU Aero
    "DE0008430026": "MUV2.DE",  # Münchener Rück
    "DE000PAG9113": "P911.DE",  # Porsche
    "DE0006969603": "PUM.DE",   # Puma
    "NL0012169213": "QIA.DE",   # Qiagen
    "DE0007030009": "RHM.DE",   # Rheinmetall
    "DE0007037129": "RWE.DE",   # RWE
    "DE0007164600": "SAP.DE",   # SAP
    "DE0007165631": "SRT3.DE",  # Sartorius
    "DE0007236101": "SIE.DE",   # Siemens
    "DE000ENER6Y0": "ENR.DE",   # Siemens Energy
    "DE000SHL1006": "SHL.DE",   # Siemens Healthineers
    "DE000SYM9999": "SY1.DE",   # Symrise
    "DE0007664039": "VOW3.DE",  # Volkswagen
    "DE000A1ML7J1": "VNA.DE",   # Vonovia
    "DE000ZAL1111": "ZAL.DE",   # Zalando
}

EUREX_URL = (
    "https://www.eurex.com/ex-en/markets/equ/equ-opt/options/Rheinmetall-951816"
)
WS_URL = "wss://eurex-api.factsetdigitalsolutions.com/ws"


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_price(sigma: float, S: float, K: float, T: float, r: float,
             option_type: str) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CALL":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def iv_from_price(price: float, S: float, K: float, T: float, r: float,
                  option_type: str) -> float:
    if T <= 0 or price <= 0:
        return np.nan
    try:
        return brentq(lambda s: bs_price(s, S, K, T, r, option_type) - price,
                       0.01, 3.0, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------------
# Eurex WebSocket client
# ---------------------------------------------------------------------------

class _EurexWSClient:
    """Fetches option contracts from Eurex via WebSocket."""

    def __init__(self, token: str, isins: List[str]):
        self.token = token
        self.isins = list(isins)
        self.contracts: List[Dict] = []
        self.authenticated = False
        self.pending = 0
        self.isin_map: Dict[int, str] = {}
        self.isin_queue: List[str] = []
        self.max_concurrent = 5
        self.completed = False
        self._next_job = 0

        self.ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            subprotocols=["v2.ws-jsjson.mdgms.com"],
        )

    # -- WS callbacks -------------------------------------------------------

    def _on_open(self, ws):
        ws.send(self._auth_msg())

    def _on_message(self, ws, message):
        msg = json.loads(message)
        mtype = msg.get("Message", "")

        if not self.authenticated and "AuthenticationByTokenResponse" in mtype:
            if msg.get("server_info") or "error" not in msg:
                self.authenticated = True
                self.isin_queue = self.isins.copy()
                self._send_batch(ws)
            else:
                logger.error("Eurex auth failed: %s", msg)
                ws.close()
                self.completed = True
            return

        if "HighLevelResponse" in mtype:
            self._handle_data(ws, msg)
        elif "ErrorResponse" in mtype:
            self._handle_error(ws, msg)

    def _on_error(self, ws, error):
        logger.error("Eurex WS error: %s", error)
        self.completed = True

    def _on_close(self, ws, code, reason):
        logger.info("Eurex WS closed (code=%s)", code)
        self.completed = True

    # -- Internals ----------------------------------------------------------

    def _auth_msg(self) -> str:
        return json.dumps({
            "Message": "AuthenticationByTokenRequest",
            "Version": 1,
            "token": {"value": {"b64": self.token}},
            "software": json.dumps({
                "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "platform": "Win32", "version": "5.5.0",
                "package": "@fds/wm-typescript-mdg2-client",
                "build": "esnext", "mobile": False,
            }),
            "os": "Win32",
            "feature_flags_wanted": {"value": 0},
            "maximum_idle_interval": 45000000,
            "maximum_receivable_message_size": 1048576,
            "flags": 0,
            "cache_authentication_salt": {"value": []},
            "cache_authentication_encrypted_secret": {"encrypted_secret": []},
        })

    def _request_msg(self, isin: str, job_id: int) -> str:
        return json.dumps({
            "header": {
                "dataset": {"id_dataset": 0}, "id_job": job_id,
                "flags_r2": 0, "resend_counter": 0, "timeout": 60000,
                "authentication_identifiers": {"id_application": -2, "id_user": -2},
                "cache_key": {"value": []},
                "previous_response_hash": {"value": []},
                "tracing": {"value": {"value": ""}},
            },
            "Message": "HighLevelRequest", "Version": 3,
            "accept": "application/json", "content_type": "application/json",
            "body": {"value": []},
            "query": f"isin={isin}&zeroValues=true&_paginationOffset=0&_paginationLimit=3000",
            "path": "/api/v2/custom/prices/get",
            "method": {"value": 1},
        })

    def _send_batch(self, ws):
        while self.pending < self.max_concurrent and self.isin_queue:
            isin = self.isin_queue.pop(0)
            jid = self._next_job
            self._next_job += 1
            ws.send(self._request_msg(isin, jid))
            self.isin_map[jid] = isin
            self.pending += 1

    def _decode_body(self, msg: dict) -> List[Dict]:
        body = msg.get("body", {})
        val = body.get("value")
        if not val:
            return msg.get("items", [])

        try:
            if isinstance(val, str):
                js = json.loads(val)
            elif isinstance(val, list):
                js = json.loads(bytes(val).decode("utf-8"))
            else:
                return []
            return js.get("data", {}).get("contracts", js.get("items", []))
        except Exception:
            return []

    def _handle_data(self, ws, msg):
        header = msg.get("header", {})
        job_id = header.get("id_job") if isinstance(header, dict) else None
        underlying_isin = self.isin_map.get(job_id)

        items = self._decode_body(msg)
        if items:
            # Derive underlying name from first contract
            underlying_name = ""
            if items[0].get("name"):
                underlying_name = items[0]["name"].split(" - ")[0].split(" (")[0]

            for item in items:
                otype = "PUT" if "/PUT/" in item.get("name", "") else "CALL"
                self.contracts.append({
                    "isin": item.get("isin"),
                    "underlying_isin": underlying_isin,
                    "name": item.get("name", ""),
                    "strikePrice": item.get("strikePrice", 0),
                    "dateMaturity": item.get("dateMaturity", ""),
                    "underlying_name": underlying_name,
                    "option_type": otype,
                    "price": item.get("settlementPrice", np.nan),
                })
            logger.info("Fetched %d contracts for %s", len(items), underlying_name)

        self.pending -= 1
        self._send_batch(ws)
        if self.pending <= 0 and not self.isin_queue:
            ws.close()

    def _handle_error(self, ws, msg):
        job_id = msg.get("id_job")
        if job_id is None:
            h = msg.get("header", {})
            job_id = h.get("id_job") if isinstance(h, dict) else None
        failed = self.isin_map.get(job_id, "Unknown")
        logger.warning("Eurex error for %s: %s", failed,
                        msg.get("details") or msg.get("error"))
        self.pending -= 1
        self._send_batch(ws)
        if self.pending <= 0 and not self.isin_queue:
            ws.close()

    # -- Public entry -------------------------------------------------------

    def fetch(self, timeout: float = 60) -> List[Dict]:
        def _timeout():
            time.sleep(timeout)
            if not self.completed:
                self.ws.close()

        threading.Thread(target=_timeout, daemon=True).start()
        self.ws.run_forever(origin="https://www.eurex.com")
        return self.contracts


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class ForwardVolatilityScanner(BaseScanner):
    """
    Scans DAX options for forward-volatility calendar spread opportunities.
    Runs daily at 10:00 CET (weekdays) after European market open.
    """

    def __init__(self, use_ib: bool = False,
                 ib_host: str = "127.0.0.1", ib_port: int = 4002):
        super().__init__("Forward Volatility")
        self.use_ib = use_ib
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.data_file = "data/forward_volatility_results.feather"

    @property
    def schedule(self) -> str:
        return "0 10 * * 1-5"

    # -- Eurex token --------------------------------------------------------

    @staticmethod
    def _fetch_connection_token() -> Optional[str]:
        import requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/141.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(EUREX_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        m = re.search(r'"connectionToken":\s*"([^"]+)"', resp.text)
        if m:
            logger.info("Eurex token fetched")
            return m.group(1)
        logger.error("No connectionToken in Eurex page")
        return None

    # -- Spot prices --------------------------------------------------------

    @staticmethod
    def fetch_spot_prices(isins: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for isin in isins:
            ticker = ISIN_TO_TICKER.get(isin)
            if not ticker:
                continue
            try:
                t = yf.Ticker(ticker)
                spot = None
                # fast_info
                try:
                    spot = t.fast_info.get("lastPrice")
                except Exception:
                    pass
                # info dict
                if spot is None or (isinstance(spot, float) and (np.isnan(spot) or spot == 0)):
                    try:
                        info = t.info
                        spot = info.get("currentPrice") or info.get("regularMarketPrice")
                    except Exception:
                        pass
                # history fallback
                if spot is None or (isinstance(spot, float) and (np.isnan(spot) or spot == 0)):
                    try:
                        h = t.history(period="1d")
                        if not h.empty:
                            spot = h["Close"].iloc[-1]
                    except Exception:
                        pass

                if spot and not (isinstance(spot, float) and np.isnan(spot)) and spot > 0:
                    out[isin] = float(spot)
            except Exception as exc:
                logger.warning("Price fetch failed for %s: %s", ticker, exc)
        logger.info("Fetched %d/%d spot prices", len(out), len(isins))
        return out

    # -- Option contracts ---------------------------------------------------

    def fetch_option_contracts(self, isins: List[str]) -> pd.DataFrame:
        token = self._fetch_connection_token()
        if not token:
            return pd.DataFrame()

        client = _EurexWSClient(token, isins)
        contracts = client.fetch()

        if not contracts:
            return pd.DataFrame()

        df = pd.DataFrame(contracts)
        df["expiry_date"] = pd.to_datetime(df["dateMaturity"], utc=True)
        now = pd.Timestamp.now(tz="UTC")
        df["days_to_expiry"] = df["expiry_date"].apply(lambda x: (x - now).days)
        df["fetch_timestamp"] = now
        df = df.sort_values(["underlying_isin", "expiry_date", "option_type", "strikePrice"])
        return df.reset_index(drop=True)

    # -- Forward volatility calculation -------------------------------------

    def calculate_forward_vols(self, df_options: pd.DataFrame,
                               spot_prices: Dict[str, float]) -> pd.DataFrame:
        if self.use_ib:
            return self._fwd_from_ib(df_options, spot_prices)
        return self._fwd_from_eurex(df_options, spot_prices)

    def _fwd_from_eurex(self, df: pd.DataFrame,
                        spot: Dict[str, float]) -> pd.DataFrame:
        r = 0.025
        rows = []
        for isin, grp in df.groupby("underlying_isin"):
            S = spot.get(isin)
            if S is None:
                continue
            expiries = sorted(grp["expiry_date"].unique())
            for i in range(len(expiries) - 1):
                near_exp, far_exp = expiries[i], expiries[i + 1]
                ng = grp[grp["expiry_date"] == near_exp]
                fg = grp[grp["expiry_date"] == far_exp]
                if ng.empty or fg.empty:
                    continue

                near_days = ng["days_to_expiry"].iloc[0]
                far_days = fg["days_to_expiry"].iloc[0]
                if not (25 <= (far_days - near_days) <= 40):
                    continue

                # ATM ±5%
                near_atm = ng[(ng["strikePrice"] / S >= 0.95) & (ng["strikePrice"] / S <= 1.05)]
                far_atm = fg[(fg["strikePrice"] / S >= 0.95) & (fg["strikePrice"] / S <= 1.05)]
                if near_atm.empty or far_atm.empty:
                    continue

                def _avg_iv(group, days):
                    ivs = []
                    for _, opt in group.iterrows():
                        T = days / 365.0
                        price = opt.get("price", np.nan)
                        if np.isnan(price) or price <= 0:
                            continue
                        iv = iv_from_price(price, S, opt["strikePrice"], T, r, opt["option_type"])
                        if not np.isnan(iv) and 0.01 < iv < 3.0:
                            ivs.append(iv)
                    return float(np.mean(ivs)) if ivs else None

                near_iv = _avg_iv(near_atm, near_days)
                far_iv = _avg_iv(far_atm, far_days)
                if near_iv is None or far_iv is None:
                    continue

                T1, T2 = near_days / 365.0, far_days / 365.0
                var_fwd = (T2 * far_iv**2 - T1 * near_iv**2) / (T2 - T1)
                if var_fwd > 0:
                    fwd_vol = np.sqrt(var_fwd)
                    strikes = near_atm["strikePrice"].values
                    atm_strike = strikes[np.argmin(np.abs(strikes - S))]
                    rows.append({
                        "underlying_isin": isin,
                        "underlying_name": ng["underlying_name"].iloc[0],
                        "spot_price": S,
                        "atm_strike": atm_strike,
                        "near_expiry": near_exp,
                        "far_expiry": far_exp,
                        "near_days": near_days,
                        "far_days": far_days,
                        "near_iv": near_iv,
                        "far_iv": far_iv,
                        "forward_vol": fwd_vol,
                        "near_iv_count": len(near_atm),
                        "far_iv_count": len(far_atm),
                    })

        out = pd.DataFrame(rows)
        if not out.empty:
            out["fwd_vol_ratio"] = out["forward_vol"] / out["near_iv"]
        logger.info("Calculated %d forward-vol pairs", len(out))
        return out

    def _fwd_from_ib(self, df: pd.DataFrame,
                     spot: Dict[str, float]) -> pd.DataFrame:
        try:
            from ib_data_fetcher import IBDataFetcher  # type: ignore
        except ImportError:
            logger.error("ib_insync not installed; cannot use IB mode")
            return pd.DataFrame()

        fetcher = IBDataFetcher(self.ib_host, self.ib_port)
        rows = []
        try:
            fetcher.connect()
            for isin, grp in df.groupby("underlying_isin"):
                S = spot.get(isin)
                if S is None:
                    continue
                stock = fetcher.get_eurex_stock(isin)
                if not stock:
                    continue
                name = grp["underlying_name"].iloc[0]
                expiries = sorted(grp["expiry_date"].unique())
                for i in range(len(expiries) - 1):
                    near_exp, far_exp = expiries[i], expiries[i + 1]
                    near_days = (near_exp - pd.Timestamp.now(tz="UTC")).days
                    far_days = (far_exp - pd.Timestamp.now(tz="UTC")).days
                    if not (25 <= (far_days - near_days) <= 40):
                        continue
                    n_iv = fetcher.get_atm_options_iv(stock, S, near_exp.strftime("%Y%m%d"))
                    f_iv = fetcher.get_atm_options_iv(stock, S, far_exp.strftime("%Y%m%d"))
                    if n_iv["average_iv"] is None or f_iv["average_iv"] is None:
                        continue
                    T1, T2 = near_days / 365.0, far_days / 365.0
                    var = (T2 * f_iv["average_iv"]**2 - T1 * n_iv["average_iv"]**2) / (T2 - T1)
                    if var > 0:
                        strikes = grp[grp["expiry_date"] == near_exp]["strikePrice"].values
                        atm = strikes[np.argmin(np.abs(strikes - S))]
                        rows.append({
                            "underlying_isin": isin, "underlying_name": name,
                            "spot_price": S, "atm_strike": atm,
                            "near_expiry": near_exp, "far_expiry": far_exp,
                            "near_days": near_days, "far_days": far_days,
                            "near_iv": n_iv["average_iv"], "far_iv": f_iv["average_iv"],
                            "forward_vol": np.sqrt(var),
                            "near_iv_count": n_iv["num_calls"] + n_iv["num_puts"],
                            "far_iv_count": f_iv["num_calls"] + f_iv["num_puts"],
                        })
        finally:
            fetcher.disconnect()

        out = pd.DataFrame(rows)
        if not out.empty:
            out["fwd_vol_ratio"] = out["forward_vol"] / out["near_iv"]
        return out

    # -- Scoring ------------------------------------------------------------

    @staticmethod
    def score_opportunities(df_fwd: pd.DataFrame) -> pd.DataFrame:
        if df_fwd.empty:
            return df_fwd

        df = df_fwd.copy()
        if "fwd_vol_ratio" not in df.columns:
            df["fwd_vol_ratio"] = df["forward_vol"] / df["near_iv"]
        df = df[df["fwd_vol_ratio"] > 1.1].copy()
        if df.empty:
            return df

        def _score(row):
            ratio = row["fwd_vol_ratio"]
            if ratio >= 1.3:
                score, grade = 50, "🔴 Excellent"
            elif ratio >= 1.2:
                score, grade = 35, "🟡 Good"
            elif ratio >= 1.1:
                score, grade = 20, "🔵 Fair"
            else:
                score, grade = 0, "⚪ Weak"

            tdiff = row["far_days"] - row["near_days"]
            if 50 <= tdiff <= 70:
                score += 15
            elif 30 <= tdiff < 50 or 70 < tdiff <= 90:
                score += 10
            elif tdiff > 90:
                score += 5

            if row["near_days"] < 60:
                score += 10
            elif row["near_days"] < 90:
                score += 5

            return pd.Series({"score": score, "grade": grade})

        df[["score", "grade"]] = df.apply(_score, axis=1)
        return df.sort_values("score", ascending=False)

    # -- Main scan ----------------------------------------------------------

    def scan(self) -> Dict[str, Any]:
        try:
            isins = list(ISIN_TO_TICKER.keys())
            logger.info("Scanning %d DAX stocks", len(isins))

            spot = self.fetch_spot_prices(isins)
            df_opts = self.fetch_option_contracts(isins)

            if df_opts.empty:
                return {"success": False, "error": "No option data retrieved",
                        "timestamp": datetime.now(timezone.utc)}

            df_fwd = self.calculate_forward_vols(df_opts, spot)
            if df_fwd.empty:
                return {"success": False, "error": "No forward-vol pairs",
                        "timestamp": datetime.now(timezone.utc)}

            df_scored = self.score_opportunities(df_fwd)

            fields = [{
                "name": "📊 Summary",
                "value": (f"Found {len(df_scored)} calendar spread opportunities\n"
                          f"Scanned {len(isins)} stocks, analyzed {len(df_fwd)} pairs"),
                "inline": False,
            }]

            for _, row in df_scored.head(5).iterrows():
                ne = row["near_expiry"].strftime("%Y-%m-%d") if hasattr(row["near_expiry"], "strftime") else str(row["near_expiry"])
                fe = row["far_expiry"].strftime("%Y-%m-%d") if hasattr(row["far_expiry"], "strftime") else str(row["far_expiry"])
                fields.append({
                    "name": f"📈 {row['underlying_name']}",
                    "value": (
                        f"{row['grade']} FV Ratio: {row['fwd_vol_ratio']:.2f}x\n"
                        f"Spot: €{row['spot_price']:.2f}\n"
                        f"📤 SELL {ne} €{row['atm_strike']:.0f} ({row['near_days']}d)\n"
                        f"📥 BUY  {fe} €{row['atm_strike']:.0f} ({row['far_days']}d)\n"
                        f"Near IV: {row['near_iv']*100:.1f}% | Far IV: {row['far_iv']*100:.1f}%\n"
                        f"Forward Vol: {row['forward_vol']*100:.1f}%"
                    ),
                    "inline": False,
                })

            if len(df_scored) > 5:
                fields.append({
                    "name": "➕ More",
                    "value": f"...and {len(df_scored) - 5} more",
                    "inline": False,
                })

            # Persist scored results
            if not df_scored.empty:
                try:
                    df_scored.to_feather(self.data_file)
                except Exception:
                    pass

            return {
                "success": True,
                "embed": {
                    "title": "Forward Volatility Opportunities",
                    "color": 3066993,
                    "fields": fields,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "recommendations": df_scored.to_dict("records"),
                "metrics": {
                    "total_opportunities": len(df_scored),
                    "total_pairs": len(df_fwd),
                    "stocks_scanned": len(isins),
                    "contracts_fetched": len(df_opts),
                },
                "timestamp": datetime.now(timezone.utc),
            }

        except Exception as exc:
            logger.exception("Forward-vol scan failed")
            return {"success": False, "error": str(exc),
                    "timestamp": datetime.now(timezone.utc)}
