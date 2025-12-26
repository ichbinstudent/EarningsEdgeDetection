"""
Forward Volatility Scanner that inherits from BaseScanner.
"""

from scanner_base import BaseScanner
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
import os
import requests
import websocket
import json
import re
import time
import threading
import random
import yfinance as yf

warnings.filterwarnings("ignore")


class ForwardVolatilityScanner(BaseScanner):
    """
    Scanner for forward implied volatility opportunities in DAX options.
    Runs daily at 10:00 EST (after European market open).
    """

    def __init__(self, use_ib: bool = False, ib_host: str = "127.0.0.1", ib_port: int = 4002):
        super().__init__("Forward Volatility")
        # Data will be fetched fresh each run
        self.data_file = "forward_volatility_results.feather"
        self.eurex_url = (
            "https://www.eurex.com/ex-en/markets/equ/equ-opt/options/Rheinmetall-951816"
        )
        self.websocket_url = "wss://eurex-api.factsetdigitalsolutions.com/ws"
        
        # IB integration settings
        self.use_ib = use_ib
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.ib_fetcher = None

    @property
    def schedule(self) -> str:
        """
        Run daily at 10:00 EST (4:00 PM UTC) on weekdays.
        This is after European market open when fresh data is available.
        """
        return "0 10 * * 1-5"

    def get_isin_to_ticker_mapping(self) -> Dict[str, str]:
        """Return mapping of DAX ISINs to Yahoo Finance tickers."""
        return {
            "DE000A1EWWW0": "ADS.DE",  # Adidas
            "NL0000235190": "AIR.PA",  # Airbus (Paris)
            "DE0008404005": "ALV.DE",  # Allianz
            "DE000BASF111": "BAS.DE",  # BASF
            "DE000BAY0017": "BAYN.DE",  # Bayer
            "DE0005200000": "BEI.DE",  # Beiersdorf
            "DE0005190003": "BMW.DE",  # BMW
            "DE000A1DAHH0": "BNR.DE",  # Brenntag
            "DE0005439004": "CON.DE",  # Continental
            "DE000CBK1001": "CBK.DE",  # Commerzbank
            "DE000DTR0CK8": "DTG.DE",  # Daimler Truck
            "DE0005140008": "DBK.DE",  # Deutsche Bank
            "DE0005810055": "DB1.DE",  # Deutsche Börse
            "DE0005552004": "DHL.DE",  # Deutsche Post (CORRECTED)
            "DE0005557508": "DTE.DE",  # Deutsche Telekom
            "DE000ENAG999": "EOAN.DE",  # E.ON
            "DE0005785802": "FME.DE",  # Fresenius Medical
            "DE0005785604": "FRE.DE",  # Fresenius
            "DE0008402215": "HNR1.DE",  # Hannover Rück
            "DE0006047004": "HEI.DE",  # Heidelberg Materials
            "DE0006048432": "HEN3.DE",  # Henkel
            "DE0006231004": "IFX.DE",  # Infineon
            "DE0007100000": "MBG.DE",  # Mercedes-Benz
            "DE0006599905": "MRK.DE",  # Merck
            "DE000A0D9PT0": "MTX.DE",  # MTU Aero
            "DE0008430026": "MUV2.DE",  # Münchener Rück
            "DE000PAG9113": "P911.DE",  # Porsche
            "DE0006969603": "PUM.DE",  # Puma
            "NL0012169213": "QIA.DE",  # Qiagen
            "DE0007030009": "RHM.DE",  # Rheinmetall
            "DE0007037129": "RWE.DE",  # RWE
            "DE0007164600": "SAP.DE",  # SAP
            "DE0007165631": "SRT3.DE",  # Sartorius
            "DE0007236101": "SIE.DE",  # Siemens
            "DE000ENER6Y0": "ENR.DE",  # Siemens Energy
            "DE000SHL1006": "SHL.DE",  # Siemens Healthineers
            "DE000SYM9999": "SY1.DE",  # Symrise
            "DE0007664039": "VOW3.DE",  # Volkswagen
            "DE000A1ML7J1": "VNA.DE",  # Vonovia
            "DE000ZAL1111": "ZAL.DE",  # Zalando
        }

    def fetch_spot_prices(self, isins: List[str]) -> Dict[str, float]:
        """
        Fetch current spot prices from Yahoo Finance for given ISINs.
        Returns dict mapping ISIN to spot price.
        """
        isin_to_ticker = self.get_isin_to_ticker_mapping()
        spot_prices = {}

        print(f"📈 Fetching spot prices for {len(isins)} stocks from Yahoo Finance...")

        for isin in isins:
            ticker_symbol = isin_to_ticker.get(isin)
            if not ticker_symbol:
                print(f"  ⚠️ No ticker mapping for ISIN {isin}, skipping...")
                continue

            try:
                ticker = yf.Ticker(ticker_symbol)

                # Try multiple methods to get the spot price
                spot = None

                # Method 1: fast_info (fastest, most recent)
                try:
                    spot = ticker.fast_info.get("lastPrice")
                except:
                    pass

                # Method 2: info dict (slower but comprehensive)
                if spot is None or pd.isna(spot) or spot == 0:
                    try:
                        info = ticker.info
                        spot = info.get("currentPrice") or info.get(
                            "regularMarketPrice"
                        )
                    except:
                        pass

                # Method 3: history (fallback to last close)
                if spot is None or pd.isna(spot) or spot == 0:
                    try:
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            spot = hist["Close"].iloc[-1]
                    except:
                        pass

                if spot and not pd.isna(spot) and spot > 0:
                    spot_prices[isin] = float(spot)
                    print(f"  ✓ {ticker_symbol}: €{spot:.2f}")
                else:
                    print(
                        f"  ⚠️ Could not fetch price for {ticker_symbol} (ISIN: {isin})"
                    )

            except Exception as e:
                print(f"  ❌ Error fetching {ticker_symbol}: {str(e)}")
                continue

        print(f"✓ Successfully fetched {len(spot_prices)} spot prices")
        return spot_prices

    def fetch_connection_token(self) -> Optional[str]:
        """
        Fetch the connection token from the Eurex website.
        
        Args:
            url: The URL to fetch the token from
            
        Returns:
            The connection token string or None if not found
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0'
            }
            
            response = requests.get(self.eurex_url, headers=headers)
            response.raise_for_status()
            
            # Search for the connectionToken pattern
            pattern = r'"connectionToken":\s*"([^"]+)"'
            match = re.search(pattern, response.text)
            
            if match:
                token = match.group(1)
                print(f"✓ Token fetched successfully: {token}")
                return token
            else:
                print("✗ Connection token not found in response")
                return None
                
        except Exception as e:
            print(f"✗ Error fetching token: {e}")
            return None

    def fetch_option_contracts(self, isins: List[str]) -> pd.DataFrame:
        """
        Fetch option contracts for given ISINs from Eurex WebSocket API.
        Uses sequential approach: one connection per ISIN.
        """
        token = self.fetch_connection_token()
        
        if not token:
            print("❌ Failed to fetch connection token, cannot proceed")
            return pd.DataFrame()
        
        all_contracts = []

        class FetchClient:
            def __init__(self, ws_url, token, isins):
                self.ws_url = ws_url
                self.token = token
                self.contracts = []
                self.authenticated = False
                self.isins = list(isins)  # Make a copy to avoid mutating original
                self.pending_requests = 0
                self.isin_map = {}  # Map job_id to ISIN for tracking
                self.isin_queue = []  # Queue of ISINs to request
                self.max_concurrent = 5  # Maximum concurrent requests
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    subprotocols=["v2.ws-jsjson.mdgms.com"]
                )
                self.completed = False
                self.next_job_id = 0

            def create_auth_message(self):
                return json.dumps({
                    "Message": "AuthenticationByTokenRequest",
                    "Version": 1,
                    "token": {"value": {"b64": self.token}},
                    "software": json.dumps({
                        "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "platform": "Win32",
                        "version": "5.5.0",
                        "package": "@fds/wm-typescript-mdg2-client",
                        "build": "esnext",
                        "mobile": False
                    }),
                    "os": "Win32",
                    "feature_flags_wanted": {"value": 0},
                    "maximum_idle_interval": 45000000,
                    "maximum_receivable_message_size": 1048576,
                    "flags": 0,
                    "cache_authentication_salt": {"value": []},
                    "cache_authentication_encrypted_secret": {"encrypted_secret": []}
                })

            def send_next_batch(self, ws):
                """Send next batch of requests from the queue."""
                while self.pending_requests < self.max_concurrent and self.isin_queue:
                    isin = self.isin_queue.pop(0)
                    job_id = self.next_job_id
                    self.next_job_id += 1
                    
                    request_msg = self.create_request(isin, job_id)
                    ws.send(request_msg)
                    self.isin_map[job_id] = isin
                    self.pending_requests += 1
                    print(f"  📤 Sent request for {isin} (job_id: {job_id}, pending: {self.pending_requests})")

            def create_request(self, isin, job_id):
                return json.dumps({
                    "header": {
                        "dataset": {"id_dataset": 0},
                        "id_job": job_id,
                        "flags_r2": 0,
                        "resend_counter": 0,
                        "timeout": 60000,
                        "authentication_identifiers": {"id_application": -2, "id_user": -2},
                        "cache_key": {"value": []},
                        "previous_response_hash": {"value": []},
                        "tracing": {"value": {"value": ""}}
                    },
                    "Message": "HighLevelRequest",
                    "Version": 3,
                    "accept": "application/json",
                    "content_type": "application/json",
                    "body": {"value": []},
                    "query": f"isin={isin}&zeroValues=true&_paginationOffset=0&_paginationLimit=3000",
                    "path": "/api/v2/custom/prices/get",
                    "method": {"value": 1}
                })

            def on_open(self, ws):
                auth_msg = self.create_auth_message()
                ws.send(auth_msg)

            def on_message(self, ws, message):
                try:
                    msg = json.loads(message)
                    msg_type = msg.get("Message", "Unknown")
                    
                    # Debug: print message type
                    print(f"  📩 Received message type: {msg_type}")
                    
                    # Handle authentication response - check for both formats
                    if (
                        not self.authenticated
                        and ("AuthenticationByTokenResponse" in msg_type)
                    ):
                        # Check if authentication was successful
                        # The response doesn't have an 'authenticated' field if successful
                        # Instead, check for server_info presence or lack of error
                        if msg.get("server_info") or "error" not in msg:
                            self.authenticated = True
                            print(f"  ✓ Authenticated successfully")
                            # Initialize queue with all ISINs
                            self.isin_queue = self.isins.copy()
                            # Send first batch of requests
                            self.send_next_batch(ws)
                            print(f"  ✓ Initialized with {len(self.isins)} ISINs, sending in batches of {self.max_concurrent}")
                        else:
                            print(f"  ❌ Authentication failed: {msg}")
                            ws.close()
                            self.completed = True
                        return
                    
                    # Handle data response - check for HighLevelResponse
                    if "HighLevelResponse" in msg_type:
                        # Get the job_id to identify which ISIN this response is for
                        header = msg.get("header", {})
                        job_id = header.get("id_job") if isinstance(header, dict) else None
                        underlying_isin = self.isin_map.get(job_id) if job_id is not None else None
                        
                        # Decode the body if present
                        body_data = msg.get("body", {})
                        items = []
                        
                        if body_data and "value" in body_data:
                            body_value = body_data["value"]
                            
                            try:
                                # The body.value is a JSON string
                                if isinstance(body_value, str):
                                    body_json = json.loads(body_value)
                                    # The structure is: {"data": {"contracts": [...]}}
                                    if "data" in body_json and "contracts" in body_json["data"]:
                                        items = body_json["data"]["contracts"]
                                        print(f"  � Decoded {len(items)} contracts from response body")
                                    else:
                                        print(f"  ⚠️ Unexpected JSON structure: {list(body_json.keys())}")
                                        items = body_json.get("items", [])
                                elif isinstance(body_value, list):
                                    # If it's a list of integers (byte array), convert to bytes
                                    body_bytes = bytes(body_value)
                                    decoded_str = body_bytes.decode('utf-8')
                                    body_json = json.loads(decoded_str)
                                    if "data" in body_json and "contracts" in body_json["data"]:
                                        items = body_json["data"]["contracts"]
                                        print(f"  � Decoded {len(items)} contracts from byte array")
                                    else:
                                        items = body_json.get("items", [])
                                else:
                                    print(f"  ⚠️ Unexpected body_value type: {type(body_value)}")
                                    items = []
                                
                            except Exception as e:
                                import traceback
                                print(f"  ⚠️ Body decode failed: {e}")
                                print(f"  ⚠️ Traceback: {traceback.format_exc()}")
                                items = msg.get("items", [])
                        else:
                            items = msg.get("items", [])
                        
                        if items:
                            # Extract underlying name from the first item's name field
                            underlying_name = ""
                            if len(items) > 0 and "name" in items[0]:
                                # Parse "adidas AG (ADS) - EUX/CALL/..." to get "adidas AG"
                                name_parts = items[0]["name"].split(" - ")
                                if len(name_parts) > 0:
                                    underlying_name = name_parts[0].split(" (")[0]
                            
                            for item in items:
                                # Extract option type from name (CALL or PUT)
                                option_type = "CALL"  # default
                                if "name" in item:
                                    if "/PUT/" in item["name"]:
                                        option_type = "PUT"
                                    elif "/CALL/" in item["name"]:
                                        option_type = "CALL"
                                
                                # print(item)
                                contract = {
                                    "isin": item.get("isin"),
                                    "underlying_isin": underlying_isin,  # Use the ISIN from our request
                                    "name": item.get("name", ""),
                                    "strikePrice": item.get("strikePrice", 0),
                                    "dateMaturity": item.get("dateMaturity", ""),
                                    "underlying_name": underlying_name,
                                    "option_type": option_type,
                                    "price": item.get("settlementPrice", np.nan),
                                }
                                self.contracts.append(contract)
                            print(f"  ✅ Added {len(items)} contracts for {underlying_name} ({underlying_isin})")
                        else:
                            print(f"  ℹ️ No items in response")
                        
                        # Mark request as complete
                        self.pending_requests -= 1
                        print(f"  ✓ Received response ({self.pending_requests} pending, {len(self.contracts)} total contracts)")
                        
                        # Send next batch of requests if queue not empty
                        self.send_next_batch(ws)
                        
                        # Close connection when all responses received and queue is empty
                        if self.pending_requests <= 0 and not self.isin_queue:
                            print(f"  ✓ All requests completed, closing connection")
                            ws.close()
                    
                    # Handle error responses
                    elif "ErrorResponse" in msg_type:
                        # Get job_id to identify which ISIN failed
                        # In error responses, id_job is at root level, not in header
                        job_id = msg.get("id_job")
                        if job_id is None:
                            header = msg.get("header", {})
                            job_id = header.get("id_job") if isinstance(header, dict) else None
                        
                        failed_isin = self.isin_map.get(job_id) if job_id is not None else "Unknown"
                        
                        # Get error details
                        error_details = msg.get("details") or msg.get("error") or "Unknown error"
                        reason_code = msg.get("reason", {}).get("value") if isinstance(msg.get("reason"), dict) else None
                        
                        # Only print summary
                        if reason_code == 11:  # "send buffer full" - rate limiting
                            print(f"  ⚠️ Rate limit for {failed_isin}: {error_details}")
                        elif job_id is not None:
                            print(f"  ⚠️ Error for {failed_isin}: {error_details}")
                        
                        # Still count as a response
                        self.pending_requests -= 1
                        
                        # Send next batch of requests if queue not empty
                        self.send_next_batch(ws)
                        
                        if self.pending_requests <= 0 and not self.isin_queue:
                            print(f"  ✓ All requests completed, closing connection")
                            ws.close()
                    else:
                        # Debug: print unexpected message types
                        print(f"  ℹ️ Unhandled message type: {msg_type}")

                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON decode error: {e}")
                    print(f"  📄 Raw message: {message[:200]}...")

            def on_error(self, ws, error):
                print(f"  ❌ WebSocket error: {error}")
                self.completed = True

            def on_close(self, ws, code, msg):
                print(f"  🔌 WebSocket closed (code: {code})")
                self.completed = True

            def fetch(self):
                def timeout_close():
                    time.sleep(60)  # Increased timeout to 60 seconds
                    if not self.completed:
                        print(f"  ⏱️ Timeout reached after 60s")
                        self.ws.close()

                timeout_thread = threading.Thread(target=timeout_close, daemon=True)
                timeout_thread.start()

                print(f"  🔌 Connecting to WebSocket...")
                self.ws.run_forever(origin="https://www.eurex.com")
                print(f"  🔌 WebSocket connection ended")

                return self.contracts

        # Fetch contracts sequentially for each ISIN
        client = FetchClient(self.websocket_url, token, list(isins))
        all_contracts = client.fetch()

        if not all_contracts:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(all_contracts)

        # Enrich the data
        # option_type is already included in the contracts
        df["expiry_date"] = pd.to_datetime(df["dateMaturity"], utc=True)
        now = pd.Timestamp.now(tz="UTC")
        df["days_to_expiry"] = df["expiry_date"].apply(lambda x: (x - now).days)
        df["fetch_timestamp"] = pd.Timestamp.now(tz="UTC")

        # Sort by ISIN, expiry, strike
        df = df.sort_values(
            ["underlying_isin", "expiry_date", "option_type", "strikePrice"]
        )
        df = df.reset_index(drop=True)

        return df

    def score_opportunities(self, df_forward):
        """Score forward volatility opportunities based on FV ratio criteria."""
        if df_forward.empty:
            return pd.DataFrame()

        df = df_forward.copy()
        df["fwd_vol_ratio"] = df["forward_vol"] / df["near_iv"]

        # Filter to only include opportunities with FV ratio > 1.1
        df = df[df["fwd_vol_ratio"] > 1.1].copy()

        print(f"📊 Filtered to {len(df)} opportunities with FV ratio > 1.1")

        if df.empty:
            return pd.DataFrame()

        # Simplified scoring based primarily on FV ratio
        def score_calendar_spread(row):
            score = 0
            ratio = row["fwd_vol_ratio"]

            # FV ratio scoring (primary factor)
            if ratio >= 1.3:
                score += 50  # Excellent
                grade = "🔴 Excellent"
            elif ratio >= 1.2:
                score += 35  # Good
                grade = "🟡 Good"
            elif ratio >= 1.1:
                score += 20  # Fair
                grade = "🔵 Fair"
            else:
                grade = "⚪ Weak"

            # Time spread quality (10-30 days is optimal for quarterly spreads)
            time_diff = row["far_days"] - row["near_days"]
            if 50 <= time_diff <= 70:
                score += 15  # Ideal quarterly spread
            elif 30 <= time_diff < 50 or 70 < time_diff <= 90:
                score += 10  # Good spread
            elif time_diff > 90:
                score += 5  # Wide spread (less ideal)

            # Liquidity bonus - prefer nearer expirations (more liquid)
            if row["near_days"] < 60:
                score += 10
            elif row["near_days"] < 90:
                score += 5

            return pd.Series({"score": score, "grade": grade})

        df[["score", "grade"]] = df.apply(score_calendar_spread, axis=1)
        df = df.sort_values("score", ascending=False)

        return df

    def calculate_iv_from_price(self, option_price, S, K, T, r, option_type):
        """Calculate implied volatility from option price using Black-Scholes."""
        if T <= 0 or option_price <= 0:
            return np.nan

        def bs_price(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "CALL":
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # PUT
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        def objective(sigma):
            return bs_price(sigma) - option_price

        try:
            iv = brentq(objective, 0.01, 3.0, maxiter=100)
            return iv
        except (ValueError, RuntimeError):
            return np.nan

    def calculate_forward_vols(self, df_options, spot_prices):
        """
        Calculate forward implied volatility from option data.
        Uses IB data if use_ib=True, otherwise falls back to Eurex contract specs.
        """
        print(f"📊 Calculating forward volatilities...")

        if self.use_ib:
            return self.calculate_forward_vols_from_ib(df_options, spot_prices)
        else:
            print("⚠️ WARNING: Using Eurex contract specs without pricing data.")
            print("   IVs will be estimated (unreliable). Enable IB integration for accurate data.")
            return self.calculate_forward_vols_from_eurex(df_options, spot_prices)

    def calculate_forward_vols_from_ib(self, df_options, spot_prices):
        """Calculate forward volatility using IB market data."""
        try:
            from ib_data_fetcher import IBDataFetcher
        except ImportError:
            print("❌ ib_insync not installed. Run: pip install ib_insync")
            return pd.DataFrame()

        forward_vols = []
        
        # Initialize IB connection
        self.ib_fetcher = IBDataFetcher(self.ib_host, self.ib_port)
        
        try:
            self.ib_fetcher.connect()
            
            # Group by underlying
            for underlying_isin, group in df_options.groupby("underlying_isin"):
                spot = spot_prices.get(underlying_isin)
                if spot is None:
                    print(f"  ⚠️ No spot price for {underlying_isin}, skipping...")
                    continue

                # Get stock contract from IB
                stock = self.ib_fetcher.get_eurex_stock(underlying_isin)
                if not stock:
                    print(f"  ⚠️ Could not find IB contract for {underlying_isin}, skipping...")
                    continue

                underlying_name = group["underlying_name"].iloc[0]
                print(f"  📈 Processing {underlying_name} (spot: €{spot:.2f})...")

                # Get unique expiries
                expiries = sorted(group["expiry_date"].unique())

                # For each pair of consecutive expiries
                for i in range(len(expiries) - 1):
                    near_expiry = expiries[i]
                    far_expiry = expiries[i + 1]

                    near_days = (near_expiry - pd.Timestamp.now(tz="UTC")).days
                    far_days = (far_expiry - pd.Timestamp.now(tz="UTC")).days

                    # Only consider pairs 25-40 days apart (quarterly spreads)
                    time_diff = far_days - near_days
                    if not (25 <= time_diff <= 40):
                        continue
                        
                    # Convert to IB date format (YYYYMMDD)
                    near_expiry_ib = near_expiry.strftime("%Y%m%d")
                    far_expiry_ib = far_expiry.strftime("%Y%m%d")

                    # Fetch IVs from IB
                    near_iv_data = self.ib_fetcher.get_atm_options_iv(
                        stock, spot, near_expiry_ib
                    )
                    far_iv_data = self.ib_fetcher.get_atm_options_iv(
                        stock, spot, far_expiry_ib
                    )

                    if (
                        near_iv_data["average_iv"] is None
                        or far_iv_data["average_iv"] is None
                    ):
                        continue

                    near_iv = near_iv_data["average_iv"]
                    far_iv = far_iv_data["average_iv"]

                    print(
                        f"    {near_expiry.strftime('%Y-%m-%d')}: IV={near_iv*100:.1f}% "
                        f"({near_iv_data['num_calls']}C + {near_iv_data['num_puts']}P)"
                    )
                    print(
                        f"    {far_expiry.strftime('%Y-%m-%d')}: IV={far_iv*100:.1f}% "
                        f"({far_iv_data['num_calls']}C + {far_iv_data['num_puts']}P)"
                    )

                    # Calculate forward volatility
                    T1 = near_days / 365.0
                    T2 = far_days / 365.0

                    var_forward = (T2 * far_iv**2 - T1 * near_iv**2) / (T2 - T1)

                    if var_forward > 0:
                        forward_vol = np.sqrt(var_forward)

                        # Find the ATM strike
                        near_group = group[group["expiry_date"] == near_expiry]
                        near_atm_strikes = near_group["strikePrice"].values
                        atm_strike = near_atm_strikes[
                            np.argmin(np.abs(near_atm_strikes - spot))
                        ]

                        forward_vols.append(
                            {
                                "underlying_isin": underlying_isin,
                                "underlying_name": underlying_name,
                                "spot_price": spot,
                                "atm_strike": atm_strike,
                                "near_expiry": near_expiry,
                                "far_expiry": far_expiry,
                                "near_days": near_days,
                                "far_days": far_days,
                                "near_iv": near_iv,
                                "far_iv": far_iv,
                                "forward_vol": forward_vol,
                                "near_iv_count": near_iv_data["num_calls"]
                                + near_iv_data["num_puts"],
                                "far_iv_count": far_iv_data["num_calls"]
                                + far_iv_data["num_puts"],
                            }
                        )

                        print(
                            f"    ✓ Forward Vol: {forward_vol*100:.1f}% "
                            f"(FV Ratio: {forward_vol/near_iv:.2f}x)\n"
                        )

        finally:
            if self.ib_fetcher:
                self.ib_fetcher.disconnect()

        df_fv = pd.DataFrame(forward_vols)
        print(f"✓ Calculated {len(df_fv)} forward volatility spreads")

        if not df_fv.empty:
            # Debug: print distribution of FV ratios
            df_fv["fwd_vol_ratio"] = df_fv["forward_vol"] / df_fv["near_iv"]
            print(f"\n📈 FV Ratio Distribution:")
            print(f"  Min: {df_fv['fwd_vol_ratio'].min():.3f}")
            print(f"  25%: {df_fv['fwd_vol_ratio'].quantile(0.25):.3f}")
            print(f"  50%: {df_fv['fwd_vol_ratio'].median():.3f}")
            print(f"  75%: {df_fv['fwd_vol_ratio'].quantile(0.75):.3f}")
            print(f"  Max: {df_fv['fwd_vol_ratio'].max():.3f}")
            print(f"  Count > 1.1: {(df_fv['fwd_vol_ratio'] > 1.1).sum()}")
            print(f"  Count > 1.2: {(df_fv['fwd_vol_ratio'] > 1.2).sum()}")
            print(f"  Count > 1.3: {(df_fv['fwd_vol_ratio'] > 1.3).sum()}\n")

        return df_fv

    def calculate_forward_vols_from_eurex(self, df_options, spot_prices):
        """
        Calculate forward implied volatility from Eurex contract specs (legacy method).
        WARNING: This uses estimated prices and will give unreliable IVs!
        """
        forward_vols = []
        r = 0.025  # Risk-free rate

        # Group by underlying
        for underlying_isin, group in df_options.groupby("underlying_isin"):
            spot = spot_prices.get(underlying_isin)
            if spot is None:
                print(f"  ⚠️ No spot price for {underlying_isin}, skipping...")
                continue

            # Get unique expiries
            expiries = sorted(group["expiry_date"].unique())

            # For each pair of consecutive expiries
            for i in range(len(expiries) - 1):
                near_expiry = expiries[i]
                far_expiry = expiries[i + 1]

                near_group = group[group["expiry_date"] == near_expiry]
                far_group = group[group["expiry_date"] == far_expiry]

                if near_group.empty or far_group.empty:
                    continue

                near_days = near_group["days_to_expiry"].iloc[0]
                far_days = far_group["days_to_expiry"].iloc[0]

                # Only consider pairs 60-120 days apart (quarterly spreads)
                time_diff = far_days - near_days
                if not (25 <= time_diff <= 40):
                    continue

                # Filter to near-ATM options (0.90-1.10 moneyness)
                near_atm = near_group[
                    (near_group["strikePrice"] / spot >= 0.95)
                    & (near_group["strikePrice"] / spot <= 1.05)
                ]
                far_atm = far_group[
                    (far_group["strikePrice"] / spot >= 0.95)
                    & (far_group["strikePrice"] / spot <= 1.05)
                ]

                if near_atm.empty or far_atm.empty:
                    continue

                # Calculate average IV for near expiry (ATM calls and puts)
                near_ivs = []
                for _, opt in near_atm.iterrows():
                    T = near_days / 365.0
                    # Use mid-price if bid/ask available, otherwise use last price
                    if (
                        "bid" in opt
                        and "ask" in opt
                        and pd.notna(opt["bid"])
                        and pd.notna(opt["ask"])
                    ):
                        price = (opt["bid"] + opt["ask"]) / 2
                    else:
                        price = opt.get(
                            "lastPrice", opt.get("price")
                        )  # Fallback

                    iv = self.calculate_iv_from_price(
                        price, spot, opt["strikePrice"], T, r, opt["option_type"]
                    )
                    if pd.notna(iv) and 0.01 < iv < 3.0:
                        near_ivs.append(iv)

                # Calculate average IV for far expiry
                far_ivs = []
                for _, opt in far_atm.iterrows():
                    T = far_days / 365.0
                    if (
                        "bid" in opt
                        and "ask" in opt
                        and pd.notna(opt["bid"])
                        and pd.notna(opt["ask"])
                    ):
                        price = (opt["bid"] + opt["ask"]) / 2
                    else:
                        price = opt.get("lastPrice", opt.get("strikePrice") * 0.05)

                    iv = self.calculate_iv_from_price(
                        price, spot, opt["strikePrice"], T, r, opt["option_type"]
                    )
                    if pd.notna(iv) and 0.01 < iv < 3.0:
                        far_ivs.append(iv)

                if not near_ivs or not far_ivs:
                    continue

                near_iv = np.mean(near_ivs)
                far_iv = np.mean(far_ivs)

                # Calculate forward volatility
                T1 = near_days / 365.0
                T2 = far_days / 365.0

                var_forward = (T2 * far_iv**2 - T1 * near_iv**2) / (T2 - T1)

                if var_forward > 0:
                    forward_vol = np.sqrt(var_forward)

                    # Find the ATM strike (closest to spot price)
                    near_atm_strikes = near_atm["strikePrice"].values
                    atm_strike = near_atm_strikes[
                        np.argmin(np.abs(near_atm_strikes - spot))
                    ]

                    forward_vols.append(
                        {
                            "underlying_isin": underlying_isin,
                            "underlying_name": near_group["underlying_name"].iloc[0],
                            "spot_price": spot,
                            "atm_strike": atm_strike,
                            "near_expiry": near_expiry,
                            "far_expiry": far_expiry,
                            "near_days": near_days,
                            "far_days": far_days,
                            "near_iv": near_iv,
                            "far_iv": far_iv,
                            "forward_vol": forward_vol,
                            "near_iv_count": len(near_ivs),
                            "far_iv_count": len(far_ivs),
                        }
                    )

        df_fv = pd.DataFrame(forward_vols)
        print(f"✓ Calculated {len(df_fv)} forward volatility spreads")

        if not df_fv.empty:
            # Debug: print distribution of FV ratios
            df_fv["fwd_vol_ratio"] = df_fv["forward_vol"] / df_fv["near_iv"]
            print(f"\n📈 FV Ratio Distribution:")
            print(f"  Min: {df_fv['fwd_vol_ratio'].min():.3f}")
            print(f"  25%: {df_fv['fwd_vol_ratio'].quantile(0.25):.3f}")
            print(f"  50%: {df_fv['fwd_vol_ratio'].median():.3f}")
            print(f"  75%: {df_fv['fwd_vol_ratio'].quantile(0.75):.3f}")
            print(f"  Max: {df_fv['fwd_vol_ratio'].max():.3f}")
            print(f"  Count > 1.1: {(df_fv['fwd_vol_ratio'] > 1.1).sum()}")
            print(f"  Count > 1.2: {(df_fv['fwd_vol_ratio'] > 1.2).sum()}")
            print(f"  Count > 1.3: {(df_fv['fwd_vol_ratio'] > 1.3).sum()}\n")

        return df_fv

    def scan(self) -> Dict[str, Any]:
        """Main scanning logic."""
        try:
            print("🔍 Starting Forward Volatility scan...")

            # Get DAX ISINs
            dax_isins = list(self.get_isin_to_ticker_mapping().keys())
            print(f"📋 Scanning {len(dax_isins)} DAX stocks")

            # Fetch spot prices from Yahoo Finance
            spot_prices = self.fetch_spot_prices(dax_isins)

            # Fetch option contracts
            df_options = self.fetch_option_contracts(dax_isins)

            if df_options.empty:
                print("❌ No option data retrieved")
                return {
                    "success": False,
                    "error": "No option data retrieved",
                    "recommendations": [],
                    "metrics": {"error": "No option data retrieved"},
                    "timestamp": datetime.now(timezone.utc),
                }

            print(f"✓ Retrieved {len(df_options)} option contracts")

            # Calculate forward volatilities
            df_forward = self.calculate_forward_vols(df_options, spot_prices)

            if df_forward.empty:
                print("❌ No forward volatility pairs calculated")
                return {
                    "success": False,
                    "error": "No forward volatility pairs calculated",
                    "recommendations": [],
                    "metrics": {"error": "No forward volatility pairs calculated"},
                    "timestamp": datetime.now(timezone.utc),
                }

            # Score opportunities
            df_scored = self.score_opportunities(df_forward)

            if df_scored.empty:
                print("❌ No opportunities found with FV ratio > 1.1")
                return {
                    "success": True,
                    "embed": {
                        "title": "Forward Volatility Scanner Results",
                        "color": 3066993,
                        "fields": [
                            {
                                "name": "No Opportunities",
                                "value": f"No calendar spreads found with FV ratio > 1.1\nScanned {len(dax_isins)} stocks, analyzed {len(df_forward)} pairs",
                                "inline": False,
                            }
                        ],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    "recommendations": [],
                    "metrics": {"total_pairs": len(df_forward), "filtered_count": 0},
                    "timestamp": datetime.now(timezone.utc),
                }

            # Save results
            df_scored.to_feather(self.data_file)
            print(f"💾 Saved {len(df_scored)} opportunities to {self.data_file}")

            # Create embed fields from opportunities
            fields = []
            
            # Summary field
            fields.append({
                "name": "📊 Summary",
                "value": f"Found {len(df_scored)} calendar spread opportunities\nScanned {len(dax_isins)} stocks, analyzed {len(df_forward)} pairs",
                "inline": False,
            })
            
            # Individual opportunity fields (top 5)
            for _, row in df_scored.head(5).iterrows():
                near_expiry_str = row['near_expiry'].strftime('%Y-%m-%d') if hasattr(row['near_expiry'], 'strftime') else str(row['near_expiry'])
                far_expiry_str = row['far_expiry'].strftime('%Y-%m-%d') if hasattr(row['far_expiry'], 'strftime') else str(row['far_expiry'])
                
                field_value = (
                    f"{row['grade']} FV Ratio: {row['fwd_vol_ratio']:.2f}x\n"
                    f"Spot: €{row['spot_price']:.2f}\n"
                    f"📤 SELL {near_expiry_str} €{row['atm_strike']:.0f} ({row['near_days']}d)\n"
                    f"📥 BUY  {far_expiry_str} €{row['atm_strike']:.0f} ({row['far_days']}d)\n"
                    f"Near IV: {row['near_iv']*100:.1f}% | Far IV: {row['far_iv']*100:.1f}%\n"
                    f"Forward Vol: {row['forward_vol']*100:.1f}%"
                )
                
                fields.append({
                    "name": f"📈 {row['underlying_name']}",
                    "value": field_value,
                    "inline": False,
                })
            
            if len(df_scored) > 5:
                fields.append({
                    "name": "➕ More Opportunities",
                    "value": f"...and {len(df_scored) - 5} more opportunities in the results file",
                    "inline": False,
                })

            # Create embed structure for bot
            embed = {
                "title": "Forward Volatility Opportunities",
                "color": 3066993,
                "fields": fields,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Convert to list of dicts for recommendations
            recommendations = df_scored.to_dict("records")

            return {
                "success": True,
                "embed": embed,
                "recommendations": recommendations,
                "metrics": {
                    "total_opportunities": len(df_scored),
                    "total_pairs_calculated": len(df_forward),
                    "stocks_scanned": len(dax_isins),
                    "contracts_fetched": len(df_options),
                },
                "timestamp": datetime.now(timezone.utc),
            }

        except Exception as e:
            print(f"❌ Error in scan: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "recommendations": [],
                "timestamp": datetime.now(timezone.utc),
            }

    def format_message(self, df: pd.DataFrame) -> str:
        """Format the results into a message."""
        if df.empty:
            return "No forward volatility opportunities found with FV ratio > 1.1"

        message = f"*Forward Volatility Opportunities* (FV Ratio > 1.1)\n\n"
        message += f"Found {len(df)} calendar spread opportunities:\n\n"

        for _, row in df.head(10).iterrows():
            # Format expiry dates as readable strings
            near_expiry_str = row['near_expiry'].strftime('%Y-%m-%d') if hasattr(row['near_expiry'], 'strftime') else str(row['near_expiry'])
            far_expiry_str = row['far_expiry'].strftime('%Y-%m-%d') if hasattr(row['far_expiry'], 'strftime') else str(row['far_expiry'])
            
            message += f"{row['grade']} *{row['underlying_name']}*\n"
            message += f"  Spot: €{row['spot_price']:.2f}\n"
            message += f"  💰 Calendar Spread:\n"
            message += f"    • SELL {near_expiry_str} €{row['atm_strike']:.0f} Call/Put ({row['near_days']}d)\n"
            message += f"    • BUY  {far_expiry_str} €{row['atm_strike']:.0f} Call/Put ({row['far_days']}d)\n"
            message += f"  FV Ratio: {row['fwd_vol_ratio']:.2f}x\n"
            message += f"  Near IV: {row['near_iv']*100:.1f}% | Far IV: {row['far_iv']*100:.1f}%\n"
            message += f"  Forward Vol: {row['forward_vol']*100:.1f}%\n"
            message += f"  Score: {row['score']:.0f}\n\n"

        if len(df) > 10:
            message += f"...and {len(df) - 10} more opportunities\n"

        return message


if __name__ == "__main__":
    scanner = ForwardVolatilityScanner()
    results = scanner.scan()

    # Extract DataFrame from results for formatting
    if results["recommendations"]:
        df_results = pd.DataFrame(results["recommendations"])
        print("\n" + scanner.format_message(df_results))
    else:
        print("\nNo opportunities found")
        print(f"Metrics: {results.get('metrics', {})}")
