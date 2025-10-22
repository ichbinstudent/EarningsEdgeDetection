"""
Core options analysis functionality.
Handles volatility calculations and options chain analysis.
"""

import logging
import warnings
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import interp1d

from scipy.stats import norm
from scipy.optimize import brentq


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class OptionsAnalyzer:
    def __init__(self):
        self.warnings_shown = False
    
    def filter_dates(self, dates: List[str]) -> List[str]:
        """Filter option expiration dates to those 45+ days out."""
        today = datetime.today().date()
        cutoff_date = today + timedelta(days=45)
        sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() 
                            for date in dates)
        
        arr = []
        for i, date in enumerate(sorted_dates):
            if date >= cutoff_date:
                arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
                break
        
        if arr:
            if arr[0] == today.strftime("%Y-%m-%d") and len(arr) > 1:
                return arr[1:]
            return arr
        return [x.strftime("%Y-%m-%d") for x in sorted_dates]

    def yang_zhang_volatility(self, price_data: pd.DataFrame, 
                            window: int = 30,
                            trading_periods: int = 252,
                            return_last_only: bool = True) -> float:
        """Calculate Yang-Zhang volatility."""
        try:
            log_ho = np.log(price_data['High'] / price_data['Open'])
            log_lo = np.log(price_data['Low'] / price_data['Open'])
            log_co = np.log(price_data['Close'] / price_data['Open'])
            log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
            log_oc_sq = log_oc**2
            log_cc = np.log(price_data['Close'] / price_data['Close'].shift(1))
            log_cc_sq = log_cc**2
            
            rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
            close_vol = log_cc_sq.rolling(window=window).sum() / (window - 1.0)
            open_vol = log_oc_sq.rolling(window=window).sum() / (window - 1.0)
            window_rs = rs.rolling(window=window).sum() / (window - 1.0)
            
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            result = np.sqrt(open_vol + k * close_vol + (1 - k) * window_rs) * np.sqrt(trading_periods)
            
            if return_last_only:
                return result.iloc[-1]
            return result.dropna()
            
        except Exception as e:
            if not self.warnings_shown:
                warnings.warn(f"Error in volatility calculation: {str(e)}. Using simple volatility.")
                self.warnings_shown = True
            return self.calculate_simple_volatility(price_data, window, trading_periods, return_last_only)

    def calculate_simple_volatility(self, price_data: pd.DataFrame,
                                  window: int = 30,
                                  trading_periods: int = 252,
                                  return_last_only: bool = True) -> float:
        """Calculate simple volatility as fallback method."""
        try:
            returns = price_data['Close'].pct_change().dropna()
            vol = returns.rolling(window=window).std() * np.sqrt(trading_periods)
            if return_last_only:
                return vol.iloc[-1]
            return vol
        except Exception as e:
            warnings.warn(f"Error in simple volatility calculation: {str(e)}")
            return np.nan

    def build_term_structure(self, days: List[int], ivs: List[float]) -> callable:
        """Build IV term structure using linear interpolation."""
        try:
            days_arr = np.array(days)
            ivs_arr = np.array(ivs)
            sort_idx = days_arr.argsort()
            days_arr = days_arr[sort_idx]
            ivs_arr = ivs_arr[sort_idx]
            
            spline = interp1d(days_arr, ivs_arr, kind='linear', fill_value="extrapolate")
            
            def term_spline(dte: float) -> float:
                if dte < days_arr[0]:
                    return float(ivs_arr[0])
                elif dte > days_arr[-1]:
                    return float(ivs_arr[-1])
                else:
                    return float(spline(dte))
            
            return term_spline
        except Exception as e:
            warnings.warn(f"Error in term structure calculation: {str(e)}")
            return lambda x: np.nan

    def compute_recommendation(self, ticker: str, earnings_date: Optional[date] = None) -> Dict:
        """Analyze options and compute trading recommendation."""
        try:
            ticker = ticker.strip().upper()
            if not ticker:
                return {"error": "No symbol provided."}

            stock = yf.Ticker(ticker)
            if not stock.options:
                return {"error": f"No options for {ticker}."}

            exp_dates = self.filter_dates(list(stock.options))
            options_chains = {date: stock.option_chain(date) for date in exp_dates}

            # Get current price
            hist = stock.history(period='1d')
            if hist.empty:
                return {"error": "No price data available"}
            current_price = hist['Close'].iloc[-1]

            # Calculate ATM IV for each expiration
            atm_ivs = {}
            bid_ivs = {}
            ask_ivs = {}
            straddle = None
            first_chain = True
            atm_call_delta = None
            atm_put_delta = None
            today = datetime.today().date()

            # Calculate historical volatility
            hist_data = stock.history(period='3mo')
            hist_vol = self.yang_zhang_volatility(hist_data)
            hist_vol_1y = self.yang_zhang_volatility(stock.history(period='1y'))

            for exp_date, chain in options_chains.items():
                calls, puts = chain.calls, chain.puts
                if calls.empty or puts.empty:
                    continue

                call_idx = (calls['strike'] - current_price).abs().idxmin()
                put_idx = (puts['strike'] - current_price).abs().idxmin()

                call_iv = calls.loc[call_idx, 'impliedVolatility']
                put_iv = puts.loc[put_idx, 'impliedVolatility']
                atm_iv = (call_iv + put_iv) / 2.0
                atm_ivs[exp_date] = atm_iv

                # Calculate bid and ask IVs using price adjustments
                # Bid IV: more conservative (higher IV for same premium)
                # Ask IV: more aggressive (lower IV for same premium)
                call_bid = calls.loc[call_idx, 'bid']
                call_ask = calls.loc[call_idx, 'ask']
                put_bid = puts.loc[put_idx, 'bid']
                put_ask = puts.loc[put_idx, 'ask']

                if call_bid > 0 and call_ask > 0 and put_bid > 0 and put_ask > 0:
                    # Adjust IV based on bid-ask spread (rough approximation)
                    bid_iv = implied_volatility(
                        market_price=call_bid,
                        S=current_price,
                        K=calls.loc[call_idx, 'strike'],
                        T=(datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days / 365,
                        r=0.04,
                        option_type='call'
                    )
                    ask_iv = implied_volatility(
                        market_price=call_ask,
                        S=current_price,
                        K=calls.loc[call_idx, 'strike'],
                        T=(datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days / 365,
                        r=0.04,
                        option_type='call'
                    )
                else:
                    # Fallback to mid IV if bid/ask data unavailable
                    raise Exception("Bid/ask data unavailable")
                    bid_iv = atm_iv
                    ask_iv = atm_iv

                bid_ivs[exp_date] = bid_iv
                ask_ivs[exp_date] = ask_iv

                if first_chain:
                    # Calculate straddle price for first expiration
                    call_mid = (calls.loc[call_idx, 'bid'] + calls.loc[call_idx, 'ask']) / 2
                    put_mid = (puts.loc[put_idx, 'bid'] + puts.loc[put_idx, 'ask']) / 2
                    straddle = call_mid + put_mid

                    # Get ATM deltas if available
                    atm_call_delta = calls.loc[call_idx, 'delta'] if 'delta' in calls.columns else None
                    atm_put_delta = puts.loc[put_idx, 'delta'] if 'delta' in puts.columns else None

                    first_chain = False

            if not atm_ivs:
                return {"error": "Could not calculate ATM IVs"}

            # Build term structures for mid, bid, and ask IVs
            dtes = [(datetime.strptime(exp, "%Y-%m-%d").date() - today).days for exp in atm_ivs.keys()]
            ivs_mid = list(atm_ivs.values())
            ivs_bid = list(bid_ivs.values())
            ivs_ask = list(ask_ivs.values())

            term_spline_mid = self.build_term_structure(dtes, ivs_mid)
            iv30 = term_spline_mid(45)
            slope = (term_spline_mid(45) - term_spline_mid(min(dtes))) / (45 - min(dtes))

            # Find the ATM IV for the expiry closest to 365 days (1 year)
            target_days = 365
            # Calculate days to earnings date for short leg
            if earnings_date:
                today = datetime.today().date()
                # Find the option expiration that is closest to (but not before) the earnings date
                exp_date_objects = [datetime.strptime(exp, "%Y-%m-%d").date() for exp in exp_dates]
                # Filter to expirations that are at or after the earnings date
                valid_expirations = [exp_date for exp_date in exp_date_objects if exp_date >= earnings_date]
                if valid_expirations:
                    # Find the expiration with the least days from today (among those >= earnings_date)
                    short_leg_days = min((exp_date - today).days for exp_date in valid_expirations)
                    # Ensure we have at least 1 day
                    short_leg_days = max(1, short_leg_days)
                else:
                    # Fallback: use the original logic if no valid expirations found
                    short_leg_days = (earnings_date - today).days
                    short_leg_days = max(1, min(short_leg_days, 35))
            else:
                short_leg_days = 4  # fallback to hardcoded value

            if dtes:
                # idx_1y = min(range(len(dtes)), key=lambda i: abs(dtes[i] - target_days))
                sigma_baseline_mid = min(ivs_mid) # TODO: ivs_mid[idx_1y]

                T_short = short_leg_days

                idx_long = min(range(len(dtes)), key=lambda i: abs(dtes[i] - 30))
                T_long = dtes[idx_long]
                sigma_long_leg_ask = ivs_ask[idx_long]
                
                sigma_short_leg_fair = np.sqrt((sigma_long_leg_ask**2 * T_long - sigma_baseline_mid**2 * (T_long - T_short)) / T_short)

                # Actual IV of short leg for mid, bid, and ask
                idx_short = min(range(len(dtes)), key=lambda i: abs(dtes[i] - short_leg_days))
                sigma_short_leg_bid = ivs_bid[idx_short]

                actual_to_fair_ratio =  ((sigma_short_leg_bid / sigma_short_leg_fair) - 1) * 100 if sigma_short_leg_fair else np.nan
            else:
                sigma_baseline_mid = None
                sigma_short_leg_fair = None
                sigma_short_leg_bid = None
                actual_to_fair_ratio = None

            # Get volume data
            avg_volume = hist_data['Volume'].rolling(30).mean().dropna().iloc[-1]

            # Prepare result
            result_dict = {
                'avg_volume': avg_volume >= 1_500_000,
                'iv30_rv30': iv30 / hist_vol if hist_vol > 0 else 9999,
                'term_slope': slope,
                'term_structure_valid': slope <= -0.004,
                'term_structure_tier2': -0.006 < slope <= -0.004,
                'expected_move': f"{(straddle/current_price*100):.2f}%" if straddle else "N/A",
                'current_price': current_price,
                'ticker': ticker,
                'recommendation': 'BUY' if iv30 < hist_vol and avg_volume >= 1_500_000 else 'SELL' if iv30 > hist_vol * 1.2 else 'HOLD',
            }
            
            # Add sigma values if they are valid (using mid values for backward compatibility)
            if sigma_baseline_mid is not None:
                result_dict['sigma_baseline_1y'] = sigma_baseline_mid
            if sigma_short_leg_fair is not None and not np.isnan(sigma_short_leg_fair):
                result_dict['sigma_short_leg_fair'] = sigma_short_leg_fair
            if sigma_short_leg_bid is not None:
                result_dict['sigma_short_leg'] = sigma_short_leg_bid
            if actual_to_fair_ratio is not None:
                result_dict['actual_to_fair_ratio'] = actual_to_fair_ratio

            # Add ATM deltas if available
            if atm_call_delta is not None and atm_put_delta is not None:
                result_dict['atm_call_delta'] = atm_call_delta
                result_dict['atm_put_delta'] = atm_put_delta

            return result_dict
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            return {
                "error": f"Failed to compute recommendation: {str(e)}",
                "ticker": ticker if 'ticker' in locals() else "UNKNOWN",
                "status": "ERROR"
            }


def black_scholes_price(S, K, T, r, sigma, option_type='call', q=0):
    """
    Calculates the price of a European option using the Black-Scholes-Merton model.

    Parameters:
    S (float): Current stock price
    K (float): Option strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annual)
    sigma (float): Volatility (annual)
    option_type (str): 'call' or 'put'
    q (float): Annual dividend yield (default=0)

    Returns:
    float: The theoretical price of the option
    """
    
    # Ensure T and sigma are positive
    if T <= 0 or sigma <= 0:
        # Return intrinsic value if expired
        if T == 0:
            if option_type == 'call':
                return max(0.0, S - K)
            else:
                return max(0.0, K - S)
        return np.nan # Invalid input

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

    return price

def implied_volatility(market_price, S, K, T, r, option_type='call', q=0, tol=1e-6, vol_min=1e-5, vol_max=5.0):
    """
    Calculates the implied volatility (IV) of an option using the brentq root-finder.

    This function finds the volatility 'sigma' that makes the Black-Scholes
    price equal to the 'market_price'.

    Parameters:
    market_price (float): The observed market price of the option
    S (float): Current stock price
    K (float): Option strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annual)
    option_type (str): 'call' or 'put'
    q (float): Annual dividend yield (default=0)
    tol (float): Tolerance for the root-finding algorithm
    vol_min (float): Minimum volatility guess
    vol_max (float): Maximum volatility guess

    Returns:
    float: The implied volatility, or np.nan if no solution is found
    """

    # 1. Define the objective function for the root finder
    # We need to find 'sigma' where: BSM_price(sigma) - market_price = 0
    def objective_function(sigma):
        try:
            return black_scholes_price(S, K, T, r, sigma, option_type, q) - market_price
        except (ValueError, ZeroDivisionError):
            # Return a large value if BSM fails (e.g., sigma=0)
            return 1e10 

    # 2. Check for arbitrage / price bounds
    # The price cannot be below its intrinsic value
    if option_type == 'call':
        intrinsic_value = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    else: # put
        intrinsic_value = max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))

    if market_price < intrinsic_value - tol:
        # Price is below theoretical minimum
        return np.nan 

    # 3. Check if market price is achievable in the vol range
    # BSM price increases with volatility (positive vega)
    try:
        min_price = black_scholes_price(S, K, T, r, vol_min, option_type, q)
        max_price = black_scholes_price(S, K, T, r, vol_max, option_type, q)
    except Exception:
        return np.nan # Failed to price at bounds

    if market_price < min_price - tol or market_price > max_price + tol:
        # Price is outside the range of prices for vol_min to vol_max
        return np.nan

    # 4. Use brentq to find the root (the implied volatility)
    try:
        iv = brentq(
            objective_function, 
            vol_min,  # Lower bound
            vol_max,  # Upper bound
            xtol=tol,
            rtol=tol
        )
        return iv
    except ValueError:
        # This catches cases where brentq fails to converge
        # (e.g., f(a) and f(b) have the same sign, which our checks
        # should prevent, but it's a good safeguard).
        return np.nan

