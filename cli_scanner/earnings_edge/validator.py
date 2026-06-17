"""Stock validation: apply the tiered filter chain and assign tiers."""

import logging
from datetime import date, datetime
from typing import Optional

import yfinance as yf

from .analyzer import OptionsAnalyzer
from .browser import MarketChameleonBrowser
from .config import (
    get_logger, session,
    MIN_PRICE, MIN_EXPECTED_MOVE, MAX_ATM_DELTA,
    TERM_STRUCTURE_HARD_LIMIT, DEFAULT_IV_RV_PASS, DEFAULT_IV_RV_NEAR_MISS,
)
from .models import EarningsCandidate, ValidationResult, ValidationMetrics

logger = get_logger("validator")


def _fail(reason: str) -> ValidationResult:
    """Convenience constructor for failed validation."""
    return ValidationResult(passed=False, tier=0, near_miss=False, reason=reason)


class StockValidator:
    """Apply the multi-tier filter chain to an earnings candidate."""

    def __init__(self, analyzer: OptionsAnalyzer, browser: MarketChameleonBrowser) -> None:
        self.analyzer = analyzer
        self.browser = browser
        self.iv_rv_pass = DEFAULT_IV_RV_PASS
        self.iv_rv_near_miss = DEFAULT_IV_RV_NEAR_MISS

    # -- SPY threshold adjustment ----------------------------------------

    def adjust_spy_thresholds(self) -> None:
        """Relax IV/RV thresholds when overall market vol is low."""
        spy = self.analyzer.compute_recommendation("SPY")
        if not spy.ok:
            logger.warning(f"SPY analysis failed: {spy.error}")
            return

        ratio = spy.iv30_rv30
        logger.info(f"SPY IV/RV: {ratio:.2f}")

        if ratio <= 0.75:
            self.iv_rv_pass, self.iv_rv_near_miss = 0.90, 0.65
        elif ratio <= 0.85:
            self.iv_rv_pass, self.iv_rv_near_miss = 1.00, 0.75
        elif ratio <= 1.0:
            self.iv_rv_pass, self.iv_rv_near_miss = 1.10, 0.85
        else:
            self.iv_rv_pass, self.iv_rv_near_miss = DEFAULT_IV_RV_PASS, DEFAULT_IV_RV_NEAR_MISS

        logger.info(f"IV/RV thresholds → pass={self.iv_rv_pass}, near_miss={self.iv_rv_near_miss}")

    # -- main validation --------------------------------------------------

    def validate(self, candidate: EarningsCandidate) -> ValidationResult:
        ticker = candidate.ticker
        m = ValidationMetrics()
        failed: list[str] = []
        near_miss: list[str] = []

        try:
            yt = yf.Ticker(ticker, session=session)

            # 1. Price (fastest)
            hist = yt.history(period="1d")
            if hist.empty:
                return _fail("No price data")
            price = hist["Close"].iloc[-1]
            m.price = price
            if price < MIN_PRICE:
                return _fail(f"Price ${price:.2f} < ${MIN_PRICE:.0f}")

            # 2. Options availability
            options_dates = yt.options
            if not options_dates:
                return _fail("No options available")

            # 3. Expiration proximity
            first_exp = datetime.strptime(options_dates[0], "%Y-%m-%d").date()
            days_to_exp = (first_exp - datetime.now().date()).days
            if days_to_exp > 9:
                return _fail(f"Expiry too far: {days_to_exp} days")
            m.days_to_expiry = days_to_exp

            # 4. Open interest
            chain = yt.option_chain(options_dates[0])
            total_oi = int(chain.calls["openInterest"].sum() + chain.puts["openInterest"].sum())
            if total_oi < 2000:
                return _fail(f"OI {total_oi} < 2000")
            m.open_interest = total_oi

            # 5. Core analysis (term structure, IVs, etc.)
            analysis = self.analyzer.compute_recommendation(ticker, candidate.earnings_date)
            if not analysis.ok:
                return _fail(f"Analysis error: {analysis.error}")

            # Carry analysis fields into metrics (direct assignment, no dict copy)
            m.sigma_baseline_1y = analysis.sigma_baseline_1y
            m.sigma_short_leg = analysis.sigma_short_leg
            m.sigma_short_leg_fair = analysis.sigma_short_leg_fair
            m.actual_to_fair_ratio = analysis.actual_to_fair_ratio
            m.atm_call_delta = analysis.atm_call_delta
            m.atm_put_delta = analysis.atm_put_delta

            # 6. Term structure (hard gate)
            m.term_structure = analysis.term_slope
            if analysis.term_slope > TERM_STRUCTURE_HARD_LIMIT:
                return _fail(f"Term structure {analysis.term_slope:.4f} > {TERM_STRUCTURE_HARD_LIMIT}")

            # 7. ATM delta gate
            cd = analysis.atm_call_delta
            pd = analysis.atm_put_delta
            if cd is not None and pd is not None:
                try:
                    cd_f, pd_f = float(cd), float(pd)
                    if cd_f > MAX_ATM_DELTA or abs(pd_f) > MAX_ATM_DELTA:
                        return _fail(f"Delta > {MAX_ATM_DELTA} (C:{cd_f:.2f} P:{pd_f:.2f})")
                except (TypeError, ValueError):
                    pass

            # 8. Expected move ≥ $0.90
            raw = analysis.expected_move
            if raw != "N/A":
                em_fail = self._check_expected_move(raw, price, m, yt, options_dates)
                if em_fail:
                    return em_fail

            # 9. Volume
            vol = yt.history(period="1mo")["Volume"].mean()
            m.volume = vol
            if vol < 1_000_000:
                failed.append(f"Volume {vol:,.0f} < 1M")
            elif vol < 1_500_000:
                near_miss.append(f"Volume {vol:,.0f} < 1.5M")

            # 10. Market Chameleon win rate
            if not failed:
                wr_data = self.browser.get_win_rate(ticker)
                m.win_rate = wr_data.win_rate
                m.win_quarters = wr_data.quarters
                # If scraper returned no data (e.g. no Chrome), skip the
                # win-rate gate entirely rather than failing on 0%.
                if wr_data.quarters > 0:
                    if wr_data.win_rate < 40.0:
                        failed.append(f"Winrate {wr_data.win_rate}% < 40% ({wr_data.quarters} earnings)")
                    elif wr_data.win_rate < 50.0:
                        near_miss.append(f"Winrate {wr_data.win_rate}% < 50% ({wr_data.quarters} earnings)")
            else:
                m.win_rate = 0.0
                m.win_quarters = 0

            # 11. IV/RV ratio
            iv_rv = analysis.iv30_rv30
            m.iv_rv_ratio = iv_rv
            if iv_rv < self.iv_rv_near_miss:
                failed.append(f"IV/RV {iv_rv:.2f} < {self.iv_rv_near_miss}")
            elif iv_rv < self.iv_rv_pass:
                near_miss.append(f"IV/RV {iv_rv:.2f} < {self.iv_rv_pass}")

            # ---- Tier assignment ----
            is_pass: bool = not failed and not near_miss
            is_tier2: bool = not failed and bool(near_miss) and analysis.term_slope <= -0.006
            is_near: bool = not failed and bool(near_miss) and not is_tier2

            m.tier = 1 if is_pass else 2 if is_tier2 else 0

            reason = (
                " | ".join(failed) if failed
                else " | ".join(near_miss) if near_miss
                else "Tier 1 Trade" if is_pass
                else "Tier 2 Trade" if is_tier2
                else "Near Miss"
            )
            return ValidationResult(
                passed=bool(is_pass or is_tier2),
                tier=m.tier,
                near_miss=is_near,
                reason=reason,
                metrics=m,
            )

        except Exception as exc:
            logger.warning(f"Validation error for {ticker}: {exc}")
            return ValidationResult(passed=False, tier=0, near_miss=False, reason=f"Validation error: {exc}")

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _check_expected_move(raw, price, m: ValidationMetrics, yt, options_dates) -> Optional[ValidationResult]:
        """Return a fail-result if expected move < $0.90, else None."""
        try:
            pct = float(str(raw).strip("%")) / 100 if isinstance(raw, str) else float(raw) / 100
        except (ValueError, TypeError):
            # Fallback: straddle from option chain
            try:
                ch = yt.option_chain(options_dates[0])
                ci = (ch.calls["strike"] - price).abs().idxmin()
                pi = (ch.puts["strike"] - price).abs().idxmin()
                pct = ((ch.calls.loc[ci, "bid"] + ch.calls.loc[ci, "ask"]) / 2 +
                       (ch.puts.loc[pi, "bid"] + ch.puts.loc[pi, "ask"]) / 2) / price
            except Exception:
                return None

        dollars = price * pct
        m.expected_move_dollars = dollars
        m.expected_move_pct = pct * 100
        if dollars < MIN_EXPECTED_MOVE:
            return _fail(f"Expected move ${dollars:.2f} < ${MIN_EXPECTED_MOVE}")
        return None
