"""
Earnings Calendar Scanner that inherits from BaseScanner.
"""

from scanner_base import BaseScanner
from earningsCalendarCall.core.scanner import EarningsScanner
from datetime import datetime, timezone
from typing import Dict, List, Any


class EarningsCalendarScanner(BaseScanner):
    """
    Scanner for earnings-based options opportunities.
    Runs daily at 16:15 EST (weekdays).
    """

    def __init__(self):
        super().__init__("Earnings Calendar")
        self.scanner = EarningsScanner()

    @property
    def schedule(self) -> str:
        """
        Run daily at 16:15 EST (4:15 PM Eastern) on weekdays (Monday-Friday).
        Cron format: minute hour day month day-of-week
        """
        return "30 21 * * 1-5"

    def scan(self) -> Dict[str, Any]:
        """
        Perform the earnings scan and return formatted results.
        """
        try:
            # Run the scan
            recommended, near_misses, stock_metrics = self.scanner.scan_earnings(
                workers=8,
                use_finnhub=True
            )

            # Format results similar to the original
            fields = []
            recommended = sorted(
                [t for t in recommended if stock_metrics[t].get('tier') == 1 or stock_metrics[t].get('tier') == 2],
                key=lambda x: stock_metrics[x].get('actual_to_fair_ratio', 0),
                reverse=True
            )

            # Tier 1 details
            if recommended:
                for tick in recommended:
                    m = stock_metrics[tick]
                    name = f"{tick}"
                    value_lines = [
                        f"• Price: `${m['price']:.2f}`",
                        f"• Volume: `{m['volume']:,.0f}`",
                        f"• Winrate: `{m['win_rate']:.1f}%` over last `{m['win_quarters']}` earnings",
                        f"• IV/RV Ratio: `{m['iv_rv_ratio']:.2f}`",
                        f"• Term Structure: `{m['term_structure']:.3f}`",
                        f"• Tier: `{m.get('tier')}`",
                        f'• 1Y ATM IV (Baseline): {m["sigma_baseline_1y"]:.4f}',
                        f'• Fair IV (Short Leg): {m["sigma_short_leg_fair"]:.4f}',
                        f'• Actual IV (Short Leg): {m["sigma_short_leg"]:.4f}',
                        f'• Actual to Fair Ratio: {m["actual_to_fair_ratio"]:.2f}%',
                    ]
                    fields.append({'name': name, 'value': "\n".join(value_lines), 'inline': False})

            # Near misses
            if near_misses:
                for tick, reason in near_misses:
                    m = stock_metrics[tick]
                    name = f"Near Miss — {tick}"
                    value = (
                        f"• Failed: `{reason}`\n"
                        f"• Price: `${m['price']:.2f}`\n"
                        f"• Volume: `{m['volume']:,.0f}`\n"
                        f"• Winrate: `{m['win_rate']:.1f}%` over last `{m['win_quarters']}` earnings\n"
                        f"• IV/RV Ratio: `{m['iv_rv_ratio']:.2f}`\n"
                        f"• Term Structure: `{m['term_structure']:.3f}`\n",
                        f'• 1Y ATM IV (Baseline): {m["sigma_baseline_1y"]:.4f}\n',
                        f'• Fair IV (Short Leg): {m["sigma_short_leg_fair"]:.4f}\n',
                        f'• Actual IV (Short Leg): {m["sigma_short_leg"]:.4f}\n',
                        f'• Actual to Fair Ratio: {m["actual_to_fair_ratio"]:.2f}%',
                    )
                    fields.append({'name': name, 'value': value, 'inline': False})

            # Fallback if nothing to show
            if not fields:
                fields.append({'name': 'No recommendations', 'value': 'None found', 'inline': False})

            embed = {
                'title': 'Earnings Scanner Results',
                'color': 3066993,
                'fields': fields,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            return {
                'success': True,
                'embed': embed,
                'recommendations': recommended,
                'near_misses': near_misses,
                'metrics': stock_metrics,
                'timestamp': datetime.now(timezone.utc)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc)
            }