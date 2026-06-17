from __future__ import annotations

from datetime import date
from typing import Iterable


def select_calendar_expiries(
    expiries: Iterable[date],
    *,
    min_far_days: int = 21,
    target_far_days: int = 28,
    max_far_days: int = 35,
) -> tuple[date, date]:
    """Pick the near leg plus the preferred farther-dated calendar expiry.

    Prefer a far leg roughly one month beyond the near expiry so weekly options
    between the legs do not collapse the intended calendar width. If no expiry is
    available in the preferred window, fall back to the next listed expiry.
    """

    ordered = sorted(set(expiries))
    if len(ordered) < 2:
        raise ValueError("Need at least two expiries for a calendar spread")

    near_expiry = ordered[0]
    preferred_far = [
        expiry
        for expiry in ordered[1:]
        if min_far_days <= (expiry - near_expiry).days <= max_far_days
    ]
    if preferred_far:
        far_expiry = min(preferred_far, key=lambda expiry: abs((expiry - near_expiry).days - target_far_days))
    else:
        far_expiry = ordered[1]
    return near_expiry, far_expiry
