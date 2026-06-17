"""Monkey-patch yfinance cookie handling for robustness."""

from requests.cookies import create_cookie
import yfinance.data as _data


def _wrap_cookie(cookie, session):
    """Convert a plain cookie-name string into a real Cookie object."""
    if isinstance(cookie, str):
        value = session.cookies.get(cookie)
        return create_cookie(name=cookie, value=value)
    return cookie


def apply() -> None:
    """Patch ``YfData._get_cookie_basic`` to always return a proper Cookie."""
    original = _data.YfData._get_cookie_basic

    def _patched(self, timeout=30):
        cookie = original(self, timeout)
        return _wrap_cookie(cookie, self._session)

    _data.YfData._get_cookie_basic = _patched
