"""Selenium-based Market Chameleon scraping for historical win-rate data."""

import logging
import re
import threading
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .config import get_logger
from .models import WinRateData

logger = get_logger("browser")


class MarketChameleonBrowser:
    """Lazy-initialised, thread-safe headless Chrome for Market Chameleon."""

    _MAX_RETRIES = 3

    def __init__(self) -> None:
        self._driver = None
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    # -- internal ---------------------------------------------------------

    def _init_driver(self) -> None:
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass

        opts = webdriver.ChromeOptions()
        for flag in (
            "--headless", "--no-sandbox", "--disable-dev-shm-usage",
            "--disable-gpu", "--disable-extensions", "--disable-infobars",
            "--blink-settings=imagesEnabled=false", "--disable-3d-apis",
            "--mute-audio", "--no-first-run", "--no-default-browser-check",
            "--disable-translate", "--disable-plugins",
            "--disable-software-rasterizer", "--window-size=1920,1080",
        ):
            opts.add_argument(flag)

        service = Service(ChromeDriverManager().install())
        self._driver = webdriver.Chrome(service=service, options=opts)
        self._driver.set_page_load_timeout(10)

    # -- public API -------------------------------------------------------

    def get_win_rate(self, ticker: str) -> WinRateData:
        """
        Return historical win-rate for *ticker*.

        Defaults to ``WinRateData()`` (zeroed) on any failure.
        """
        default = WinRateData()

        with self._lock:
            if self._driver is None:
                try:
                    self._init_driver()
                except Exception as exc:
                    logger.error(f"Browser init failed: {exc}")
                    return default

            for attempt in range(1, self._MAX_RETRIES + 1):
                try:
                    # Health check
                    try:
                        self._driver.window_handles
                    except Exception:
                        self._init_driver()

                    self._driver.get(
                        f"https://marketchameleon.com/Overview/{ticker}/Earnings/Earnings-Charts/"
                    )
                    wait = WebDriverWait(self._driver, 8)
                    section = wait.until(
                        EC.presence_of_element_located(
                            (By.CLASS_NAME, "symbol-section-header-descr")
                        )
                    )

                    win_rate, quarters = 0.0, 0
                    for span in section.find_elements(By.TAG_NAME, "span"):
                        if "overestimated" not in span.text:
                            continue
                        try:
                            strong = span.find_element(By.TAG_NAME, "strong")
                            win_rate = float(strong.text.strip("%"))
                            m = re.search(r"in the last (\d+) quarters", span.text)
                            if m:
                                quarters = int(m.group(1))
                        except Exception as inner:
                            logger.debug(f"Parse error for {ticker}: {inner}")
                        break

                    return WinRateData(win_rate=win_rate, quarters=quarters)

                except Exception as exc:
                    logger.warning(f"MC scrape attempt {attempt}/{self._MAX_RETRIES} for {ticker}: {exc}")
                    try:
                        self._init_driver()
                    except Exception:
                        pass
                    time.sleep(1)

        logger.error(f"MC data failed for {ticker} after {self._MAX_RETRIES} attempts")
        return default
