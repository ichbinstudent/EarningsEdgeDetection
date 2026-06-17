"""Discord webhook helper."""

import logging
from typing import Any, Dict, Union

import requests

from .config import get_logger

logger = get_logger("webhook")


def send_webhook(url: str, message: Union[str, Dict[str, Any]], log: logging.Logger) -> None:
    """Send a string (code-block) or embed dict to a Discord webhook."""
    try:
        if isinstance(message, str):
            payload = {"content": f"```\n{message}\n```"}
        else:
            payload = {"embeds": [message]}

        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if resp.status_code >= 400:
            log.error(f"Webhook failed ({resp.status_code}): {resp.text}")
    except Exception as exc:
        log.error(f"Webhook error: {exc}")
