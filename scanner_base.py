"""
Base scanner interface for trading signal scanners.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import datetime


class BaseScanner(ABC):
    """
    Abstract base class for all trading signal scanners.
    Each scanner should implement the scan method and define its schedule.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def scan(self) -> Dict[str, Any]:
        """
        Perform the scan and return results.

        Returns:
            Dict containing scan results with keys like:
            - 'recommendations': List of recommended trades
            - 'near_misses': List of near miss trades
            - 'metrics': Dict of additional metrics
            - 'timestamp': When the scan was performed
        """
        pass

    @property
    @abstractmethod
    def schedule(self) -> str:
        """
        Return the cron schedule string for when this scanner should run.
        For example: "0 16 * * 1-5" for weekdays at 16:00 (4 PM)
        """
        pass

    def get_schedule_description(self) -> str:
        """
        Return a human-readable description of the schedule.
        """
        return f"Scanner '{self.name}' runs on schedule: {self.schedule}"