"""Base scraper class with common functionality"""

from playwright.sync_api import sync_playwright
import logging
from abc import ABC, abstractmethod

class BaseScraper(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_browser(self):
        """Initialize browser"""
        return sync_playwright().start()

    @abstractmethod
    def scrape(self):
        """Main scraping method to be implemented by child classes"""
        pass

    def handle_pagination(self, page, next_button_selector):
        """Handle pagination if it exists"""
        try:
            next_button = page.query_selector(next_button_selector)
            if next_button and next_button.is_visible():
                next_button.click()
                page.wait_for_timeout(3000)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error handling pagination: {e}")
            return False
