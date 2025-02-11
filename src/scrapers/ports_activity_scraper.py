from playwright.async_api import async_playwright
from datetime import datetime, UTC
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional
from src.scrapers.utils.vessel_types import get_vessel_type
import asyncio

from src.database.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortsActivityScraper:
    def __init__(self):
        self.db = DatabaseManager()
        self.ports_df = self._load_ports_data()
        self.context = None

    @staticmethod
    def _load_ports_data():
        """Load enriched ports data"""
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]
        ports_path = project_root / 'src' / 'data_collection' / 'ports' / 'ports_enriched.csv'
        return pd.read_csv(ports_path)

    @staticmethod
    def _construct_urls(port_id: str) -> Dict[str, str]:
        """Construct URLs for a given port ID"""
        return {
            'in_port': f"https://www.myshiptracking.com/el/inport?pid={port_id}",
            'expected': f"https://www.myshiptracking.com/el/estimate?pid={port_id}",
            'port_calls': f"https://www.myshiptracking.com/el/ports-arrivals-departures/?pid={port_id}"
        }

    async def scrape_port_activity(self, page, port_name: str, port_id: str):
        """Scrape current activity for a specific port"""
        try:
            # Get URLs for this port
            urls = self._construct_urls(port_id)
            timestamp = datetime.now(UTC)

            # Scrape in_port vessels
            logger.info(f"Scraping in-port vessels for {port_name}")
            in_port_vessels = await self._scrape_in_port(page, urls['in_port'])
            if in_port_vessels:
                self.db.save_port_data(
                    port_id=port_id,
                    port_name=port_name,
                    data_type="in_port",
                    vessels=in_port_vessels,
                    timestamp=timestamp
                )

            # Scrape expected arrivals
            logger.info(f"Scraping expected arrivals for {port_name}")
            expected_vessels = await self._scrape_expected(page, urls['expected'])
            if expected_vessels:
                self.db.save_port_data(
                    port_id=port_id,
                    port_name=port_name,
                    data_type="expected",
                    vessels=expected_vessels,
                    timestamp=timestamp
                )

            # Scrape port calls
            logger.info(f"Scraping port calls for {port_name}")
            port_calls = await self._scrape_port_calls(page, urls['port_calls'])
            if port_calls:
                self.db.save_port_data(
                    port_id=port_id,
                    port_name=port_name,
                    data_type="port_calls",
                    vessels=port_calls,
                    timestamp=timestamp
                )

        except Exception as e:
            logger.error(f"Error scraping {port_name}: {e}")

    @staticmethod
    async def _extract_port_call(row) -> Optional[Dict]:
        """Extract vessel data from port calls table row"""
        try:
            cells = await row.query_selector_all('td')
            if len(cells) < 5:
                logger.warning(f"Insufficient cells in row: found {len(cells)}, expected 5")
                return None

            # Get event type from icon
            icon_html = await cells[0].inner_html()
            event_type = None
            if 'fa-flag-checkered' in icon_html:
                event_type = 'Arrival'
            elif 'fa-sign-out' in icon_html:
                event_type = 'Departure'

            if not event_type:
                logger.debug(f"Could not determine event type from icon HTML: {icon_html}")
                return None

            # Extract other fields
            try:
                timestamp = await cells[2].inner_text()
                port = await cells[3].inner_text()
                vessel_name = await cells[4].inner_text()
                vessel_type = await get_vessel_type(await cells[4].query_selector('img'))
            except Exception as e:
                logger.debug(f"Error extracting basic fields: {e}")
                return None

            # Get vessel identifiers if available
            vessel_link = await cells[4].query_selector('a')
            imo, mmsi = await PortsActivityScraper._extract_vessel_identifiers(vessel_link)

            # Validate essential data
            if not all([event_type, timestamp, vessel_name]):
                logger.debug("Missing essential data")
                return None

            vessel_data = {
                'event_type': event_type,
                'timestamp': timestamp.strip(),
                'vessel_name': vessel_name.strip(),
                'vessel_type': vessel_type,
                'port': port.strip(),
                'imo': imo,
                'mmsi': mmsi
            }

            return vessel_data

        except Exception as e:
            logger.error(f"Error extracting port call data: {e}")
            return None

    @staticmethod
    async def _extract_vessel_identifiers(vessel_link) -> tuple[Optional[str], Optional[str]]:
        """Extract IMO and MMSI from vessel link"""
        imo = None
        mmsi = None

        try:
            if vessel_link:
                href = await vessel_link.get_attribute('href')
                if href:
                    # Extract IMO if exists
                    if 'imo-' in href:
                        try:
                            imo = href.split('imo-')[1].split()[0]
                            imo = 'NaN' if imo == '0' else imo
                        except (IndexError, AttributeError):
                            logger.debug("Could not extract IMO from href")

                    # Extract MMSI if exists
                    if 'mmsi-' in href:
                        try:
                            mmsi = href.split('mmsi-')[1].split('-')[0]
                            mmsi = 'NaN' if mmsi == '0' else mmsi
                        except (IndexError, AttributeError):
                            logger.debug("Could not extract MMSI from href")
        except Exception as e:
            logger.debug(f"Error extracting vessel identifiers: {e}")

        return imo or 'NaN', mmsi or 'NaN'

    @staticmethod
    async def _scrape_with_pagination(page, base_url: str, extract_func) -> List[Dict]:
        """Generic pagination scraping function"""
        vessels = []
        page_num = 1

        while True:
            try:
                url = f"{base_url}&page={page_num}" if page_num > 1 else base_url
                logger.debug(f"Scraping page {page_num}: {url}")

                await page.goto(url, timeout=60000)
                await page.wait_for_selector('table', timeout=60000)
                await asyncio.sleep(2)

                rows = await page.query_selector_all('table tbody tr')
                if not rows:
                    break

                page_vessels = [await extract_func(row) for row in rows]
                vessels.extend([v for v in page_vessels if v])

                if len(rows) < 50:
                    break

                page_num += 1

            except Exception as e:
                logger.error(f"Error on page {page_num}: {e}")
                break

        return vessels

    @staticmethod
    async def _extract_in_port_vessel(row) -> Optional[Dict]:
        """Extract vessel data from in-port table row"""
        def clean_measurement(value: str) -> tuple[str, str]:
            if value in ["---", "", "0"]:
                return "NaN", ""
            if "Τόνοι" in value:
                clean_value = value.replace("Τόνοι", "").replace(".", "").strip()
                return clean_value if clean_value != "0" else "NaN", "t"
            elif "μ" in value:
                clean_value = value.replace("μ", "").replace(".", "").strip()
                return clean_value if clean_value != "0" else "NaN", "m"
            return value if value != "0" else "NaN", ""

        try:
            cells = await row.query_selector_all('td')
            if not cells:
                logger.debug("Empty row found")
                return None

            vessel_cell = cells[0]
            vessel_name = await vessel_cell.inner_text()
            vessel_name = vessel_name.strip()
            vessel_img = await vessel_cell.query_selector('img')
            vessel_type = await get_vessel_type(vessel_img)

            if not vessel_name or vessel_name == "---":
                logger.debug("No vessel name found")
                return None

            dwt_raw = await cells[2].inner_text() if len(cells) > 2 else "---"
            grt_raw = await cells[3].inner_text() if len(cells) > 3 else "---"
            built_raw = await cells[4].inner_text() if len(cells) > 4 else "---"
            size_raw = await cells[5].inner_text() if len(cells) > 5 else "---"

            dwt_value, _ = clean_measurement(dwt_raw.strip())
            grt_value, _ = clean_measurement(grt_raw.strip())
            size_value, _ = clean_measurement(size_raw.strip())

            vessel_data = {
                'vessel_name': vessel_name,
                'vessel_type': vessel_type,
                'arrival': (await cells[1].inner_text()).strip() if len(cells) > 1 else "NaN",
                'dwt_t': dwt_value,
                'grt_t': grt_value,
                'built': "NaN" if built_raw in ["---", "", "0"] else built_raw,
                'size_m': size_value,
                'imo': None,
                'mmsi': None
            }

            try:
                vessel_link = await cells[0].query_selector('a')
                if vessel_link:
                    imo, mmsi = await PortsActivityScraper._extract_vessel_identifiers(vessel_link)
                    vessel_data['imo'] = imo
                    vessel_data['mmsi'] = mmsi
                else:
                    vessel_data['imo'] = "NaN"
                    vessel_data['mmsi'] = "NaN"
            except Exception as e:
                logger.debug(f"Could not extract vessel identifiers: {e}")
                vessel_data['imo'] = "NaN"
                vessel_data['mmsi'] = "NaN"

            return vessel_data

        except Exception as e:
            logger.error(f"Error processing row: {e}")
            return None

    @staticmethod
    async def _extract_expected_vessel(row) -> Optional[Dict]:
        """Extract vessel data from expected arrivals table row"""
        try:
            cells = await row.query_selector_all('td')
            if len(cells) >= 4:
                vessel_link = await cells[1].query_selector('a')
                imo, mmsi = await PortsActivityScraper._extract_vessel_identifiers(vessel_link)

                img = await cells[1].query_selector('img')
                vessel_type = await get_vessel_type(img) if img else 'unknown'

                return {
                    'mmsi': (await cells[0].inner_text()).strip(),
                    'vessel_name': (await cells[1].inner_text()).strip(),
                    'vessel_type': vessel_type,
                    'port': (await cells[2].inner_text()).strip(),
                    'eta': (await cells[3].inner_text()).strip(),
                    'imo': imo,
                    'mmsi_from_link': mmsi
                }
        except Exception as e:
            logger.error(f"Error extracting expected vessel: {e}")
        return None

    async def _scrape_in_port(self, page, url: str) -> List[Dict]:
        """Scrape vessels currently in port"""
        return await self._scrape_with_pagination(page, url, self._extract_in_port_vessel)

    async def _scrape_expected(self, page, url: str) -> List[Dict]:
        """Scrape expected arrivals"""
        return await self._scrape_with_pagination(page, url, self._extract_expected_vessel)

    async def _scrape_port_calls(self, page, url: str) -> List[Dict]:
        """Scrape recent port calls"""
        return await self._scrape_with_pagination(page, url, self._extract_port_call)

    async def run_scraping_cycle(self):
        """Run one complete scraping cycle for all ports"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                for _, row in self.ports_df.iterrows():
                    logger.info(f"Scraping activity for {row['name']}")
                    await self.scrape_port_activity(page, row['name'], row['port_id'])
                    await asyncio.sleep(2)

            finally:
                await browser.close()


if __name__ == "__main__":
    scraper = PortsActivityScraper()
    asyncio.run(scraper.run_scraping_cycle())
