"""Main port scraper implementation"""

from datetime import datetime
import pandas as pd
from playwright.sync_api import sync_playwright
from src.scrapers.base_scraper import BaseScraper
from src.scrapers.utils.vessel_types import VESSEL_TYPES, BASE_URLS
import os
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)

class ScrapingError(Exception):
    """Custom exception for scraping errors"""
    pass

class PortScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        self.vessel_types = VESSEL_TYPES

    @staticmethod
    def safe_extract_cell(cell, error_msg):
        """Safely extract content from a table cell"""
        try:
            if not cell:
                raise ScrapingError("Cell is None")
            content = cell.text_content().strip()
            if not content:
                logger.warning(f"{error_msg}: Empty content")
                return None
            return content
        except Exception as e:
            logger.error(f"{error_msg}: {str(e)}")
            return None

    # Base methods
    def scrape(self):
        """Main scraping method"""
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            current_vessels = self.scrape_current_vessels(page)
            expected_arrivals = self.scrape_expected_arrivals(page)
            port_calls = self.scrape_port_calls(page)

            # Save data without timestamp parameter
            if current_vessels:
                self.save_data(current_vessels, 'current_vessels')
                print(f"Saved {len(current_vessels)} current vessel records")

            if expected_arrivals:
                self.save_expected_arrivals(expected_arrivals)
                print(f"Saved {len(expected_arrivals)} expected arrival records")

            if port_calls:
                self.save_port_calls(port_calls)
                print(f"Saved {len(port_calls)} port call records")

            return True

        except Exception as e:
            print(f"Error during scraping: {e}")
            return False

        finally:
            browser.close()
            playwright.stop()

    @staticmethod
    def get_vessel_type(img_src):
        """Get vessel type from image source"""
        if not img_src:
            return None

        # Map icon numbers to vessel types
        icon_types = {
            'icon3': 'passenger',
            'icon4': 'high_speed',
            'icon6': 'tanker',
            'icon7': 'cargo',
            'icon8': 'tug',
            'icon9': 'pilot'
        }

        for icon_id, vessel_type in icon_types.items():
            if icon_id in img_src:
                return vessel_type

        return None

    # In Port current vessels methods
    def scrape_current_vessels(self, page):
        """Scrape vessels currently in port"""
        vessels = []
        page_count = 1

        try:
            print("\nStarting current vessels scraping...")

            while True:
                # Construct URL for current page
                current_url = f"{BASE_URLS['in_port']}&page={page_count}" if page_count > 1 else BASE_URLS['in_port']
                print(f"\nProcessing page {page_count}...")
                print(f"URL: {current_url}")

                # Navigate to page
                page.goto(current_url)
                page.wait_for_timeout(3000)

                # Get total number of rows in table
                table = page.query_selector('table')
                if table:
                    rows = table.query_selector_all('tr')
                    total_rows = len(rows) - 1  # Excluding header
                    print(f"Found {total_rows} rows in table (excluding header)")

                    if total_rows == 0:  # If no rows found, we've reached the end
                        print("No rows found, ending scraping")
                        break

                    for row in rows[1:]:  # Skip header
                        vessel_data = self.extract_vessel_data(row)
                        if vessel_data:
                            vessels.append(vessel_data)

                    # Move to next page
                    page_count += 1

                    # If we've processed both pages, stop
                    if page_count > 2:
                        print("Reached last page")
                        break
                else:
                    print("No table found on page!")
                    break

            print(f"\nScraping completed:")
            print(f"Total pages processed: {page_count - 1}")
            print(f"Total vessels extracted: {len(vessels)}")
            return vessels

        except Exception as e:
            print(f"Error in scrape_current_vessels: {e}")
            return vessels

    def extract_vessel_data(self, row):
        """Extract vessel data from table row"""
        try:
            cells = row.query_selector_all('td')
            if len(cells) < 6:
                logger.warning("Row has insufficient cells")
                return None

            # Extract vessel type from image
            img = cells[0].query_selector('img')
            vessel_type = self.get_vessel_type(img.get_attribute('src')) if img else None
            if not vessel_type:
                logger.warning("Could not determine vessel type")
                return None

            data = {
                'vessel_name': self.safe_extract_cell(cells[0], "Error extracting vessel name"),
                'vessel_type': vessel_type,
                'arrival': self.safe_extract_cell(cells[1], "Error extracting arrival time"),
                'dwt': self.safe_extract_cell(cells[2], "Error extracting DWT"),
                'grt': self.safe_extract_cell(cells[3], "Error extracting GRT"),
                'built': self.safe_extract_cell(cells[4], "Error extracting built year"),
                'size': self.safe_extract_cell(cells[5], "Error extracting size"),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Validate essential data
            if not data['vessel_name']:
                logger.warning("Vessel name is missing")
                return None

            logger.debug(f"Successfully extracted vessel data: {data['vessel_name']}")
            return data

        except Exception as e:
            logger.error(f"Error extracting vessel data: {str(e)}")
            return None

    @staticmethod
    def save_data(data, prefix):
        """Save current vessels data to CSV file"""
        if data:
            df = pd.DataFrame(data)

            # Create directory path
            directory = f'src/data_collection/raw_data/Piraeus'
            os.makedirs(directory, exist_ok=True)

            # Single file for each type
            filepath = f'{directory}/{prefix}.csv'

            if os.path.exists(filepath):
                # Read existing data
                existing_df = pd.read_csv(filepath)
                # Update with new data
                df = pd.concat([existing_df, df]).drop_duplicates(subset=['vessel_name'])

            # Sort by timestamp
            df = df.sort_values('timestamp', ascending=False)

            # Keep only the most recent records (e.g., last 1000)
            df = df.head(1000)

            # Save updated data
            df.to_csv(filepath, index=False)
            print(f"Updated {prefix} file with {len(df)} records")

    # Expected arrivals methods
    def scrape_expected_arrivals(self, page):
        """Scrape expected vessel arrivals"""
        arrivals = []
        page_count = 1

        try:
            print("\nStarting expected arrivals scraping...")

            while True:
                # Construct URL for current page
                current_url = f"{BASE_URLS['expected']}&page={page_count}" if page_count > 1 else BASE_URLS['expected']
                print(f"\nProcessing page {page_count}...")
                print(f"URL: {current_url}")

                # Navigate to page
                page.goto(current_url)
                page.wait_for_timeout(3000)

                # Get total number of rows in table
                table = page.query_selector('table')
                if table:
                    rows = table.query_selector_all('tr')
                    total_rows = len(rows) - 1  # Excluding header
                    print(f"Found {total_rows} rows in table (excluding header)")

                    if total_rows == 0:  # If no rows found, we've reached the end
                        print("No rows found, ending scraping")
                        break

                    for row in rows[1:]:  # Skip header
                        arrival_data = self.extract_expected_arrival_data(row)
                        if arrival_data:
                            arrivals.append(arrival_data)

                    # Move to next page
                    page_count += 1

                    # Check if there's a next page by looking for rows
                    if total_rows < 10:  # Assuming each page has 10 rows when full
                        print("Last page reached (incomplete page)")
                        break
                else:
                    print("No table found on page!")
                    break

            print(f"\nExpected arrivals scraping completed:")
            print(f"Total pages processed: {page_count - 1}")
            print(f"Total expected arrivals extracted: {len(arrivals)}")
            return arrivals

        except Exception as e:
            print(f"Error in scrape_expected_arrivals: {e}")
            return arrivals

    def extract_expected_arrival_data(self, row):
        """Extract expected arrival data from table row"""
        try:
            cells = row.query_selector_all('td')
            if len(cells) < 3:
                logger.warning("Row has insufficient cells for expected arrival data")
                return None

            # Extract vessel type from image
            img = cells[1].query_selector('img')
            vessel_type = self.get_vessel_type(img.get_attribute('src')) if img else None
            if not vessel_type:
                logger.warning("Could not determine vessel type for expected arrival")
                return None

            # Extract data using safe method
            mmsi = self.safe_extract_cell(cells[0], "Error extracting MMSI")
            vessel_name = self.safe_extract_cell(cells[1], "Error extracting vessel name")
            eta = self.safe_extract_cell(cells[2], "Error extracting ETA")

            # Validate essential data
            if not all([mmsi, vessel_name, eta]):
                logger.warning("Missing essential data for expected arrival")
                return None

            data = {
                'mmsi': mmsi,
                'vessel_name': vessel_name,
                'vessel_type': vessel_type,
                'eta': eta,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.debug(f"Successfully extracted expected arrival data: {data['vessel_name']}")
            return data

        except Exception as e:
            logger.error(f"Error extracting expected arrival data: {str(e)}")
            return None

    @staticmethod
    def save_expected_arrivals(data):
        """Save expected arrivals data"""
        if data:
            df = pd.DataFrame(data)

            directory = f'src/data_collection/raw_data/Piraeus'
            os.makedirs(directory, exist_ok=True)

            # Single file
            filepath = f'{directory}/expected_arrivals.csv'

            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                df = pd.concat([existing_df, df]).drop_duplicates(subset=['mmsi', 'vessel_name'])

            # Sort by timestamp
            df = df.sort_values('timestamp', ascending=False)

            # Keep most recent records
            df = df.head(1000)

            df.to_csv(filepath, index=False)
            print(f"Updated expected arrivals file with {len(df)} records")

    def scrape_port_calls(self, page):
        """Scrape port activity (arrivals/departures)"""
        calls = []
        page_count = 1
        total_processed = 0
        expected_columns = 5  #

        try:
            logger.info("\nStarting port calls scraping...")

            while True:
                # Construct URL for current page
                current_url = f"{BASE_URLS['port_calls']}&page={page_count}" if page_count > 1 else BASE_URLS[
                    'port_calls']
                logger.info(f"\nProcessing page {page_count}...")
                logger.debug(f"URL: {current_url}")

                # Safely navigate to page
                if not self.safe_navigate(page, current_url):
                    logger.error(f"Failed to navigate to page {page_count}")
                    break

                # Validate table structure
                table = self.validate_table(page, expected_columns)
                if not table:
                    logger.error(f"Invalid table structure on page {page_count}")
                    break

                # Get rows and process
                rows = table.query_selector_all('tr')
                total_rows = len(rows) - 1  # Excluding header
                logger.info(f"Found {total_rows} rows in table (excluding header)")

                if total_rows == 0:
                    logger.info("No rows found, ending scraping")
                    break

                # Process rows
                successful_extractions = 0
                for row in rows[1:]:  # Skip header
                    call_data = self.extract_port_call_data(row)
                    if call_data:
                        calls.append(call_data)
                        successful_extractions += 1

                # Update progress
                total_processed += total_rows
                self.track_progress(successful_extractions, total_rows, "Port calls")

                # Move to next page
                page_count += 1

                # Check if there's a next page
                if total_rows < 50:  # Assuming each page has 50 rows when full
                    logger.info("Last page reached (incomplete page)")
                    break

                # Add small delay between pages
                page.wait_for_timeout(2000)  # 2'' second delay

            # Final statistics
            logger.info(f"\nPort calls scraping completed:")
            logger.info(f"Total pages processed: {page_count - 1}")
            logger.info(f"Total rows processed: {total_processed}")
            logger.info(f"Successfully extracted calls: {len(calls)}")
            logger.info(f"Success rate: {(len(calls) / total_processed * 100):.2f}%")

            return calls

        except Exception as e:
            logger.error(f"Error in scrape_port_calls: {str(e)}")
            return calls

        finally:
            # Save partial results even if something fails
            if calls:
                try:
                    self.save_port_calls(calls)
                    logger.info(f"Saved {len(calls)} port call records")
                except Exception as e:
                    logger.error(f"Failed to save port calls: {str(e)}")
            return calls

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    def safe_navigate(self, page, url):
        """Safely navigate to URL with retry mechanism"""
        try:
            page.goto(url)
            page.wait_for_selector('table', timeout=10000)
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {str(e)}")
            return False

    @staticmethod
    def validate_table(page, expected_columns):
        """Validate table structure before scraping"""
        try:
            table = page.query_selector('table')
            if not table:
                logger.error("Table not found")
                return None

            headers = table.query_selector_all('th')
            if not headers or len(headers) < expected_columns:
                logger.error(
                    f"Invalid table structure. Expected {expected_columns} columns, found {len(headers) if headers else 0}")
                return None

            return table
        except Exception as e:
            logger.error(f"Table validation failed: {str(e)}")
            return None

    @staticmethod
    def track_progress(extracted_count, total_count, data_type):
        """Track scraping progress"""
        progress = (extracted_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"{data_type} scraping progress: {progress:.2f}% ({extracted_count}/{total_count})")

    def extract_port_call_data(self, row):
        """Extract port call data from table row"""
        try:
            cells = row.query_selector_all('td')
            if len(cells) < 5:
                logger.warning("Row has insufficient cells for port call data")
                return None

            # Extract vessel type from image
            vessel_cell = cells[4]
            img = vessel_cell.query_selector('img')
            vessel_type = self.get_vessel_type(img.get_attribute('src')) if img else None
            if not vessel_type:
                logger.warning("Could not determine vessel type for port call")
                return None

            # Extract event type
            event_type = self.safe_extract_cell(cells[1], "Error extracting event type")
            if event_type:
                event_type = 'arrival' if event_type == 'Άφιξη' else 'departure' if event_type == 'Αναχώρηση' else None

            if not event_type:
                logger.warning("Invalid event type")
                return None

            # Extract vessel name (either from link or direct text)
            vessel_name = None
            vessel_link = vessel_cell.query_selector('a')
            if vessel_link:
                vessel_name = self.safe_extract_cell(vessel_link, "Error extracting vessel name from link")
            if not vessel_name:
                vessel_name = self.safe_extract_cell(vessel_cell, "Error extracting vessel name from cell")

            # Extract timestamp
            timestamp = self.safe_extract_cell(cells[2], "Error extracting timestamp")

            # Validate essential data
            if not all([vessel_name, timestamp]):
                logger.warning("Missing essential data for port call")
                return None

            data = {
                'event_type': event_type,
                'timestamp': timestamp,
                'vessel_name': vessel_name,
                'vessel_type': vessel_type,
                'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.debug(f"Successfully extracted port call data: {data['vessel_name']} - {data['event_type']}")
            return data

        except Exception as e:
            logger.error(f"Error extracting port call data: {str(e)}")
            return None

    @staticmethod
    def save_port_calls(data):
        """Save port calls data"""
        if data:
            df = pd.DataFrame(data)

            directory = 'src/data_collection/raw_data/Piraeus'
            os.makedirs(directory, exist_ok=True)
            filepath = f'{directory}/port_calls.csv'

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(filepath):
                        existing_df = pd.read_csv(filepath)
                        df = pd.concat([existing_df, df]).drop_duplicates(
                            subset=['timestamp', 'vessel_name', 'event_type']
                        )

                    df = df.sort_values('timestamp', ascending=False)
                    df = df.head(1000)

                    df.to_csv(filepath, index=False)
                    print(f"Updated port calls file with {len(df)} records")
                    break

                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"File is locked, retrying... (attempt {attempt + 1})")
                        time.sleep(1)  # Wait a second before retrying
                    else:
                        print("Could not save port calls data - file is locked")
                        raise
