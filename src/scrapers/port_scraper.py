"""Main port scraper implementation"""

from datetime import datetime
import pandas as pd
from playwright.sync_api import sync_playwright
from .base_scraper import BaseScraper
from .utils.vessel_types import VESSEL_TYPES, BASE_URLS
import os
import time

class PortScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        self.vessel_types = VESSEL_TYPES

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
            'icon7': 'cargo'
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
                return None

            img = cells[0].query_selector('img')
            vessel_type = self.get_vessel_type(img.get_attribute('src')) if img else None

            if not vessel_type:
                return None

            data = {
                'vessel_name': cells[0].text_content().strip(),
                'vessel_type': vessel_type,
                'arrival': cells[1].text_content().strip(),
                'dwt': cells[2].text_content().strip(),
                'grt': cells[3].text_content().strip(),
                'built': cells[4].text_content().strip(),
                'size': cells[5].text_content().strip(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return data

        except Exception as e:
            print(f"Error extracting data: {e}")
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
                return None

            img = cells[1].query_selector('img')
            vessel_type = self.get_vessel_type(img.get_attribute('src')) if img else None

            if not vessel_type:
                return None

            data = {
                'mmsi': cells[0].text_content().strip(),
                'vessel_name': cells[1].text_content().strip(),
                'vessel_type': vessel_type,
                'eta': cells[2].text_content().strip(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return data

        except Exception as e:
            print(f"Error extracting expected arrival data: {e}")
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

    # Port calls methods
    def scrape_port_calls(self, page):
        """Scrape port activity (arrivals/departures)"""
        calls = []
        page_count = 1

        try:
            print("\nStarting port calls scraping...")

            while True:
                # Construct URL for current page
                current_url = f"{BASE_URLS['port_calls']}&page={page_count}" if page_count > 1 else BASE_URLS[
                    'port_calls']
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
                        call_data = self.extract_port_call_data(row)
                        if call_data:
                            calls.append(call_data)

                    # Move to next page
                    page_count += 1

                    # Check if there's a next page by looking for rows
                    if total_rows < 50:  # Assuming each page has 50 rows when full
                        print("Last page reached (incomplete page)")
                        break
                else:
                    print("No table found on page!")
                    break

            print(f"\nPort calls scraping completed:")
            print(f"Total pages processed: {page_count - 1}")
            print(f"Total port calls extracted: {len(calls)}")
            return calls

        except Exception as e:
            print(f"Error in scrape_port_calls: {e}")
            return calls

    def extract_port_call_data(self, row):
        """Extract port call data from table row"""
        try:
            cells = row.query_selector_all('td')
            if len(cells) < 5:
                return None

            vessel_cell = cells[4]
            img = vessel_cell.query_selector('img')
            vessel_type = self.get_vessel_type(img.get_attribute('src')) if img else None

            if not vessel_type:
                return None

            event_type = cells[1].text_content().strip()
            if event_type == 'Άφιξη':
                event_type = 'arrival'
            elif event_type == 'Αναχώρηση':
                event_type = 'departure'

            data = {
                'event_type': event_type,
                'timestamp': cells[2].text_content().strip(),
                'vessel_name': vessel_cell.query_selector('a').text_content().strip() if vessel_cell.query_selector(
                    'a') else vessel_cell.text_content().strip(),
                'vessel_type': vessel_type,
                'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return data

        except Exception as e:
            print(f"Error extracting port call data: {e}")
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