from playwright.sync_api import sync_playwright, TimeoutError
import pandas as pd
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_page_load(page, url, max_retries=3):
    """Safely load a page with retries"""
    for retry in range(max_retries):
        try:
            page.goto(url, timeout=60000)
            page.wait_for_selector('table', state='visible', timeout=60000)
            time.sleep(3)
            return True
        except TimeoutError:
            if retry == max_retries - 1:
                logger.error(f"Failed to load page after {max_retries} attempts")
                return False
            logger.warning(f"Attempt {retry + 1} failed, retrying...")
            time.sleep(5)


def extract_port_data(row):
    """Extract data from a table row"""
    try:
        cells = row.query_selector_all('td')
        if len(cells) > 0:
            img = cells[0].query_selector('img')
            port_type = cells[2].inner_text()

            # Filter only for actual ports (exclude marinas)
            if img and 'GR.png' in img.get_attribute('src') and port_type == 'Port':
                port_link = cells[1].query_selector('a')
                port_url = port_link.get_attribute('href') if port_link else None

                return {
                    'name': cells[1].inner_text(),
                    'type': port_type,
                    'size': cells[3].inner_text(),
                    'url': port_url
                }
    except Exception as e:
        logger.error(f"Error extracting port data: {e}")
    return None


def discover_ports():
    """Main function to discover Greek ports"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        greek_ports_data = []
        page_num = 1
        total_pages = 144

        try:
            while page_num <= total_pages:
                current_url = f'https://www.myshiptracking.com/ports?sort=ID&page={page_num}'
                logger.info(f"Processing page {page_num}/{total_pages}")

                if not safe_page_load(page, current_url):
                    continue

                rows = page.query_selector_all('table tbody tr')
                logger.info(f"Found {len(rows)} entries on page {page_num}")

                # Process current page
                page_data = []
                for row in rows:
                    port_data = extract_port_data(row)
                    if port_data:
                        page_data.append(port_data)
                        greek_ports_data.append(port_data)

                # Save after each page
                if page_data:
                    df_all = pd.DataFrame(greek_ports_data)
                    save_data(df_all)
                    logger.info(f"Saved {len(greek_ports_data)} total ports after page {page_num}")

                page_num += 1

        except Exception as e:
            logger.error(f"Error during port discovery: {e}")

        finally:
            browser.close()

        return greek_ports_data


def save_data(df, filename='ports_catalog.csv'):
    """Save DataFrame to CSV with proper error handling"""
    # Debug prints
    current_file = Path(__file__).resolve()
    print(f"\nCurrent file: {current_file}")

    for i, parent in enumerate(current_file.parents):
        print(f"Level {i}: {parent}")

    # Get project root and construct path
    project_root = current_file.parents[4]  # port_delays_monitor
    save_path = project_root / 'src' / 'data_collection' / 'ports' / filename

    print(f"\nFinal save path: {save_path}")
    """Save DataFrame to CSV with proper error handling"""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[4]
    save_path = project_root / "src" / "data_collection" / "ports" / "ports_catalog.csv"

    logger.info(f"Attempting to save to: {save_path}")

    try:
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if df.empty:
            logger.warning("No data to save!")
            return False

        df.to_csv(save_path, index=False)
        logger.info(f"Successfully saved {len(df)} records")
        return True

    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        return False


if __name__ == "__main__":
    discover_ports()
