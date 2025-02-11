from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_ports_catalog():
    """Read the ports catalog CSV"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[4]
    catalog_path = project_root / 'src' / 'data_collection' / 'ports' / 'ports_catalog.csv'
    return pd.read_csv(catalog_path)


def extract_port_details(page, port_url):
    """Extract detailed information from port page"""
    try:
        # Extract port ID from URL (e.g., "id-528" -> "528")
        port_id = port_url.split('id-')[1]

        full_url = f"https://www.myshiptracking.com{port_url}"
        page.goto(full_url, timeout=60000)
        page.wait_for_selector('table', timeout=60000)
        time.sleep(2)

        details = {
            'port_id': port_id,
            'un_locode': None,
            'longitude': None,
            'latitude': None,
            'url_in_port': f"https://www.myshiptracking.com/el/inport?pid={port_id}",
            'url_expected': f"https://www.myshiptracking.com/el/estimate?pid={port_id}",
            'url_port_calls': f"https://www.myshiptracking.com/el/ports-arrivals-departures/?pid={port_id}"
        }

        # Extract coordinates and UN/LOCODE from table
        rows = page.query_selector_all('table tr')
        for row in rows:
            text = row.inner_text()
            if 'UN/LOCODE' in text:
                details['un_locode'] = text.split('\t')[1].strip()
            elif 'Longitude' in text:
                details['longitude'] = text.split('\t')[1].strip().replace('°', '')
            elif 'Latitude' in text:
                details['latitude'] = text.split('\t')[1].strip().replace('°', '')

        return details

    except Exception as e:
        logger.error(f"Error extracting details from {port_url}: {e}")
        return None


def save_enriched_data(df, filename='ports_enriched.csv'):
    """Save enriched data"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[4]
    save_path = project_root / 'src' / 'data_collection' / 'ports' / filename

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Successfully saved enriched data with {len(df)} records")
        return True
    except Exception as e:
        logger.error(f"Failed to save enriched data: {e}")
        return False


def enrich_port_data():
    """Main function to enrich port data"""
    # Read existing ports
    df = read_ports_catalog()
    logger.info(f"Read {len(df)} ports from catalog")

    enriched_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        total_ports = len(df)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            logger.info(f"Processing port {i}/{total_ports}: {row['name']}")

            details = extract_port_details(page, row['url'])
            if details:
                port_data = {
                    'name': row['name'],
                    'type': row['type'],
                    'size': row['size'],
                    'url': row['url'],
                    **details
                }
                enriched_data.append(port_data)

                # Save after each successful extraction
                enriched_df = pd.DataFrame(enriched_data)
                save_enriched_data(enriched_df)

            time.sleep(2)  # Prevent overloading

        browser.close()


if __name__ == "__main__":
    enrich_port_data()