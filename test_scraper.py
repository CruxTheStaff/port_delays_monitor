from src.scrapers.port_scraper import PortScraper
from playwright.sync_api import sync_playwright


def test_scraping():
    scraper = PortScraper()
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    page = browser.new_page()

    try:
        print("\n=== Starting scraping process ===")

        print("\n1. Scraping current vessels...")
        current_vessels = scraper.scrape_current_vessels(page)
        print(f"Found {len(current_vessels)} current vessels")

        print("\n2. Scraping expected arrivals...")
        expected_arrivals = scraper.scrape_expected_arrivals(page)
        print(f"Found {len(expected_arrivals)} expected arrivals")

        print("\n3. Scraping port calls...")
        port_calls = scraper.scrape_port_calls(page)
        print(f"Found {len(port_calls)} port calls")

        # Save data without timestamp parameter
        if current_vessels:
            scraper.save_data(current_vessels, 'current_vessels')
        if expected_arrivals:
            scraper.save_expected_arrivals(expected_arrivals)
        if port_calls:
            scraper.save_port_calls(port_calls)

        print("\n=== Scraping process completed! ===")

    except Exception as e:
        print(f"Error during scraping: {e}")

    finally:
        browser.close()
        playwright.stop()


if __name__ == "__main__":
    test_scraping()