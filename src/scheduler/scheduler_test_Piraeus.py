import schedule
import time
from datetime import datetime
from src.scrapers.port_scraper import PortScraper


def scraping_job():
    """Run the scraping job"""
    print(f"\n=== Starting scraping job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    scraper = PortScraper()
    try:
        result = scraper.scrape()
        if result:
            print("Scraping completed successfully")
        else:
            print("Scraping failed")
    except Exception as e:
        print(f"Error during scraping: {e}")

    print(f"=== Job finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")


def main():
    print("Starting scheduler...")
    print("Will run scraping every 15 minutes")

    # Run immediately when starting
    scraping_job()

    # Schedule to run every 15 minutes
    schedule.every(15).minutes.do(scraping_job)

    # Keep running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            break
        except Exception as e:
            print(f"Error in scheduler: {e}")
            continue


if __name__ == "__main__":
    main()