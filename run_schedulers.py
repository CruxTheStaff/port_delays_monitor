# src/run_schedulers.py

import logging
from multiprocessing import Process
from src.scheduler.cluster_scheduler import ClusterScheduler
from src.scheduler.scraping_scheduler import ScrapingScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cluster_scheduler():
    """Run the cluster analysis scheduler"""
    try:
        scheduler = ClusterScheduler()
        scheduler.run()
    except Exception as error:
        logger.error(f"Cluster scheduler failed: {error}")


def run_scraping_scheduler():
    """Run the scraping scheduler"""
    try:
        scheduler = ScrapingScheduler()
        scheduler.run()
    except Exception as error:
        logger.error(f"Scraping scheduler failed: {error}")


def main():
    """Start both schedulers as separate processes"""
    logger.info("Starting schedulers...")

    # Create processes
    cluster_process = Process(target=run_cluster_scheduler)
    scraping_process = Process(target=run_scraping_scheduler)

    try:
        # Start processes
        cluster_process.start()
        logger.info("Cluster scheduler started")

        scraping_process.start()
        logger.info("Scraping scheduler started")

        # Wait for processes
        cluster_process.join()
        scraping_process.join()

    except KeyboardInterrupt:
        logger.info("Shutting down schedulers...")

        # Terminate processes
        if cluster_process.is_alive():
            cluster_process.terminate()
            cluster_process.join()

        if scraping_process.is_alive():
            scraping_process.terminate()
            scraping_process.join()

        logger.info("Schedulers shut down successfully")

    except Exception as error:
        logger.error(f"Error running schedulers: {error}")

        # Ensure processes are terminated
        for process in [cluster_process, scraping_process]:
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == "__main__":
    main()
