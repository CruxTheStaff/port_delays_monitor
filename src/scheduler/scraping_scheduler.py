from datetime import datetime, timedelta
import asyncio
from typing import Dict, Tuple
from colorama import Fore, Style, init
from playwright.async_api import async_playwright
from src.config.config_manager import ConfigManager
from src.database.db_manager import DatabaseManager
from src.scrapers.ports_activity_scraper import PortsActivityScraper
from dataclasses import dataclass
from typing import List


# Initialize colorama
init()

config_manager = ConfigManager()
logger = config_manager.setup_logging()



@dataclass
class ClusterConfig:
    interval: int  # minutes
    ports: List[str]
    max_concurrent: int
    retry_attempts: int
    color: str


class ScrapingStats:
    def __init__(self):
        self.scrape_counts = {}
        self.successful_scrapes = {}
        self.failed_scrapes = {}
        self.start_time = datetime.now()

    def add_scrape(self, port_name: str, success: bool):
        if port_name not in self.scrape_counts:
            self.scrape_counts[port_name] = 0
            self.successful_scrapes[port_name] = 0
            self.failed_scrapes[port_name] = 0

        self.scrape_counts[port_name] += 1
        if success:
            self.successful_scrapes[port_name] += 1
        else:
            self.failed_scrapes[port_name] += 1

    def print_stats(self):
        runtime = datetime.now() - self.start_time
        total_scrapes = sum(self.scrape_counts.values())
        successful = sum(self.successful_scrapes.values())

        print(f"\n{Fore.CYAN}=== Scraping Statistics ==={Style.RESET_ALL}")
        print(f"Runtime: {str(runtime).split('.')[0]}")
        print(f"Total Scrapes: {total_scrapes}")

        # Fix the success rate calculation
        success_rate = f"{(successful / total_scrapes * 100):.1f}%" if total_scrapes > 0 else "N/A"
        print(f"Success Rate: {success_rate}")

        print("\nTop 5 Most Scraped Ports:")

        # Get top 5 ports by scrape count
        top_ports = sorted(self.scrape_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for port, count in top_ports:
            success_rate = (self.successful_scrapes[port] / count * 100)
            print(f"  {port}: {count} scrapes ({success_rate:.1f}% success)")
        print(f"{Fore.CYAN}========================={Style.RESET_ALL}\n")

class ScrapingScheduler:
    def __init__(self):
        self.db = DatabaseManager()
        self.scraper = PortsActivityScraper()
        self._last_scrape_times: Dict[str, datetime] = {}
        self._active_scrapes: int = 0
        self.stats = ScrapingStats()
        self.stop_event = asyncio.Event()
        self.config_manager = ConfigManager()

        # Full cluster configuration
        self.cluster_configs = {
            0: ClusterConfig(
                interval=15,
                ports=['PIRAEUS'],
                max_concurrent=1,
                retry_attempts=3,
                color=Fore.RED
            ),
            1: ClusterConfig(
                interval=15,
                ports=['PERAMA', 'SALAMINA'],
                max_concurrent=1,
                retry_attempts=2,
                color=Fore.YELLOW
            ),
            2: ClusterConfig(
                interval=30,
                ports=['AGIOI THEODORI', 'ELEFSIS', 'MEGARA', 'SALAMINA MEG'],
                max_concurrent=1,
                retry_attempts=2,
                color=Fore.GREEN
            ),
            3: ClusterConfig(
                interval=60,
                ports=['AEGINA', 'AGIA MARINA', 'ARGOSTOLI', 'ASPROPYRGOS', 'ASTAKOS', 'CHIOS', 'ERETRIA',
                       'HERACLIO', 'IGOUMENITSA', 'KEFALONIA-LIXOURI', 'KYLLINI', 'LAVRIO', 'MILOS', 'NAXOS',
                       'NEA STIRA', 'PAROS', 'PATRA', 'POROS', 'PSYTTALEIA', 'REVITHOUSSA', 'RHODES', 'SANTORINI',
                       'SOUDA', 'SOUSAKI', 'SPETSES', 'SYROS', 'THESSALONIKI', 'TINOS', 'VOLOS'],
                max_concurrent=2,
                retry_attempts=1,
                color=Fore.BLUE
            ),
            4: ClusterConfig(
                interval=120,
                ports=['ACHLADI', 'AETOS', 'AGATHONISI', 'AGIA KYRIAKI', 'AGIA ROUMELI', 'AGIOKAMPOS',
                       'AGIOS CHARALAMPOS', 'AGIOS DIMITRIOS', 'AGIOS EYSTRATIOS', 'AGIOS GEORGIOS',
                       'AGIOS KONSTANTINOS', 'AGIOS KYRIKOS', 'AGIOS NIKOLAOS 1', 'AGIOS NIKOLAOS 2',
                       'AGIOS NIKOLAOS 3', 'AGIOS ONOUFRIOS', 'AGIOS STEFANOS', 'AGISTRI', 'AGNONTAS',
                       'AIDIPSOS', 'AIGIALI', 'AIGIO', 'ALEXANDROUPOLI', 'ALIVERI', 'ALONNISOS', 'ALTSI',
                       'AMALIAPOLI', 'AMFILOCHIA', 'AMMOULINI', 'ANAFI', 'ANDROS', 'ANTIKITHIRA', 'ANTIPAROS',
                       'ANTIRIO', 'ARKITSA', 'ARKOI', 'ASTAKOS 2', 'ASTYPALEA', 'ATHERINOLAKOS', 'BRAXATI',
                       'CHALKI', 'CHALKIS', 'CHANIA MARINA', 'CHRISOMILIA', 'CHRISSI', 'CORINTHOS', 'DAFNI',
                       'DELOS', 'DIAFANI', 'DOMVRAINA', 'DONOUSA', 'DREPANO', 'ELAFONISOS', 'ELEFTHERES',
                       'EREIKOUSSA', 'ERMIONI', 'EVDILOS', 'FOLEGANDROS', 'FOURNI', 'GAIOS', 'GALAKSIDI',
                       'GAVDOS', 'GLOSSA', 'GYALI', 'GYTHEIO', 'HYDRA', 'IERAPETRA', 'IOS', 'IRAKLIA', 'ITEA',
                       'ITHAKI', 'KALAMATA', 'KALI LIMENES', 'KALLITHEA', 'KALYMNOS', 'KAMARI', 'KAMIROS',
                       'KANAVA', 'KARDAMAINA', 'KARPATHOS', 'KARYSTOS', 'KASOS', 'KASTELLORIZO', 'KATAKOLO',
                       'KATAPOLA', 'KAVALA', 'KEA', 'KERAMOTI', 'KERKYRA', 'KIATO', 'KIMOLOS', 'KISSAMOS',
                       'KITHIRA', 'KLIMA', 'KOLYMPARI', 'KOS', 'KOSTA', 'KOUFONISI', 'KYMI', 'KYTHNOS', 'LARYMNA',
                       'LEFKIMMI', 'LEIPSI', 'LEROS', 'LIMNI', 'LIMNOS', 'LINDOS', 'LOUTRO', 'MANTOUDI', 'MARMARI',
                       'MASTICHARI', 'MATHRAKI', 'MESTA', 'MILAKI', 'MOUDROS', 'MYKONOS', 'MYKONOS OLD PORT',
                       'MYRTIES', 'MYTIKAS', 'MYTILINI', 'NAFPLIO', 'NAOUSA', 'NEA KARVALI', 'NEA MICHANIONA',
                       'NEA MOUDANIA', 'NEAPOLI', 'NEOS MARMARAS', 'NISYROS', 'NYDRI', 'OINOUSSES', 'ORMOS PRINOU',
                       'OROPOS', 'OTHONI', 'OURANOUPOLI', 'PACHI', 'PALAIA EPIDAVROS', 'PALAIO TRIKERI',
                       'PALEOCHORA', 'PANORMITIS', 'PATMOS', 'PEFKOCHORI', 'PERAMA MYT', 'PETRIES', 'PLATANIA',
                       'POLLONIA', 'POROS KEFALONIA', 'PORTO LAGOS', 'PORTOCHELI', 'POUNDA', 'POUNTA', 'PREVEZA',
                       'PSACHNA', 'PSARA', 'PYTHAGOREIO', 'RAFINA', 'RETHYMNO', 'RIO', 'SAMI', 'SAMOS-KARLOVASI',
                       'SAMOS-VATHI', 'SAMOTHRAKI', 'SERIFOS', 'SIFNOS', 'SIKINOS', 'SITIA', 'SKARAMAGAS',
                       'SKIATHOS', 'SKOPELOS', 'SKYROS', 'SOUGIA', 'SPARTOCHORI', 'STAVROS', 'STYLIDA', 'SXOINOUSA',
                       'SYMI', 'SYRI', 'THASOS', 'THIRASSIA', 'THYMAINA', 'TILOS', 'TRYPITI', 'TSINGELI',
                       'TSOURAKI', 'VASSILIKI', 'VOUDIA', 'ZAKYNTHOS'],
                max_concurrent=3,
                retry_attempts=1,
                color=Fore.WHITE
            )
        }

    def _log_scraping_status(self, cluster: int, port_name: str, status: str, message: str = ""):
        """Enhanced logging with color coding and visual structure"""
        config = self.cluster_configs[cluster]
        timestamp = datetime.now().strftime("%H:%M:%S")

        status_color = {
            "START": Fore.CYAN,
            "SUCCESS": Fore.GREEN,
            "ERROR": Fore.RED,
            "SKIP": Fore.YELLOW
        }.get(status, Fore.WHITE)

        print(f"{config.color}[Cluster {cluster}]{Style.RESET_ALL} "
              f"{status_color}[{status}]{Style.RESET_ALL} "
              f"{timestamp} - {port_name}: {message}")

    async def get_current_clusters(self) -> Dict[str, Tuple[int, str]]:
        """Get cluster assignments with fallback to predefined configs"""
        try:
            # Try to get from database first
            latest = self.db.db.cluster_analysis_daily.find_one(
                sort=[('timestamp', -1)]
            )

            if latest:
                logger.info("Using cluster assignments from database")
                ports = {
                    port['name']: port['port_id']
                    for port in self.db.get_all_ports()
                }

                assignments = {
                    port['port_name']: (
                        port['cluster'],
                        str(ports.get(port['port_name'], ''))
                    )
                    for port in latest['port_assignments']
                    if port['port_name'] in ports
                }
            else:
                logger.info("Using predefined cluster configurations")
                ports = {
                    port['name']: port['port_id']
                    for port in self.db.get_all_ports()
                }

                assignments = {}
                for cluster, config in self.cluster_configs.items():
                    for port_name in config.ports:
                        if port_name in ports:
                            assignments[port_name] = (cluster, str(ports.get(port_name, '')))

            logger.info(f"Using {len(assignments)} port assignments")
            return assignments

        except Exception as error:
            logger.error(f"Error getting cluster assignments: {error}")
            return {}

    async def _should_scrape(self, port_name: str, cluster: int) -> bool:
        """Determine if port should be scraped based on last scrape time"""
        last_scrape = self._last_scrape_times.get(port_name)
        if not last_scrape:
            self._log_scraping_status(cluster, port_name, "START", "First scrape")
            return True

        config = self.cluster_configs[cluster]
        next_scrape = last_scrape + timedelta(minutes=config.interval)
        should_scrape = datetime.now() >= next_scrape

        # Log only if we're close to next scrape (e.g., within 5 minutes)
        if not should_scrape:
            wait_time = (next_scrape - datetime.now()).total_seconds() / 60
            if wait_time <= 5:  # Only log if within 5 minutes
                self._log_scraping_status(
                    cluster, port_name, "SKIP",
                    f"Next scrape in {wait_time:.1f} minutes"
                )

        return should_scrape

    async def scrape_port(self, port_name: str, port_id: str, cluster: int):
        """Scrape single port with detailed logging and statistics"""
        success = False
        try:
            self._log_scraping_status(cluster, port_name, "START")

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await self.scraper.scrape_port_activity(
                    page=page,
                    port_name=port_name,
                    port_id=port_id
                )

                self._log_scraping_status(
                    cluster, port_name, "SUCCESS",
                    f"Scrape #{self.stats.scrape_counts.get(port_name, 0) + 1}"
                )
                success = True
                await browser.close()

        except Exception as e:
            self._log_scraping_status(
                cluster, port_name, "ERROR",
                f"Failed: {str(e)}"
            )
        finally:
            self.stats.add_scrape(port_name, success)
            self._last_scrape_times[port_name] = datetime.now()

    async def schedule_cluster(self, cluster: int):
        """Handle scheduling for a specific cluster"""
        config = self.cluster_configs[cluster]
        while not self.stop_event.is_set():
            try:
                current_clusters = await self.get_current_clusters()
                eligible_ports = [
                    (name, pid) for name, (c, pid) in current_clusters.items()
                    if c == cluster and await self._should_scrape(name, cluster)
                ]

                # Process ports in this cluster concurrently up to max_concurrent
                if eligible_ports:
                    for i in range(0, len(eligible_ports), config.max_concurrent):
                        batch = eligible_ports[i:i + config.max_concurrent]
                        tasks = [
                            self.scrape_port(port_name, port_id, cluster)
                            for port_name, port_id in batch
                        ]
                        # Wait for all tasks in this batch to complete
                        await asyncio.gather(*tasks)

                # Wait for the cluster's interval before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in cluster {cluster}: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def report_stats(self):
        """Periodic statistics reporting"""
        while not self.stop_event.is_set():
            await asyncio.sleep(30 * 60)  # Every 30 minutes
            self.stats.print_stats()

    async def run(self):
        """Run the scheduler with async tasks per cluster"""
        logger.info(f"{Fore.CYAN}Starting Scraping Scheduler{Style.RESET_ALL}")

        # Print initial configuration
        logger.info("Cluster configurations:")
        for cluster, config in self.cluster_configs.items():
            print(f"{config.color}Cluster {cluster}:{Style.RESET_ALL}")
            print(f"  Interval: {config.interval} minutes")
            print(f"  Ports: {len(config.ports)}")
            print(f"  Max concurrent: {config.max_concurrent}")
            print()

        try:
            # Create tasks for each cluster
            cluster_tasks = [
                asyncio.create_task(self.schedule_cluster(cluster))
                for cluster in self.cluster_configs
            ]

            # Add stats reporting task
            stats_task = asyncio.create_task(self.report_stats())

            # Wait for all tasks
            await asyncio.gather(*cluster_tasks, stats_task)

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as error:
            logger.error(f"Error in scheduler: {error}")
        finally:
            self.stop_event.set()
            # Wait a bit for tasks to clean up
            await asyncio.sleep(1)

def main():
    """Entry point for the scheduler"""
    scheduler = ScrapingScheduler()
    try:
        asyncio.run(scheduler.run())
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as error:
        logger.error(f"Scheduler failed: {error}")


if __name__ == "__main__":
    main()