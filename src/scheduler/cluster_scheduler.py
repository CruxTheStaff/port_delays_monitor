"""
Cluster Scheduler Module
Handles daily cluster analysis and scraping interval adjustments
"""

import logging
from datetime import datetime, timedelta
import schedule
import time
from typing import Literal, Dict, Optional
from src.database.db_manager import DatabaseManager
from src.analytics.ports_clusters import PortClusterAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TimeWindow = Literal['daily', 'weekly', 'monthly']


class ClusterScheduler:
    def __init__(self):
        self.db = DatabaseManager()
        self.cluster_analyzer = PortClusterAnalyzer()
        self.last_analysis: Optional[Dict] = None

    @staticmethod
    def _get_time_params(time_window: TimeWindow) -> tuple[datetime, str]:
        """Get start time and collection name for given time window"""
        now = datetime.now()

        params = {
            'daily': (now - timedelta(days=1), 'cluster_analysis_daily'),
            'weekly': (now - timedelta(weeks=1), 'cluster_analysis_weekly'),
            'monthly': (now - timedelta(days=30), 'cluster_analysis_monthly')
        }

        if time_window not in params:
            raise ValueError(f"Invalid time window: {time_window}")

        return params[time_window]

    def _compare_with_previous(self, new_analysis: Dict) -> Dict:
        """Compare new analysis with previous one and log significant changes"""
        changes = {
            'cluster_changes': [],
            'interval_changes': [],
            'total_ports_changed': 0
        }

        if not self.last_analysis:
            logger.info("First analysis - no comparison available")
            self.last_analysis = new_analysis
            return changes

        # Compare cluster assignments
        old_assignments = {
            port['port_name']: port['cluster']
            for port in self.last_analysis.get('port_assignments', [])
        }

        new_assignments = {
            port['port_name']: port['cluster']
            for port in new_analysis.get('port_assignments', [])
        }

        for port_name, new_cluster in new_assignments.items():
            old_cluster = old_assignments.get(port_name)
            if old_cluster is not None and old_cluster != new_cluster:
                changes['cluster_changes'].append({
                    'port': port_name,
                    'old_cluster': old_cluster,
                    'new_cluster': new_cluster
                })
                changes['total_ports_changed'] += 1

        # Log significant changes
        if changes['cluster_changes']:
            logger.info(f"Cluster changes detected for {len(changes['cluster_changes'])} ports")
            for change in changes['cluster_changes']:
                logger.info(
                    f"Port {change['port']} moved from cluster {change['old_cluster']} "
                    f"to cluster {change['new_cluster']}"
                )

        self.last_analysis = new_analysis
        return changes

    def update_clusters(self, time_window: TimeWindow) -> None:
        """Update clusters based on specified time window"""
        try:
            logger.info(f"Starting {time_window} cluster analysis")
            start_time = datetime.now()

            # Get parameters for time window
            analysis_start, collection_name = self._get_time_params(time_window)

            # Perform clustering
            df = self.cluster_analyzer.collect_port_data()
            if df.empty:
                logger.error("No data available for clustering")
                return

            clustered_df = self.cluster_analyzer.perform_clustering(df)
            cluster_stats = self.cluster_analyzer.analyze_clusters(clustered_df)

            # Prepare analysis document
            analysis_doc = {
                'timestamp': datetime.now(),
                'time_window': time_window,
                'analysis_duration': (datetime.now() - start_time).total_seconds(),
                'cluster_stats': cluster_stats,
                'port_assignments': clustered_df[['port_name', 'cluster']].to_dict('records')
            }

            # Compare with previous analysis
            changes = self._compare_with_previous(analysis_doc)
            analysis_doc['changes'] = changes

            # Save to MongoDB
            self.db.db[collection_name].insert_one(analysis_doc)

            # Save CSV files
            if time_window == 'daily':
                self.cluster_analyzer.save_results(clustered_df, cluster_stats)

            duration = datetime.now() - start_time
            logger.info(
                f"Updated clusters for {time_window} analysis in {duration.total_seconds():.2f}s. "
                f"Changes detected in {changes['total_ports_changed']} ports"
            )

        except Exception as error:
            logger.error(f"Error updating {time_window} clusters: {error}", exc_info=True)

    def schedule_updates(self) -> None:
        """Schedule all cluster updates"""
        # Daily update at midnight
        schedule.every().day.at("00:00").do(
            self.update_clusters, time_window='daily'
        )

        # Weekly update on Monday at midnight
        schedule.every().monday.at("00:00").do(
            self.update_clusters, time_window='weekly'
        )

        # Monthly update on first day of month at midnight
        schedule.every().day.at("00:00").do(
            lambda: self.update_clusters('monthly')
            if datetime.now().day == 1 else None
        )

    def run(self) -> None:
        """Run the cluster scheduler"""
        logger.info("Starting cluster scheduler")

        try:
            # Initial cluster analysis
            self.update_clusters('daily')

            # Schedule future updates
            self.schedule_updates()

            # Run continuously
            while True:
                schedule.run_pending()
                time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Cluster scheduler stopped by user")
        except Exception as error:
            logger.error(f"Error in cluster scheduler: {error}", exc_info=True)
            raise


def main() -> None:
    try:
        scheduler = ClusterScheduler()
        scheduler.run()
    except KeyboardInterrupt:
        logger.info("Cluster scheduler stopped by user")
    except Exception as error:
        logger.error(f"Cluster scheduler failed: {error}", exc_info=True)


if __name__ == "__main__":
    main()