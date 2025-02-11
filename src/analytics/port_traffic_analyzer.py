# src/analytics/port_traffic_analyzer.py

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortTrafficAnalyzer:
    def __init__(self):
        self.db = DatabaseManager()
        self._setup_collections()

    def _setup_collections(self):
        """Setup time-series collection for traffic analysis"""
        try:
            self.db.db.create_collection(
                "traffic_analysis",
                timeseries={
                    "timeField": "timestamp",
                    "metaField": "port_name",
                    "granularity": "hours",
                    "expireAfterSeconds": 30 * 24 * 60 * 60  # 30 days retention
                }
            )
            logger.info("Created time-series collection for traffic analysis")
        except Exception as e:
            logger.debug(f"Collection might already exist: {e}")

        self.analysis_collection = self.db.db['traffic_analysis']

    def generate_daily_summary(self) -> pd.DataFrame:
        """Generate daily summary for all ports"""
        summary_data = []
        logger.info("Starting daily summary generation")

        try:
            ports = list(self.db.get_all_ports())
            logger.info(f"Found {len(ports)} ports to analyze")

            for port in ports:
                port_name = port['name']
                logger.info(f"Processing {port_name}")

                try:
                    # Get today's data
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

                    # Get in-port vessels
                    in_port_data = self.db.get_latest_port_data(port_name, "in_port")
                    vessels_count = len(in_port_data['vessels']) if in_port_data and 'vessels' in in_port_data else 0

                    # Get port calls
                    port_calls_data = self.db.get_latest_port_data(port_name, "port_calls")
                    port_calls = port_calls_data['vessels'] if port_calls_data and 'vessels' in port_calls_data else []

                    # Calculate metrics
                    total_calls = len(port_calls)
                    arrivals = sum(1 for call in port_calls if call.get('event_type') == 'Arrival')
                    departures = sum(1 for call in port_calls if call.get('event_type') == 'Departure')

                    summary_data.append({
                        'port_name': port_name,
                        'vessels_in_port': vessels_count,
                        'total_calls': total_calls,
                        'arrivals': arrivals,
                        'departures': departures,
                        'total_activity': arrivals + departures
                    })

                except Exception as e:
                    logger.error(f"Error processing port {port_name}: {e}")
                    continue

            df = pd.DataFrame(summary_data)
            logger.info(f"Generated summary for {len(df)} ports")
            return df

        except Exception as e:
            logger.error(f"Error in generate_daily_summary: {e}")
            return pd.DataFrame()

    def plot_activity_distributions(self, df: pd.DataFrame, save_path: str = None):
        """Create histograms for port activity metrics"""
        try:
            if df.empty:
                logger.warning("No data to plot")
                return

            plt.style.use('seaborn')  # Use a nicer style
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Vessels in port
            sns.histplot(data=df, x='vessels_in_port', ax=axes[0, 0], bins=20)
            axes[0, 0].set_title('Distribution of Vessels in Port')
            axes[0, 0].set_xlabel('Number of Vessels')
            axes[0, 0].set_ylabel('Number of Ports')

            # Total calls
            sns.histplot(data=df, x='total_calls', ax=axes[0, 1], bins=20)
            axes[0, 1].set_title('Distribution of Total Port Calls')
            axes[0, 1].set_xlabel('Number of Calls')
            axes[0, 1].set_ylabel('Number of Ports')

            # Total activity
            sns.histplot(data=df, x='total_activity', ax=axes[1, 0], bins=20)
            axes[1, 0].set_title('Distribution of Total Activity')
            axes[1, 0].set_xlabel('Total Activity (Arrivals + Departures)')
            axes[1, 0].set_ylabel('Number of Ports')

            # Top 10 ports by activity
            top_10 = df.nlargest(10, 'total_activity')
            sns.barplot(data=top_10, x='port_name', y='total_activity', ax=axes[1, 1])
            axes[1, 1].set_title('Top 10 Most Active Ports')
            axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Saved plot to {save_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Error creating plots: {e}")

    def generate_activity_report(self):
        """Generate comprehensive activity report"""
        try:
            # Get summary data
            df = self.generate_daily_summary()

            if df.empty:
                print("No data available for report")
                return None

            # Sort by total activity
            df_sorted = df.sort_values('total_activity', ascending=False)

            # Print summary table
            print("\nPort Activity Summary (Top 20 Most Active Ports)")
            print("=" * 100)
            print(df_sorted.head(20).to_string(index=False))

            # Print overall statistics
            print("\nOverall Statistics")
            print("=" * 100)
            print(f"Total ports analyzed: {len(df)}")
            print(f"Total vessels in ports: {df['vessels_in_port'].sum()}")
            print(f"Total vessel movements: {df['total_activity'].sum()}")
            print(f"Average vessels per port: {df['vessels_in_port'].mean():.2f}")
            print(f"Maximum vessels in a single port: {df['vessels_in_port'].max()}")
            print(f"Total arrivals: {df['arrivals'].sum()}")
            print(f"Total departures: {df['departures'].sum()}")

            # Create distribution plots
            self.plot_activity_distributions(df, 'port_activity_distribution.png')

            return df

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None


if __name__ == "__main__":
    analyzer = PortTrafficAnalyzer()
    summary_df = analyzer.generate_activity_report()

    if summary_df is not None:
        # Save detailed data to CSV
        summary_df.to_csv('port_activity_summary.csv', index=False)
        logger.info("Saved detailed summary to port_activity_summary.csv")