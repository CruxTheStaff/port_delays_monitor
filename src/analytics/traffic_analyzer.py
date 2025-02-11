# src/analytics/port_traffic_analyzer.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from src.database.db_manager import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortTrafficAnalyzer:
    def __init__(self):
        self.db = DatabaseManager()

    def get_traffic_data(self) -> pd.DataFrame:
        """Get detailed traffic data for analysis"""
        all_ports = self.db.get_all_ports()
        traffic_data = []

        for port in all_ports:
            port_name = port['name']
            try:
                # Get current vessels
                in_port = self.db.get_port_data(port_name, "in_port")
                current_vessels = len(in_port[0]['vessels']) if in_port else 0

                # Get port calls
                port_calls = self.db.get_port_data(port_name, "port_calls")
                if port_calls:
                    arrivals = sum(1 for v in port_calls[0]['vessels']
                                   if v.get('event_type') == 'Arrival')
                    departures = sum(1 for v in port_calls[0]['vessels']
                                     if v.get('event_type') == 'Departure')
                else:
                    arrivals = departures = 0

                traffic_data.append({
                    'port_name': port_name,
                    'current_vessels': current_vessels,
                    'arrivals': arrivals,
                    'departures': departures,
                    'total_movements': arrivals + departures,
                    'latitude': float(port['latitude']),
                    'longitude': float(port['longitude'])
                })

            except Exception as e:
                logger.error(f"Error processing {port_name}: {e}")

        return pd.DataFrame(traffic_data)

    # # filtering of the active ports, to not have false outliers
    @staticmethod
    def analyze_active_ports(df: pd.DataFrame, min_activity: int = 1):
        """Analyze only ports with significant activity"""
        # Filter active ports
        active_ports = df[
            (df['current_vessels'] > min_activity) |
            (df['total_movements'] > min_activity)
            ].copy()

        logger.info(f"Analyzing {len(active_ports)} active ports out of {len(df)} total ports")

        features = ['current_vessels', 'total_movements']
        X = active_ports[features].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means clustering (probably better than DBSCAN for this case)
        n_clusters = 4  # Based on your original clustering logic
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Add results to dataframe
        active_ports['cluster'] = clusters

        # Visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(active_ports['current_vessels'],
                              active_ports['total_movements'],
                              c=clusters,
                              cmap='viridis')

        # Annotate major ports
        for idx, row in active_ports.iterrows():
            if row['current_vessels'] > 20 or row['total_movements'] > 50:
                plt.annotate(row['port_name'],
                             (row['current_vessels'], row['total_movements']))

        plt.xlabel('Current Vessels in Port')
        plt.ylabel('Total Movements (Arrivals + Departures)')
        plt.title('Active Ports Traffic Analysis')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.savefig('active_ports_analysis.png')
        plt.close()

        # Print cluster analysis
        print("\n=== Active Ports Analysis ===")
        for i in range(n_clusters):
            cluster_ports = active_ports[active_ports['cluster'] == i]
            print(f"\nCluster {i}:")
            print(f"Number of ports: {len(cluster_ports)}")
            print("Major ports in cluster:")
            top_ports = cluster_ports.nlargest(3, 'total_movements')
            for _, port in top_ports.iterrows():
                print(f"  - {port['port_name']}: {port['current_vessels']} vessels, "
                      f"{port['total_movements']} movements")

        return active_ports

    @staticmethod
    def analyze_outliers(df: pd.DataFrame):
        """Analyze potential outliers in port traffic"""
        features = ['current_vessels', 'total_movements']
        X = df[features].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN for outlier detection
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)

        # Add results to dataframe
        df['cluster'] = clusters
        outliers = df[clusters == -1]

        # Visualization
        plt.figure(figsize=(12, 8))
        plt.scatter(df[clusters != -1]['current_vessels'],
                    df[clusters != -1]['total_movements'],
                    c='blue', label='Normal')
        plt.scatter(outliers['current_vessels'],
                    outliers['total_movements'],
                    c='red', label='Outlier')

        # Annotate outliers
        for idx, row in outliers.iterrows():
            plt.annotate(row['port_name'],
                         (row['current_vessels'], row['total_movements']))

        plt.xlabel('Current Vessels in Port')
        plt.ylabel('Total Movements (Arrivals + Departures)')
        plt.title('Port Traffic Analysis - Outlier Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('port_traffic_outliers.png')
        plt.close()

        # Print detailed analysis
        print("\n=== Port Traffic Analysis ===")
        print(f"Total ports analyzed: {len(df)}")
        print(f"Potential outliers detected: {len(outliers)}")

        if not outliers.empty:
            print("\nDetailed outlier analysis:")
            for _, port in outliers.iterrows():
                print(f"\nPort: {port['port_name']}")
                print(f"Current vessels: {port['current_vessels']}")
                print(f"Total movements: {port['total_movements']}")
                print(f"Arrivals: {port['arrivals']}")
                print(f"Departures: {port['departures']}")

        return outliers


def main():
    analyzer = PortTrafficAnalyzer()

    # Get traffic data
    logger.info("Collecting traffic data...")
    traffic_df = analyzer.get_traffic_data()

    # Analyze outliers
    logger.info("Analyzing traffic patterns...")
    outliers = analyzer.analyze_outliers(traffic_df)

    # Save results
    traffic_df.to_csv('port_traffic_analysis.csv', index=False)
    if not outliers.empty:
        outliers.to_csv('port_traffic_outliers.csv', index=False)
        logger.info(f"Found {len(outliers)} outliers. Check port_traffic_outliers.csv")

    logger.info("Analysis complete. Check port_traffic_outliers.png for visualization")

    return traffic_df, outliers


if __name__ == "__main__":
    traffic_df, outliers = main()