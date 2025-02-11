# src/data_collection/analyze_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import logging
from src.database.db_manager import DatabaseManager
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortActivityAnalyzer:
    def __init__(self):
        self.db = DatabaseManager()

    def get_port_activity(self) -> pd.DataFrame:
        """Get total activity for all ports"""
        all_ports = self.db.get_all_ports()
        activity_data = []

        for port in all_ports:
            port_name = port['name']
            try:
                # Get vessels in port
                in_port = self.db.get_port_data(port_name, "in_port")
                vessels_in_port = len(in_port[0]['vessels']) if in_port else 0

                # Get port calls
                port_calls = self.db.get_port_data(port_name, "port_calls")
                total_calls = len(port_calls[0]['vessels']) if port_calls else 0

                # Get expected arrivals
                expected = self.db.get_port_data(port_name, "expected")
                expected_arrivals = len(expected[0]['vessels']) if expected else 0

                activity_data.append({
                    'port_name': port_name,
                    'vessels_in_port': vessels_in_port,
                    'total_calls': total_calls,
                    'expected_arrivals': expected_arrivals,
                    'total_activity': vessels_in_port + total_calls + expected_arrivals
                })

            except Exception as e:
                logger.error(f"Error processing {port_name}: {e}")

        return pd.DataFrame(activity_data)

    @staticmethod
    def create_activity_distribution(df: pd.DataFrame):
        """Create activity distribution using Sturges' rule"""
        # Calculate number of bins using Sturges' rule
        n = len(df)
        num_bins = int(1 + 3.322 * math.log10(n))

        # Create histogram
        plt.figure(figsize=(12, 6))
        plt.hist(df['total_activity'], bins=num_bins, edgecolor='black')
        plt.title('Distribution of Port Activity')
        plt.xlabel('Total Activity (Vessels)')
        plt.ylabel('Number of Ports')
        plt.grid(True, alpha=0.3)

        # Add mean and median lines
        plt.axvline(df['total_activity'].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(df['total_activity'].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
        plt.legend()

        plt.savefig('port_activity_distribution.png')
        plt.close()

    @staticmethod
    def create_activity_summary(df: pd.DataFrame):
        """Create summary statistics of port activity"""
        summary = {
            'Total Ports': len(df),
            'Active Ports': len(df[df['total_activity'] > 0]),
            'Inactive Ports': len(df[df['total_activity'] == 0]),
            'Mean Activity': df['total_activity'].mean(),
            'Median Activity': df['total_activity'].median(),
            'Max Activity': df['total_activity'].max(),
            'Most Active Ports': df.nlargest(10, 'total_activity')[['port_name', 'total_activity']].to_dict('records')
        }

        # Print summary
        print("\n=== Port Activity Summary ===")
        print(f"Total Ports: {summary['Total Ports']}")
        print(f"Active Ports: {summary['Active Ports']}")
        print(f"Inactive Ports: {summary['Inactive Ports']}")
        print(f"\nActivity Statistics:")
        print(f"Mean Activity: {summary['Mean Activity']:.2f} vessels")
        print(f"Median Activity: {summary['Median Activity']:.2f} vessels")
        print(f"Maximum Activity: {summary['Max Activity']} vessels")

        print("\nTop 10 Most Active Ports:")
        for port in summary['Most Active Ports']:
            print(f"{port['port_name']}: {port['total_activity']} vessels")

        return summary


def main():
    analyzer = PortActivityAnalyzer()

    # Get activity data
    logger.info("Collecting port activity data...")
    activity_df = analyzer.get_port_activity()

    # Create distribution plot
    logger.info("Creating activity distribution plot...")
    analyzer.create_activity_distribution(activity_df)

    # Create and print summary
    logger.info("Creating activity summary...")
    analyzer.create_activity_summary(activity_df)

    # Save detailed data to CSV
    activity_df.to_csv('port_activity_details.csv', index=False)
    logger.info("Analysis complete. Check port_activity_distribution.png and port_activity_details.csv")


if __name__ == "__main__":
    main()