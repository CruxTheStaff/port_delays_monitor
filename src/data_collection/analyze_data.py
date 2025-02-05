import pandas as pd
import os


def analyze_csv_folder(folder_name, base_path='src/data_collection/raw_data'):
    """Analyze CSV files in a specific folder"""
    folder_path = f"{base_path}/{folder_name}"
    if not os.path.exists(folder_path):
        print(f"\nNo data found for {folder_name}")
        return None

    latest_file = max([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    df = pd.read_csv(f"{folder_path}/{latest_file}")
    return df


def analyze_data():
    """Analyze all CSV files in the raw_data folders"""

    # Current Vessels
    print("\n=== Current Vessels Analysis ===")
    df_current = analyze_csv_folder('current_vessels')
    if df_current is not None:
        print(f"\nTotal current vessels: {len(df_current)}")
        print("\nVessels by type:")
        print(df_current['vessel_type'].value_counts())
        print("\nFirst few entries:")
        print(df_current.head())

    # Expected Arrivals
    print("\n=== Expected Arrivals Analysis ===")
    df_expected = analyze_csv_folder('expected_arrivals')
    if df_expected is not None:
        print(f"\nTotal expected arrivals: {len(df_expected)}")
        print("\nVessels by type:")
        print(df_expected['vessel_type'].value_counts())
        print("\nFirst few entries:")
        print(df_expected.head())

    # Port Calls
    print("\n=== Port Calls Analysis ===")
    df_calls = analyze_csv_folder('port_calls')
    if df_calls is not None:
        print(f"\nTotal port calls: {len(df_calls)}")
        print("\nEvents by type:")
        print(df_calls['event_type'].value_counts())
        print("\nVessels by type:")
        print(df_calls['vessel_type'].value_counts())
        print("\nFirst few entries:")
        print(df_calls.head())


if __name__ == "__main__":
    analyze_data()