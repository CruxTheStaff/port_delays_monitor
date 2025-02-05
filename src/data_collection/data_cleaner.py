import pandas as pd
import numpy as np
import os


class DataCleaner:
    def __init__(self, port_name='Piraeus'):
        # Get absolute paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_data_path = os.path.join(self.project_root, 'src', 'data_collection', 'raw_data', port_name)
        self.cleaned_data_path = os.path.join(self.project_root, 'src', 'data_collection', 'cleaned_data', port_name)

        print(f"Raw data path: {self.raw_data_path}")
        print(f"Cleaned data path: {self.cleaned_data_path}")

    def clean_current_vessels(self, df=None):
        """Clean current vessels data"""
        if df is None:
            df = pd.read_csv(f'{self.raw_data_path}/current_vessels.csv')

        df = df.copy()

        # Clean DWT
        df['dwt'] = df['dwt'].replace('---', np.nan)
        df['dwt_tons'] = pd.to_numeric(
            df['dwt'].str.replace(' Τόνοι', '') \
                .str.replace(',', ''),
            errors='coerce'
        )

        # Clean GRT
        df['grt'] = df['grt'].replace('---', np.nan)
        df['grt_tons'] = pd.to_numeric(
            df['grt'].str.replace(' Τόνοι', '') \
                .str.replace(',', ''),
            errors='coerce'
        )

        # Clean size
        df['size_meters'] = pd.to_numeric(
            df['size'].str.replace(' μ', '') \
                .replace('---', np.nan),
            errors='coerce'
        )
        # Convert zero sizes to NaN
        df.loc[df['size_meters'] == 0, 'size_meters'] = np.nan

        # Clean built year
        df['built_year'] = pd.to_numeric(
            df['built'].replace('---', np.nan),
            errors='coerce'
        )

        # Drop original columns
        df = df.drop(['dwt', 'grt', 'size', 'built'], axis=1)

        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=False)

        # Add stats columns
        df['has_complete_data'] = df[['dwt_tons', 'grt_tons', 'size_meters', 'built_year']].notna().all(axis=1)

        # Remove duplicates
        df = self.remove_duplicates(df, 'current_vessels')

        return df

    def clean_expected_arrivals(self, df=None):
        """Clean expected arrivals data"""
        if df is None:
            df = pd.read_csv(f'{self.raw_data_path}/expected_arrivals.csv')

        df = df.copy()

        # Convert mmsi to string to preserve leading zeros
        df['mmsi'] = df['mmsi'].astype(str)

        # Clean vessel names
        df['vessel_name'] = df['vessel_name'].str.strip()

        # Remove eta column as it's always "PIRAEUS"
        df = df.drop('eta', axis=1)

        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=False)

        # Remove duplicates
        df = self.remove_duplicates(df, 'expected_arrivals')

        return df

    def clean_port_calls(self, df=None):
        """Clean port calls data"""
        if df is None:
            df = pd.read_csv(f'{self.raw_data_path}/port_calls.csv')

        df = df.copy()

        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['record_time'] = pd.to_datetime(df['record_time'])

        # Clean vessel names
        df['vessel_name'] = df['vessel_name'].str.strip()

        # Convert event_type to numeric (1 for arrival, 0 for departure)
        df['is_arrival'] = np.where(df['event_type'] == 'arrival', 1, 0)

        # Drop original event_type column
        df = df.drop('event_type', axis=1)

        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=False)

        # Remove duplicates
        df = self.remove_duplicates(df, 'port_calls')

        # Manage historical data
        df = self.manage_historical_data(df)

        return df

    @staticmethod
    def analyze_port_calls_continuity(df):
        """Analyze if we're missing events between scrapes"""
        df = df.sort_values('timestamp')

        # Convert timestamps to datetime if they aren't already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['record_time'] = pd.to_datetime(df['record_time'])

        # Calculate time difference between consecutive events
        time_diffs = df['timestamp'].diff()

        print("\nPort Calls Continuity Analysis:")
        print(f"Time gaps between events:")
        print(time_diffs.describe())

        # Find large gaps (e.g., > 30 minutes)
        large_gaps = df[time_diffs > pd.Timedelta(minutes=30)]
        if not large_gaps.empty:
            print(f"\nFound {len(large_gaps)} gaps larger than 30 minutes:")
            for _, row in large_gaps.iterrows():
                print(f"Gap before {row['timestamp']}: {row['vessel_name']} ({row['vessel_type']})")

    @staticmethod
    def analyze_scraping_coverage(df):
        """Analyze if our scraping interval is adequate"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['record_time'] = pd.to_datetime(df['record_time'])

        # Calculate delay between event and recording
        df['recording_delay'] = df['record_time'] - df['timestamp']

        print("\nScraping Coverage Analysis:")
        print("Delay between events and recording:")
        print(df['recording_delay'].describe())

        # Check for events that were recorded much later
        late_records = df[df['recording_delay'] > pd.Timedelta(minutes=30)]
        if not late_records.empty:
            print(f"\nFound {len(late_records)} events recorded >30 min after occurrence:")
            for _, row in late_records.iterrows():
                print(f"Event: {row['vessel_name']} at {row['timestamp']}, recorded at {row['record_time']}")

    @staticmethod
    def manage_historical_data(df, max_days=30):
        """Manage historical data keeping only recent records"""
        df = df.copy()

        # Convert timestamps to datetime if they aren't already
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Get current time and calculate cutoff date
        current_time = pd.Timestamp.now()
        cutoff_date = current_time - pd.Timedelta(days=max_days)

        # Keep only recent records
        df_recent = df[df['timestamp'] > cutoff_date]

        # Print statistics
        print(f"\nHistorical Data Management:")
        print(f"Total records: {len(df)}")
        print(f"Records within last {max_days} days: {len(df_recent)}")
        print(f"Removed records: {len(df) - len(df_recent)}")

        return df_recent.sort_values('timestamp', ascending=False)

    @staticmethod
    def remove_duplicates(df, data_type):
        """Remove duplicates based on data type specific criteria"""
        df = df.copy()

        if data_type == 'port_calls':
            # Για port calls, θεωρούμε διπλοεγγραφή αν έχουμε ίδιο πλοίο,
            # χρόνο και τύπο συμβάντος μέσα σε διάστημα 5 λεπτών
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['time_group'] = df['timestamp'].dt.floor('5min')
            duplicates = df.duplicated(
                subset=['vessel_name', 'time_group', 'is_arrival'],
                keep='last'
            )
            df = df[~duplicates].drop('time_group', axis=1)

        elif data_type == 'current_vessels':
            # Για current vessels, κρατάμε την πιο πρόσφατη εγγραφή για κάθε πλοίο
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            df = df.drop_duplicates(subset=['vessel_name'], keep='first')

        elif data_type == 'expected_arrivals':
            # Για expected arrivals, κρατάμε την πιο πρόσφατη εγγραφή για κάθε MMSI
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            df = df.drop_duplicates(subset=['mmsi'], keep='first')

        print(f"\nDuplicate removal for {data_type}:")
        print(f"Original records: {len(df)}")
        print(f"After removing duplicates: {len(df)}")

        return df

    def clean_all_data(self):
        """Clean all data files and save cleaned versions"""
        os.makedirs(self.cleaned_data_path, exist_ok=True)

        # Clean and save current vessels
        print("\nCleaning current vessels data...")
        current_vessels = self.clean_current_vessels()
        current_vessels.to_csv(f'{self.cleaned_data_path}/current_vessels.csv', index=False)

        print("\nCurrent Vessels Statistics:")
        print(f"Total vessels: {len(current_vessels)}")
        print(f"Vessels with complete data: {current_vessels['has_complete_data'].sum()}")
        print("\nSize statistics (meters):")
        print(current_vessels['size_meters'].describe())
        print("\nVessels by type:")
        print(current_vessels['vessel_type'].value_counts())
        print("\nMissing values per column:")
        print(current_vessels.isna().sum())

        # Clean and save expected arrivals
        print("\nCleaning expected arrivals data...")
        expected_arrivals = self.clean_expected_arrivals()
        expected_arrivals.to_csv(f'{self.cleaned_data_path}/expected_arrivals.csv', index=False)

        print("\nExpected Arrivals Statistics:")
        print(f"Total expected arrivals: {len(expected_arrivals)}")
        print("\nVessels by type:")
        print(expected_arrivals['vessel_type'].value_counts())

        # Clean and save port calls
        print("\nCleaning port calls data...")
        port_calls = self.clean_port_calls()
        port_calls.to_csv(f'{self.cleaned_data_path}/port_calls.csv', index=False)

        print("\nPort Calls Statistics:")
        print(f"Total port calls: {len(port_calls)}")
        print("\nEvents distribution:")
        event_counts = port_calls['is_arrival'].value_counts()
        print(f"Arrivals: {event_counts.get(1, 0)}")
        print(f"Departures: {event_counts.get(0, 0)}")
        print("\nVessels by type:")
        print(port_calls['vessel_type'].value_counts())

        # Analyze port calls timing
        print("\nAnalyzing port calls timing...")
        self.analyze_port_calls_continuity(port_calls)
        self.analyze_scraping_coverage(port_calls)

def main():
    cleaner = DataCleaner()
    cleaner.clean_all_data()


if __name__ == "__main__":
    main()