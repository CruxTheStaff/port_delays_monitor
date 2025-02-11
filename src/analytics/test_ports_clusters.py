"""
Port Clustering Analysis Module

Features:
- Data collection and validation
- Noise detection and handling
- Cache management
- Data consistency analysis
- Dynamic interval suggestions
- Advanced clustering with monitoring
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortClusterAnalyzer:
    def __init__(self, n_clusters: int = 5):
        self.db = DatabaseManager()
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.consistency_cache = {}
        self.noise_threshold = 3.0  # Z-score threshold για outliers
        self.cache_ttl = timedelta(hours=1)
        self.cache_timestamp = {}

    def detect_data_noise(self, df: pd.DataFrame) -> Dict:
        """Detecting noise in data with advanced statistical measures"""
        noise_report = {
            'outliers': defaultdict(list),
            'timestamp_anomalies': defaultdict(list),
            'missing_data': defaultdict(int),
            'data_quality_score': 1.0
        }

        # Detecting outliers with Robust Z-score
        numeric_cols = ['vessels_in_port', 'total_calls', 'total_activity']
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            if mad == 0:
                continue
                
            modified_z = 0.6745 * (df[col] - median) / mad
            outliers = df[modified_z.abs() > self.noise_threshold]['port_name'].tolist()
            noise_report['outliers'][col] = outliers

        # Time consistency analysis
        if 'timestamp' in df.columns:
            df['time_diff'] = pd.to_datetime(df['timestamp']).diff().dt.total_seconds()
            time_std = df['time_diff'].std()
            if time_std > 3600:
                noise_report['timestamp_anomalies']['irregular_intervals'] = df[
                    df['time_diff'] > 3 * time_std]['port_name'].tolist()

        # Percentage of missing values
        missing_percent = df.isna().mean().mean()
        noise_report['data_quality_score'] = 1.0 - missing_percent

        return noise_report

    @staticmethod
    def handle_noise(df: pd.DataFrame, noise_report: Dict) -> pd.DataFrame:
        """Advanced noise handling with multiple strategies"""
        # Winsorization for outliers
        for col, ports in noise_report['outliers'].items():
            if ports:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df[col] = np.clip(df[col], q1 - 1.5*iqr, q3 + 1.5*iqr)

        # Time correction
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp').sort_index()
            df = df.interpolate(method='time').reset_index()

        # Dynamic imputation
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('unknown')
                else:
                    imputer = SimpleImputer(strategy='mean')
                    df[col] = imputer.fit_transform(df[[col]])

        return df

    def analyze_data_consistency(self, port_name: str, data_type: str, days: int = 30) -> Dict:
        """data sufficiency checking and caching"""
        cache_key = (port_name, data_type, days)
        if cache_key in self.consistency_cache:
            return self.consistency_cache[cache_key]

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            data = self.db.get_port_data(port_name, data_type, start_date, end_date)
            
            if not data:
                result = {'status': 'no_data'}
                self.consistency_cache[cache_key] = result
                return result
                
            df = pd.DataFrame(data)
            if len(df) < 3:  # Minimum registration check
                result = {'status': 'insufficient_data'}
                self.consistency_cache[cache_key] = result
                return result
                
            # Consistency analysis
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            intervals = df['timestamp'].diff()
            
            metrics = {
                'status': 'ok',
                'min_interval': intervals.min(),
                'max_interval': intervals.max(),
                'median_interval': intervals.median(),
                'std_interval': intervals.std(),
                'missing_periods': self._find_missing_periods(df['timestamp']),
                'update_consistency': self._calculate_consistency_score(intervals)
            }
            
            self.consistency_cache[cache_key] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing consistency for {port_name}: {e}")
            return {'status': 'error', 'message': str(e)}

    @staticmethod
    def _find_missing_periods(timestamps: pd.Series) -> List[Dict]:
        """Identifying blank periods in data"""
        expected_interval = timedelta(hours=1)
        gaps = []
        
        for i in range(len(timestamps) - 1):
            interval = timestamps.iloc[i+1] - timestamps.iloc[i]
            if interval > expected_interval * 2:
                gaps.append({
                    'start': timestamps.iloc[i],
                    'end': timestamps.iloc[i+1],
                    'duration': interval
                })
        
        return gaps

    @staticmethod
    def _calculate_consistency_score(intervals: pd.Series) -> float:
        """Consistency score calculation"""
        if intervals.empty:
            return 0
            
        cv = intervals.std() / intervals.mean() if intervals.mean() != 0 else float('inf')
        return 1 / (1 + cv)

    def collect_port_data(self) -> pd.DataFrame:
        """Collect and prepare data for clustering"""
        logger.info("Collecting port data for clustering")
        port_data = []

        # Get all unique port names from existing collections
        port_names = set()
        for collection in self.db.db.list_collection_names():
            try:
                if '_' in collection:
                    parts = collection.split('_')
                    if len(parts) >= 2 and parts[-1] in ['in_port', 'port_calls', 'expected']:
                        port_name = '_'.join(parts[:-1])
                        port_names.add(port_name)
            except Exception as e:
                logger.warning(f"Invalid collection name format: {collection}. Error: {e}")
                continue

        if not port_names:
            logger.error("No valid port collections found in database")
            return pd.DataFrame()

        logger.info(f"Found {len(port_names)} unique ports")

        for port_name in port_names:
            try:
                # Get latest data for each type
                in_port = self.db.get_latest_port_data(port_name, "in_port")
                port_calls = self.db.get_latest_port_data(port_name, "port_calls")
                expected = self.db.get_latest_port_data(port_name, "expected")

                if not any([in_port, port_calls, expected]):
                    logger.warning(f"No data found for port {port_name}")
                    continue

                # Calculate metrics
                vessels_in_port = len(in_port.get('vessels', [])) if in_port else 0
                port_calls_list = port_calls.get('vessels', []) if port_calls else []
                expected_count = len(expected.get('vessels', [])) if expected else 0

                # Calculate vessel type distributions
                vessel_types = defaultdict(int)
                if in_port and 'vessels' in in_port:
                    for vessel in in_port['vessels']:
                        try:
                            v_type = vessel.get('vessel_type', 'unknown')
                            vessel_types[v_type] += 1
                        except Exception as e:
                            logger.warning(f"Error processing vessel in {port_name}: {e}")
                            continue

                # Calculate arrivals and departures
                arrivals = sum(1 for v in port_calls_list 
                             if isinstance(v, dict) and v.get('event_type') == 'Arrival')
                departures = sum(1 for v in port_calls_list 
                               if isinstance(v, dict) and v.get('event_type') == 'Departure')

                port_data.append({
                    'port_name': port_name,
                    'port_id': port_name.lower(),
                    'vessels_in_port': vessels_in_port,
                    'total_calls': len(port_calls_list),
                    'arrivals': arrivals,
                    'departures': departures,
                    'expected_arrivals': expected_count,
                    'total_activity': arrivals + departures,
                    **dict(vessel_types)
                })

            except Exception as e:
                logger.error(f"Error processing port {port_name}: {str(e)}")
                continue

        df = pd.DataFrame(port_data)
        
        if df.empty:
            logger.error("No valid port data collected")
            return df

        logger.info(f"Successfully collected data for {len(df)} ports")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df

    def perform_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improved clustering process with noise-aware preprocessing"""
        # Pre-processing phase
        noise_report = self.detect_data_noise(df)
        logger.info(f"Data quality score before cleaning: {noise_report['data_quality_score']:.2f}")
        
        df_clean = self.handle_noise(df, noise_report)
        
        # Repeated quality control
        post_cleaning_check = self.detect_data_noise(df_clean)
        logger.info(f"Data quality score after cleaning: {post_cleaning_check['data_quality_score']:.2f}")

        # Select features for clustering
        features = [
            'vessels_in_port', 'total_calls', 'arrivals',
            'departures', 'expected_arrivals', 'total_activity'
        ]

        # Add vessel type columns
        vessel_type_columns = [col for col in df_clean.columns
                             if col not in features + ['port_name', 'port_id']]
        features.extend(vessel_type_columns)

        # Normalize features
        X = self.scaler.fit_transform(df_clean[features])

        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df_clean['cluster'] = kmeans.fit_predict(X)

        # Sort clusters by activity level
        cluster_activity = []
        for i in range(self.n_clusters):
            cluster_df = df_clean[df_clean['cluster'] == i]
            avg_activity = cluster_df['total_activity'].mean()
            cluster_activity.append((i, avg_activity))

        # Remap clusters based on activity
        cluster_map = {old: new for new, (old, _)
                      in enumerate(sorted(cluster_activity, key=lambda x: x[1], reverse=True))}
        df_clean['cluster'] = df_clean['cluster'].map(cluster_map)

        logger.info("Clustering completed")
        return df_clean

    def _get_suggested_interval(self, cluster: int, patterns: Dict = None) -> str:
        """dynamic calculation and control of edge cases"""
        base_intervals = {
            0: timedelta(minutes=15),
            1: timedelta(minutes=30),
            2: timedelta(hours=1),
            3: timedelta(hours=2),
            4: timedelta(hours=4)
        }
        
        # Fallback for missing data
        if not patterns or f'cluster_{cluster}' not in patterns:
            return str(base_intervals.get(cluster, timedelta(hours=4)))
        
        cluster_data = patterns[f'cluster_{cluster}']
        
        # Check for inactive clusters
        if cluster_data.get('avg_activity', 0) == 0:
            return str(timedelta(hours=24))
        
        # Performance optimization: sampling up to 5 ports per cluster
        ports = cluster_data.get('ports', [])[:5]
        consistency_metrics = []
        
        for port_name in ports:
            for data_type in ['in_port', 'port_calls', 'expected']:
                metrics = self.analyze_data_consistency(port_name, data_type)
                if metrics['status'] == 'ok':
                    consistency_metrics.append(metrics)
        
        if not consistency_metrics:
            return str(base_intervals.get(cluster, timedelta(hours=4)))
        
        # Dynamic calculation based on consistency and recent activity
        median_intervals = [
            m['median_interval'].total_seconds() 
            for m in consistency_metrics 
            if 'median_interval' in m
        ]
        consistency_scores = [
            m['update_consistency'] 
            for m in consistency_metrics 
            if 'update_consistency' in m
        ]
        
        if not median_intervals or not consistency_scores:
            return str(base_intervals.get(cluster, timedelta(hours=4)))
        
        # Combining metrics with weights
        weighted_interval = np.average(median_intervals, weights=consistency_scores)
        
        # Adaptation based on temporal patterns
        variation = cluster_data.get('temporal_patterns', {}).get('variation', 0)
        if variation > 0.5:
            weighted_interval *= 0.6  # More often in peaks
        elif variation < 0.2:
            weighted_interval *= 1.5  # More sparse in steady states
        
        # Adjustment based on data quality score
        quality_score = cluster_data.get('data_quality_score', 1.0)
        if quality_score < 0.7:
            weighted_interval *= 0.8  # More frequent sampling if data is noisy
        
        # Setting limits and conversion
        optimal_interval = timedelta(
            seconds=max(min(float(weighted_interval), 86400.0), 900.0)  # 15min - 24h
        )
        
        # Custom rounding to multiples of 15 minutes
        rounded_seconds = 900 * round(optimal_interval.total_seconds() / 900)
        return str(timedelta(seconds=rounded_seconds))

    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze temporal patterns in port activity with consistent array shapes

        Args:
            df: DataFrame with clustered port data

        Returns:
            Dictionary with temporal patterns for each cluster
        """
        patterns = {}

        for cluster in range(self.n_clusters):
            cluster_df = df[df['cluster'] == cluster]

            # Initialize patterns with fixed shapes
            activity_patterns = {
                'hourly': np.zeros((len(cluster_df), 24)),  # 24 hours
                'daily': np.zeros((len(cluster_df), 7))  # 7 days
            }

            for idx, port_name in enumerate(cluster_df['port_name']):
                try:
                    port_calls = self.db.get_port_data(
                        port_name,
                        "port_calls",
                        start_date=datetime.now() - timedelta(days=30)
                    )

                    if port_calls:
                        df_calls = pd.DataFrame(port_calls)
                        df_calls['hour'] = pd.to_datetime(df_calls['timestamp']).dt.hour
                        df_calls['day'] = pd.to_datetime(df_calls['timestamp']).dt.dayofweek

                        # Calculate hourly pattern
                        hourly_counts = df_calls.groupby('hour').size()
                        activity_patterns['hourly'][idx, hourly_counts.index] = hourly_counts

                        # Calculate daily pattern
                        daily_counts = df_calls.groupby('day').size()
                        activity_patterns['daily'][idx, daily_counts.index] = daily_counts

                except Exception as e:
                    logger.error(f"Error analyzing patterns for {port_name}: {e}")
                    continue

            # Calculate patterns only if we have data
            if len(cluster_df) > 0:
                patterns[f'cluster_{cluster}'] = {
                    'hourly_pattern': activity_patterns['hourly'].mean(axis=0).tolist(),
                    'daily_pattern': activity_patterns['daily'].mean(axis=0).tolist(),
                    'variation': float(activity_patterns['hourly'].std()),
                    'sample_size': len(cluster_df)
                }

        return patterns

    def analyze_clusters(self, df: pd.DataFrame) -> Dict:
        """
        Generate detailed analysis of clusters with temporal patterns

        Args:
            df: DataFrame with clustered port data

        Returns:
            Dictionary containing detailed statistics for each cluster
        """
        logger.info("Analyzing clusters")

        # Get temporal patterns for better interval suggestions
        patterns = self.analyze_temporal_patterns(df)

        cluster_stats = {}
        for i in range(self.n_clusters):
            cluster_df = df[df['cluster'] == i]

            # Calculate basic statistics
            stats = {
                'size': len(cluster_df),
                'avg_vessels': cluster_df['vessels_in_port'].mean(),
                'avg_calls': cluster_df['total_calls'].mean(),
                'avg_activity': cluster_df['total_activity'].mean(),
                'ports': cluster_df['port_name'].tolist(),
                'suggested_interval': self._get_suggested_interval(i, patterns)
            }

            # Add vessel type distributions if available
            vessel_type_columns = [col for col in df.columns
                                   if col not in ['port_name', 'port_id', 'cluster',
                                                  'vessels_in_port', 'total_calls',
                                                  'arrivals', 'departures',
                                                  'expected_arrivals', 'total_activity']]

            if vessel_type_columns:
                stats['vessel_type_distribution'] = {
                    col: cluster_df[col].mean()
                    for col in vessel_type_columns
                }

            # Add temporal patterns if available
            if patterns and f'cluster_{i}' in patterns:
                stats['temporal_patterns'] = patterns[f'cluster_{i}']

            cluster_stats[f'cluster_{i}'] = stats

        return cluster_stats

    def save_results(self, df: pd.DataFrame, cluster_stats: Dict):
        """
        Save clustering results and generate visualizations

        Args:
            df: DataFrame with clustering results
            cluster_stats: Dictionary with cluster statistics
        """
        # Save main results
        df.to_csv('clusters/clusters_first/port_clusters_results.csv', index=False)

        # Save individual cluster files
        for i in range(self.n_clusters):
            cluster_df = df[df['cluster'] == i]
            cluster_df.to_csv(f'cluster_{i}_ports.csv', index=False)

        # Save cluster statistics
        stats_df = pd.DataFrame.from_dict(cluster_stats, orient='index')
        stats_df.to_csv('cluster_statistics.csv')

        # Generate visualizations
        self.generate_visualizations(df, cluster_stats)

        logger.info("Results and visualizations saved")

    def generate_visualizations(self, df: pd.DataFrame, cluster_stats: Dict):
        """
        Generate comprehensive visualizations of clustering results

        Args:
            df: DataFrame with clustering results
            cluster_stats: Dictionary with cluster statistics
        """
        plt.style.use('seaborn')

        # Create main figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        # 1. Activity Distribution
        sns.boxplot(x='cluster', y='total_activity', data=df, ax=axes[0, 0])
        axes[0, 0].set_title('Activity Distribution by Cluster')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Total Activity')

        # 2. Vessel Type Distribution
        vessel_types = [col for col in df.columns
                        if col not in ['port_name', 'port_id', 'cluster',
                                       'vessels_in_port', 'total_calls',
                                       'arrivals', 'departures',
                                       'expected_arrivals', 'total_activity']]
        if vessel_types:
            cluster_vessel_types = df.groupby('cluster')[vessel_types].mean()
            sns.heatmap(cluster_vessel_types, annot=True, fmt='.2f',
                        cmap='YlOrRd', ax=axes[0, 1])
            axes[0, 1].set_title('Vessel Type Distribution by Cluster')

        # 3. Temporal Patterns
        for i in range(self.n_clusters):
            if f'cluster_{i}' in cluster_stats and 'temporal_patterns' in cluster_stats[f'cluster_{i}']:
                patterns = cluster_stats[f'cluster_{i}']['temporal_patterns']
                axes[1, 0].plot(patterns['hourly_pattern'],
                                label=f'Cluster {i}')
        axes[1, 0].set_title('Hourly Activity Patterns')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Activity')
        axes[1, 0].legend()

        # 4. Cluster Sizes
        sizes = [stats['size'] for stats in cluster_stats.values()]
        axes[1, 1].pie(sizes, labels=[f'Cluster {i}' for i in range(len(sizes))],
                       autopct='%1.1f%%')
        axes[1, 1].set_title('Cluster Size Distribution')

        # Adjust layout and save
        fig.tight_layout()
        fig.savefig('cluster_analysis.png')
        plt.close(fig)
