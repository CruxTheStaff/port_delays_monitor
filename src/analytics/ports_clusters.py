"""
Port Clustering Analysis Module V2
Compatible with current db_manager implementation
"""
from kneed import KneeLocator
import logging
from typing import Dict, Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from src.database.db_manager import DatabaseManager
import math
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from datetime import datetime
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortClusterAnalyzer:
    def __init__(self, n_clusters: Optional[int]=None):
        self.db = DatabaseManager()
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.results_dir = self._setup_results_dir()
        self.features = [
            'vessels_in_port',
            'total_calls',
            'arrivals',
            'departures',
            'expected_arrivals',
            'total_activity'
        ]

    @staticmethod
    def _log_cluster_analysis(cluster_stats: Dict):
        """Enhanced logging of cluster analysis"""
        for cluster, stats in cluster_stats.items():
            logger.info(f"\n=== {cluster.upper()} ===")
            logger.info(f"Ports: {stats['size']}")
            logger.info(f"Average Activity: {stats['avg_activity']:.2f}")
            logger.info(f"Suggested Interval: {stats['suggested_interval']} minutes")
            logger.info(f"Peak Ratio: {stats['peak_ratio']:.2f}")
            logger.info(f"Trend: {stats['trend']:.3f}")
            logger.info("Sample ports: " + ", ".join(stats['ports'][:5]))

    # retrieval from database
    def collect_port_data(self) -> pd.DataFrame:
        """Collect and prepare data for clustering"""
        logger.info("Collecting port data for clustering")
        port_data = []

        # Get all ports
        all_ports = self.db.get_all_ports()
        logger.info(f"Found {len(all_ports)} ports in database")

        for port in all_ports:
            try:
                port_name = port['name']
                port_id = port['port_id']

                # Get latest data for each type
                in_port = self.db.get_latest_port_data(port_name, "in_port")
                port_calls = self.db.get_latest_port_data(port_name, "port_calls")
                expected = self.db.get_latest_port_data(port_name, "expected")

                # Calculate metrics
                vessels_in_port = len(in_port.get('vessels', [])) if in_port else 0
                port_calls_list = port_calls.get('vessels', []) if port_calls else []
                expected_count = len(expected.get('vessels', [])) if expected else 0

                # Calculate arrivals and departures
                arrivals = sum(1 for v in port_calls_list 
                             if isinstance(v, dict) and v.get('event_type') == 'Arrival')
                departures = sum(1 for v in port_calls_list 
                               if isinstance(v, dict) and v.get('event_type') == 'Departure')

                port_data.append({
                    'port_name': port_name,
                    'port_id': port_id,
                    'vessels_in_port': vessels_in_port,
                    'total_calls': len(port_calls_list),
                    'arrivals': arrivals,
                    'departures': departures,
                    'expected_arrivals': expected_count,
                    'total_activity': arrivals + departures
                })

            except Exception as e:
                logger.error(f"Error processing port {port.get('name', 'unknown')}: {e}")
                continue

        df = pd.DataFrame(port_data)
        
        if df.empty:
            logger.error("No port data collected")
            return df

        logger.info(f"Successfully collected data for {len(df)} ports")
        return df

    # cleaning and normalization
    def _normalize_features(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize features using StandardScaler"""
        df_clean = df.copy()
        for feature in self.features:
            df_clean[feature] = df_clean[feature].fillna(0)
        return self.scaler.fit_transform(df_clean[self.features])

    # number of clusters decision-making
    def _evaluate_clustering_methods(self, X: np.ndarray, max_clusters: int = 10) -> Dict:
        """
        Enhanced clustering evaluation with advanced metrics and validation
        """
        results = {
            'elbow': {'method': 'KMeans', 'metrics': []},
            'silhouette': {'method': 'KMeans', 'metrics': []},
            'dbscan': {'method': 'Density-Based', 'metrics': []},
            'performance': {},
            'validation': {}
        }

        # Fit KMeans models
        kmeans_models = []
        inertias = []
        for k in range(1, max_clusters + 1):
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(X)
            kmeans_models.append(model)
            inertias.append(model.inertia_)

        # Automatic "elbow" detection with intelligent algorithm
        knee = KneeLocator(
            range(1, max_clusters + 1),
            inertias,
            curve='convex',
            direction='decreasing'
        )
        elbow_k = knee.elbow or 3  # Fallback if no output

        results['elbow'].update({
            'optimal_k': elbow_k,
            'inertias': inertias,
            'knee_point': knee.elbow,
        })

        # 2. Silhouette Analysis
        silhouette_scores = []
        sample_analysis = []  # for visualization

        for k in range(2, max_clusters + 1):
            labels = kmeans_models[k - 1].labels_
            score = silhouette_score(X, labels)
            samples = silhouette_samples(X, labels)

            cluster_stats = []
            for i in range(k):
                cluster_samples = samples[labels == i]
                cluster_stats.append({
                    'cluster': i,
                    'mean_silhouette': cluster_samples.mean(),
                    'std_silhouette': cluster_samples.std(),
                    'min_silhouette': cluster_samples.min(),
                    'size': len(cluster_samples)
                })

            silhouette_scores.append(score)
            sample_analysis.append(cluster_stats)

        # Find the optimal k with filter for stability
        smoothed_scores = pd.Series(silhouette_scores).rolling(2).mean().fillna(0).tolist()
        silhouette_k = np.argmax(smoothed_scores) + 2  # +2 because it starts from k=2

        results['silhouette'].update({
            'optimal_k': silhouette_k,
            'scores': silhouette_scores,
            'sample_analysis': sample_analysis,
            'smoothed_scores': smoothed_scores
        })

        # DBSCAN with optimal parameter search
        from sklearn.neighbors import NearestNeighbors
        from sklearn.model_selection import ParameterGrid

        # Automatic eps range calculation
        nn = NearestNeighbors(n_neighbors=2).fit(X)
        distances, _ = nn.kneighbors(X)
        distances = np.sort(distances[:, 1])
        eps_grid = np.linspace(distances.mean(), distances[-int(0.02 * len(distances))], 20)

        # Search in parameter space
        param_grid = ParameterGrid({
            'eps': eps_grid,
            'min_samples': [3, 5, 7]
        })

        best_dbscan = {'silhouette': -1}
        for params in param_grid:
            dbscan = DBSCAN(**params)
            labels = dbscan.fit_predict(X)

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

            if 1 < n_clusters < 10:
                score = silhouette_score(X, labels) if len(unique_labels) > 1 else -1
                if score > best_dbscan['silhouette']:
                    best_dbscan = {
                        **params,
                        'n_clusters': n_clusters,
                        'silhouette': score,
                        'noise_ratio': np.sum(labels == -1) / len(labels)
                    }

        results['dbscan'].update(best_dbscan)

        # validation metrics
        for method in ['elbow', 'silhouette']:
            k = results[method]['optimal_k']
            labels = kmeans_models[k - 1].labels_
            results['validation'][method] = {
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels)
            }

        # 5. Recommendation with ensemble approach
        final_k = self._ensemble_cluster_selection(results)
        results['recommendation'] = {
            'k': final_k,
            'method': self._select_best_method(results),
            'confidence': self._calculate_confidence(results)
        }

        # Visualization if distances are provided
       # if distances is not None:
        #    self._advanced_visualization(results, distances)

        return results

    @staticmethod
    def _select_best_method(results: Dict) -> str:
        """Select best clustering method based on validation metrics"""
        method_scores = {}

        for method in ['elbow', 'silhouette']:
            ch_score = results['validation'][method]['calinski_harabasz']
            db_score = results['validation'][method]['davies_bouldin']
            method_scores[method] = ch_score / (1 + db_score)

        # Add DBSCAN if it found reasonable clusters
        if results['dbscan']['n_clusters'] > 1:
            method_scores['dbscan'] = results['dbscan']['silhouette'] * 100

        return max(method_scores.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _calculate_confidence(results: Dict) -> float:
        """Calculate confidence in clustering recommendation"""
        factors = []

        # Agreement between methods
        k_values = [
            results['elbow']['optimal_k'],
            results['silhouette']['optimal_k'],
            results['dbscan']['n_clusters']
        ]
        k_std = np.std(k_values)
        agreement_score = 1 / (1 + k_std)
        factors.append(agreement_score)

        # Silhouette stability
        sil_scores = results['silhouette']['scores']
        sil_stability = 1 - np.std(sil_scores) / np.mean(sil_scores)
        factors.append(sil_stability)

        # Validation metrics
        for method in ['elbow', 'silhouette']:
            ch_score = results['validation'][method]['calinski_harabasz']
            db_score = results['validation'][method]['davies_bouldin']
            validation_score = ch_score / (1 + db_score) / 1000  # Normalize
            factors.append(validation_score)

        # DBSCAN quality if applicable
        if results['dbscan']['n_clusters'] > 1:
            dbscan_quality = 1 - results['dbscan'].get('noise_ratio', 0)
            factors.append(dbscan_quality)

        return float(np.mean(factors))

    @staticmethod
    def _ensemble_cluster_selection(results: Dict) -> int:
        """ Combination of results based on statistical significance"""
        scores = {
            'elbow': results['validation']['elbow']['calinski_harabasz'],
            'silhouette': results['silhouette']['smoothed_scores'][results['silhouette']['optimal_k'] - 2],
            'dbscan': results['dbscan']['silhouette'] * 10  # Scale normalization
        }

        # Weights based on methods' performance
        weights = {
            'elbow': 0.3,
            'silhouette': 0.5,
            'dbscan': 0.2
        }

        # Conversion to probabilities
        total = sum(scores.values())
        probabilities = {k: (v / total) * weights[k] for k, v in scores.items()}

        # Selecting k with the largest weighted probability
        candidates = {
            results['elbow']['optimal_k']: probabilities['elbow'],
            results['silhouette']['optimal_k']: probabilities['silhouette'],
            results['dbscan']['n_clusters']: probabilities['dbscan']
        }

        return max(candidates, key=candidates.get)

    def _calculate_optimal_interval(self, cluster_data: pd.DataFrame) -> Union[int, str]:
        """
        Calculates optimal scraping interval using multifactor analysis

        Args:
            cluster_data: DataFrame containing historical port activity data

        Returns:
            Optimal interval in minutes (multiple of 15) or 'adaptive' mode recommendation
        """
        try:
            # Calculate key metrics
            avg_activity = cluster_data['total_activity'].mean()
            activity_std = cluster_data['total_activity'].std()
            peak_ratio = cluster_data['total_activity'].max() / (avg_activity + 1e-9)
            trend = self._calculate_activity_trend(cluster_data)

            # Handle zero activity case
            if avg_activity < 1:
                return "24 hours"  # Minimal monitoring

            # Core calculation using exponential scaling
            activity_factor = math.log(avg_activity ** 2 + 1)
            variability_factor = 1 + math.erf(activity_std / (avg_activity + 1e-9))
            trend_factor = 1 + abs(trend)  # Magnitude of trend regardless of direction

            # Combine factors with weights
            weighted_score = (
                    0.6 * activity_factor +
                    0.25 * variability_factor +
                    0.15 * trend_factor
            )

            # Dynamic interval mapping
            interval_ranges = {
                15: (6.0, math.inf),  # Ultra-high activity
                30: (4.0, 6.0),
                60: (2.5, 4.0),
                120: (1.5, 2.5),
                240: (-math.inf, 1.5)  # Low activity
            }

            # Find matching interval
            for interval, (lower, upper) in interval_ranges.items():
                if lower <= weighted_score < upper:
                    # Apply peak ratio modifier
                    if peak_ratio > 3:
                        return max(15, interval // 2)
                    return interval

            # Fallback for edge cases
            return 240 if weighted_score < 1 else 15

        except Exception as e:
            logger.error(f"Interval calculation failed: {e}")
            return 240  # Fallback to safe default

    def _get_suggested_interval(self, cluster_data: pd.DataFrame) -> str:
        """Dynamic interval suggestion based on cluster analysis"""
        interval = self._calculate_optimal_interval(cluster_data)
        return f"{interval} minutes" if isinstance(interval, int) else interval

    @staticmethod
    def _calculate_activity_trend(cluster_data: pd.DataFrame) -> float:
        """Calculates activity trend using simple linear regression"""
        try:
            x = np.arange(len(cluster_data))
            y = cluster_data['total_activity'].values
            slope = np.polyfit(x, y, 1)[0]
            return float(slope / (np.mean(y) + 1e-9))  # Normalized trend
        except (ValueError, np.linalg.LinAlgError) as e:
            # ValueError: for NaN or invalid data
            # LinAlgError: polyfit related errors
            logger.warning(f"Could not calculate trend: {e}")
            return 0.0
        except KeyError as e:
            # if 'total_activity' column is missing
            logger.error(f"Missing required column: {e}")
            return 0.0
        except Exception as e:
            # For other unexpected errors
            logger.error(f"Unexpected error in trend calculation: {e}")
            return 0.0

    def perform_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform clustering analysis"""
        logger.info("Starting clustering analysis")

        # Normalize features
        X = self._normalize_features(df)
        logger.info("Features normalized")

        # Calculate optimal clusters if not specified
        if self.n_clusters is None:
            logger.info("Determining optimal number of clusters...")
            evaluation_results = self._evaluate_clustering_methods(X)
            self.n_clusters = evaluation_results['recommendation']['k']
            logger.info(f"Determined optimal number of clusters: {self.n_clusters}")
        else:
            logger.info(f"Using specified number of clusters: {self.n_clusters}")

        # Perform clustering
        logger.info("Performing KMeans clustering...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)

        # Sort clusters by activity
        logger.info("Sorting clusters by activity level...")
        cluster_activity = [
            (i, df[df['cluster'] == i]['total_activity'].mean())
            for i in range(self.n_clusters)
        ]

        # Remap clusters
        cluster_map = {old: new for new, (old, _)
                       in enumerate(sorted(cluster_activity, key=lambda x: x[1], reverse=True))}
        df['cluster'] = df['cluster'].map(cluster_map)

        logger.info("Clustering completed")
        return df

    def analyze_clusters(self, df: pd.DataFrame) -> Dict:
        """Generate detailed analysis of clusters"""
        logger.info("Analyzing clusters")

        cluster_stats = {}
        for i in range(self.n_clusters):
            cluster_df = df[df['cluster'] == i]

            # Calculate optimal interval
            optimal_interval = self._get_suggested_interval(cluster_df)

            stats = {
                'size': len(cluster_df),
                'avg_vessels': cluster_df['vessels_in_port'].mean(),
                'avg_calls': cluster_df['total_calls'].mean(),
                'avg_activity': cluster_df['total_activity'].mean(),
                'ports': cluster_df['port_name'].tolist(),
                'suggested_interval': optimal_interval,
                'peak_ratio': cluster_df['total_activity'].max() /
                              (cluster_df['total_activity'].mean() + 1e-9),
                'trend': self._calculate_activity_trend(cluster_df)
            }
            cluster_stats[f'cluster_{i}'] = stats

        return cluster_stats

    def generate_visualizations(self, df: pd.DataFrame, cluster_stats: Dict):
        """Generate visualizations of clustering results"""
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Activity Distribution
        axes[0, 0].boxplot([df[df['cluster'] == i]['total_activity'] 
                           for i in range(self.n_clusters)])
        axes[0, 0].set_title('Activity Distribution by Cluster')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Total Activity')
        
        # 2. Average Metrics by Cluster
        metrics_df = df.groupby('cluster')[self.features].mean()
        im = axes[0, 1].imshow(metrics_df.values, aspect='auto')
        plt.colorbar(im, ax=axes[0, 1])
        axes[0, 1].set_title('Average Metrics by Cluster')
        axes[0, 1].set_yticks(range(len(metrics_df.index)))
        axes[0, 1].set_yticklabels([f'Cluster {i}' for i in metrics_df.index])
        axes[0, 1].set_xticks(range(len(metrics_df.columns)))
        axes[0, 1].set_xticklabels(self.features, rotation=45)
        
        # 3. Cluster Sizes
        sizes = [stats['size'] for stats in cluster_stats.values()]
        axes[1, 0].pie(sizes, labels=[f'Cluster {i}' for i in range(len(sizes))],
                       autopct='%1.1f%%')
        axes[1, 0].set_title('Cluster Size Distribution')
        
        # 4. Activity vs Vessels
        scatter = axes[1, 1].scatter(df['vessels_in_port'], df['total_activity'],
                                   c=df['cluster'], cmap='viridis')
        axes[1, 1].set_xlabel('Vessels in Port')
        axes[1, 1].set_ylabel('Total Activity')
        axes[1, 1].set_title('Activity vs Vessels by Cluster')
        plt.colorbar(scatter, ax=axes[1, 1])
        
        fig.tight_layout()
        fig.savefig('cluster_analysis.png')
        plt.close(fig)

    @staticmethod
    def _setup_results_dir() -> Path:
        """Setup directory structure for results"""
        # Base directory for all cluster analyses
        base_dir = Path(__file__).parent / 'clusters'

        # Create timestamp-based directory for this analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = base_dir / timestamp

        # Create directories if they don't exist
        results_dir.mkdir(parents=True, exist_ok=True)

        return results_dir

    def save_results(self, df: pd.DataFrame, cluster_stats: Dict):
        """Save clustering results and generate visualizations"""
        # Save main results
        df.to_csv(self.results_dir / 'port_clusters_results.csv', index=False)

        # Save individual cluster files
        for i in range(self.n_clusters):
            cluster_df = df[df['cluster'] == i]
            cluster_df.to_csv(self.results_dir / f'cluster_{i}_ports.csv', index=False)

        # Save cluster statistics
        stats_df = pd.DataFrame.from_dict(cluster_stats, orient='index')
        stats_df.to_csv(self.results_dir / 'cluster_statistics.csv')

        # Generate visualizations
        self.generate_visualizations(df, cluster_stats)

        # Save summary report
        self._save_summary_report(cluster_stats)

        logger.info(f"Results saved in: {self.results_dir}")

    def _save_summary_report(self, cluster_stats: Dict):
        """Save a text summary of the analysis"""
        with open(self.results_dir / 'analysis_summary.txt', 'w') as f:
            f.write("Port Clusters Analysis\n")
            f.write("=====================\n\n")

            for i in range(self.n_clusters):
                stats = cluster_stats[f'cluster_{i}']
                f.write(f"Cluster {i}\n")
                f.write(f"Number of ports: {stats['size']}\n")
                f.write(f"Average vessels in port: {stats['avg_vessels']:.2f}\n")
                f.write(f"Average daily activity: {stats['avg_activity']:.2f}\n")
                f.write(f"Suggested scraping interval: {stats['suggested_interval']}\n")
                f.write(f"Sample ports: {', '.join(stats['ports'][:5])}\n\n")


def main():
    analyzer = PortClusterAnalyzer(n_clusters=5)

    # Collect and prepare data
    df = analyzer.collect_port_data()
    if df.empty:
        logger.error("No data to analyze")
        return

    # Perform clustering
    df = analyzer.perform_clustering(df)

    # Analyze clusters
    cluster_stats = analyzer.analyze_clusters(df)

    # Save results
    analyzer.save_results(df, cluster_stats)

    # Print summary
    print("\nPort Clusters Analysis")
    print("=====================")
    for i in range(analyzer.n_clusters):
        stats = cluster_stats[f'cluster_{i}']
        print(f"\nCluster {i}")
        print(f"Number of ports: {stats['size']}")
        print(f"Average vessels in port: {stats['avg_vessels']:.2f}")
        print(f"Average daily activity: {stats['avg_activity']:.2f}")
        print(f"Suggested scraping interval: {stats['suggested_interval']}")
        print("Sample ports:", ', '.join(stats['ports'][:5]))


if __name__ == "__main__":
    main()