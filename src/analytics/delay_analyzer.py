from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import logging
from pydantic import BaseModel

from ..database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class DelayPattern(BaseModel):
    """Pattern identified in delay data"""
    pattern_type: str
    confidence: float
    frequency: int
    average_duration: timedelta
    impact_score: float
    contributing_factors: Dict[str, float]


class DelayAnalytics:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def analyze_port_delays(self, port_id: str,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict:
        """Analyze delays for specific port"""
        delays = self.db.get_port_delays(port_id, start_date, end_date)

        return {
            "total_delays": len(delays),
            "delay_types": self._analyze_delay_types(delays),
            "temporal_patterns": self._analyze_temporal_patterns(delays),
            "operation_impact": self._analyze_operation_impact(delays),
            "average_durations": self._calculate_average_durations(delays),
            "identified_patterns": self._identify_patterns(delays),
            "recommendations": self._generate_recommendations(delays)
        }

    async def analyze_vessel_delays(self, vessel_id: str) -> Dict:
        """Analyze delays for specific vessel"""
        delays = self.db.get_vessel_delays(vessel_id)

        return {
            "total_delays": len(delays),
            "delay_patterns": self._analyze_delay_types(delays),
            "port_performance": self._analyze_port_performance(delays),
            "efficiency_metrics": self._calculate_efficiency_metrics(delays)
        }

    @staticmethod
    def _analyze_delay_types(delays: List[Dict]) -> Dict:
        """Analyze frequency and impact of different delay types"""
        type_stats = defaultdict(lambda: {
            "count": 0,
            "total_duration": timedelta(0),
            "average_duration": timedelta(0),
            "impact_score": 0.0
        })

        for delay in delays:
            delay_type = delay.get('delay_type')
            if not delay_type:
                continue

            stats = type_stats[delay_type]
            stats["count"] += 1

            duration = delay.get('verified_duration') or timedelta(0)
            stats["total_duration"] += duration
            stats["average_duration"] = stats["total_duration"] / stats["count"]

            stats["impact_score"] = (stats["count"] *
                                   stats["average_duration"].total_seconds() / 3600)

        return dict(type_stats)

    @staticmethod
    def _analyze_temporal_patterns(delays: List[Dict]) -> Dict:
        """Analyze temporal patterns in delays"""
        hourly_distribution = defaultdict(int)
        daily_distribution = defaultdict(int)
        monthly_distribution = defaultdict(int)
        seasonal_patterns = defaultdict(int)

        for delay in delays:
            created_at = delay.get('created_at')
            if not created_at:
                continue

            hourly_distribution[created_at.hour] += 1
            daily_distribution[created_at.strftime('%A')] += 1
            monthly_distribution[created_at.strftime('%B')] += 1

            month = created_at.month
            season = (
                'Winter' if month in [12, 1, 2]
                else 'Spring' if month in [3, 4, 5]
                else 'Summer' if month in [6, 7, 8]
                else 'Fall'
            )
            seasonal_patterns[season] += 1

        return {
            "hourly": dict(hourly_distribution),
            "daily": dict(daily_distribution),
            "monthly": dict(monthly_distribution),
            "seasonal": dict(seasonal_patterns)
        }

    @staticmethod
    def _analyze_operation_impact(delays: List[Dict]) -> Dict:
        """Analyze impact of delays on different operations"""
        operation_impact = defaultdict(lambda: {
            "delay_count": 0,
            "total_duration": timedelta(0),
            "average_duration": timedelta(0),
            "efficiency_impact": 0.0
        })

        for delay in delays:
            operation = delay.get('operation_type')
            if not operation:
                continue

            impact = operation_impact[operation]
            impact["delay_count"] += 1

            duration = delay.get('verified_duration') or timedelta(0)
            impact["total_duration"] += duration
            impact["average_duration"] = (impact["total_duration"] /
                                        impact["delay_count"])

            impact["efficiency_impact"] = (impact["delay_count"] *
                                         impact["average_duration"].total_seconds() / 3600)

        return dict(operation_impact)

    def _calculate_average_durations(self, delays: List[Dict]) -> Dict:
        """Calculate average delay durations by type and operation"""
        duration_stats = {
            "overall": self._calculate_overall_average(delays),
            "by_type": self._calculate_type_averages(delays),
            "by_operation": self._calculate_operation_averages(delays)
        }
        return duration_stats

    def _identify_patterns(self, delays: List[Dict]) -> List[DelayPattern]:
        """Identify significant patterns in delay data"""
        patterns = []

        temporal_clusters = self._find_temporal_clusters(delays)
        for cluster in temporal_clusters:
            patterns.append(DelayPattern(
                pattern_type="temporal_cluster",
                confidence=cluster["confidence"],
                frequency=cluster["frequency"],
                average_duration=cluster["average_duration"],
                impact_score=cluster["impact_score"],
                contributing_factors=cluster["factors"]
            ))

        causal_patterns = self._find_causal_patterns(delays)
        patterns.extend(causal_patterns)

        return patterns

    def _generate_recommendations(self, delays: List[Dict]) -> List[Dict]:
        """Generate recommendations based on delay analysis"""
        recommendations = []

        delay_patterns = self._analyze_delay_types(delays)
        for delay_type, stats in delay_patterns.items():
            if stats["impact_score"] > 10:
                recommendations.append({
                    "type": "high_impact_delay",
                    "delay_type": delay_type,
                    "impact_score": stats["impact_score"],
                    "suggestion": self._get_recommendation_for_delay_type(delay_type)
                })

        operation_impact = self._analyze_operation_impact(delays)
        for operation, impact in operation_impact.items():
            if impact["efficiency_impact"] > 5:
                recommendations.append({
                    "type": "operation_optimization",
                    "operation": operation,
                    "impact": impact["efficiency_impact"],
                    "suggestion": self._get_recommendation_for_operation(operation)
                })

        return recommendations

    @staticmethod
    def _calculate_overall_average(delays: List[Dict]) -> timedelta:
        """Calculate overall average delay duration"""
        durations = [d.get('verified_duration') for d in delays if d.get('verified_duration')]
        if not durations:
            return timedelta(0)
        return sum(durations, timedelta()) / len(durations)

    @staticmethod
    def _calculate_type_averages(delays: List[Dict]) -> Dict[str, timedelta]:
        """Calculate average duration by delay type"""
        type_durations = defaultdict(list)
        for delay in delays:
            if delay.get('verified_duration') and delay.get('delay_type'):
                type_durations[delay['delay_type']].append(delay['verified_duration'])

        return {
            delay_type: sum(durations, timedelta()) / len(durations)
            for delay_type, durations in type_durations.items()
        }

    @staticmethod
    def _calculate_operation_averages(delays: List[Dict]) -> Dict[str, timedelta]:
        """Calculate average duration by operation type"""
        operation_durations = defaultdict(list)
        for delay in delays:
            if delay.get('verified_duration') and delay.get('operation_type'):
                operation_durations[delay['operation_type']].append(
                    delay['verified_duration']
                )

        return {
            op_type: sum(durations, timedelta()) / len(durations)
            for op_type, durations in operation_durations.items()
        }

    @staticmethod
    def _get_recommendation_for_delay_type(delay_type: str) -> str:
        """Get recommendation based on delay type"""
        recommendations = {
            "BERTH_UNAVAILABLE": "Consider implementing a berth allocation system",
            "STAFF_SHORTAGE": "Review staffing levels during peak hours",
            "EQUIPMENT_FAILURE": "Implement preventive maintenance schedule",
            "DOCUMENTATION": "Streamline documentation processes",
            "WEATHER": "Develop contingency plans for adverse weather",
            "CARGO_ISSUES": "Review cargo handling procedures"
        }
        return recommendations.get(delay_type, "Review processes and procedures")

    @staticmethod
    def _get_recommendation_for_operation(operation: str) -> str:
        """Get recommendation based on operation type"""
        recommendations = {
            "BERTHING": "Review berthing procedures and pilot availability",
            "LOADING": "Optimize loading equipment utilization",
            "UNLOADING": "Review unloading process efficiency",
            "DEPARTURE": "Streamline departure procedures"
        }
        return recommendations.get(operation, "Review operation procedures")

    @staticmethod
    def _analyze_port_performance(delays: List[Dict]) -> Dict:
        """Analyze performance metrics for each port"""
        port_performance = defaultdict(lambda: {
            "delay_count": 0,
            "total_duration": timedelta(0),
            "efficiency_score": 0.0,
            "common_issues": defaultdict(int)
        })

        for delay in delays:
            port_id = delay.get('port_id')
            if not port_id:
                continue

            perf = port_performance[port_id]
            perf["delay_count"] += 1

            duration = delay.get('verified_duration') or timedelta(0)
            perf["total_duration"] += duration

            if delay.get('delay_type'):
                perf["common_issues"][delay['delay_type']] += 1

            perf["efficiency_score"] = (
                                               perf["delay_count"] *
                                               perf["total_duration"].total_seconds() / 3600
                                       ) / max(1, len(delays))

        return dict(port_performance)

    def _calculate_efficiency_metrics(self, delays: List[Dict]) -> Dict:
        """Calculate various efficiency metrics"""
        total_delays = len(delays)
        if not total_delays:
            return {
                "delay_frequency": 0,
                "average_delay_duration": timedelta(0),
                "efficiency_score": 0,
                "performance_trend": "N/A"
            }

        total_duration = sum(
            (delay.get('verified_duration') or timedelta(0) for delay in delays),
            timedelta(0)
        )

        return {
            "delay_frequency": total_delays,
            "average_delay_duration": total_duration / total_delays,
            "efficiency_score": self._calculate_efficiency_score(delays),
            "performance_trend": self._analyze_performance_trend(delays)
        }

    def _find_temporal_clusters(self, delays: List[Dict]) -> List[Dict]:
        """Find temporal clusters in delay data"""
        clusters = []
        sorted_delays = sorted(delays, key=lambda x: x.get('created_at', datetime.min))
        current_cluster = []
        cluster_threshold = timedelta(hours=4)

        for delay in sorted_delays:
            if not current_cluster:
                current_cluster.append(delay)
                continue

            time_diff = (delay.get('created_at', datetime.min) -
                         current_cluster[-1].get('created_at', datetime.min))

            if time_diff <= cluster_threshold:
                current_cluster.append(delay)
            else:
                if len(current_cluster) > 2:
                    clusters.append(self._analyze_cluster(current_cluster))
                current_cluster = [delay]

        if len(current_cluster) > 2:
            clusters.append(self._analyze_cluster(current_cluster))

        return clusters

    def _find_causal_patterns(self, delays: List[Dict]) -> List[DelayPattern]:
        """Identify potential causal relationships between delays"""
        patterns = []
        delays_by_type = defaultdict(list)

        for delay in delays:
            if delay.get('delay_type'):
                delays_by_type[delay['delay_type']].append(delay)

        for delay_type, type_delays in delays_by_type.items():
            if len(type_delays) < 3:
                continue

            avg_interval = self._calculate_average_interval(type_delays)

            if avg_interval and avg_interval < timedelta(days=7):
                patterns.append(DelayPattern(
                    pattern_type="sequential",
                    confidence=self._calculate_pattern_confidence(type_delays),
                    frequency=len(type_delays),
                    average_duration=self._calculate_average_duration(type_delays),
                    impact_score=self._calculate_impact_score(type_delays),
                    contributing_factors=self._identify_contributing_factors(type_delays)
                ))

        return patterns

    def _analyze_cluster(self, cluster: List[Dict]) -> Dict:
        """Analyze characteristics of a delay cluster"""
        return {
            "confidence": self._calculate_cluster_confidence(cluster),
            "frequency": len(cluster),
            "average_duration": self._calculate_average_duration(cluster),
            "impact_score": self._calculate_impact_score(cluster),
            "factors": self._identify_contributing_factors(cluster)
        }

    @staticmethod
    def _calculate_cluster_confidence(cluster: List[Dict]) -> float:
        """Calculate confidence score for cluster"""
        if not cluster:
            return 0.0
        return min(0.95, len(cluster) * 0.15)

    @staticmethod
    def _calculate_pattern_confidence(delays: List[Dict]) -> float:
        """Calculate confidence score for pattern"""
        if not delays:
            return 0.0
        return min(0.90, len(delays) * 0.10)

    @staticmethod
    def _calculate_average_interval(delays: List[Dict]) -> Optional[timedelta]:
        """Calculate average time interval between delays"""
        if len(delays) < 2:
            return None

        intervals = []
        sorted_delays = sorted(delays, key=lambda x: x.get('created_at', datetime.min))

        for i in range(1, len(sorted_delays)):
            interval = (sorted_delays[i].get('created_at', datetime.min) -
                        sorted_delays[i - 1].get('created_at', datetime.min))
            intervals.append(interval)

        return sum(intervals, timedelta()) / len(intervals)

    @staticmethod
    def _identify_contributing_factors(delays: List[Dict]) -> Dict[str, float]:
        """Identify factors contributing to delays"""
        factors = defaultdict(int)
        total_delays = len(delays)

        for delay in delays:
            if delay.get('delay_type'):
                factors[f"delay_type_{delay['delay_type']}"] += 1
            if delay.get('operation_type'):
                factors[f"operation_{delay['operation_type']}"] += 1

        return {k: v / total_delays for k, v in factors.items()} if total_delays else {}

    @staticmethod
    def _calculate_average_duration(delays: List[Dict]) -> timedelta:
        """Calculate average duration for a set of delays"""
        if not delays:
            return timedelta(0)

        durations = [
            delay.get('verified_duration') or timedelta(0)
            for delay in delays
        ]
        return sum(durations, timedelta()) / len(durations)

    @staticmethod
    def _calculate_impact_score(delays: List[Dict]) -> float:
        """Calculate impact score based on frequency and duration"""
        if not delays:
            return 0.0

        total_duration = sum(
            (delay.get('verified_duration') or timedelta(0)).total_seconds()
            for delay in delays
        )

        # Impact score calculation:
        # - Considers both frequency (number of delays) and total duration
        # - Normalized by converting to hours
        # - Higher score means bigger impact
        frequency_factor = len(delays)
        duration_factor = total_duration / 3600  # Convert seconds to hours

        return frequency_factor * duration_factor

    def _calculate_efficiency_score(self, delays: List[Dict]) -> float:
        """Calculate efficiency score (0-100) based on delays"""
        if not delays:
            return 100.0  # Perfect score if no delays

        # Calculate base metrics
        total_delays = len(delays)
        avg_duration = self._calculate_average_duration(delays)
        impact_score = self._calculate_impact_score(delays)

        # Factors that reduce efficiency
        delay_penalty = min(total_delays * 2, 40)  # Up to 40 points for frequency
        duration_penalty = min(avg_duration.total_seconds() / 3600 * 5, 40)  # Up to 40 points for duration
        impact_penalty = min(impact_score / 10, 20)  # Up to 20 points for overall impact

        # Calculate final score (100 - penalties)
        efficiency_score = 100 - (delay_penalty + duration_penalty + impact_penalty)

        return max(0.0, efficiency_score)  # Ensure score doesn't go below 0

    def _analyze_performance_trend(self, delays: List[Dict]) -> str:
        """Analyze performance trend over time"""
        if not delays or len(delays) < 2:
            return "Insufficient data"

        # Sort delays by date
        sorted_delays = sorted(delays, key=lambda x: x.get('created_at', datetime.min))

        # Split into two periods for comparison
        mid_point = len(sorted_delays) // 2
        first_period = sorted_delays[:mid_point]
        second_period = sorted_delays[mid_point:]

        # Calculate metrics for each period
        first_metrics = {
            'avg_duration': self._calculate_average_duration(first_period),
            'efficiency': self._calculate_efficiency_score(first_period),
            'impact': self._calculate_impact_score(first_period)
        }

        second_metrics = {
            'avg_duration': self._calculate_average_duration(second_period),
            'efficiency': self._calculate_efficiency_score(second_period),
            'impact': self._calculate_impact_score(second_period)
        }

        # Compare periods
        improvements = 0
        deteriorations = 0

        if second_metrics['avg_duration'] < first_metrics['avg_duration']:
            improvements += 1
        elif second_metrics['avg_duration'] > first_metrics['avg_duration']:
            deteriorations += 1

        if second_metrics['efficiency'] > first_metrics['efficiency']:
            improvements += 1
        elif second_metrics['efficiency'] < first_metrics['efficiency']:
            deteriorations += 1

        if second_metrics['impact'] < first_metrics['impact']:
            improvements += 1
        elif second_metrics['impact'] > first_metrics['impact']:
            deteriorations += 1

        # Determine trend
        if improvements > deteriorations:
            return "Improving"
        elif deteriorations > improvements:
            return "Deteriorating"
        else:
            return "Stable"
