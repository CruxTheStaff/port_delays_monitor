# src/data_collection/raw_data_validation.py

from datetime import datetime, timedelta
from typing import Dict, List, Set
import logging
from collections import defaultdict
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class RawDataValidator:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def validate_port_data(self, port_name: str, time_window: timedelta) -> Dict:
        """
        Validate raw data consistency for a specific port
        """
        validation_results = {
            'port_name': port_name,
            'time_window': str(time_window),
            'timestamp': datetime.now(),
            'issues': [],
            'stats': defaultdict(int)
        }

        try:
            # Get all data types for the port
            in_port_data = self.db.get_port_data(port_name, "in_port")
            expected_data = self.db.get_port_data(port_name, "expected")
            port_calls = self.db.get_port_data(port_name, "port_calls")

            # Basic data presence check
            if not in_port_data:
                validation_results['issues'].append("No in_port data found")
            if not expected_data:
                validation_results['issues'].append("No expected arrivals data found")
            if not port_calls:
                validation_results['issues'].append("No port calls data found")

            # Check data consistency
            for i in range(len(in_port_data) - 1):
                current = in_port_data[i]
                next_data = in_port_data[i + 1]

                # Track vessels that disappeared without port_calls record
                current_vessels = {v['mmsi'] for v in current['vessels']}
                next_vessels = {v['mmsi'] for v in next_data['vessels']}

                disappeared = current_vessels - next_vessels
                if disappeared:
                    # Check if we have departure records
                    departures = {
                        v['mmsi'] for v in port_calls
                        if v['event_type'] == 'Departure' and
                           current['timestamp'] <= v['timestamp'] <= next_data['timestamp']
                    }

                    missing = disappeared - departures
                    if missing:
                        validation_results['issues'].append(
                            f"Vessels disappeared without departure record: {missing}"
                        )
                        validation_results['stats']['unexplained_disappearances'] += len(missing)

            # Check expected arrivals vs actual
            for expected in expected_data:
                vessel = expected['vessels']
                # Look for matching arrival in port_calls
                found = False
                for call in port_calls:
                    if (call['mmsi'] == vessel['mmsi'] and
                            call['event_type'] == 'Arrival' and
                            abs(call['timestamp'] - vessel['eta']) < timedelta(hours=24)):
                        found = True
                        break

                if not found:
                    validation_results['stats']['missing_expected_arrivals'] += 1

            # Calculate statistics
            validation_results['stats'].update({
                'total_records': len(in_port_data),
                'unique_vessels': len({v['mmsi'] for d in in_port_data for v in d['vessels']}),
                'total_port_calls': len(port_calls),
                'data_completeness': len(in_port_data) / (time_window.total_seconds() / 3600)  # records per hour
            })

        except Exception as e:
            validation_results['issues'].append(f"Validation error: {str(e)}")

        return validation_results

    def check_cross_port_consistency(self, time_window: timedelta) -> Dict:
        """
        Check for vessels appearing in multiple ports simultaneously
        """
        results = {
            'timestamp': datetime.now(),
            'time_window': str(time_window),
            'issues': [],
            'stats': defaultdict(int)
        }

        try:
            # Get all ports data
            all_ports = self.db.get_all_ports()
            vessel_locations = defaultdict(list)

            for port in all_ports:
                in_port_data = self.db.get_port_data(port['name'], "in_port")

                for record in in_port_data:
                    for vessel in record['vessels']:
                        vessel_locations[vessel['mmsi']].append({
                            'port': port['name'],
                            'timestamp': record['timestamp']
                        })

            # Check for simultaneous appearances
            for mmsi, locations in vessel_locations.items():
                sorted_locs = sorted(locations, key=lambda x: x['timestamp'])

                for i in range(len(sorted_locs) - 1):
                    if (sorted_locs[i + 1]['timestamp'] - sorted_locs[i]['timestamp']
                            < timedelta(hours=1) and
                            sorted_locs[i]['port'] != sorted_locs[i + 1]['port']):
                        results['issues'].append(
                            f"Vessel {mmsi} appeared in {sorted_locs[i]['port']} and "
                            f"{sorted_locs[i + 1]['port']} within one hour"
                        )
                        results['stats']['suspicious_movements'] += 1

            results['stats']['total_vessels'] = len(vessel_locations)

        except Exception as e:
            results['issues'].append(f"Cross-port validation error: {str(e)}")

        return results


def main():
    """Run validation on collected data"""
    db = DatabaseManager()
    validator = RawDataValidator(db)

    # Run validation for last 24 hours
    time_window = timedelta(hours=24)

    # Validate individual ports
    for port in db.get_all_ports():
        results = validator.validate_port_data(port['name'], time_window)
        if results['issues']:
            logger.warning(f"Validation issues found for {port['name']}:")
            for issue in results['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info(f"No issues found for {port['name']}")

    # Check cross-port consistency
    cross_port_results = validator.check_cross_port_consistency(time_window)
    if cross_port_results['issues']:
        logger.warning("Cross-port validation issues found:")
        for issue in cross_port_results['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("No cross-port consistency issues found")


if __name__ == "__main__":
    main()