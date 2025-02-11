from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import logging

from ..models.delay_record import DelayRecord, DelayReport
from ..models.types import DelayType, ReportSource, VerificationStatus
from ..database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class VerificationRule(BaseModel):
    """Base class for verification rules"""
    rule_id: str
    description: str
    severity: int  # 1-5, where 5 is most severe

    def verify(self, delay_record: DelayRecord) -> Tuple[bool, str]:
        """Verify delay record against rule"""
        raise NotImplementedError


class ReportConsistencyRule(VerificationRule):
    """Verify consistency between different delay reports"""

    def verify(self, delay_record: DelayRecord) -> Tuple[bool, str]:
        if len(delay_record.reports) < 2:
            return True, "Insufficient reports for consistency check"

        durations = [report.duration for report in delay_record.reports if report.duration]
        if not durations:
            return True, "No duration data available"

        max_duration = max(durations)
        min_duration = min(durations)

        # Check if duration difference exceeds threshold
        if (max_duration - min_duration) > timedelta(hours=1):
            return False, f"Inconsistent durations reported: variance {max_duration - min_duration}"
        return True, "Durations are consistent"


class EvidenceValidityRule(VerificationRule):
    """Verify validity of provided evidence"""

    def verify(self, delay_record: DelayRecord) -> Tuple[bool, str]:
        for report in delay_record.reports:
            if not report.evidence:
                return False, f"No evidence provided for report {report.report_id}"

            for evidence in report.evidence:
                if (datetime.now(UTC) - evidence.timestamp) > timedelta(hours=24):
                    return False, f"Evidence {evidence.source} is too old"

        return True, "All evidence is valid"


class DelayVerifier:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.rules: List[VerificationRule] = self._initialize_rules()

    @staticmethod
    def _initialize_rules() -> List[VerificationRule]:
        """Initialize verification rules"""
        return [
            ReportConsistencyRule(
                rule_id="CONS_001",
                description="Check consistency between delay reports",
                severity=4
            ),
            EvidenceValidityRule(
                rule_id="EVID_001",
                description="Verify evidence validity",
                severity=3
            )
        ]

    async def verify_delay(self, delay_record: DelayRecord) -> Dict:
        """Verify delay record using all rules"""
        verification_results = []
        total_severity = 0
        failed_rules = 0

        for rule in self.rules:
            passed, message = rule.verify(delay_record)
            verification_results.append({
                "rule_id": rule.rule_id,
                "passed": passed,
                "message": message,
                "severity": rule.severity
            })

            if not passed:
                failed_rules += 1
                total_severity += rule.severity

        # Determine verification status
        if failed_rules == 0:
            status = VerificationStatus.VERIFIED
        elif total_severity >= 7:  # High severity threshold
            status = VerificationStatus.DISPUTED
        else:
            status = VerificationStatus.PENDING

        verification_result = {
            "status": status,
            "timestamp": datetime.now(UTC),
            "results": verification_results,
            "failed_rules": failed_rules,
            "total_severity": total_severity
        }

        # Update delay record
        delay_record.verification_status = status
        if status == VerificationStatus.VERIFIED:
            delay_record.verified_duration = self._calculate_verified_duration(delay_record)

        # Save updated record
        self.db.save_delay_record(delay_record)

        return verification_result

    @staticmethod
    def _calculate_verified_duration(self, delay_record: DelayRecord) -> timedelta:
        """Calculate verified delay duration from reports"""
        durations = [report.duration for report in delay_record.reports
                     if report.duration is not None]
        if not durations:
            return timedelta(0)

        # Use median duration as verified duration
        sorted_durations = sorted(durations)
        mid = len(sorted_durations) // 2
        if len(sorted_durations) % 2 == 0:
            return (sorted_durations[mid - 1] + sorted_durations[mid]) / 2
        return sorted_durations[mid]

    async def cross_reference_data(self, delay_record: DelayRecord) -> Dict:
        """Cross-reference delay data with other sources"""
        vessel_call = self.db.get_vessel_call(delay_record.port_call_id)
        if not vessel_call:
            return {"error": "Vessel call not found"}

        # Get other delays for same port and time period
        similar_delays = self.db.get_port_delays(
            delay_record.port_id,
            start_date=delay_record.created_at - timedelta(hours=12),
            end_date=delay_record.created_at + timedelta(hours=12)
        )

        # Analyze patterns
        return {
            "similar_delays_count": len(similar_delays),
            "common_delay_types": self._analyze_delay_types(similar_delays),
            "average_duration": self._calculate_average_duration(similar_delays)
        }

    @staticmethod
    def _analyze_delay_types(delays: List[Dict]) -> Dict[DelayType, int]:
        """Analyze frequency of delay types"""
        type_counts = {}
        for delay in delays:
            delay_type = delay.get('delay_type')
            if delay_type:
                type_counts[delay_type] = type_counts.get(delay_type, 0) + 1
        return type_counts

    @staticmethod
    def _calculate_average_duration(delays: List[Dict]) -> timedelta:
        """Calculate average delay duration"""
        durations = [delay.get('verified_duration') for delay in delays
                     if delay.get('verified_duration')]
        if not durations:
            return timedelta(0)
        return sum(durations, timedelta()) / len(durations)