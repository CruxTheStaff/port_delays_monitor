from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from bson import ObjectId

from .types import DelayType, ReportSource, VerificationStatus, PortOperationType

class DelayEvidence(BaseModel):
    """Supporting evidence for delay verification"""
    source: str
    timestamp: datetime
    evidence_type: str    # e.g., "document", "system_log", "weather_report"
    content: Dict
    metadata: Optional[Dict] = None

class DelayReport(BaseModel):
    """Individual delay report from a single source"""
    report_id: str = Field(default_factory=lambda: str(ObjectId()))
    source: ReportSource
    reported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    delay_type: DelayType
    operation_type: PortOperationType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    description: str
    evidence: List[DelayEvidence] = Field(default_factory=list)
    reporter_details: Dict

class DelayRecord(BaseModel):
    """Main delay record combining all reports and verification data"""
    record_id: str = Field(default_factory=lambda: str(ObjectId()))
    port_call_id: str
    vessel_id: str
    port_id: str
    delay_type: DelayType
    operation_type: PortOperationType
    reports: List[DelayReport] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    verified_duration: Optional[timedelta] = None
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict = Field(default_factory=dict)

    def add_report(self, report: DelayReport) -> None:
        """Add a new delay report and update record status"""
        self.reports.append(report)
        self.updated_at = datetime.now(UTC)
        self._evaluate_verification_status()

    def verify_delay(self, verifier: str, verified_duration: timedelta) -> None:
        """Mark delay as verified with official duration"""
        self.verification_status = VerificationStatus.VERIFIED
        self.verified_duration = verified_duration
        self.verified_by = verifier
        self.verified_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def dispute_delay(self, disputed_by: str, reason: str) -> None:
        """Mark delay as disputed"""
        self.verification_status = VerificationStatus.DISPUTED
        self.metadata['dispute'] = {
            'disputed_by': disputed_by,
            'reason': reason,
            'disputed_at': datetime.now(UTC)
        }
        self.updated_at = datetime.now(UTC)

    def _evaluate_verification_status(self) -> None:
        """Evaluate verification status based on available reports"""
        if len(self.reports) < 2:
            return

        # Check if reports from both port and vessel exist
        sources = {report.source for report in self.reports}
        if ReportSource.PORT in sources and ReportSource.VESSEL in sources:
            # Compare reported durations
            self._compare_report_durations()

    def _compare_report_durations(self) -> None:
        """Compare durations from different sources"""
        port_reports = [r for r in self.reports if r.source == ReportSource.PORT]
        vessel_reports = [r for r in self.reports if r.source == ReportSource.VESSEL]

        if not (port_reports and vessel_reports):
            return

        # Calculate average durations
        port_duration = sum((r.duration or timedelta(0) for r in port_reports), timedelta(0))
        vessel_duration = sum((r.duration or timedelta(0) for r in vessel_reports), timedelta(0))

        # If durations differ significantly, mark as disputed
        if abs((port_duration - vessel_duration).total_seconds()) > 3600:  # 1 hour threshold
            self.verification_status = VerificationStatus.DISPUTED
            self.metadata['duration_discrepancy'] = {
                'port_duration': port_duration,
                'vessel_duration': vessel_duration
            }

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: str(v),
            ObjectId: lambda v: str(v)
        }
