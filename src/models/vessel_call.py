from datetime import datetime, UTC
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from bson import ObjectId

from .types import PortOperationType
from .delay_record import DelayRecord


class VesselDetails(BaseModel):
    """Essential vessel information for port calls"""
    vessel_id: str
    name: str
    type: str
    length: float
    draft: float
    cargo_type: Optional[str] = None
    cargo_quantity: Optional[float] = None


class OperationSchedule(BaseModel):
    """Schedule for a specific port operation"""
    operation_type: PortOperationType
    scheduled_start: datetime
    scheduled_end: datetime
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    status: str = "scheduled"  # scheduled, in_progress, completed, delayed
    notes: Optional[str] = None


class VesselCall(BaseModel):
    """Represents a vessel's visit to a port"""
    call_id: str = Field(default_factory=lambda: str(ObjectId()))
    vessel: VesselDetails
    port_id: str
    berth_id: Optional[str] = None

    # Arrival/Departure times
    eta: datetime
    etd: datetime
    actual_arrival: Optional[datetime] = None
    actual_departure: Optional[datetime] = None

    # Operations schedule
    operations: List[OperationSchedule] = Field(default_factory=list)

    # Status tracking
    status: str = "scheduled"  # scheduled, arrived, in_progress, completed
    delays: List[DelayRecord] = Field(default_factory=list)

    # Additional info
    pilot_required: bool = True
    tug_required: bool = True
    special_requirements: Optional[Dict] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
