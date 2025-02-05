from datetime import datetime, UTC
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from bson import ObjectId

from .vessel_call import VesselCall


class BerthDetails(BaseModel):
    """Details for a specific berth in the port"""
    berth_id: str = Field(default_factory=lambda: str(ObjectId()))
    name: str
    length: float
    max_draft: float
    cargo_types: Set[str]
    equipment: List[str] = Field(default_factory=list)
    status: str = "available"  # available, occupied, maintenance
    current_vessel: Optional[str] = None
    next_available: Optional[datetime] = None


class PortFacility(BaseModel):
    """Port facility information"""
    facility_id: str = Field(default_factory=lambda: str(ObjectId()))
    name: str
    type: str  # storage, crane, fuel, repair, etc.
    capacity: float
    current_utilization: float = 0
    status: str = "operational"
    maintenance_schedule: List[Dict] = Field(default_factory=list)


class PortCapacity(BaseModel):
    """Port capacity and utilization metrics"""
    max_vessels: int
    current_vessels: int = 0
    waiting_vessels: int = 0
    berth_utilization: float = 0.0
    storage_utilization: float = 0.0
    equipment_utilization: float = 0.0


class Port(BaseModel):
    """Main port model"""
    port_id: str = Field(default_factory=lambda: str(ObjectId()))
    name: str
    location: Dict[str, float]  # lat, lon
    timezone: str

    # Port infrastructure
    berths: List[BerthDetails] = Field(default_factory=list)
    facilities: List[PortFacility] = Field(default_factory=list)
    capacity: PortCapacity

    # Vessel tracking
    active_calls: List[str] = Field(default_factory=list)  # List of active vessel_call_ids
    scheduled_calls: List[str] = Field(default_factory=list)  # List of scheduled vessel_call_ids
    waiting_vessels: List[str] = Field(default_factory=list)  # List of waiting vessel_ids

    # Operational parameters
    operating_hours: Dict[str, Dict[str, str]] = Field(default_factory=dict)  # Day -> {open, close}
    pilot_service: bool = True
    tug_service: bool = True

    # Status tracking
    status: str = "operational"
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def add_berth(self, berth: BerthDetails) -> None:
        """Add new berth to port"""
        self.berths.append(berth)
        self._update_capacity()


    def update_berth_status(self, berth_id: str, status: str,
                           vessel_id: Optional[str] = None) -> None:
        """Update berth status and occupancy"""
        for berth in self.berths:
            if berth.berth_id == berth_id:
                berth.status = status
                berth.current_vessel = vessel_id
                if status == "available":
                    berth.next_available = datetime.now(UTC)
                elif status == "occupied":
                    berth.next_available = None
                break
        self._update_capacity()

    def schedule_vessel_call(self, vessel_call: VesselCall) -> bool:
        """Schedule new vessel call"""
        if self._can_accommodate_vessel(vessel_call):
            self.scheduled_calls.append(vessel_call.call_id)
            self._assign_berth(vessel_call)
            return True
        return False

    def register_vessel_arrival(self, vessel_call_id: str) -> None:
        """Register vessel arrival at port"""
        if vessel_call_id in self.scheduled_calls:
            self.scheduled_calls.remove(vessel_call_id)
            self.active_calls.append(vessel_call_id)
            self._update_capacity()

    def register_vessel_departure(self, vessel_call_id: str) -> None:
        """Register vessel departure from port"""
        if vessel_call_id in self.active_calls:
            self.active_calls.remove(vessel_call_id)
            self._update_capacity()

    def get_available_berths(self, vessel_requirements: Dict) -> List[BerthDetails]:
        """Get list of berths matching vessel requirements"""
        return [
            berth for berth in self.berths
            if (berth.status == "available" and
                berth.length >= vessel_requirements.get('length', 0) and
                berth.max_draft >= vessel_requirements.get('draft', 0) and
                vessel_requirements.get('cargo_type') in berth.cargo_types)
        ]

    def get_port_congestion_status(self) -> Dict:
        """Calculate current port congestion status"""
        capacity = self.capacity
        utilization = capacity.current_vessels / capacity.max_vessels

        return {
            "utilization_rate": utilization,
            "waiting_vessels": len(self.waiting_vessels),
            "berth_availability": self._get_berth_availability(),
            "congestion_level": self._calculate_congestion_level(utilization)
        }

    def _update_capacity(self) -> None:
        """Update port capacity metrics"""
        self.capacity.current_vessels = len(self.active_calls)
        self.capacity.waiting_vessels = len(self.waiting_vessels)

        total_berths = len(self.berths)
        occupied_berths = len([b for b in self.berths if b.status == "occupied"])
        self.capacity.berth_utilization = occupied_berths / total_berths if total_berths > 0 else 0

        self.last_updated = datetime.now(UTC)

    def _can_accommodate_vessel(self, vessel_call: VesselCall) -> bool:
        """Check if port can accommodate vessel"""
        available_berths = self.get_available_berths({
            'length': vessel_call.vessel.length,
            'draft': vessel_call.vessel.draft,
            'cargo_type': vessel_call.vessel.cargo_type
        })
        return len(available_berths) > 0

    def _assign_berth(self, vessel_call: VesselCall) -> None:
        """Assign appropriate berth to vessel"""
        available_berths = self.get_available_berths({
            'length': vessel_call.vessel.length,
            'draft': vessel_call.vessel.draft,
            'cargo_type': vessel_call.vessel.cargo_type
        })
        if available_berths:
            # Assign first available matching berth
            vessel_call.berth_id = available_berths[0].berth_id

    def _get_berth_availability(self) -> Dict:
        """Get detailed berth availability status"""
        return {
            "total": len(self.berths),
            "available": len([b for b in self.berths if b.status == "available"]),
            "occupied": len([b for b in self.berths if b.status == "occupied"]),
            "maintenance": len([b for b in self.berths if b.status == "maintenance"])
        }
    @staticmethod
    def _calculate_congestion_level(utilization: float) -> str:
        """Calculate port congestion level based on utilization"""
        if utilization < 0.5:
            return "LOW"
        elif utilization < 0.75:
            return "MEDIUM"
        elif utilization < 0.9:
            return "HIGH"
        return "CRITICAL"

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }
