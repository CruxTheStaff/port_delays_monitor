from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from ..database.db_manager import DatabaseManager
from ..models.port import Port, BerthDetails
from ..models.vessel_call import VesselCall

router = APIRouter(prefix="/api/ports", tags=["ports"])
db = DatabaseManager()


@router.post("/", response_model=str)
async def create_port(port: Port):
    """Create new port"""
    try:
        port_id = db.save_port(port)
        return port_id
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{port_id}", response_model=Port)
async def get_port(port_id: str):
    """Get port details"""
    port = db.get_port(port_id)
    if not port:
        raise HTTPException(status_code=404, detail="Port not found")
    return port


@router.get("/{port_id}/congestion", response_model=dict)
async def get_port_congestion(port_id: str):
    """Get port congestion status"""
    port_data = db.get_port(port_id)
    if not port_data:
        raise HTTPException(status_code=404, detail="Port not found")

    port = Port(**port_data)
    return port.get_port_congestion_status()


@router.get("/{port_id}/berths", response_model=List[BerthDetails])
async def get_available_berths(
        port_id: str,
        vessel_length: Optional[float] = Query(None),
        vessel_draft: Optional[float] = Query(None),
        cargo_type: Optional[str] = Query(None)
):
    """Get available berths matching criteria"""
    port_data = db.get_port(port_id)
    if not port_data:
        raise HTTPException(status_code=404, detail="Port not found")

    port = Port(**port_data)
    requirements = {
        'length': vessel_length,
        'draft': vessel_draft,
        'cargo_type': cargo_type
    }
    return port.get_available_berths(requirements)


@router.get("/{port_id}/schedule", response_model=List[VesselCall])
async def get_port_schedule(
        port_id: str,
        start_date: Optional[datetime] = Query(None),
        end_date: Optional[datetime] = Query(None)
):
    """Get port schedule within date range"""
    calls = db.get_port_vessel_calls(port_id, start_date, end_date)
    return calls


@router.post("/{port_id}/vessel-calls", response_model=str)
async def schedule_vessel_call(port_id: str, vessel_call: VesselCall):
    """Schedule new vessel call"""
    port_data = db.get_port(port_id)
    if not port_data:
        raise HTTPException(status_code=404, detail="Port not found")

    port = Port(**port_data)
    if not port.schedule_vessel_call(vessel_call):
        raise HTTPException(
            status_code=400,
            detail="Cannot accommodate vessel at specified time"
        )

    return db.save_vessel_call(vessel_call)
