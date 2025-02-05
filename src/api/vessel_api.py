from fastapi import APIRouter, HTTPException, Query
from typing import List
from datetime import datetime, UTC
from ..database.db_manager import DatabaseManager
from ..models.vessel_call import VesselCall
from ..models.delay_record import DelayRecord, DelayReport
from ..models.port import Port

router = APIRouter(prefix="/api/vessels", tags=["vessels"])

# Create database manager instance
db_manager = DatabaseManager()

@router.get("/calls/{call_id}", response_model=VesselCall)
async def get_vessel_call(call_id: str):
    """Get vessel call details"""
    call = db_manager.get_vessel_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Vessel call not found")
    return call

@router.post("/calls/{call_id}/delays", response_model=str)
async def report_delay(call_id: str, delay_report: DelayReport):
    """Report new delay for vessel call"""
    call = db_manager.get_vessel_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Vessel call not found")

    vessel_call = VesselCall(**call)
    delay_record = DelayRecord(
        port_call_id=call_id,
        vessel_id=vessel_call.vessel.vessel_id,
        port_id=vessel_call.port_id,
        delay_type=delay_report.delay_type,
        operation_type=delay_report.operation_type
    )
    delay_record.add_report(delay_report)

    record_id = db_manager.save_delay_record(delay_record)
    vessel_call.add_delay(delay_record)
    db_manager.save_vessel_call(vessel_call)

    return record_id

@router.get("/calls/{call_id}/delays", response_model=List[DelayRecord])
async def get_vessel_delays(call_id: str):
    """Get all delays for vessel call"""
    call = db_manager.get_vessel_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Vessel call not found")

    vessel_call = VesselCall(**call)
    return vessel_call.delays

@router.post("/calls/{call_id}/arrival", response_model=VesselCall)
async def record_vessel_arrival(
    call_id: str,
    arrival_time: datetime = Query(default_factory=lambda: datetime.now(UTC))
):
    """Record vessel arrival and update port status"""
    # Get vessel call
    call = db_manager.get_vessel_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Vessel call not found")

    # Update vessel call
    vessel_call = VesselCall(**call)
    vessel_call.record_arrival(arrival_time)
    db_manager.save_vessel_call(vessel_call)

    # Update port status if available
    port = db_manager.get_port(vessel_call.port_id)
    if port:
        port_obj = Port(**port)
        port_obj.register_vessel_arrival(call_id)
        db_manager.save_port(port_obj)

    return vessel_call

@router.post("/calls/{call_id}/departure", response_model=VesselCall)
async def record_vessel_departure(
    call_id: str,
    departure_time: datetime = Query(default_factory=lambda: datetime.now(UTC))
):
    """Record vessel departure and update port status"""
    # Get vessel call
    call = db_manager.get_vessel_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Vessel call not found")

    # Update vessel call
    vessel_call = VesselCall(**call)
    vessel_call.record_departure(departure_time)
    db_manager.save_vessel_call(vessel_call)

    # Update port status if available
    port = db_manager.get_port(vessel_call.port_id)
    if port:
        port_obj = Port(**port)
        port_obj.register_vessel_departure(call_id)
        db_manager.save_port(port_obj)

    return vessel_call