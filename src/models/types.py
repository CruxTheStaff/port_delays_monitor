from enum import Enum

class DelayType(Enum):
    BERTH_UNAVAILABLE = "berth_unavailable"    # No available berth
    STAFF_SHORTAGE = "staff_shortage"          # Lack of port personnel
    EQUIPMENT_FAILURE = "equipment_failure"     # Equipment malfunction
    DOCUMENTATION = "documentation"            # Documentation/customs delays
    WEATHER = "weather"                        # Weather-related delays
    CARGO_ISSUES = "cargo_issues"              # Cargo-related problems

class ReportSource(Enum):
    PORT = "port"
    VESSEL = "vessel"
    EXTERNAL = "external"

class VerificationStatus(Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    REJECTED = "rejected"

class PortOperationType(Enum):
    BERTHING = "berthing"          # Berthing process
    UNLOADING = "unloading"        # Cargo unloading
    LOADING = "loading"            # Cargo loading
    DEPARTURE = "departure"        # Departure process