# vessel types based on color

"""Constants and configurations for vessel types"""

VESSEL_TYPES = {
    'icon3': 'passenger',    # Κανονικά επιβατηγά
    'icon4': 'high_speed',   # Ταχύπλοα
    'icon6': 'tanker',       # Δεξαμενόπλοια
    'icon7': 'cargo'         # Φορτηγά/Bulk
}

# URLs for Piraeus port
BASE_URLS = {
    'in_port': "https://www.myshiptracking.com/el/inport?pid=445",
    'expected': "https://www.myshiptracking.com/el/estimate?pid=445",
    'port_calls': "https://www.myshiptracking.com/el/ports-arrivals-departures/?pid=445"
}