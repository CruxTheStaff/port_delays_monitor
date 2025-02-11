import logging
from typing import Any

logger = logging.getLogger(__name__)

# Base URLs for myshiptracking.com
BASE_URLS = {
    'in_port': "https://www.myshiptracking.com/el/inport",
    'expected': "https://www.myshiptracking.com/el/estimate",
    'port_calls': "https://www.myshiptracking.com/el/ports-arrivals-departures"
}

# vessel types based on color
VESSEL_TYPES = {
    'icon9': 'yacht',
    'icon8': 'tanker',
    'icon7': 'cargo',
    'icon6': 'passenger',
    'icon4': 'high_speed',
    'icon3': 'tug_pilot',
    'icon10': 'fishing'
}

async def get_vessel_type(img_element_or_src: str | Any) -> str:
    """Get vessel type from image element or source"""
    try:
        # If we got an element instead of src string
        if hasattr(img_element_or_src, 'get_attribute'):
            img_src = await img_element_or_src.get_attribute('src')
        else:
            img_src = img_element_or_src

        if not img_src:
            logger.debug("No image source provided")
            return 'unknown'

        for icon_id, vessel_type in VESSEL_TYPES.items():
            if icon_id in img_src:
                return vessel_type

        logger.debug(f"Unknown vessel type for image: {img_src}")
        return 'unknown'

    except Exception as e:
        logger.error(f"Error determining vessel type: {e}")
        return 'unknown'