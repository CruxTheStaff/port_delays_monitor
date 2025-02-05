import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from datetime import UTC, datetime

# Direct imports
from src.api.port_api import router as port_router
from src.api.vessel_api import router as vessel_router
from src.config.config_manager import ConfigManager
from src.database.db_manager import DatabaseManager
from src.verification.delay_verifier import DelayVerifier
from src.analytics import DelayAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('port_delays_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize configuration
config_manager = ConfigManager()

# Initialize components
db_manager = DatabaseManager(
    connection_url=config_manager.get_database_config().connection_string
)
delay_verifier = DelayVerifier(db_manager)
delay_analyzer = DelayAnalytics(db_manager)

# Create FastAPI app instance
app = FastAPI(
    title="Port Delays Monitor",
    description="API for monitoring and analyzing port delays",
    version="1.0.0"
)

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info(f"Starting Port Delays Monitor at {datetime.now(UTC)}")
    logger.info("Initializing components...")

    # Store instances in app state
    app_.state.db = db_manager
    app_.state.verifier = delay_verifier
    app_.state.analyzer = delay_analyzer
    app_.state.config = config_manager
    app_.state.startup_time = datetime.now(UTC)

    logger.info("All components initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Port Delays Monitor...")
    db_manager.close()
    logger.info("Shutdown complete")

# Add lifespan handler
app.lifespan = lifespan

# Include routers
app.include_router(port_router)
app.include_router(vessel_router)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC),
        "components": {
            "database": "connected" if db_manager.client else "disconnected",
            "api": "running"
        }
    }

@app.get("/version")
async def version():
    """Get API version"""
    return {
        "version": "1.0.0",
        "environment": config_manager.config.environment,
        "last_startup": app.state.startup_time
    }

if __name__ == "__main__":
    # Get API configuration
    api_config = config_manager.get_api_config()

    # Run application
    uvicorn.run(
        app,
        host=api_config.host,
        port=api_config.port,
        reload=api_config.debug,
        log_level=config_manager.config.logging_level.lower()
    )