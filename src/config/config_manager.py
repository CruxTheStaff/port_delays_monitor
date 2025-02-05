from typing import Dict, Any, Optional
import yaml
import json
import os
from pathlib import Path
import logging
from datetime import timedelta
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 27017
    database: str = "port_delays_monitor"
    username: Optional[str] = None
    password: Optional[str] = None
    auth_source: Optional[str] = None

    @property
    def connection_string(self) -> str:
        """Generate MongoDB connection string"""
        if self.username and self.password:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"mongodb://{self.host}:{self.port}"


class VerificationConfig(BaseModel):
    """Verification system configuration"""
    min_reports_required: int = 2
    max_duration_variance: timedelta = timedelta(hours=1)
    evidence_expiry: timedelta = timedelta(hours=24)
    verification_timeout: timedelta = timedelta(minutes=30)
    auto_verify_threshold: float = 0.8


class AnalyticsConfig(BaseModel):
    """Analytics system configuration"""
    analysis_window: timedelta = timedelta(days=30)
    min_data_points: int = 10
    confidence_threshold: float = 0.7
    pattern_detection_window: timedelta = timedelta(days=7)
    cache_duration: timedelta = timedelta(hours=1)


class APIConfig(BaseModel):
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = Field(default_factory=lambda: ["*"])
    rate_limit: int = 100
    rate_limit_period: int = 60


class Config(BaseModel):
    """Main configuration"""
    environment: str = "development"
    database: DatabaseConfig
    verification: VerificationConfig
    analytics: AnalyticsConfig
    api: APIConfig
    logging_level: str = "INFO"


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

    @staticmethod
    def _get_default_config_path() -> str:
        """Get default configuration file path"""
        return os.path.join(
            Path(__file__).parent.parent.parent,
            "config",
            "config.yaml"
        )

    def _load_config(self) -> Config:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Override with environment variables
            config_data = self._override_from_env(config_data)

            return Config(**config_data)

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _override_from_env(self, config: Dict) -> Dict:
        """Override configuration with environment variables"""
        env_mapping = {
            "PDM_DB_HOST": ("database", "host"),
            "PDM_DB_PORT": ("database", "port"),
            "PDM_DB_NAME": ("database", "database"),
            "PDM_DB_USER": ("database", "username"),
            "PDM_DB_PASS": ("database", "password"),
            "PDM_API_PORT": ("api", "port"),
            "PDM_ENV": ("environment",),
            "PDM_LOG_LEVEL": ("logging_level",)
        }

        for env_var, config_path in env_mapping.items():
            if env_var in os.environ:
                self._set_nested_dict(config, config_path, os.environ[env_var])

        return config

    @staticmethod
    def _set_nested_dict(d: Dict, keys: tuple, value: Any) -> None:
        """Set value in nested dictionary"""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.config.database

    def get_verification_config(self) -> VerificationConfig:
        """Get verification system configuration"""
        return self.config.verification

    def get_analytics_config(self) -> AnalyticsConfig:
        """Get analytics system configuration"""
        return self.config.analytics

    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        return self.config.api

    def update_config(self, new_config: Dict) -> None:
        """Update configuration"""
        try:
            # Validate new configuration
            Config(**new_config)

            # Save to file
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(new_config, f)

            # Reload configuration
            self.config = self._load_config()

        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise

    def export_config(self, export_path: str) -> None:
        """Export configuration to file"""
        try:
            config_dict = self.config.model_dump()

            if export_path.endswith('.json'):
                with open(export_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            elif export_path.endswith('.yaml'):
                with open(export_path, 'w') as f:
                    yaml.safe_dump(config_dict, f)
            else:
                raise ValueError("Unsupported export format")

        except Exception as e:
            logger.error(f"Error exporting configuration: {str(e)}")
            raise
