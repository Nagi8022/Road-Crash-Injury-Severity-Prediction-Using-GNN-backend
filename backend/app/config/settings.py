import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Configuration
    api_title: str = "Road Crash Analytics API"
    api_version: str = "1.0.0"
    api_description: str = "Advanced ML-powered crash severity prediction system"
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./crash_analytics.db", env="DATABASE_URL")
    
    # Model Configuration
    models_path: Path = Field(default=Path("backend/models"), env="MODELS_PATH")
    model_cache_size: int = Field(default=100, env="MODEL_CACHE_SIZE")
    
    # API Security
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="crash_analytics.log", env="LOG_FILE")
    
    # File Upload Configuration
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(default=[".csv", ".json"], env="ALLOWED_EXTENSIONS")
    
    # Performance Configuration
    prediction_timeout: int = Field(default=30, env="PREDICTION_TIMEOUT")
    batch_size_limit: int = Field(default=1000, env="BATCH_SIZE_LIMIT")
    
    # Analytics Configuration
    analytics_retention_days: int = Field(default=365, env="ANALYTICS_RETENTION_DAYS")
    
    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'protected_namespaces': ()
    }

# Global settings instance
settings = Settings()