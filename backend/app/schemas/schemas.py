from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class SeverityLevel(str, Enum):
    SLIGHT = "Slight"
    SERIOUS = "Serious"
    FATAL = "Fatal"

class RiskLevel(str, Enum):
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class PredictionRequest(BaseModel):
    """Schema for single prediction requests"""
    road_type: str = Field(..., description="Type of road (Urban/Rural)")
    road_class: str = Field(..., description="Classification of road (Motorway/A/B/C/Unclassified)")
    speed_limit: int = Field(..., ge=20, le=70, description="Speed limit in mph")
    area_type: str = Field(..., description="Area type (Urban/Suburban/Rural)")
    junction_location: str = Field(..., description="Junction location details")
    junction_control: str = Field(..., description="Junction control type")
    junction_detail: str = Field(..., description="Detailed junction information")
    hazards: str = Field(..., description="Hazards present at scene")
    road_surface_conditions: str = Field(..., description="Road surface conditions")
    vehicle_type: str = Field(..., description="Type of vehicle involved")
    light_conditions: str = Field(..., description="Lighting conditions")
    weather_conditions: str = Field(..., description="Weather conditions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "road_type": "Urban",
                "road_class": "A",
                "speed_limit": 30,
                "area_type": "Urban",
                "junction_location": "Not at junction",
                "junction_control": "Give way or uncontrolled",
                "junction_detail": "Not at junction",
                "hazards": "None",
                "road_surface_conditions": "Dry",
                "vehicle_type": "Car",
                "light_conditions": "Daylight",
                "weather_conditions": "Fine no high winds"
            }
        }
        protected_namespaces = ()

class PredictionResponse(BaseModel):
    """Schema for prediction responses"""
    severity: SeverityLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    probability_distribution: Dict[str, float]
    model_agreement: bool
    risk_level: RiskLevel
    
    class Config:
        json_schema_extra = {
            "example": {
                "severity": "Serious",
                "confidence": 0.87,
                "probability_distribution": {
                    "Slight": 0.15,
                    "Serious": 0.72,
                    "Fatal": 0.13
                },
                "model_agreement": True,
                "risk_level": "High"
            }
        }
        protected_namespaces = ()

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses"""
    message: str
    total_records: int
    results: List[Dict[str, Any]]
    summary: Dict[str, int]

class AnalyticsResponse(BaseModel):
    """Schema for analytics overview"""
    total_predictions: int
    severity_distribution: Dict[str, int]
    average_confidence: float
    high_risk_alerts: int
    model_agreement_rate: float
    period: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 1247,
                "severity_distribution": {
                    "slight": 856,
                    "serious": 312,
                    "fatal": 79
                },
                "average_confidence": 0.834,
                "high_risk_alerts": 127,
                "model_agreement_rate": 0.91,
                "period": "Last 30 days"
            }
        }
        protected_namespaces = ()

class ChartDataPoint(BaseModel):
    """Schema for chart data points"""
    label: str
    value: float
    color: Optional[str] = None

class VisualizationResponse(BaseModel):
    """Schema for visualization data"""
    chart_type: str
    title: str
    data: List[ChartDataPoint]
    metadata: Optional[Dict[str, Any]] = None

class ModelPerformanceResponse(BaseModel):
    """Schema for model performance metrics"""
    model_architecture: str
    training_accuracy: float
    validation_accuracy: float
    f1_score: float
    precision: float
    recall: float
    model_size_mb: float
    inference_time_ms: float
    last_updated: str

    class Config:
        protected_namespaces = ()

class PredictionHistoryItem(BaseModel):
    """Schema for prediction history items"""
    id: int
    timestamp: datetime
    severity: SeverityLevel
    confidence: float
    risk_level: RiskLevel
    model_agreement: bool

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str] = None