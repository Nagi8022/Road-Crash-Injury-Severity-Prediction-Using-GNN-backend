from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from ..services.visualization import DataVisualizer
from ..database.database import get_db

router = APIRouter(prefix="/api/visualization", tags=["visualization"])
logger = logging.getLogger(__name__)

# Initialize visualization service
viz_service = DataVisualizer()

@router.get("/severity-distribution")
async def get_severity_distribution(
    days: int = Query(30, description="Number of days of data to include"),
    db: Session = Depends(get_db)
):
    """Get a pie chart of accident severity distribution."""
    try:
        # Get data from database
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # This is a placeholder - replace with actual database query
        # accidents = db.query(Accident).filter(Accident.timestamp >= start_date).all()
        # data = [{"severity": a.severity} for a in accidents]
        
        # For now, return sample data
        data = [
            {"severity": "Slight", "count": 65},
            {"severity": "Serious", "count": 25},
            {"severity": "Fatal", "count": 10}
        ]
        
        img_bytes = viz_service.create_severity_pie_chart(data)
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating severity distribution: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating visualization")

@router.get("/time-series")
async def get_time_series(
    days: int = Query(30, description="Number of days of data to include"),
    db: Session = Depends(get_db)
):
    """Get a time series plot of accidents over time."""
    try:
        # Get data from database
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # This is a placeholder - replace with actual database query
        # accidents = db.query(Accident).filter(Accident.timestamp >= start_date).all()
        # data = [{"timestamp": a.timestamp, "severity": a.severity} for a in accidents]
        
        # For now, return sample data
        data = []
        for i in range(30):
            date = (end_date - timedelta(days=i)).date()
            data.append({"timestamp": date.isoformat(), "severity": "Slight"})
            if i % 2 == 0:
                data.append({"timestamp": date.isoformat(), "severity": "Serious"})
            if i % 5 == 0:
                data.append({"timestamp": date.isoformat(), "severity": "Fatal"})
        
        img_bytes = viz_service.create_time_series_plot(data)
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating time series: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating visualization")

@router.get("/feature-importance")
async def get_feature_importance():
    """Get a bar chart of feature importances."""
    try:
        # Get feature importances from your model
        # This is a placeholder - replace with actual feature importances from your model
        importances = {
            "speed_limit": 0.25,
            "road_type": 0.20,
            "weather_conditions": 0.18,
            "light_conditions": 0.15,
            "vehicle_type": 0.12,
            "road_surface": 0.10
        }
        
        img_bytes = viz_service.create_feature_importance_plot(importances)
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating visualization")

@router.get("/dashboard")
async def get_dashboard(
    days: int = Query(30, description="Number of days of data to include"),
    db: Session = Depends(get_db)
):
    """Get a comprehensive dashboard with multiple visualizations."""
    try:
        # Get data from database
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # This is a placeholder - replace with actual database query
        # accidents = db.query(Accident).filter(Accident.timestamp >= start_date).all()
        # data = [{"timestamp": a.timestamp, "severity": a.severity} for a in accidents]
        
        # For now, return sample data
        data = []
        for i in range(30):
            date = (end_date - timedelta(days=i)).date()
            data.append({"timestamp": date.isoformat(), "severity": "Slight"})
            if i % 2 == 0:
                data.append({"timestamp": date.isoformat(), "severity": "Serious"})
            if i % 5 == 0:
                data.append({"timestamp": date.isoformat(), "severity": "Fatal"})
        
        # Sample feature importances
        importances = {
            "speed_limit": 0.25,
            "road_type": 0.20,
            "weather_conditions": 0.18,
            "light_conditions": 0.15,
            "vehicle_type": 0.12,
            "road_surface": 0.10
        }
        
        img_bytes = viz_service.create_dashboard(data, importances)
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating dashboard")
