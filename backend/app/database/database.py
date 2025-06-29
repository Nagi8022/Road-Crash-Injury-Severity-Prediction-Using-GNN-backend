from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./crash_analytics.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PredictionRecord(Base):
    """Database model for storing predictions"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    input_data = Column(Text)  # JSON string
    predicted_severity = Column(String, index=True)
    confidence_score = Column(Float)
    probability_distribution = Column(Text)  # JSON string
    risk_level = Column(String)
    model_agreement = Column(Boolean)
    processing_time_ms = Column(Float)
    user_session = Column(String, nullable=True)

class AnalyticsMetrics(Base):
    """Database model for analytics metrics"""
    __tablename__ = "analytics_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    total_predictions = Column(Integer, default=0)
    slight_predictions = Column(Integer, default=0)
    serious_predictions = Column(Integer, default=0)
    fatal_predictions = Column(Integer, default=0)
    average_confidence = Column(Float, default=0.0)
    high_risk_alerts = Column(Integer, default=0)

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    async def initialize(self):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            db = self.SessionLocal()
            db.execute("SELECT 1")
            db.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

    async def store_prediction(
        self, 
        db: Session, 
        input_data: Dict[str, Any], 
        prediction_result: Dict[str, Any],
        user_session: Optional[str] = None
    ):
        """Store a single prediction in database"""
        try:
            record = PredictionRecord(
                input_data=json.dumps(input_data),
                predicted_severity=prediction_result["severity"],
                confidence_score=prediction_result["confidence"],
                probability_distribution=json.dumps(prediction_result["probability_distribution"]),
                risk_level=prediction_result["risk_level"],
                model_agreement=prediction_result["model_agreement"],
                user_session=user_session
            )
            
            db.add(record)
            db.commit()
            db.refresh(record)
            
            # Update analytics
            await self._update_analytics(db)
            
            logger.info(f"Prediction stored with ID: {record.id}")
            return record.id
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {str(e)}")
            db.rollback()
            raise

    async def store_batch_predictions(
        self, 
        db: Session, 
        batch_results: List[Dict[str, Any]]
    ):
        """Store batch predictions in database"""
        try:
            records = []
            for result in batch_results:
                record = PredictionRecord(
                    input_data=json.dumps(result.get("input_data", {})),
                    predicted_severity=result["severity"],
                    confidence_score=result["confidence"],
                    probability_distribution=json.dumps(result["probability_distribution"]),
                    risk_level=result["risk_level"],
                    model_agreement=result["model_agreement"]
                )
                records.append(record)
            
            db.add_all(records)
            db.commit()
            
            # Update analytics
            await self._update_analytics(db)
            
            logger.info(f"Batch of {len(records)} predictions stored")
            
        except Exception as e:
            logger.error(f"Failed to store batch predictions: {str(e)}")
            db.rollback()
            raise

    async def get_analytics_overview(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive analytics overview"""
        try:
            # Get recent predictions (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            recent_predictions = db.query(PredictionRecord).filter(
                PredictionRecord.timestamp >= thirty_days_ago
            ).all()
            
            total_predictions = len(recent_predictions)
            
            if total_predictions == 0:
                return self._empty_analytics()
            
            # Calculate metrics
            severity_counts = {
                "slight": sum(1 for p in recent_predictions if p.predicted_severity == "Slight"),
                "serious": sum(1 for p in recent_predictions if p.predicted_severity == "Serious"),
                "fatal": sum(1 for p in recent_predictions if p.predicted_severity == "Fatal")
            }
            
            avg_confidence = sum(p.confidence_score for p in recent_predictions) / total_predictions
            high_risk_count = sum(1 for p in recent_predictions if p.risk_level in ["High", "Critical"])
            
            return {
                "total_predictions": total_predictions,
                "severity_distribution": severity_counts,
                "average_confidence": round(avg_confidence, 3),
                "high_risk_alerts": high_risk_count,
                "model_agreement_rate": sum(1 for p in recent_predictions if p.model_agreement) / total_predictions,
                "period": "Last 30 days"
            }
            
        except Exception as e:
            logger.error(f"Analytics overview failed: {str(e)}")
            return self._empty_analytics()

    async def get_severity_distribution(self, db: Session) -> List[Dict[str, Any]]:
        """Get severity distribution for visualization"""
        try:
            results = db.query(
                PredictionRecord.predicted_severity,
                func.count(PredictionRecord.id).label('count')
            ).group_by(PredictionRecord.predicted_severity).all()
            
            return [
                {"severity": result.predicted_severity, "count": result.count}
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Severity distribution failed: {str(e)}")
            return []

    async def get_prediction_trends(self, db: Session, days: int = 30) -> List[Dict[str, Any]]:
        """Get prediction trends over time"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            results = db.query(
                func.date(PredictionRecord.timestamp).label('date'),
                func.count(PredictionRecord.id).label('count'),
                PredictionRecord.predicted_severity
            ).filter(
                PredictionRecord.timestamp >= start_date
            ).group_by(
                func.date(PredictionRecord.timestamp),
                PredictionRecord.predicted_severity
            ).all()
            
            return [
                {
                    "date": str(result.date),
                    "count": result.count,
                    "severity": result.predicted_severity
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Prediction trends failed: {str(e)}")
            return []

    async def get_prediction_history(self, db: Session, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent prediction history"""
        try:
            predictions = db.query(PredictionRecord).order_by(
                PredictionRecord.timestamp.desc()
            ).limit(limit).all()
            
            return [
                {
                    "id": p.id,
                    "timestamp": p.timestamp.isoformat(),
                    "severity": p.predicted_severity,
                    "confidence": p.confidence_score,
                    "risk_level": p.risk_level,
                    "model_agreement": p.model_agreement
                }
                for p in predictions
            ]
            
        except Exception as e:
            logger.error(f"Prediction history failed: {str(e)}")
            return []

    async def _update_analytics(self, db: Session):
        """Update daily analytics metrics"""
        try:
            today = datetime.utcnow().date()
            
            # Check if today's metrics exist
            existing_metric = db.query(AnalyticsMetrics).filter(
                func.date(AnalyticsMetrics.date) == today
            ).first()
            
            # Count today's predictions
            today_predictions = db.query(PredictionRecord).filter(
                func.date(PredictionRecord.timestamp) == today
            ).all()
            
            metrics_data = {
                "total_predictions": len(today_predictions),
                "slight_predictions": sum(1 for p in today_predictions if p.predicted_severity == "Slight"),
                "serious_predictions": sum(1 for p in today_predictions if p.predicted_severity == "Serious"),
                "fatal_predictions": sum(1 for p in today_predictions if p.predicted_severity == "Fatal"),
                "average_confidence": sum(p.confidence_score for p in today_predictions) / len(today_predictions) if today_predictions else 0,
                "high_risk_alerts": sum(1 for p in today_predictions if p.risk_level in ["High", "Critical"])
            }
            
            if existing_metric:
                for key, value in metrics_data.items():
                    setattr(existing_metric, key, value)
            else:
                new_metric = AnalyticsMetrics(**metrics_data)
                db.add(new_metric)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Analytics update failed: {str(e)}")
            db.rollback()

    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure"""
        return {
            "total_predictions": 0,
            "severity_distribution": {"slight": 0, "serious": 0, "fatal": 0},
            "average_confidence": 0.0,
            "high_risk_alerts": 0,
            "model_agreement_rate": 0.0,
            "period": "No data available"
        }

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()