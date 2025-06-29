import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import threading
import time

from .model_manager import ModelManager
from .drift_detection import DriftDetector
from ..database.database import DatabaseManager

logger = logging.getLogger(__name__)

class AutoRetrainer:
    """Automatic model retraining service"""
    
    def __init__(self, model_manager: ModelManager, drift_detector: DriftDetector, db_manager: DatabaseManager):
        self.model_manager = model_manager
        self.drift_detector = drift_detector
        self.db_manager = db_manager
        self.scheduler = AsyncIOScheduler()
        
        # Configuration
        self.retrain_config = {
            "schedule_type": "weekly",  # "daily", "weekly", "monthly", "drift_based"
            "retrain_day": "sunday",    # For weekly schedule
            "retrain_time": "02:00",    # 2 AM
            "min_samples_for_retraining": 1000,
            "performance_threshold": 0.80,  # Minimum accuracy to avoid retraining
            "drift_threshold": 0.15,    # Drift score threshold for retraining
            "max_retrain_frequency": 7,  # Maximum days between retrains
            "enable_drift_based_retraining": True,
            "enable_performance_based_retraining": True
        }
        
        # State tracking
        self.last_retrain_date = None
        self.retrain_history = []
        self.is_retraining = False
        self.retrain_callbacks = []
        
    async def initialize(self):
        """Initialize the auto-retrainer"""
        try:
            # Load configuration
            await self._load_config()
            
            # Start scheduler
            self.scheduler.start()
            
            # Schedule retraining jobs
            await self._schedule_retraining_jobs()
            
            logger.info("Auto-retrainer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Auto-retrainer initialization failed: {str(e)}")
            return False
    
    async def _load_config(self):
        """Load retraining configuration"""
        try:
            config_path = Path("backend/config/retrain_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.retrain_config.update(saved_config)
                    logger.info("Loaded retraining configuration from file")
        except Exception as e:
            logger.warning(f"Failed to load retraining config: {str(e)}")
    
    async def _save_config(self):
        """Save retraining configuration"""
        try:
            config_path = Path("backend/config/retrain_config.json")
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.retrain_config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save retraining config: {str(e)}")
    
    async def _schedule_retraining_jobs(self):
        """Schedule retraining jobs based on configuration"""
        try:
            # Clear existing jobs
            self.scheduler.remove_all_jobs()
            
            schedule_type = self.retrain_config["schedule_type"]
            
            if schedule_type == "daily":
                # Daily retraining at specified time
                self.scheduler.add_job(
                    self._scheduled_retrain,
                    CronTrigger(hour=int(self.retrain_config["retrain_time"].split(":")[0]),
                               minute=int(self.retrain_config["retrain_time"].split(":")[1])),
                    id="daily_retrain",
                    name="Daily Model Retraining"
                )
                
            elif schedule_type == "weekly":
                # Weekly retraining on specified day
                day_map = {
                    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
                    "friday": 5, "saturday": 6, "sunday": 0
                }
                
                self.scheduler.add_job(
                    self._scheduled_retrain,
                    CronTrigger(day_of_week=day_map[self.retrain_config["retrain_day"]],
                               hour=int(self.retrain_config["retrain_time"].split(":")[0]),
                               minute=int(self.retrain_config["retrain_time"].split(":")[1])),
                    id="weekly_retrain",
                    name="Weekly Model Retraining"
                )
                
            elif schedule_type == "monthly":
                # Monthly retraining on first day of month
                self.scheduler.add_job(
                    self._scheduled_retrain,
                    CronTrigger(day=1,
                               hour=int(self.retrain_config["retrain_time"].split(":")[0]),
                               minute=int(self.retrain_config["retrain_time"].split(":")[1])),
                    id="monthly_retrain",
                    name="Monthly Model Retraining"
                )
            
            # Add drift monitoring job (runs every 6 hours)
            if self.retrain_config["enable_drift_based_retraining"]:
                self.scheduler.add_job(
                    self._check_drift_and_retrain,
                    IntervalTrigger(hours=6),
                    id="drift_monitoring",
                    name="Drift Monitoring and Retraining"
                )
            
            # Add performance monitoring job (runs every 12 hours)
            if self.retrain_config["enable_performance_based_retraining"]:
                self.scheduler.add_job(
                    self._check_performance_and_retrain,
                    IntervalTrigger(hours=12),
                    id="performance_monitoring",
                    name="Performance Monitoring and Retraining"
                )
            
            logger.info(f"Scheduled retraining jobs: {schedule_type}")
            
        except Exception as e:
            logger.error(f"Failed to schedule retraining jobs: {str(e)}")
    
    async def _scheduled_retrain(self):
        """Scheduled retraining job"""
        try:
            logger.info("Starting scheduled retraining...")
            
            # Check if retraining is needed
            if not await self._should_retrain():
                logger.info("Scheduled retraining skipped - conditions not met")
                return
            
            # Perform retraining
            await self._perform_retraining("scheduled")
            
        except Exception as e:
            logger.error(f"Scheduled retraining failed: {str(e)}")
    
    async def _check_drift_and_retrain(self):
        """Check for drift and retrain if necessary"""
        try:
            logger.info("Checking for data drift...")
            
            # Get recent data for drift detection
            recent_data = await self._get_recent_data(days=7)
            
            if recent_data.empty:
                logger.info("No recent data available for drift detection")
                return
            
            # Perform drift analysis
            drift_result = await self.drift_detector.comprehensive_drift_analysis(recent_data)
            
            # Check if drift exceeds threshold
            if drift_result.get("summary", {}).get("total_alerts", 0) > 0:
                high_severity_alerts = drift_result["summary"].get("high_alerts", 0) + drift_result["summary"].get("critical_alerts", 0)
                
                if high_severity_alerts > 0:
                    logger.warning(f"High severity drift detected: {high_severity_alerts} alerts")
                    
                    if await self._should_retrain():
                        await self._perform_retraining("drift_based", drift_result)
                    else:
                        logger.info("Drift-based retraining skipped - conditions not met")
                else:
                    logger.info("Low severity drift detected - monitoring")
            else:
                logger.info("No significant drift detected")
                
        except Exception as e:
            logger.error(f"Drift monitoring failed: {str(e)}")
    
    async def _check_performance_and_retrain(self):
        """Check model performance and retrain if necessary"""
        try:
            logger.info("Checking model performance...")
            
            # Get recent performance metrics
            recent_performance = await self._get_recent_performance(days=7)
            
            if not recent_performance:
                logger.info("No recent performance data available")
                return
            
            # Calculate average performance
            avg_accuracy = np.mean([p["accuracy"] for p in recent_performance])
            
            if avg_accuracy < self.retrain_config["performance_threshold"]:
                logger.warning(f"Performance below threshold: {avg_accuracy:.3f} < {self.retrain_config['performance_threshold']}")
                
                if await self._should_retrain():
                    await self._perform_retraining("performance_based", {"avg_accuracy": avg_accuracy})
                else:
                    logger.info("Performance-based retraining skipped - conditions not met")
            else:
                logger.info(f"Performance acceptable: {avg_accuracy:.3f}")
                
        except Exception as e:
            logger.error(f"Performance monitoring failed: {str(e)}")
    
    async def _should_retrain(self) -> bool:
        """Determine if retraining should proceed"""
        try:
            # Check if already retraining
            if self.is_retraining:
                logger.info("Retraining already in progress")
                return False
            
            # Check minimum time between retrains
            if self.last_retrain_date:
                days_since_last = (datetime.now() - self.last_retrain_date).days
                if days_since_last < self.retrain_config["max_retrain_frequency"]:
                    logger.info(f"Too soon for retraining: {days_since_last} days since last retrain")
                    return False
            
            # Check if enough new data is available
            recent_data_count = await self._get_recent_data_count(days=30)
            if recent_data_count < self.retrain_config["min_samples_for_retraining"]:
                logger.info(f"Insufficient data for retraining: {recent_data_count} samples")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking retraining conditions: {str(e)}")
            return False
    
    async def _perform_retraining(self, trigger: str, context: Optional[Dict] = None):
        """Perform the actual retraining"""
        try:
            self.is_retraining = True
            start_time = datetime.now()
            
            logger.info(f"Starting {trigger} retraining...")
            
            # Notify callbacks
            await self._notify_retrain_start(trigger, context)
            
            # Get training data
            training_data = await self._prepare_training_data()
            
            if training_data.empty:
                raise Exception("No training data available")
            
            # Split data
            train_data, val_data = self._split_data(training_data)
            
            # Train new model
            training_result = await self.model_manager.train_new_model(
                training_data=train_data,
                validation_data=val_data
            )
            
            if not training_result["success"]:
                raise Exception(f"Training failed: {training_result.get('error', 'Unknown error')}")
            
            # Update retraining history
            retrain_record = {
                "trigger": trigger,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "new_version": training_result["version"],
                "metrics": training_result["metrics"],
                "context": context
            }
            
            self.retrain_history.append(retrain_record)
            self.last_retrain_date = datetime.now()
            
            # Notify callbacks
            await self._notify_retrain_complete(retrain_record)
            
            logger.info(f"Retraining completed successfully: version {training_result['version']}")
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
            await self._notify_retrain_failed(trigger, str(e))
            
        finally:
            self.is_retraining = False
    
    async def _prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data from recent predictions and actual outcomes"""
        try:
            # Get recent predictions with actual outcomes
            recent_predictions = await self.db_manager.get_recent_predictions_with_outcomes(days=90)
            
            if not recent_predictions:
                # Generate synthetic training data for demo
                return self._generate_synthetic_training_data()
            
            # Convert to DataFrame
            df = pd.DataFrame(recent_predictions)
            
            # Add features and labels
            training_data = []
            for _, row in df.iterrows():
                # Extract features from input data
                features = row.get("input_data", {})
                actual_severity = row.get("actual_severity", "Slight")  # Default if not available
                
                # Create training sample
                sample = features.copy()
                sample["severity"] = actual_severity
                training_data.append(sample)
            
            return pd.DataFrame(training_data)
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_synthetic_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for demo purposes"""
        try:
            n_samples = 2000
            
            # Generate synthetic features
            data = {
                "Road Type": np.random.choice(["Urban", "Rural", "Motorway"], n_samples),
                "Road Class": np.random.choice(["A", "B", "C", "Motorway", "Unclassified"], n_samples),
                "Speed Limit": np.random.choice([20, 30, 40, 50, 60, 70], n_samples),
                "Area Type": np.random.choice(["Urban", "Suburban", "Rural"], n_samples),
                "Junction Location": np.random.choice(["Not at junction", "At junction", "Approaching junction"], n_samples),
                "Junction Control": np.random.choice(["Give way or uncontrolled", "Stop sign", "Traffic signals"], n_samples),
                "Junction Detail": np.random.choice(["Not at junction", "Roundabout", "Crossroads"], n_samples),
                "Hazards": np.random.choice(["None", "Vehicle load on road", "Other object on road"], n_samples),
                "Road Surface Conditions": np.random.choice(["Dry", "Wet or damp", "Snow", "Frost or ice"], n_samples),
                "Vehicle Type": np.random.choice(["Car", "Motorcycle", "Bus or coach", "Goods vehicle"], n_samples),
                "Light Conditions": np.random.choice(["Daylight", "Darkness - lights lit", "Darkness - lights unlit"], n_samples),
                "Weather Conditions": np.random.choice(["Fine no high winds", "Raining no high winds", "Snowing no high winds"], n_samples)
            }
            
            # Generate synthetic labels with some logic
            severities = []
            for i in range(n_samples):
                # Simple logic for severity based on features
                speed = data["Speed Limit"][i]
                road_type = data["Road Type"][i]
                weather = data["Weather Conditions"][i]
                
                if speed > 60 and road_type == "Motorway":
                    severity = np.random.choice(["Slight", "Serious", "Fatal"], p=[0.4, 0.4, 0.2])
                elif "Snow" in weather or "Frost" in weather:
                    severity = np.random.choice(["Slight", "Serious", "Fatal"], p=[0.3, 0.5, 0.2])
                elif speed > 40:
                    severity = np.random.choice(["Slight", "Serious", "Fatal"], p=[0.6, 0.3, 0.1])
                else:
                    severity = np.random.choice(["Slight", "Serious", "Fatal"], p=[0.8, 0.15, 0.05])
                
                severities.append(severity)
            
            data["severity"] = severities
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {str(e)}")
            return pd.DataFrame()
    
    def _split_data(self, data: pd.DataFrame) -> tuple:
        """Split data into training and validation sets"""
        try:
            # Simple 80/20 split
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            val_data = data.iloc[split_idx:]
            
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"Failed to split data: {str(e)}")
            return data, data
    
    async def _get_recent_data(self, days: int) -> pd.DataFrame:
        """Get recent data for drift detection"""
        try:
            recent_predictions = await self.db_manager.get_recent_predictions(days=days)
            
            if not recent_predictions:
                return pd.DataFrame()
            
            # Extract features from predictions
            features_list = []
            for pred in recent_predictions:
                features = pred.get("input_data", {})
                if features:
                    features_list.append(features)
            
            return pd.DataFrame(features_list)
            
        except Exception as e:
            logger.error(f"Failed to get recent data: {str(e)}")
            return pd.DataFrame()
    
    async def _get_recent_data_count(self, days: int) -> int:
        """Get count of recent data points"""
        try:
            recent_predictions = await self.db_manager.get_recent_predictions(days=days)
            return len(recent_predictions) if recent_predictions else 0
        except Exception as e:
            logger.error(f"Failed to get recent data count: {str(e)}")
            return 0
    
    async def _get_recent_performance(self, days: int) -> List[Dict]:
        """Get recent performance metrics"""
        try:
            # This would typically come from actual performance tracking
            # For demo, return mock performance data
            return [
                {"accuracy": 0.85, "timestamp": datetime.now().isoformat()},
                {"accuracy": 0.83, "timestamp": (datetime.now() - timedelta(days=1)).isoformat()},
                {"accuracy": 0.87, "timestamp": (datetime.now() - timedelta(days=2)).isoformat()}
            ]
        except Exception as e:
            logger.error(f"Failed to get recent performance: {str(e)}")
            return []
    
    async def _notify_retrain_start(self, trigger: str, context: Optional[Dict]):
        """Notify callbacks of retraining start"""
        for callback in self.retrain_callbacks:
            try:
                await callback("start", {"trigger": trigger, "context": context})
            except Exception as e:
                logger.error(f"Callback notification failed: {str(e)}")
    
    async def _notify_retrain_complete(self, result: Dict):
        """Notify callbacks of retraining completion"""
        for callback in self.retrain_callbacks:
            try:
                await callback("complete", result)
            except Exception as e:
                logger.error(f"Callback notification failed: {str(e)}")
    
    async def _notify_retrain_failed(self, trigger: str, error: str):
        """Notify callbacks of retraining failure"""
        for callback in self.retrain_callbacks:
            try:
                await callback("failed", {"trigger": trigger, "error": error})
            except Exception as e:
                logger.error(f"Callback notification failed: {str(e)}")
    
    def add_retrain_callback(self, callback: Callable):
        """Add callback for retraining events"""
        self.retrain_callbacks.append(callback)
    
    async def update_config(self, new_config: Dict[str, Any]):
        """Update retraining configuration"""
        try:
            self.retrain_config.update(new_config)
            await self._save_config()
            await self._schedule_retraining_jobs()
            
            logger.info("Retraining configuration updated")
            
        except Exception as e:
            logger.error(f"Failed to update retraining config: {str(e)}")
    
    async def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status"""
        try:
            return {
                "is_retraining": self.is_retraining,
                "last_retrain_date": self.last_retrain_date.isoformat() if self.last_retrain_date else None,
                "config": self.retrain_config,
                "history": self.retrain_history[-10:],  # Last 10 retrains
                "next_scheduled": self._get_next_scheduled_retrain()
            }
        except Exception as e:
            logger.error(f"Failed to get retraining status: {str(e)}")
            return {"error": str(e)}
    
    def _get_next_scheduled_retrain(self) -> Optional[str]:
        """Get next scheduled retraining time"""
        try:
            jobs = self.scheduler.get_jobs()
            for job in jobs:
                if "retrain" in job.id:
                    next_run = job.next_run_time
                    return next_run.isoformat() if next_run else None
            return None
        except Exception as e:
            logger.error(f"Failed to get next scheduled retrain: {str(e)}")
            return None
    
    async def manual_retrain(self) -> Dict[str, Any]:
        """Trigger manual retraining"""
        try:
            if self.is_retraining:
                return {"success": False, "error": "Retraining already in progress"}
            
            if not await self._should_retrain():
                return {"success": False, "error": "Retraining conditions not met"}
            
            await self._perform_retraining("manual")
            return {"success": True, "message": "Manual retraining completed"}
            
        except Exception as e:
            logger.error(f"Manual retraining failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def shutdown(self):
        """Shutdown the auto-retrainer"""
        try:
            self.scheduler.shutdown()
            logger.info("Auto-retrainer shutdown complete")
        except Exception as e:
            logger.error(f"Auto-retrainer shutdown failed: {str(e)}") 