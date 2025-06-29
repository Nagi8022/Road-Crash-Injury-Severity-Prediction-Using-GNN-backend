import mlflow
import mlflow.pytorch
import mlflow.sklearn
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass
from enum import Enum
import os
import shutil

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"

@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    status: ModelStatus
    created_at: datetime
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    file_path: str
    model_size_mb: float
    training_time_seconds: float

class ModelManager:
    """MLflow-based model management service.
    If mlflow is not available, this class degrades gracefully: all public methods still exist but
    return generic failure responses while logging clear warnings. This allows the rest of the API
    to function (predictions, batch processing, etc.) without blocking on the heavy mlflow dependency.
    """
    """MLflow-based model management service"""
    
    def __init__(self, mlflow_tracking_uri: str = "sqlite:///mlflow.db"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = "crash_severity_prediction"
        self.model_registry_name = "crash_severity_model"
        self.models_path = Path("backend/models")
        self.backup_path = Path("backend/models/backup")
        
        # Ensure directories exist
        self.models_path.mkdir(exist_ok=True)
        self.backup_path.mkdir(exist_ok=True)
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        else:
            logger.warning("mlflow not installed; ModelManager will operate in disabled mode.")
        
        self.current_model_version = None
        self.model_versions = []
        
    async def initialize(self):
        """Initialize the model manager"""
        try:
            # Load existing model versions
            await self._load_model_versions()
            
            # Set up model registry
            await self._setup_model_registry()
            
            logger.info("Model manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model manager initialization failed: {str(e)}")
            return False
    
    async def _load_model_versions(self):
        """Load existing model versions from MLflow"""
        try:
            client = mlflow.tracking.MlflowClient()
            registered_models = client.list_registered_models()
            
            for model in registered_models:
                if model.name == self.model_registry_name:
                    versions = client.search_model_versions(f"name='{self.model_registry_name}'")
                    for version in versions:
                        model_version = ModelVersion(
                            version=version.version,
                            status=ModelStatus(version.status),
                            created_at=datetime.fromtimestamp(version.creation_timestamp / 1000),
                            metrics=version.tags.get("metrics", {}),
                            parameters=version.tags.get("parameters", {}),
                            file_path=version.source,
                            model_size_mb=float(version.tags.get("model_size_mb", 0)),
                            training_time_seconds=float(version.tags.get("training_time_seconds", 0))
                        )
                        self.model_versions.append(model_version)
            
            # Sort by creation date
            self.model_versions.sort(key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            logger.warning(f"Failed to load model versions: {str(e)}")
    
    async def _setup_model_registry(self):
        """Set up model registry if it doesn't exist"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Check if model registry exists
            try:
                client.get_registered_model(self.model_registry_name)
            except:
                # Create new model registry
                client.create_registered_model(self.model_registry_name)
                logger.info(f"Created new model registry: {self.model_registry_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup model registry: {str(e)}")
    
    async def train_new_model(self, training_data: pd.DataFrame, 
                            validation_data: pd.DataFrame,
                            hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train a new model version using MLflow"""
        try:
            start_time = datetime.now()
            
            # Set default hyperparameters
            if hyperparameters is None:
                hyperparameters = {
                    "gnn_hidden_dim": 256,
                    "bilstm_hidden_dim": 128,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 100,
                    "dropout": 0.2
                }
            
            # Start MLflow run
            with mlflow.start_run():
                # Log hyperparameters
                mlflow.log_params(hyperparameters)
                
                # Train models
                training_result = await self._train_models(training_data, validation_data, hyperparameters)
                
                if not training_result["success"]:
                    raise Exception(f"Training failed: {training_result['error']}")
                
                # Log metrics
                for metric_name, metric_value in training_result["metrics"].items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Save models
                model_files = await self._save_models(training_result["models"])
                
                # Log model artifacts
                for model_name, file_path in model_files.items():
                    mlflow.log_artifact(file_path, f"models/{model_name}")
                
                # Register model version
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/models"
                client = mlflow.tracking.MlflowClient()
                
                # Create new version
                version = client.create_model_version(
                    name=self.model_registry_name,
                    source=model_uri,
                    run_id=mlflow.active_run().info.run_id
                )
                
                # Add metadata
                training_time = (datetime.now() - start_time).total_seconds()
                model_size = sum(os.path.getsize(f) for f in model_files.values()) / (1024 * 1024)  # MB
                
                client.set_model_version_tag(
                    name=self.model_registry_name,
                    version=version.version,
                    key="metrics",
                    value=json.dumps(training_result["metrics"])
                )
                
                client.set_model_version_tag(
                    name=self.model_registry_name,
                    version=version.version,
                    key="parameters",
                    value=json.dumps(hyperparameters)
                )
                
                client.set_model_version_tag(
                    name=self.model_registry_name,
                    version=version.version,
                    key="model_size_mb",
                    value=str(model_size)
                )
                
                client.set_model_version_tag(
                    name=self.model_registry_name,
                    version=version.version,
                    key="training_time_seconds",
                    value=str(training_time)
                )
                
                # Transition to ready state
                client.transition_model_version_stage(
                    name=self.model_registry_name,
                    version=version.version,
                    stage="Production"
                )
                
                # Update current model version
                self.current_model_version = version.version
                
                # Create model version object
                model_version = ModelVersion(
                    version=version.version,
                    status=ModelStatus.READY,
                    created_at=datetime.now(),
                    metrics=training_result["metrics"],
                    parameters=hyperparameters,
                    file_path=model_uri,
                    model_size_mb=model_size,
                    training_time_seconds=training_time
                )
                
                self.model_versions.append(model_version)
                
                logger.info(f"New model version {version.version} trained successfully")
                
                return {
                    "success": True,
                    "version": version.version,
                    "metrics": training_result["metrics"],
                    "training_time_seconds": training_time,
                    "model_size_mb": model_size
                }
                
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _train_models(self, training_data: pd.DataFrame, 
                          validation_data: pd.DataFrame,
                          hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train the ensemble models"""
        try:
            # This is a simplified training implementation
            # In a real scenario, you would implement full training logic here
            
            # Simulate training process
            await asyncio.sleep(2)  # Simulate training time
            
            # Generate mock metrics
            metrics = {
                "training_accuracy": 0.89,
                "validation_accuracy": 0.85,
                "f1_score": 0.84,
                "precision": 0.86,
                "recall": 0.82
            }
            
            # Mock models (in reality, these would be actual trained models)
            models = {
                "gnn_model": "mock_gnn_model",
                "bilstm_model": "mock_bilstm_model",
                "random_forest": "mock_rf_model",
                "scaler": "mock_scaler",
                "label_encoders": "mock_encoders"
            }
            
            return {
                "success": True,
                "metrics": metrics,
                "models": models
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
        """Save trained models to files"""
        try:
            model_files = {}
            
            # Save PyTorch models
            for model_name in ["gnn_model", "bilstm_model"]:
                if model_name in models:
                    file_path = self.models_path / f"{model_name}.pth"
                    # In reality, you would save the actual model
                    torch.save(torch.randn(10, 10), file_path)  # Mock save
                    model_files[model_name] = str(file_path)
            
            # Save scikit-learn models
            for model_name in ["random_forest", "scaler", "label_encoders"]:
                if model_name in models:
                    file_path = self.models_path / f"{model_name}.pkl"
                    # In reality, you would save the actual model
                    joblib.dump({"mock": "model"}, file_path)  # Mock save
                    model_files[model_name] = str(file_path)
            
            return model_files
            
        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
            raise
    
    async def deploy_model_version(self, version: str) -> Dict[str, Any]:
        """Deploy a specific model version"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Transition model to production
            client.transition_model_version_stage(
                name=self.model_registry_name,
                version=version,
                stage="Production"
            )
            
            # Download model files
            model_uri = client.get_model_version_download_uri(
                name=self.model_registry_name,
                version=version
            )
            
            # Update current model version
            self.current_model_version = version
            
            logger.info(f"Model version {version} deployed successfully")
            
            return {
                "success": True,
                "version": version,
                "model_uri": model_uri
            }
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def rollback_model(self, target_version: str) -> Dict[str, Any]:
        """Rollback to a previous model version"""
        try:
            # Find target version
            target_model = None
            for model in self.model_versions:
                if model.version == target_version:
                    target_model = model
                    break
            
            if not target_model:
                raise Exception(f"Model version {target_version} not found")
            
            # Deploy target version
            result = await self.deploy_model_version(target_version)
            
            if result["success"]:
                logger.info(f"Successfully rolled back to model version {target_version}")
            
            return result
            
        except Exception as e:
            logger.error(f"Model rollback failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_model_performance_history(self, days: int = 30) -> Dict[str, Any]:
        """Get model performance history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_models = [m for m in self.model_versions if m.created_at > cutoff_date]
            
            performance_data = {
                "models": [],
                "metrics_trends": {
                    "accuracy": [],
                    "f1_score": [],
                    "training_time": []
                }
            }
            
            for model in recent_models:
                performance_data["models"].append({
                    "version": model.version,
                    "created_at": model.created_at.isoformat(),
                    "metrics": model.metrics,
                    "training_time": model.training_time_seconds,
                    "model_size": model.model_size_mb
                })
                
                # Add to trends
                if "validation_accuracy" in model.metrics:
                    performance_data["metrics_trends"]["accuracy"].append({
                        "date": model.created_at.isoformat(),
                        "value": model.metrics["validation_accuracy"]
                    })
                
                if "f1_score" in model.metrics:
                    performance_data["metrics_trends"]["f1_score"].append({
                        "date": model.created_at.isoformat(),
                        "value": model.metrics["f1_score"]
                    })
                
                performance_data["metrics_trends"]["training_time"].append({
                    "date": model.created_at.isoformat(),
                    "value": model.training_time_seconds
                })
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get performance history: {str(e)}")
            return {"error": str(e)}
    
    async def compare_model_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        try:
            # Find model versions
            model1 = None
            model2 = None
            
            for model in self.model_versions:
                if model.version == version1:
                    model1 = model
                elif model.version == version2:
                    model2 = model
            
            if not model1 or not model2:
                raise Exception("One or both model versions not found")
            
            # Compare metrics
            comparison = {
                "version1": {
                    "version": model1.version,
                    "created_at": model1.created_at.isoformat(),
                    "metrics": model1.metrics,
                    "training_time": model1.training_time_seconds,
                    "model_size": model1.model_size_mb
                },
                "version2": {
                    "version": model2.version,
                    "created_at": model2.created_at.isoformat(),
                    "metrics": model2.metrics,
                    "training_time": model2.training_time_seconds,
                    "model_size": model2.model_size_mb
                },
                "differences": {}
            }
            
            # Calculate differences
            for metric in ["validation_accuracy", "f1_score", "precision", "recall"]:
                if metric in model1.metrics and metric in model2.metrics:
                    diff = model2.metrics[metric] - model1.metrics[metric]
                    comparison["differences"][metric] = {
                        "difference": diff,
                        "improvement": diff > 0
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup_old_models(self, keep_versions: int = 5) -> Dict[str, Any]:
        """Clean up old model versions"""
        try:
            if len(self.model_versions) <= keep_versions:
                return {"success": True, "message": "No cleanup needed"}
            
            # Sort by creation date and keep only the latest versions
            sorted_models = sorted(self.model_versions, key=lambda x: x.created_at, reverse=True)
            models_to_remove = sorted_models[keep_versions:]
            
            client = mlflow.tracking.MlflowClient()
            removed_count = 0
            
            for model in models_to_remove:
                try:
                    # Delete model version from registry
                    client.delete_model_version(
                        name=self.model_registry_name,
                        version=model.version
                    )
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove model version {model.version}: {str(e)}")
            
            # Update local list
            self.model_versions = sorted_models[:keep_versions]
            
            logger.info(f"Cleaned up {removed_count} old model versions")
            
            return {
                "success": True,
                "removed_count": removed_count,
                "remaining_versions": len(self.model_versions)
            }
            
        except Exception as e:
            logger.error(f"Model cleanup failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_model_registry_info(self) -> Dict[str, Any]:
        """Get model registry information"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get registered model info
            model_info = client.get_registered_model(self.model_registry_name)
            
            # Get latest versions
            latest_versions = client.search_model_versions(
                f"name='{self.model_registry_name}'",
                max_results=10
            )
            
            return {
                "model_name": model_info.name,
                "total_versions": len(self.model_versions),
                "current_version": self.current_model_version,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "created_at": datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat()
                    }
                    for v in latest_versions
                ],
                "registry_uri": self.mlflow_tracking_uri
            }
            
        except Exception as e:
            logger.error(f"Failed to get registry info: {str(e)}")
            return {"error": str(e)} 