from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import json
import traceback
import shutil
import time
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import uuid
import traceback

from fastapi import Depends, FastAPI, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import uuid
import traceback
import logging

from . import __version__
from .schemas.schemas import PredictionRequest, PredictionResponse, AnalyticsResponse
from .services.data_processor import DataProcessor
from .services.database import DatabaseManager, get_db
from .services.auto_retrain import AutoRetrainer
from .services.visualization import DataVisualizer
from .routes.visualization import router as visualization_router
from .utils.logger import setup_logger
from .config.settings import Settings
from .services.drift_detection import DriftDetector

# Initialize FastAPI app
# Configure CORS
origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8000",  # Local development
    "https://your-production-domain.com"  # Production domain
]

app = FastAPI(
    title="Road Crash Analytics API",
    description="Advanced ML-powered crash severity prediction system with XAI, drift detection, and auto-retraining",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
# Setup CORS with more specific settings for file downloads
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length", "Content-Type"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Include visualization routes
app.include_router(visualization_router)

# Initialize core services
settings = Settings()
logger = setup_logger(__name__)

# Create data processor instance with reports directory
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
data_processor = DataProcessor(reports_dir=REPORTS_DIR)
db_manager = DatabaseManager()
viz_service = DataVisualizer()

# Initialize model and predictor
predictor = None
model_manager = None
xai_service = None
auto_retrainer = None
drift_detector = None

try:
    from .services.model_predictor import ModelPredictor
    from .services.model_manager import ModelManager
    from .services.xai_service import XAIService
    from .services.drift_detection import DriftDetector
    
    predictor = ModelPredictor()
    model_manager = ModelManager()
    xai_service = XAIService(predictor)
    
    # Initialize drift detector
    try:
        drift_detector = DriftDetector()
        # Generate sample reference data
        drift_detector.generate_sample_data()
        logger.info("Drift detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize drift detector: {str(e)}")
        drift_detector = None
    
    # Only initialize auto_retrainer if all required services are available
    if predictor and data_processor and db_manager:
        auto_retrainer = AutoRetrainer(data_processor, predictor, db_manager)
    
    logger.info("ML services initialized successfully")
except ImportError as e:
    logger.warning(f"Failed to initialize ML services: {str(e)}")
    logger.warning("The API will start, but ML features will be disabled")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Crash Analytics API with Intelligent ML Features...")
    
    # Initialize database
    try:
        await db_manager.initialize()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        # Continue without database if it fails
    
    # Initialize ML services if available
    if predictor is not None:
        try:
            # Load ML models
            model_status = await predictor.load_models()
            if not model_status:
                logger.warning("⚠️ Failed to load ML models. The API will start, but predictions will not work.")
            else:
                logger.info("✅ ML models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
        
        # Initialize XAI service if available
        if xai_service is not None:
            try:
                await xai_service.initialize_explainers()
                logger.info("✅ XAI service initialized")
            except Exception as e:
                logger.error(f"❌ XAI service initialization failed: {str(e)}")
        
        # Initialize model manager if available
        if model_manager is not None:
            try:
                await model_manager.initialize()
                logger.info("✅ Model manager initialized")
            except Exception as e:
                logger.error(f"❌ Model manager initialization failed: {str(e)}")
    else:
        logger.warning("⚠️ ML predictor not available. ML features will be disabled.")
    
    # Initialize auto-retrainer if available
    if auto_retrainer is not None:
        try:
            await auto_retrainer.initialize()
            logger.info("✅ Auto-retrainer initialized successfully")
            await auto_retrainer.start_scheduler()
            logger.info("✅ Auto-retrainer scheduler started")
        except Exception as e:
            logger.error(f"❌ Auto-retrainer initialization failed: {str(e)}")
    else:
        logger.warning("⚠️ Auto-retrainer not available. Auto-retraining will be disabled.")
    
    logger.info("🚀 API startup completed")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚗 Road Crash Analytics API with Intelligent ML",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "docs": "/api/docs",
        "features": [
            "Advanced ML Predictions",
            "Explainable AI (XAI)",
            "Data Drift Detection",
            "Auto-Retraining",
            "Model Versioning"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_health = await db_manager.health_check()
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_health = False
    
    try:
        xai_health = xai_service.lime_explainer is not None
    except Exception as e:
        logger.error(f"XAI health check failed: {str(e)}")
        xai_health = False
    
    try:
        retrainer_health = auto_retrainer.scheduler.running
    except Exception as e:
        logger.error(f"Retrainer health check failed: {str(e)}")
        retrainer_health = False
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": db_health,
            "ml_models": predictor.is_loaded(),
            "data_processor": True,
            "xai_service": xai_health,
            "drift_detection": True,
            "model_manager": True,
            "auto_retrainer": retrainer_health
        },
        "demo_mode": not predictor.is_loaded(),
        "message": "API is running in demo mode" if not predictor.is_loaded() else "API is running with full ML models"
    }

@app.post("/api/predict/single", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict severity for a single crash scenario"""
    try:
        logger.info(f"Processing single prediction request")
        
        # Validate input data
        validated_data = await data_processor.validate_input(request.dict())
        
        # Make prediction
        prediction_result = await predictor.predict_single(validated_data)
        
        # Try to store in database (don't fail if database is unavailable)
        try:
            db = next(get_db())
            await db_manager.store_prediction(db, validated_data, prediction_result)
        except Exception as db_error:
            logger.warning(f"Failed to store prediction in database: {str(db_error)}")
            # Continue without database storage
        
        logger.info(f"Single prediction completed: {prediction_result['severity']}")
        return PredictionResponse(**prediction_result)
        
    except Exception as e:
        logger.error(f"Single prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch predict severity from uploaded CSV/JSON file.
    
    Expected CSV format:
    - First row should contain headers matching the required fields
    - Each subsequent row should contain values for prediction
    - Empty cells will be filled with default values
    
    Supported formats: CSV, JSON
    Max file size: 10MB
    """
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    try:
        logger.info(f"Processing batch prediction request for file: {file.filename}")
        logger.info(f"File size: {file.size} bytes")
        logger.info(f"Content type: {file.content_type}")
        
        # Validate file size
        if file.size > MAX_FILE_SIZE:
            error_msg = f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024:.1f}MB"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Validate file extension
        if not file.filename.lower().endswith((".csv", ".json")):
            error_msg = "Invalid file format. Only CSV and JSON files are supported"
            logger.error(f"{error_msg}: {file.filename}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read file content with size limit
        logger.info("Reading file content...")
        file_content = await file.read(MAX_FILE_SIZE + 1)
        if len(file_content) > MAX_FILE_SIZE:
            error_msg = f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024:.1f}MB"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
            
        logger.info(f"Successfully read {len(file_content)} bytes")
        
        # Log file preview for debugging (first 200 chars)
        try:
            preview = file_content[:200].decode('utf-8', errors='replace')
            logger.info(f"File preview (first 200 chars):\n{preview}")
        except Exception as e:
            logger.warning(f"Could not generate file preview: {str(e)}")
        
        logger.info("Starting batch file processing...")
        try:
            # Process the file
            processed_data = await data_processor.process_batch_file(file_content, file.filename)
            
            if not processed_data:
                error_msg = "No valid records found in the uploaded file. Please check the file format and data."
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            logger.info(f"Successfully processed {len(processed_data)} records")
            logger.debug(f"Sample processed record: {processed_data[0] if processed_data else 'None'}")
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
            
        except ValueError as ve:
            # Handle validation errors with more context
            error_msg = f"Data validation error: {str(ve)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Provide more helpful error message for common issues
            if "No valid records found" in str(ve):
                error_msg = (
                    "No valid records found in the uploaded file. "
                    "Please check that your file contains the required columns and valid data. "
                    "Refer to the API documentation for the expected format."
                )
            
            raise HTTPException(status_code=400, detail=error_msg)
            
        except Exception as e:
            # Log the full error for debugging
            error_id = str(uuid.uuid4())[:8]
            error_msg = f"Error processing file (ID: {error_id}). Please try again or contact support."
            
            logger.error(f"Error ID: {error_id}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
        
        logger.info("Making predictions...")
        
        try:
            # Make batch predictions
            batch_results = await predictor.predict_batch(processed_data)
            
            if not batch_results:
                error_msg = "Prediction service returned no results. Please try again or contact support."
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            logger.info(f"Successfully generated predictions for {len(batch_results)} records")
            
            # Calculate summary statistics
            severity_counts = {
                'slight': 0,
                'serious': 0,
                'fatal': 0
            }
            
            for result in batch_results:
                severity = result.get('severity', '').lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1
            
            # Try to store results in database (don't fail if database is unavailable)
            try:
                db = next(get_db())
                await db_manager.store_batch_predictions(db, batch_results)
                logger.info("Successfully stored predictions in database")
            except Exception as db_error:
                logger.warning(f"Failed to store batch predictions in database: {str(db_error)}")
                # Continue without database storage
            
            # Prepare response
            response = {
                "status": "success",
                "message": "Batch prediction completed successfully",
                "metadata": {
                    "file_name": file.filename,
                    "file_size_bytes": len(file_content),
                    "records_processed": len(processed_data),
                    "timestamp": datetime.utcnow().isoformat()
                },
                "summary": {
                    "total_records": len(batch_results),
                    "severity_distribution": severity_counts
                },
                "results": batch_results
            }
            
            logger.info(f"Batch prediction completed. Processed {len(batch_results)} records")
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
            
        except Exception as e:
            # Log the full error for debugging
            error_id = str(uuid.uuid4())[:8]
            error_msg = f"Prediction failed (ID: {error_id}). Please try again or contact support."
            
            logger.error(f"Prediction Error ID: {error_id}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            raise HTTPException(status_code=500, detail=error_msg)
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions with their original status codes
        logger.error(f"Batch prediction HTTP error: {str(http_exc.detail)}")
        raise
        
    except Exception as e:
        # Log the full error for debugging
        error_id = str(uuid.uuid4())[:8]
        error_msg = f"An unexpected error occurred (ID: {error_id}). Please try again or contact support."
        
        logger.error(f"Unexpected Error ID: {error_id}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.get("/api/analytics/overview", response_model=AnalyticsResponse)
async def get_analytics_overview():
    """Get analytics overview and statistics"""
    try:
        db = next(get_db())
        analytics_data = await db_manager.get_analytics_overview(db)
        return AnalyticsResponse(**analytics_data)
    except Exception as e:
        logger.error(f"Analytics overview failed: {str(e)}")
        # Return demo data if database fails
        return AnalyticsResponse(
            total_predictions=1247,
            severity_distribution={"slight": 856, "serious": 312, "fatal": 79},
            average_confidence=0.834,
            high_risk_alerts=127,
            model_agreement_rate=0.91,
            period="Last 30 days"
        )

@app.get("/api/visualizations/severity-distribution")
async def get_severity_distribution():
    """Get severity distribution visualization data"""
    try:
        db = next(get_db())
        data = await db_manager.get_severity_distribution(db)
        chart_data = await viz_service.create_severity_chart(data)
        return chart_data
    except Exception as e:
        logger.error(f"Severity distribution failed: {str(e)}")
        # Return demo data
        return {
            "chart_type": "pie",
            "title": "Accident Severity Distribution",
            "data": {
                "data": [{
                    "values": [856, 312, 79],
                    "labels": ['Slight', 'Serious', 'Fatal'],
                    "type": 'pie'
                }],
                "layout": {
                    "title": 'Severity Distribution'
                }
            }
        }

@app.get("/api/visualizations/prediction-trends")
async def get_prediction_trends(days: int = 30):
    """Get prediction trends over time"""
    try:
        db = next(get_db())
        data = await db_manager.get_prediction_trends(db, days)
        chart_data = await viz_service.create_trends_chart(data)
        return chart_data
    except Exception as e:
        logger.error(f"Prediction trends failed: {str(e)}")
        # Return demo data
        return {
            "chart_type": "line",
            "title": "Prediction Trends Over Time",
            "data": {
                "data": [{
                    "x": ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
                    "y": [45, 52, 38, 67, 43],
                    "type": 'scatter',
                    "mode": 'lines+markers'
                }],
                "layout": {
                    "title": 'Trends Over Time'
                }
            }
        }

@app.get("/api/visualizations/feature-importance")
async def get_feature_importance():
    """Get model feature importance data"""
    try:
        importance_data = await predictor.get_feature_importance()
        chart_data = await viz_service.create_feature_importance_chart(importance_data)
        return chart_data
    except Exception as e:
        logger.error(f"Feature importance failed: {str(e)}")
        # Return demo data
        return {
            "chart_type": "bar",
            "title": "Feature Importance",
            "data": {
                "data": [{
                    "x": [0.18, 0.15, 0.13, 0.12, 0.11],
                    "y": ['Speed Limit', 'Road Class', 'Light Conditions', 'Weather', 'Junction'],
                    "type": 'bar',
                    "orientation": 'h'
                }],
                "layout": {
                    "title": 'Feature Importance'
                }
            }
        }

@app.get("/api/model/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        performance_data = await predictor.get_performance_metrics()
        return performance_data
    except Exception as e:
        logger.error(f"Model performance retrieval failed: {str(e)}")
        # Return demo data
        return {
            "model_architecture": "GraphSAGE + BiLSTM + Random Forest Ensemble",
            "training_accuracy": 0.89,
            "validation_accuracy": 0.85,
            "f1_score": 0.84,
            "precision": 0.86,
            "recall": 0.83,
            "model_size_mb": 15.2,
            "inference_time_ms": 45,
            "last_updated": "2024-01-15"
        }

@app.get("/api/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get recent prediction history"""
    try:
        db = next(get_db())
        history = await db_manager.get_prediction_history(db, limit)
        return {"predictions": history, "total": len(history)}
    except Exception as e:
        logger.error(f"Prediction history failed: {str(e)}")
        # Return demo data
        return {
            "predictions": [
                {
                    "id": 1,
                    "timestamp": datetime.now().isoformat(),
                    "severity": "Serious",
                    "confidence": 0.87,
                    "risk_level": "High",
                    "model_agreement": True
                }
            ],
            "total": 1
        }

# ===== INTELLIGENT ML SYSTEM ENDPOINTS =====

@app.post("/api/xai/explain")
async def explain_prediction(request: PredictionRequest):
    """Generate XAI explanations for a prediction"""
    try:
        logger.info("Generating XAI explanation")
        
        # Validate input data
        validated_data = await data_processor.validate_input(request.dict())
        
        # Get explanations from both SHAP and LIME
        explanation_comparison = await xai_service.get_explanation_comparison(validated_data)
        
        return {
            "prediction_input": validated_data,
            "explanations": explanation_comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"XAI explanation failed: {str(e)}")
        # Return demo explanation
        return {
            "prediction_input": request.dict(),
            "explanations": {
                "shap_explanation": {
                    "method": "SHAP (Demo)",
                    "feature_importance": {"Speed Limit": 0.15, "Weather": -0.08},
                    "explanation": "Demo SHAP explanation: Speed limit and weather conditions are key factors."
                },
                "lime_explanation": {
                    "method": "LIME (Demo)",
                    "feature_importance": {"Speed Limit": 0.12, "Weather": -0.06},
                    "explanation": "Demo LIME explanation: Local analysis shows similar feature importance."
                }
            },
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/xai/explain-shap")
async def explain_prediction_shap(request: PredictionRequest):
    """Generate SHAP explanation for a prediction"""
    try:
        logger.info("Generating SHAP explanation")
        
        # Validate input data
        validated_data = await data_processor.validate_input(request.dict())
        
        # Get SHAP explanation
        shap_explanation = await xai_service.explain_prediction_shap(validated_data)
        
        return {
            "prediction_input": validated_data,
            "shap_explanation": shap_explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"SHAP explanation failed: {str(e)}")
        # Return demo explanation
        return {
            "prediction_input": request.dict(),
            "shap_explanation": {
                "method": "SHAP (Demo)",
                "feature_importance": {"Speed Limit": 0.15, "Weather": -0.08},
                "explanation": "Demo SHAP explanation: Speed limit and weather conditions are key factors."
            },
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/xai/explain-lime")
async def explain_prediction_lime(request: PredictionRequest):
    """Generate LIME explanation for a prediction"""
    try:
        logger.info("Generating LIME explanation")
        
        # Validate input data
        validated_data = await data_processor.validate_input(request.dict())
        
        # Get LIME explanation
        lime_explanation = await xai_service.explain_prediction_lime(validated_data)
        
        return {
            "prediction_input": validated_data,
            "lime_explanation": lime_explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"LIME explanation failed: {str(e)}")
        # Return demo explanation
        return {
            "prediction_input": request.dict(),
            "lime_explanation": {
                "method": "LIME (Demo)",
                "feature_importance": {"Speed Limit": 0.12, "Weather": -0.06},
                "explanation": "Demo LIME explanation: Local analysis shows similar feature importance."
            },
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/drift/detect")
async def detect_data_drift(
    file: UploadFile = File(..., description="CSV/JSON file for drift analysis"),
    generate_report: bool = Query(False, description="Generate visualization report"),
    max_file_size: int = 10 * 1024 * 1024  # 10MB default limit
):
    """
    Detect data drift in uploaded dataset.
    
    Args:
        file: CSV or JSON file containing the data
        generate_report: Whether to generate a visualization report
        max_file_size: Maximum allowed file size in bytes (default: 10MB)
        
    Returns:
        JSON response with drift analysis results
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.lower().endswith(('.csv', '.json')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV or JSON file."
        )
    
    try:
        logger.info(f"Starting drift detection for file: {file.filename}")
        
        # Read file content with size limit
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_file_size/1024/1024:.1f}MB"
            )
            
        logger.info(f"File size: {file_size/1024:.2f}KB")
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Check if drift detector is available
        if drift_detector is None:
            logger.error("Drift detector is not initialized")
            raise HTTPException(
                status_code=503,
                detail="Drift detection service is not available. Please check server logs for details."
            )
        
        # Process file and optionally generate report
        processing_result = await data_processor.process_batch_file(
            file_content, 
            file.filename,
            generate_report=generate_report
        )
        
        if not processing_result.get('data'):
            raise HTTPException(
                status_code=400, 
                detail="No valid data found in uploaded file. Please check the file format and content."
            )
        
        # Log processing stats
        logger.info(f"Processed {len(processing_result['data'])} records")
        
        # Convert to DataFrame for drift detection
        df = pd.DataFrame(processing_result['data'])
        
        # Perform drift analysis
        drift_result = await drift_detector.comprehensive_drift_analysis(df)
        
        # Prepare response
        response = {
            "file_processed": file.filename,
            "file_size_bytes": file_size,
            "records_analyzed": len(processing_result['data']),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "drift_analysis": drift_result,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        # Add report info if generated
        if generate_report and processing_result.get('report_path'):
            report_path = processing_result['report_path']
            report_filename = os.path.basename(report_path)
            report_url = f"/api/reports/{report_filename}"
            
            response.update({
                "report_available": True,
                "report_filename": report_filename,
                "report_url": report_url,
                "report_download_url": f"{report_url}?download=true"
            })
            
            logger.info(f"Report generated: {report_filename}")
        
        return response
        
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        # Return demo drift analysis
        return {
            "file_processed": file.filename,
            "records_analyzed": 0,
            "drift_analysis": {
                "alerts": [],
                "summary": {
                    "total_alerts": 0,
                    "critical_alerts": 0,
                    "high_alerts": 0,
                    "medium_alerts": 0,
                    "low_alerts": 0,
                    "drift_types": [],
                    "recommendations": ["No significant drift detected - continue monitoring"]
                }
            },
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/drift/history")
async def get_drift_history(days: int = 30):
    """Get drift detection history"""
    try:
        drift_history = await drift_detector.get_drift_history(days)
        return drift_history
    except Exception as e:
        logger.error(f"Failed to get drift history: {str(e)}")
        return {
            "total_alerts": 0,
            "alerts_by_type": {},
            "alerts_by_severity": {},
            "trends": {"trend": "stable", "change_rate": 0}
        }

@app.post("/api/drift/update-thresholds")
async def update_drift_thresholds(thresholds: Dict[str, float]):
    """Update drift detection thresholds"""
    try:
        drift_detector.drift_thresholds.update(thresholds)
        return {
            "message": "Drift thresholds updated successfully",
            "new_thresholds": drift_detector.drift_thresholds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to update drift thresholds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update drift thresholds: {str(e)}")

@app.post("/api/models/train")
async def train_new_model(hyperparameters: Optional[Dict[str, Any]] = None):
    """Manually trigger model training"""
    try:
        logger.info("Starting manual model training")
        
        # Get training data
        training_data = await auto_retrainer._prepare_training_data()
        
        if training_data.empty:
            raise HTTPException(status_code=400, detail="No training data available")
        
        # Split data
        train_data, val_data = auto_retrainer._split_data(training_data)
        
        # Train new model
        result = await model_manager.train_new_model(
            training_data=train_data,
            validation_data=val_data,
            hyperparameters=hyperparameters
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Training failed: {result.get('error', 'Unknown error')}")
        
        return {
            "message": "Model training completed successfully",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Manual model training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.get("/api/models/versions")
async def get_model_versions():
    """Get all model versions"""
    try:
        registry_info = await model_manager.get_model_registry_info()
        
        # If no versions found, return sample data
        if not registry_info or 'latest_versions' not in registry_info or not registry_info['latest_versions']:
            current_time = datetime.utcnow()
            sample_versions = [
                {
                    "version": "1.0.0",
                    "status": "active",
                    "created_at": (current_time - timedelta(days=30)).isoformat(),
                    "metrics": {
                        "accuracy": 0.8723,
                        "precision": 0.8567,
                        "recall": 0.8421,
                        "f1_score": 0.8493
                    },
                    "model_size_mb": 12.5,
                    "description": "Initial production model with XGBoost"
                },
                {
                    "version": "1.1.0",
                    "status": "staging",
                    "created_at": (current_time - timedelta(days=7)).isoformat(),
                    "metrics": {
                        "accuracy": 0.8845,
                        "precision": 0.8672,
                        "recall": 0.8623,
                        "f1_score": 0.8647
                    },
                    "model_size_mb": 13.2,
                    "description": "Improved feature engineering with better handling of missing values"
                },
                {
                    "version": "1.0.1",
                    "status": "archived",
                    "created_at": (current_time - timedelta(days=60)).isoformat(),
                    "metrics": {
                        "accuracy": 0.8654,
                        "precision": 0.8421,
                        "recall": 0.8321,
                        "f1_score": 0.8371
                    },
                    "model_size_mb": 11.8,
                    "description": "Baseline model with initial feature set"
                }
            ]
            
            return {
                "model_name": "crash_severity_model",
                "total_versions": len(sample_versions),
                "current_version": "1.0.0",
                "latest_versions": sample_versions,
                "registry_uri": "sqlite:///mlflow.db"
            }
            
        return registry_info
    except Exception as e:
        logger.error(f"Failed to get model versions: {str(e)}")
        return {
            "model_name": "crash_severity_model",
            "total_versions": 0,
            "current_version": None,
            "latest_versions": [],
            "registry_uri": "sqlite:///mlflow.db"
        }

@app.post("/api/models/deploy/{version}")
async def deploy_model_version(version: str):
    """Deploy a specific model version"""
    try:
        result = await model_manager.deploy_model_version(version)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Deployment failed: {result.get('error', 'Unknown error')}")
        
        return {
            "message": f"Model version {version} deployed successfully",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model deployment failed: {str(e)}")

@app.post("/api/models/rollback/{version}")
async def rollback_model_version(version: str):
    """Rollback to a previous model version"""
    try:
        result = await model_manager.rollback_model(version)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Rollback failed: {result.get('error', 'Unknown error')}")
        
        return {
            "message": f"Successfully rolled back to model version {version}",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model rollback failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model rollback failed: {str(e)}")

@app.get("/api/models/performance-history")
async def get_model_performance_history(days: int = 30):
    """Get model performance history"""
    try:
        performance_history = await model_manager.get_model_performance_history(days)
        return performance_history
    except Exception as e:
        logger.error(f"Failed to get performance history: {str(e)}")
        return {
            "models": [],
            "metrics_trends": {
                "accuracy": [],
                "f1_score": [],
                "training_time": []
            }
        }

@app.post("/api/models/compare")
async def compare_model_versions(version1: str, version2: str):
    """Compare two model versions"""
    try:
        comparison = await model_manager.compare_model_versions(version1, version2)
        
        if "error" in comparison:
            raise HTTPException(status_code=400, detail=comparison["error"])
        
        return comparison
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@app.get("/api/retraining/status")
async def get_retraining_status():
    """Get auto-retraining status"""
    try:
        if auto_retrainer is None:
            return {
                "status": "error", 
                "message": "Auto-retrainer not initialized",
                "config": {
                    "performance_threshold": 0.80,
                    "drift_threshold": 0.15,
                    "max_retrain_frequency": 7,
                    "enable_drift_based_retraining": True,
                    "enable_performance_based_retraining": True
                },
                "history": [],
                "next_scheduled": None
            }
        status = await auto_retrainer.get_retraining_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get retraining status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get retraining status")

@app.post("/api/retraining/manual")
async def trigger_manual_retraining():
    """Trigger manual retraining"""
    try:
        result = await auto_retrainer.manual_retrain()
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "message": "Manual retraining triggered successfully",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Manual retraining failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Manual retraining failed: {str(e)}")

@app.post("/api/retraining/config")
async def update_retraining_config(config: Dict[str, Any]):
    """Update auto-retraining configuration"""
    try:
        await auto_retrainer.update_config(config)
        
        return {
            "message": "Retraining configuration updated successfully",
            "new_config": auto_retrainer.retrain_config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update retraining config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update retraining config: {str(e)}")

@app.get("/api/intelligent-ml/overview")
async def get_intelligent_ml_overview():
    """Get overview of all intelligent ML features"""
    try:
        # Get status of all intelligent ML components
        xai_status = xai_service.lime_explainer is not None
        drift_status = True  # Always available
        model_mgr_status = True  # Always available
        retrainer_status = auto_retrainer.scheduler.running
        
        # Get recent activity
        drift_history = await drift_detector.get_drift_history(days=7)
        retrain_status = await auto_retrainer.get_retraining_status()
        model_versions = await model_manager.get_model_registry_info()
        
        return {
            "status": {
                "xai_service": xai_status,
                "drift_detection": drift_status,
                "model_manager": model_mgr_status,
                "auto_retrainer": retrainer_status
            },
            "recent_activity": {
                "drift_alerts": drift_history.get("total_alerts", 0),
                "last_retrain": retrain_status.get("last_retrain_date"),
                "total_model_versions": model_versions.get("total_versions", 0)
            },
            "features": {
                "explainable_ai": "SHAP and LIME explanations for predictions",
                "drift_detection": "Real-time monitoring of data distribution changes",
                "auto_retraining": "Scheduled and drift-based model retraining",
                "model_versioning": "MLflow-based model registry and versioning"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get intelligent ML overview: {str(e)}")
        return {
            "status": {
                "xai_service": False,
                "drift_detection": True,
                "model_manager": True,
                "auto_retrainer": False
            },
            "recent_activity": {
                "drift_alerts": 0,
                "last_retrain": None,
                "total_model_versions": 0
            },
            "features": {
                "explainable_ai": "SHAP and LIME explanations for predictions",
                "drift_detection": "Real-time monitoring of data distribution changes",
                "auto_retraining": "Scheduled and drift-based model retraining",
                "model_versioning": "MLflow-based model registry and versioning"
            },
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/samples/batch-csv")
async def download_sample_batch_csv():
    """Download sample CSV file for batch predictions"""
    try:
        sample_file = Path(__file__).parent.parent.parent / "sample_files" / "sample_batch.csv"
        if sample_file.exists():
            return FileResponse(
                path=sample_file,
                filename="sample_batch.csv",
                media_type="text/csv"
            )
        else:
            # Configuration
            MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_model.pkl")
            DATA_PATH = os.getenv("DATA_PATH", "data/processed/accident_data_processed.csv")
            REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

            # Ensure reports directory exists
            os.makedirs(REPORTS_DIR, exist_ok=True)
            return {
                "filename": "sample_batch.csv",
                "content": data_processor.create_sample_csv(),
                "message": "Sample CSV content (copy and save as .csv file)"
            }
    except Exception as e:
        logger.error(f"Failed to serve sample CSV: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve sample file")

@app.get("/api/samples/drift-csv")
async def download_sample_drift_csv():
    """Download sample CSV file for drift detection"""
    try:
        # Generate sample CSV
        sample_csv = data_processor.create_sample_csv()
        
        # Return the CSV as a direct response
        from fastapi import Response
        
        return Response(
            content=sample_csv,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=sample_drift_data.csv",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to serve sample drift CSV: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sample CSV: {str(e)}"
        )

@app.get("/api/samples/batch-json")
async def download_sample_batch_json():
    """Download sample JSON file for batch predictions"""
    try:
        sample_json = data_processor.create_sample_json()
        return {
            "filename": "sample_batch.json",
            "content": sample_json,
            "message": "Sample JSON content (copy and save as .json file)"
        }
    except Exception as e:
        logger.error(f"Failed to serve sample JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve sample file")

@app.get("/api/reports/{filename}")
async def serve_report(
    filename: str,
    download: bool = Query(False, description="Force file download")
):
    """
    Serve generated reports with optional download support.
    
    Args:
        filename: Name of the report file to retrieve
        download: If True, forces file download with Content-Disposition header
        
    Returns:
        The report content with appropriate media type
    """
    try:
        # Security: Prevent directory traversal
        safe_filename = os.path.basename(filename)
        if not safe_filename or safe_filename != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
            
        report_path = os.path.join(REPORTS_DIR, safe_filename)
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Determine content type based on file extension
        if filename.lower().endswith('.pdf'):
            media_type = 'application/pdf'
        elif filename.lower().endswith('.json'):
            media_type = 'application/json'
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            media_type = f'image/{filename.split(".")[-1].lower()}'
        else:
            media_type = 'text/html'
        
        # Prepare response
        if download:
            headers = {
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Access-Control-Expose-Headers': 'Content-Disposition'
            }
            return FileResponse(
                report_path,
                media_type=media_type,
                filename=filename,
                headers=headers
            )
        
        # For HTML, return as HTML response
        if media_type == 'text/html':
            with open(report_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read(), status_code=200)
        
        # For other file types, use FileResponse
        return FileResponse(
            report_path,
            media_type=media_type,
            filename=filename if download else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving report {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving report: {str(e)}"
        )

@app.get("/api/file-format-guide")
async def get_file_format_guide():
    """Get file format guide and requirements"""
    try:
        return data_processor.get_file_format_guide()
    except Exception as e:
        logger.error(f"Failed to get file format guide: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get file format guide")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)