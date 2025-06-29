# 🧠 Intelligent ML System Features

This document provides a comprehensive overview of the **Intelligent ML System** features that have been integrated into the Road Crash Analytics Platform. These features enhance the system's capabilities with explainable AI, drift detection, and automated model management.

## 🎯 Overview

The Intelligent ML System consists of four main components:

1. **🤖 Explainable AI (XAI)** - SHAP and LIME explanations
2. **📊 Data Drift Detection** - Real-time monitoring and alerts
3. **🔄 Auto-Retraining** - Scheduled and drift-based retraining
4. **📦 Model Management** - MLflow-based versioning and deployment

## 🚀 Quick Start

### Prerequisites

```bash
# Install additional dependencies
pip install shap lime mlflow dvc apscheduler alibi-detect prometheus-client

# For frontend
npm install
```

### Running the System

```bash
# Backend (with intelligent ML features)
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
npm run dev
```

## 🤖 Explainable AI (XAI)

### Features

- **SHAP Explanations**: Model-agnostic explanations using SHapley Additive exPlanations
- **LIME Explanations**: Local Interpretable Model-agnostic Explanations
- **Feature Importance Analysis**: Understanding which factors influence predictions
- **Natural Language Explanations**: Human-readable explanations of model decisions
- **Comparison Analysis**: Side-by-side comparison of SHAP and LIME results

### API Endpoints

```typescript
// Generate comprehensive XAI explanation
POST /api/xai/explain
{
  "road_type": "Urban",
  "road_class": "A",
  "speed_limit": 30,
  // ... other features
}

// Generate SHAP-only explanation
POST /api/xai/explain-shap

// Generate LIME-only explanation
POST /api/xai/explain-lime
```

### Example Response

```json
{
  "prediction_input": { /* input features */ },
  "explanations": {
    "shap_explanation": {
      "method": "SHAP",
      "feature_importance": {
        "Speed Limit": 0.15,
        "Weather Conditions": -0.08,
        "Road Surface Conditions": 0.12
      },
      "top_features": [
        ["Speed Limit", 0.15],
        ["Road Surface Conditions", 0.12]
      ],
      "explanation": "The prediction is primarily influenced by: 'Speed Limit' being '30' decreases the risk; 'Road Surface Conditions' being 'Dry' decreases the risk."
    },
    "lime_explanation": {
      "method": "LIME",
      "feature_importance": { /* similar structure */ },
      "explanation": "Local explanation for this specific prediction..."
    },
    "agreement_score": 0.85,
    "consensus_features": ["Speed Limit", "Weather Conditions"]
  }
}
```

### Frontend Integration

The XAI features are accessible through the **Intelligent ML** tab in the web interface, providing:

- Interactive input forms for testing explanations
- Visual comparison of SHAP vs LIME results
- Feature importance charts and visualizations
- Natural language explanation display

## 📊 Data Drift Detection

### Features

- **Feature Drift Detection**: Monitor changes in individual feature distributions
- **Label Drift Detection**: Track changes in target variable distribution
- **Covariate Drift Detection**: Detect shifts in input feature relationships
- **Concept Drift Detection**: Monitor model performance degradation
- **Real-time Monitoring**: Continuous drift analysis with configurable thresholds
- **Alert System**: Automated notifications when drift exceeds thresholds
- **Historical Tracking**: Maintain drift detection history and trends

### Drift Types

1. **Feature Drift**: Changes in individual feature distributions
2. **Label Drift**: Changes in target variable distribution
3. **Covariate Drift**: Changes in feature relationships
4. **Concept Drift**: Changes in the underlying data-generating process

### API Endpoints

```typescript
// Detect drift in uploaded dataset
POST /api/drift/detect
Content-Type: multipart/form-data
file: <CSV/JSON file>

// Get drift detection history
GET /api/drift/history?days=30

// Update drift thresholds
POST /api/drift/update-thresholds
{
  "feature_drift": 0.1,
  "label_drift": 0.15,
  "covariate_drift": 0.2,
  "concept_drift": 0.25
}
```

### Example Drift Alert

```json
{
  "drift_type": "feature_drift",
  "severity": "high",
  "feature_name": "Speed Limit",
  "drift_score": 0.18,
  "threshold": 0.15,
  "timestamp": "2024-01-15T10:30:00Z",
  "description": "Feature 'Speed Limit' shows distribution drift (score: 0.180)",
  "recommendations": [
    "Monitor this feature more closely",
    "Consider retraining the model if drift persists",
    "Investigate if this represents a real change in the data generating process"
  ]
}
```

### Configuration

```python
# Drift detection thresholds
drift_thresholds = {
    'feature_drift': 0.1,    # 10% change in feature distribution
    'label_drift': 0.15,     # 15% change in label distribution
    'covariate_drift': 0.2,  # 20% change in covariate shift
    'concept_drift': 0.25    # 25% change in model performance
}
```

## 🔄 Auto-Retraining

### Features

- **Scheduled Retraining**: Weekly, monthly, or custom schedule
- **Drift-Based Retraining**: Automatic retraining when drift is detected
- **Performance-Based Retraining**: Retrain when model performance drops
- **Configurable Triggers**: Multiple conditions for triggering retraining
- **Training Data Management**: Automatic collection and preparation of training data
- **Model Versioning**: Track all retrained models with metadata
- **Rollback Capability**: Revert to previous model versions if needed

### Retraining Triggers

1. **Scheduled**: Based on time intervals (daily, weekly, monthly)
2. **Drift-Based**: When data drift exceeds thresholds
3. **Performance-Based**: When model accuracy drops below threshold
4. **Manual**: User-triggered retraining

### API Endpoints

```typescript
// Get retraining status
GET /api/retraining/status

// Trigger manual retraining
POST /api/retraining/manual

// Update retraining configuration
POST /api/retraining/config
{
  "schedule_type": "weekly",
  "retrain_day": "sunday",
  "retrain_time": "02:00",
  "min_samples_for_retraining": 1000,
  "performance_threshold": 0.80,
  "drift_threshold": 0.15,
  "max_retrain_frequency": 7,
  "enable_drift_based_retraining": true,
  "enable_performance_based_retraining": true
}
```

### Configuration Options

```python
retrain_config = {
    "schedule_type": "weekly",           # daily, weekly, monthly, drift_based
    "retrain_day": "sunday",            # For weekly schedule
    "retrain_time": "02:00",            # 2 AM
    "min_samples_for_retraining": 1000, # Minimum data required
    "performance_threshold": 0.80,      # Minimum accuracy
    "drift_threshold": 0.15,            # Drift score threshold
    "max_retrain_frequency": 7,         # Maximum days between retrains
    "enable_drift_based_retraining": True,
    "enable_performance_based_retraining": True
}
```

### Retraining Process

1. **Data Collection**: Gather recent predictions and outcomes
2. **Data Validation**: Ensure sufficient and quality data
3. **Model Training**: Train new model with updated data
4. **Performance Evaluation**: Compare with current model
5. **Deployment**: Deploy if performance improves
6. **Rollback**: Revert if performance degrades

## 📦 Model Management

### Features

- **MLflow Integration**: Professional model registry and tracking
- **Version Control**: Track all model versions with metadata
- **Performance Tracking**: Monitor model performance over time
- **Model Comparison**: Compare different model versions
- **Deployment Management**: Easy model deployment and rollback
- **Artifact Storage**: Store model files, metrics, and parameters
- **Experiment Tracking**: Track training experiments and hyperparameters

### API Endpoints

```typescript
// Train new model
POST /api/models/train
{
  "hyperparameters": {
    "gnn_hidden_dim": 256,
    "bilstm_hidden_dim": 128,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "dropout": 0.2
  }
}

// Get model versions
GET /api/models/versions

// Deploy specific version
POST /api/models/deploy/{version}

// Rollback to previous version
POST /api/models/rollback/{version}

// Compare model versions
POST /api/models/compare
{
  "version1": "1.0.0",
  "version2": "1.1.0"
}

// Get performance history
GET /api/models/performance-history?days=30
```

### Model Version Information

```json
{
  "version": "1.2.0",
  "status": "ready",
  "created_at": "2024-01-15T10:30:00Z",
  "metrics": {
    "training_accuracy": 0.89,
    "validation_accuracy": 0.85,
    "f1_score": 0.84,
    "precision": 0.86,
    "recall": 0.82
  },
  "parameters": {
    "gnn_hidden_dim": 256,
    "bilstm_hidden_dim": 128,
    "learning_rate": 0.001
  },
  "model_size_mb": 15.2,
  "training_time_seconds": 1800
}
```

## 🎨 Frontend Integration

### Intelligent ML Dashboard

The frontend includes a comprehensive **Intelligent ML** tab with:

1. **Overview Section**:
   - System status indicators
   - Recent activity summary
   - Feature descriptions

2. **XAI Section**:
   - Interactive prediction input form
   - SHAP and LIME explanation display
   - Feature importance visualizations
   - Natural language explanations

3. **Drift Detection Section**:
   - File upload for drift analysis
   - Drift alert display
   - Historical drift trends
   - Threshold configuration

4. **Model Management Section**:
   - Model version listing
   - Performance metrics
   - Deployment controls
   - Version comparison

5. **Auto-Retraining Section**:
   - Retraining status display
   - Configuration management
   - Manual retraining trigger
   - Schedule information

### Key Components

- **IntelligentML.tsx**: Main component for all intelligent ML features
- **XAITab**: Explainable AI functionality
- **DriftTab**: Drift detection interface
- **ModelsTab**: Model management interface
- **RetrainingTab**: Auto-retraining controls

## 🔧 Configuration

### Environment Variables

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=crash_severity_prediction

# Drift Detection
DRIFT_FEATURE_THRESHOLD=0.1
DRIFT_LABEL_THRESHOLD=0.15
DRIFT_COVARIATE_THRESHOLD=0.2
DRIFT_CONCEPT_THRESHOLD=0.25

# Auto-Retraining
RETRAIN_SCHEDULE_TYPE=weekly
RETRAIN_MIN_SAMPLES=1000
RETRAIN_PERFORMANCE_THRESHOLD=0.80
```

### Configuration Files

```json
// backend/config/retrain_config.json
{
  "schedule_type": "weekly",
  "retrain_day": "sunday",
  "retrain_time": "02:00",
  "min_samples_for_retraining": 1000,
  "performance_threshold": 0.80,
  "drift_threshold": 0.15,
  "max_retrain_frequency": 7,
  "enable_drift_based_retraining": true,
  "enable_performance_based_retraining": true
}
```

## 📈 Monitoring and Alerts

### Metrics Tracked

1. **Model Performance**:
   - Accuracy, F1-score, Precision, Recall
   - Training and validation metrics
   - Inference time and model size

2. **Drift Metrics**:
   - Feature distribution changes
   - Label distribution changes
   - Covariate shift scores
   - Concept drift indicators

3. **System Health**:
   - Service availability
   - API response times
   - Error rates and logs

### Alert Types

1. **High Severity Drift**: Immediate action required
2. **Performance Degradation**: Model retraining recommended
3. **System Errors**: Technical issues requiring attention
4. **Training Failures**: Model training process issues

## 🚀 Deployment

### Production Considerations

1. **MLflow Server**: Set up dedicated MLflow tracking server
2. **Database**: Use production-grade database (PostgreSQL)
3. **Monitoring**: Implement comprehensive logging and monitoring
4. **Security**: Secure API endpoints and model access
5. **Backup**: Regular backup of models and configurations

### Docker Deployment

```dockerfile
# Example Dockerfile for intelligent ML features
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔍 Troubleshooting

### Common Issues

1. **XAI Explanations Not Working**:
   - Check if models are loaded
   - Verify SHAP and LIME dependencies
   - Check input data format

2. **Drift Detection Alerts**:
   - Review drift thresholds
   - Check data quality
   - Verify reference data

3. **Auto-Retraining Failures**:
   - Check training data availability
   - Verify MLflow configuration
   - Review system resources

4. **Model Deployment Issues**:
   - Check MLflow server connectivity
   - Verify model file permissions
   - Review deployment logs

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Additional Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Alibi Detect Documentation](https://docs.seldon.io/projects/alibi-detect/)

## 🤝 Contributing

To contribute to the Intelligent ML features:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Follow the project's coding standards

---

**Note**: These intelligent ML features are designed to work alongside the existing crash analytics system, providing enhanced capabilities for model explainability, monitoring, and management. The system gracefully degrades to demo mode when full ML models are not available. 