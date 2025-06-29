# 🚗 Road Crash Analytics - User Guide

This comprehensive guide will help you use all the features of the Road Crash Analytics application, including Explainable AI (XAI), Drift Detection, and Batch Processing.

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Single Prediction](#single-prediction)
3. [Batch Prediction](#batch-prediction)
4. [Explainable AI (XAI)](#explainable-ai-xai)
5. [Drift Detection](#drift-detection)
6. [Intelligent ML Features](#intelligent-ml-features)
7. [Troubleshooting](#troubleshooting)
8. [Sample Files](#sample-files)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Starting the Application

#### Option 1: Automatic Startup (Recommended)
```bash
# Windows
start_app.bat

# Mac/Linux
python start_application.py
```

#### Option 2: Manual Startup
```bash
# Terminal 1 - Backend
cd project/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend
cd project
npm install
npm run dev
```

### Accessing the Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs

## 🎯 Single Prediction

### How to Use
1. Navigate to the **Prediction** tab
2. Fill in the form with crash scenario details:
   - **Speed Limit**: 10, 20, 30, 40, 50, 60, 70 mph
   - **Road Type**: Single carriageway, Dual carriageway, Motorway, Slip road
   - **Light Conditions**: Daylight, Darkness - lights lit, Darkness - no lighting
   - **Weather Conditions**: Fine no high winds, Raining without high winds, etc.
   - **Road Surface**: Dry, Wet or damp, Snow, Frost or ice
   - **Junction Details**: Not at junction, Roundabout, T junction, etc.
   - **And more...**

3. Click **Predict Severity**
4. View results including:
   - **Predicted Severity**: Slight, Serious, or Fatal
   - **Confidence Score**: 0.0 to 1.0
   - **Risk Level**: Very Low, Low, Medium, High, Critical
   - **Probability Distribution**: Breakdown for each severity level

### Example Input
```json
{
  "speed_limit": 30,
  "road_type": "Single carriageway",
  "light_conditions": "Daylight",
  "weather_conditions": "Fine no high winds",
  "road_surface_conditions": "Dry",
  "junction_detail": "Not at junction or within 20 metres",
  "junction_control": "Give way or uncontrolled",
  "pedestrian_crossing_human_control": "None within 50 metres",
  "pedestrian_crossing_physical_facilities": "No physical crossing facilities within 50 metres",
  "carriageway_hazards": "None",
  "urban_or_rural_area": "Urban",
  "did_police_officer_attend_scene_of_accident": "No",
  "trunk_road_flag": "Non-trunk",
  "lsoa_of_accident_location": "E01000001"
}
```

## 📊 Batch Prediction

### How to Use
1. Navigate to the **Batch Upload** tab
2. **Download Sample Files**:
   - Click "Download Sample CSV" for CSV format
   - Click "Download Sample JSON" for JSON format
3. **Prepare Your File**:
   - Use the sample files as templates
   - Ensure all required columns are present
   - Follow the data format guidelines
4. **Upload File**:
   - Click "Choose File" or drag and drop
   - Select your CSV or JSON file
   - Click "Upload and Predict"
5. **View Results**:
   - Summary statistics
   - Individual predictions for each record
   - Download results as CSV

### Required File Format

#### CSV Format
```csv
speed_limit,road_type,light_conditions,weather_conditions,road_surface_conditions,junction_detail,junction_control,pedestrian_crossing_human_control,pedestrian_crossing_physical_facilities,carriageway_hazards,urban_or_rural_area,did_police_officer_attend_scene_of_accident,trunk_road_flag,lsoa_of_accident_location
30,Single carriageway,Daylight,Fine no high winds,Dry,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000001
40,Dual carriageway,Darkness - lights lit,Raining without high winds,Wet or damp,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000002
```

#### JSON Format
```json
[
  {
    "speed_limit": 30,
    "road_type": "Single carriageway",
    "light_conditions": "Daylight",
    "weather_conditions": "Fine no high winds",
    "road_surface_conditions": "Dry",
    "junction_detail": "Not at junction or within 20 metres",
    "junction_control": "Give way or uncontrolled",
    "pedestrian_crossing_human_control": "None within 50 metres",
    "pedestrian_crossing_physical_facilities": "No physical crossing facilities within 50 metres",
    "carriageway_hazards": "None",
    "urban_or_rural_area": "Urban",
    "did_police_officer_attend_scene_of_accident": "No",
    "trunk_road_flag": "Non-trunk",
    "lsoa_of_accident_location": "E01000001"
  }
]
```

## 🤖 Explainable AI (XAI)

### What is XAI?
Explainable AI helps you understand **why** the model made a particular prediction by showing which features were most important.

### How to Use XAI
1. **Make a Prediction**: First, make a single prediction
2. **Get Explanation**: Click "Explain Prediction" button
3. **View Results**: See both SHAP and LIME explanations

### Understanding XAI Results

#### SHAP (SHapley Additive exPlanations)
- **Method**: Global feature importance using game theory
- **Output**: 
  - Feature importance scores (positive = increases risk, negative = decreases risk)
  - Overall explanation of prediction factors
- **Use Case**: Understanding which features most influence the model's decision

#### LIME (Local Interpretable Model-agnostic Explanations)
- **Method**: Local explanation for the specific prediction
- **Output**:
  - Local feature importance for this specific case
  - Explanation of how features contributed to this prediction
- **Use Case**: Understanding why this specific prediction was made

### Example XAI Output
```json
{
  "shap_explanation": {
    "method": "SHAP",
    "feature_importance": {
      "speed_limit": 0.15,
      "weather_conditions": -0.08,
      "light_conditions": 0.12
    },
    "explanation": "Speed limit increases risk. Light conditions increase risk. Weather conditions decrease risk."
  },
  "lime_explanation": {
    "method": "LIME",
    "feature_importance": {
      "speed_limit": 0.12,
      "weather_conditions": -0.06,
      "light_conditions": 0.10
    },
    "explanation": "Speed limit contributes positively. Light conditions contribute positively. Weather conditions contribute negatively."
  }
}
```

## 📈 Drift Detection

### What is Data Drift?
Data drift occurs when the statistical properties of incoming data change over time, potentially affecting model performance.

### How to Use Drift Detection
1. **Navigate to Intelligent ML Tab**: Go to the Intelligent ML section
2. **Upload Data**: 
   - Click "Upload File for Drift Detection"
   - Use the same format as batch prediction files
   - Download sample drift data for reference
3. **Run Analysis**: Click "Detect Drift"
4. **View Results**:
   - Drift alerts and severity levels
   - Feature-wise drift analysis
   - Recommendations for action

### Understanding Drift Results

#### Alert Types
- **Statistical Drift**: Changes in statistical properties (p-values)
- **Distribution Drift**: Changes in data distributions
- **Feature Drift**: Changes in individual feature patterns

#### Severity Levels
- **Critical**: High drift detected - immediate action required
- **High**: Significant drift - monitor closely
- **Medium**: Moderate drift - consider updates
- **Low**: Minor drift - continue monitoring

### Example Drift Output
```json
{
  "alerts": [
    {
      "type": "statistical_drift",
      "feature": "speed_limit",
      "severity": "high",
      "message": "Statistical drift detected in speed_limit (p-value: 0.0012)",
      "details": {
        "drift_detected": true,
        "p_value": 0.0012,
        "test_type": "t_test"
      }
    }
  ],
  "summary": {
    "total_alerts": 3,
    "critical_alerts": 1,
    "high_alerts": 2,
    "medium_alerts": 0,
    "low_alerts": 0,
    "overall_drift_score": 0.25,
    "recommendations": [
      "High drift detected - consider retraining the model",
      "Review data collection processes"
    ]
  }
}
```

## 🧠 Intelligent ML Features

### Available Features

#### 1. Model Management
- **Model Versions**: View all trained model versions
- **Model Comparison**: Compare performance between versions
- **Model Deployment**: Deploy specific model versions
- **Performance History**: Track model performance over time

#### 2. Auto-Retraining
- **Scheduled Retraining**: Automatic retraining on schedule
- **Drift-Based Retraining**: Retrain when drift is detected
- **Performance-Based Retraining**: Retrain when performance drops
- **Manual Retraining**: Trigger retraining manually

#### 3. Configuration
- **Drift Thresholds**: Adjust sensitivity of drift detection
- **Retraining Schedule**: Set retraining frequency
- **Performance Thresholds**: Set performance targets

### How to Use Intelligent ML
1. **Navigate to Intelligent ML Tab**
2. **Overview**: See status of all intelligent features
3. **XAI**: Generate explanations for predictions
4. **Drift Detection**: Monitor data drift
5. **Model Management**: Manage model versions
6. **Auto-Retraining**: Configure and monitor retraining

## 🔧 Troubleshooting

### Common Issues

#### 1. "Network Error" Messages
**Cause**: Backend server not running
**Solution**: 
```bash
cd project/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. File Upload Fails
**Cause**: Incorrect file format
**Solution**:
- Download and use sample files as templates
- Ensure all required columns are present
- Check file encoding (use UTF-8)

#### 3. XAI Not Working
**Cause**: Missing dependencies
**Solution**:
```bash
pip install shap lime
```

#### 4. Drift Detection Returns No Results
**Cause**: Insufficient data or format issues
**Solution**:
- Ensure file has at least 5-10 records
- Use correct column names
- Check data types match requirements

#### 5. Predictions Always Return Same Result
**Cause**: Running in demo mode
**Solution**: 
- Check if ML model files are present in `backend/models/`
- Demo mode provides realistic predictions for testing

### Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Missing required fields" | Incomplete data | Fill all required fields |
| "Invalid file format" | Wrong file type | Use CSV or JSON format |
| "No valid records found" | Data validation failed | Check data format and values |
| "Backend not available" | Server not running | Start backend server |
| "Model not loaded" | Missing model files | Check `backend/models/` directory |

## 📁 Sample Files

### Available Sample Files
1. **sample_batch.csv** - For batch predictions
2. **sample_drift_data.csv** - For drift detection
3. **sample_batch.json** - JSON format for batch predictions

### How to Download
- **Frontend**: Use download buttons in Batch Upload and Intelligent ML tabs
- **API**: 
  - `GET /api/samples/batch-csv`
  - `GET /api/samples/drift-csv`
  - `GET /api/samples/batch-json`

### File Format Guide
- **API**: `GET /api/file-format-guide`
- **Frontend**: Available in Batch Upload tab

## 📞 Getting Help

### Documentation
- **API Documentation**: http://localhost:8000/api/docs
- **Troubleshooting Guide**: See `TROUBLESHOOTING.md`
- **Technical Documentation**: See project documentation

### Support
1. **Check Logs**: Look at browser console (F12) and backend terminal
2. **Run Tests**: Use `python test_api.py` to verify functionality
3. **Health Check**: Visit `http://localhost:8000/api/health`

### System Requirements
- **Python**: 3.8+
- **Node.js**: 16+
- **RAM**: 4GB+
- **Storage**: 2GB+ free space
- **Browser**: Chrome, Firefox, Safari, Edge

---

**Remember**: The application includes a robust demo mode that works even when ML models aren't available. This ensures you can always test the interface and functionality.

For additional help, check the troubleshooting guide or run the test script to verify your setup. 