# 🚗 Road Crash Analytics Platform - Startup Guide

## 🚀 Quick Start (Recommended)

### Option 1: Use the Master Startup Script (Easiest)
```bash
# Navigate to the project directory
cd C:\Users\HP\Downloads\project-bolt-sb1-vlcuqs6j\project

# Run the master startup script
start_project.bat
```

This will automatically start both backend and frontend in separate windows.

---

## 🔧 Manual Startup (If you prefer step-by-step)

### Step 1: Start Backend Server
```bash
# Navigate to project directory
cd C:\Users\HP\Downloads\project-bolt-sb1-vlcuqs6j\project

# Start backend using the fixed script
run_backend.bat
```

**OR manually:**
```bash
cd backend
python run_backend.py
```

### Step 2: Start Frontend Server
```bash
# In a new terminal, navigate to project directory
cd C:\Users\HP\Downloads\project-bolt-sb1-vlcuqs6j\project

# Start frontend
run_frontend.bat
```

**OR manually:**
```bash
npm run dev
```

---

## 🌐 Access Your Application

Once both servers are running:

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health

---

## 📁 File Upload Testing

### Sample CSV Format
The system expects CSV files with these columns:
```csv
road_type,road_class,speed_limit,area_type,junction_location,junction_control,junction_detail,hazards,road_surface_conditions,vehicle_type,light_conditions,weather_conditions
Single carriageway,A,30,Urban,Not at junction,Give way or uncontrolled,Not at junction,None,Dry,Car,Daylight,Fine no high winds
```

### Test File Location
A sample test file is available at: `project/backend/test_data_sample.csv`

---

## 🔍 Troubleshooting

### Backend Not Starting
**Error**: `ModuleNotFoundError: No module named 'app'`

**Solution**: Use the provided `run_backend.py` script instead of direct uvicorn command.

### Frontend Can't Connect to Backend
**Error**: "Unable to connect to the server"

**Solution**: 
1. Ensure backend is running on http://localhost:8000
2. Check if port 8000 is not blocked by firewall
3. Verify backend health at http://localhost:8000/api/health

### File Upload Issues
**Error**: "Unable to process file"

**Solution**:
1. Ensure file is CSV or JSON format
2. Check column names match expected format
3. Verify backend is running and accessible

### Port Already in Use
**Error**: "Address already in use"

**Solution**:
```bash
# Find and kill process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Find and kill process using port 5173
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

---

## 🎯 Features Available

### ✅ Working Features
- **Single Prediction**: Enter crash data and get severity predictions
- **Batch Upload**: Upload CSV files with multiple crash records
- **Analytics Dashboard**: View charts and statistics
- **Demo Mode**: All features work with simulated predictions
- **Real-time Processing**: Live data validation and processing

### 📊 Batch Prediction Features
- **File Validation**: Accepts CSV and JSON files
- **Data Processing**: Validates and normalizes input data
- **Batch Processing**: Handles multiple records efficiently
- **Results Summary**: Provides severity distribution and statistics
- **Error Handling**: Graceful handling of invalid records

---

## 🛠️ Development Commands

### Backend Development
```bash
cd backend
python run_backend.py
```

### Frontend Development
```bash
npm run dev
```

### Install Dependencies (if needed)
```bash
# Backend dependencies
cd backend
pip install -r requirements.txt

# Frontend dependencies
npm install
```

---

## 📞 Support

If you encounter any issues:

1. **Check the logs** in the terminal windows
2. **Verify both servers are running** on correct ports
3. **Test the health endpoint**: http://localhost:8000/api/health
4. **Check file format** matches the expected CSV structure

The application is designed to work in demo mode even without real ML models, so all features should be functional! 