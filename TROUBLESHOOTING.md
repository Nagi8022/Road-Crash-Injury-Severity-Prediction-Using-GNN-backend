# 🚗 Road Crash Analytics - Troubleshooting Guide

This guide helps you resolve common issues when running the Road Crash Analytics application.

## 🚨 Common Issues and Solutions

### 1. Network Errors / Backend Not Available

**Symptoms:**
- "Network error" messages in frontend
- "Backend not available" errors
- API calls failing

**Solutions:**

#### A. Backend Not Starting
```bash
# Check if port 8000 is already in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Mac/Linux

# Kill process using port 8000 if needed
taskkill /PID <PID> /F        # Windows
kill -9 <PID>                 # Mac/Linux
```

#### B. Virtual Environment Issues
```bash
# Recreate virtual environment
cd backend
rm -rf venv/                  # Mac/Linux
rmdir /s venv                 # Windows
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
```

#### C. Missing Dependencies
```bash
# Install missing packages
pip install fastapi uvicorn torch pandas numpy scikit-learn sqlalchemy requests
```

### 2. Frontend Not Loading

**Symptoms:**
- Frontend shows blank page
- "Cannot connect to server" errors
- Port 5173 not accessible

**Solutions:**

#### A. Node.js Dependencies
```bash
# Reinstall node modules
rm -rf node_modules package-lock.json
npm install
```

#### B. Port Conflicts
```bash
# Check if port 5173 is in use
netstat -ano | findstr :5173  # Windows
lsof -i :5173                 # Mac/Linux
```

#### C. Vite Configuration Issues
```bash
# Clear Vite cache
npm run dev -- --force
```

### 3. Prediction Failures

**Symptoms:**
- Predictions return errors
- "Model not loaded" messages
- Demo mode always active

**Solutions:**

#### A. Model Files Missing
```bash
# Check if model files exist
ls backend/models/
# Should show: bilstm_model.pth, gnn_model.pth, label_encoders.pkl, etc.

# If missing, the app will run in demo mode
# Demo mode provides realistic predictions for testing
```

#### B. Memory Issues
```bash
# Increase Python memory limit
export PYTHONMALLOC=malloc  # Mac/Linux
set PYTHONMALLOC=malloc      # Windows
```

#### C. CUDA/GPU Issues
```bash
# Force CPU usage if GPU issues occur
export CUDA_VISIBLE_DEVICES=""  # Mac/Linux
set CUDA_VISIBLE_DEVICES=       # Windows
```

### 4. Database Issues

**Symptoms:**
- Database connection errors
- Analytics not loading
- Prediction history missing

**Solutions:**

#### A. SQLite Database
```bash
# Check database file
ls backend/crash_analytics.db

# If corrupted, delete and restart
rm backend/crash_analytics.db
# App will recreate database automatically
```

#### B. Database Permissions
```bash
# Ensure write permissions
chmod 666 backend/crash_analytics.db  # Mac/Linux
```

### 5. Intelligent ML Features Not Working

**Symptoms:**
- XAI explanations fail
- Drift detection errors
- Auto-retraining not working

**Solutions:**

#### A. Missing ML Dependencies
```bash
# Install additional ML packages
pip install shap lime mlflow dvc apscheduler
```

#### B. MLflow Issues
```bash
# Initialize MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

#### C. SHAP/LIME Issues
```bash
# Force CPU usage for SHAP
export SHAP_FORCE_CPU=1  # Mac/Linux
set SHAP_FORCE_CPU=1     # Windows
```

## 🔧 Manual Startup Commands

### Backend Only
```bash
cd backend
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Only
```bash
npm install
npm run dev
```

### Test API
```bash
cd backend
python test_api.py
```

## 📊 Health Check Endpoints

Test these endpoints to diagnose issues:

- **Health Check**: `http://localhost:8000/api/health`
- **API Docs**: `http://localhost:8000/api/docs`
- **Root**: `http://localhost:8000/`

Expected health check response:
```json
{
  "status": "healthy",
  "services": {
    "database": true,
    "ml_models": true,
    "data_processor": true,
    "xai_service": true,
    "drift_detection": true,
    "model_manager": true,
    "auto_retrainer": true
  },
  "demo_mode": false
}
```

## 🐛 Debug Mode

Enable debug logging:

```bash
# Backend debug
export LOG_LEVEL=DEBUG  # Mac/Linux
set LOG_LEVEL=DEBUG     # Windows

# Frontend debug
npm run dev -- --debug
```

## 📝 Log Files

Check these locations for error logs:

- **Backend logs**: Console output when running uvicorn
- **Frontend logs**: Browser developer console (F12)
- **Database logs**: SQLite database file

## 🔄 Reset Everything

If all else fails, reset the entire application:

```bash
# 1. Stop all servers (Ctrl+C)

# 2. Clean up
rm -rf backend/venv/
rm -rf node_modules/
rm backend/crash_analytics.db
rm backend/mlflow.db

# 3. Reinstall everything
cd backend
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

cd ..
npm install

# 4. Start fresh
python start_application.py
```

## 📞 Getting Help

If you're still experiencing issues:

1. **Check the logs** for specific error messages
2. **Run the test script**: `python backend/test_api.py`
3. **Verify system requirements**:
   - Python 3.8+
   - Node.js 16+
   - 4GB+ RAM
   - 2GB+ free disk space

## 🎯 Quick Fixes

### For Windows Users:
```batch
# Run as administrator if permission issues
# Use PowerShell instead of Command Prompt
# Ensure Python and Node.js are in PATH
```

### For Mac/Linux Users:
```bash
# Use virtual environment
# Check file permissions
# Use sudo if needed for global packages
```

### For All Users:
```bash
# Clear browser cache
# Try different browser
# Check firewall settings
# Disable antivirus temporarily
```

---

**Remember**: The application includes a robust demo mode that works even when ML models aren't available. This ensures you can always test the interface and functionality. 