@echo off
echo 🚗 Road Crash Analytics Platform - Startup Script
echo ==================================================
echo.
echo This script will start both backend and frontend servers
echo.
echo 📍 Backend: http://localhost:8000
echo 🎨 Frontend: http://localhost:5173
echo.
echo Press any key to start the servers...
pause > nul

echo.
echo 🚀 Starting Backend Server...
start "Backend Server" cmd /k "cd backend && python run_backend.py"

echo.
echo ⏳ Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak > nul

echo.
echo 🎨 Starting Frontend Server...
start "Frontend Server" cmd /k "npm run dev"

echo.
echo ✅ Both servers are starting...
echo.
echo 📍 Backend: http://localhost:8000
echo 🎨 Frontend: http://localhost:5173
echo 📚 API Docs: http://localhost:8000/api/docs
echo.
echo Press any key to exit this startup script...
pause > nul 