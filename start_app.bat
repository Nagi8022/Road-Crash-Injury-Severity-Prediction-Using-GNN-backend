@echo off
echo ============================================================
echo 🚗 ROAD CRASH ANALYTICS APPLICATION
echo ============================================================
echo Starting the application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js 16 or higher
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed or not in PATH
    echo Please install npm
    pause
    exit /b 1
)

echo ✅ Python and Node.js are installed
echo.

REM Start the application using Python script
python start_application.py

pause 