#!/usr/bin/env python3
"""
Road Crash Analytics Application Startup Script
This script helps users start both the backend and frontend with proper setup
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path
import platform

class ApplicationStarter:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_dir = self.project_root / "backend"
        self.frontend_dir = self.project_root / "src"
        self.is_windows = platform.system() == "Windows"
        
    def print_banner(self):
        """Print application banner"""
        print("=" * 60)
        print("🚗 ROAD CRASH ANALYTICS APPLICATION")
        print("=" * 60)
        print("Advanced ML-powered crash severity prediction system")
        print("with XAI, drift detection, and auto-retraining")
        print("=" * 60)
        print()
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🔍 Checking Python version...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
            return False
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("\n🔍 Checking dependencies...")
        
        required_packages = [
            "fastapi", "uvicorn", "torch", "pandas", "numpy", 
            "scikit-learn", "sqlalchemy", "requests"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (missing)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("   Run: pip install -r backend/requirements.txt")
            return False
        
        return True
    
    def setup_backend_environment(self):
        """Setup backend virtual environment"""
        print("\n🔧 Setting up backend environment...")
        
        venv_path = self.backend_dir / "venv"
        if not venv_path.exists():
            print("📦 Creating virtual environment...")
            try:
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                             check=True, cwd=self.backend_dir)
                print("✅ Virtual environment created")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to create virtual environment: {e}")
                return False
        
        # Activate virtual environment and install dependencies
        if self.is_windows:
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            activate_script = venv_path / "bin" / "activate"
            pip_path = venv_path / "bin" / "pip"
        
        if not pip_path.exists():
            print("❌ Virtual environment not properly created")
            return False
        
        # Install requirements
        requirements_file = self.backend_dir / "requirements.txt"
        if requirements_file.exists():
            print("📦 Installing Python dependencies...")
            try:
                subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], 
                             check=True, cwd=self.backend_dir)
                print("✅ Dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
                return False
        
        return True
    
    def setup_frontend_environment(self):
        """Setup frontend environment"""
        print("\n🔧 Setting up frontend environment...")
        
        package_json = self.project_root / "package.json"
        if not package_json.exists():
            print("❌ package.json not found in project root")
            return False
        
        # Check if node_modules exists
        node_modules = self.project_root / "node_modules"
        if not node_modules.exists():
            print("📦 Installing Node.js dependencies...")
            try:
                subprocess.run(["npm", "install"], check=True, cwd=self.project_root)
                print("✅ Node.js dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install Node.js dependencies: {e}")
                return False
        else:
            print("✅ Node.js dependencies already installed")
        
        return True
    
    def start_backend(self):
        """Start the backend server"""
        print("\n🚀 Starting backend server...")
        
        # Check if backend is already running
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend is already running")
                return True
        except requests.RequestException:
            pass
        
        # Start backend
        if self.is_windows:
            python_path = self.backend_dir / "venv" / "Scripts" / "python.exe"
        else:
            python_path = self.backend_dir / "venv" / "bin" / "python"
        
        if not python_path.exists():
            print("❌ Python executable not found in virtual environment")
            return False
        
        try:
            # Start backend in background
            backend_process = subprocess.Popen(
                [str(python_path), "-m", "uvicorn", "app.main:app", 
                 "--host", "0.0.0.0", "--port", "8000", "--reload"],
                cwd=self.backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for backend to start
            print("⏳ Waiting for backend to start...")
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get("http://localhost:8000/api/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ Backend started successfully")
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
            
            print("❌ Backend failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend development server"""
        print("\n🚀 Starting frontend server...")
        
        # Check if frontend is already running
        try:
            response = requests.get("http://localhost:5173", timeout=5)
            if response.status_code == 200:
                print("✅ Frontend is already running")
                return True
        except requests.RequestException:
            pass
        
        try:
            # Start frontend in background
            frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for frontend to start
            print("⏳ Waiting for frontend to start...")
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get("http://localhost:5173", timeout=2)
                    if response.status_code == 200:
                        print("✅ Frontend started successfully")
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
            
            print("❌ Frontend failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def run_tests(self):
        """Run API tests to verify everything is working"""
        print("\n🧪 Running API tests...")
        
        test_script = self.backend_dir / "test_api.py"
        if not test_script.exists():
            print("❌ Test script not found")
            return False
        
        try:
            if self.is_windows:
                python_path = self.backend_dir / "venv" / "Scripts" / "python.exe"
            else:
                python_path = self.backend_dir / "venv" / "bin" / "python"
            
            result = subprocess.run([str(python_path), str(test_script)], 
                                  cwd=self.backend_dir, capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Failed to run tests: {e}")
            return False
    
    def print_access_info(self):
        """Print access information"""
        print("\n" + "=" * 60)
        print("🌐 ACCESS INFORMATION")
        print("=" * 60)
        print("Frontend Application: http://localhost:5173")
        print("Backend API: http://localhost:8000")
        print("API Documentation: http://localhost:8000/api/docs")
        print("API Health Check: http://localhost:8000/api/health")
        print("=" * 60)
        print()
        print("📋 Available Features:")
        print("• Single Crash Severity Prediction")
        print("• Batch Prediction (CSV/JSON upload)")
        print("• Analytics Dashboard")
        print("• Explainable AI (XAI) with SHAP/LIME")
        print("• Data Drift Detection")
        print("• Auto-Retraining System")
        print("• Model Versioning")
        print()
        print("💡 Tips:")
        print("• Use the frontend for interactive predictions")
        print("• Check API docs for programmatic access")
        print("• Monitor logs for any issues")
        print("• Press Ctrl+C to stop the servers")
        print("=" * 60)
    
    def start(self):
        """Main startup process"""
        self.print_banner()
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Setup environments
        if not self.setup_backend_environment():
            return False
        
        if not self.setup_frontend_environment():
            return False
        
        # Start servers
        if not self.start_backend():
            return False
        
        if not self.start_frontend():
            return False
        
        # Run tests
        self.run_tests()
        
        # Print access information
        self.print_access_info()
        
        print("🎉 Application started successfully!")
        print("Keep this terminal open to run the application.")
        print("Press Ctrl+C to stop all servers.")
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down servers...")
            print("✅ Application stopped")
        
        return True

def main():
    """Main entry point"""
    starter = ApplicationStarter()
    success = starter.start()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 