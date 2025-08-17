#!/usr/bin/env python3
"""
Startup script for the Agricultural AI System
Performs dependency checks and starts the FastAPI backend
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('requests', 'requests'),
        ('python-multipart', 'multipart'),
        ('pillow', 'PIL'),
        ('plotly', 'plotly'),
        ('langchain-groq', 'langchain_groq')
    ]
    
    missing_packages = []
    
    print("🔍 Checking dependencies...")
    
    for package_name, import_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Installing missing packages...")
        
        for package in missing_packages:
            try:
                print(f"   Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e}")
                return False
    else:
        print("   ✅ All dependencies satisfied!")
    
    return True

def check_environment():
    """Check environment variables and configuration"""
    print("\n🔧 Checking environment...")
    
    # Check for Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"   ✅ GEMINI_API_KEY found: {gemini_key[:10]}...")
    else:
        print("   ⚠️  GEMINI_API_KEY not set")
        print("      💡 Set it to use real Gemini Vision API")
        print("      💡 Demo mode will work without it")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if "Rishikhet" in str(current_dir):
        print(f"   ✅ Working directory: {current_dir}")
    else:
        print(f"   ⚠️  Working directory: {current_dir}")
        print("      💡 Make sure you're in the Rishikhet project directory")
    
    return True

def start_backend():
    """Start the FastAPI backend"""
    print("\n🚀 Starting FastAPI backend...")
    
    try:
        # Check if backend.py exists
        if not Path("backend.py").exists():
            print("   ❌ backend.py not found in current directory")
            return False
        
        print("   ✅ backend.py found")
        print("   🌐 Starting server on http://localhost:8000")
        print("   📱 Open your browser and go to: http://localhost:8000")
        print("   🛑 Press Ctrl+C to stop the server")
        print("\n" + "="*50)
        
        # Start the backend
        subprocess.run([sys.executable, "backend.py"])
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n   ❌ Failed to start backend: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🌾 Agricultural AI System - Startup")
    print("="*40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages manually.")
        return
    
    # Check environment
    check_environment()
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main()
