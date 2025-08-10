#!/usr/bin/env python3
"""
EXPLAINIUM - Unified Application Runner

Runs both the FastAPI backend and Streamlit frontend in a coordinated way.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def run_backend():
    """Run the FastAPI backend"""
    print("🚀 Starting EXPLAINIUM Backend API...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.api.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ])

def run_frontend():
    """Run the Streamlit frontend"""
    print("🎨 Starting EXPLAINIUM Frontend Dashboard...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "src/frontend/knowledge_table.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

def main():
    """Main application runner"""
    print("=" * 60)
    print("🧠 EXPLAINIUM - Knowledge Extraction System")
    print("=" * 60)
    print("Starting full application stack...")
    print()
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = run_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start frontend
        frontend_process = run_frontend()
        time.sleep(2)  # Give frontend time to start
        
        print()
        print("✅ EXPLAINIUM is now running!")
        print("=" * 60)
        print("🌐 Frontend Dashboard: http://localhost:8501")
        print("🔧 Backend API:        http://localhost:8000")
        print("📚 API Documentation:  http://localhost:8000/docs")
        print("🏥 Health Check:       http://localhost:8000/health")
        print("=" * 60)
        print("Press Ctrl+C to stop all services")
        print()
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down EXPLAINIUM...")
        
    finally:
        # Clean shutdown
        if backend_process:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
                
        if frontend_process:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
                
        print("✅ All services stopped")

if __name__ == "__main__":
    main()