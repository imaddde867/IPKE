#!/usr/bin/env python3
"""
Simple setup script for Explainium development environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🔬 Setting up Explainium for research and development")
    
    # Check Python version
    if sys.version_info < (3, 12):
        print("❌ Python 3.12+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        sys.exit(1)
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️  spaCy model download failed, but continuing...")
    
    # Create necessary directories
    dirs_to_create = ["logs", "uploaded_files"]
    for dir_name in dirs_to_create:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_name}")
    
    # Setup models (optional)
    if Path("scripts/model_manager.py").exists():
        print("\n🤖 Setting up AI models...")
        run_command("python scripts/model_manager.py --action setup", "AI model setup")
    
    print("\n🎉 Setup complete!")
    print("\n🚀 Quick start options:")
    print("   • Frontend only: streamlit run src/frontend/knowledge_table.py")
    print("   • API backend: uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
    print("   • Full stack: docker-compose up")
    print("   • Health check: ./scripts/health_check.sh")

if __name__ == "__main__":
    main()