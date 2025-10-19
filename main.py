#!/usr/bin/env python3
"""
Explainium - Knowledge Extraction API Server

A streamlined knowledge extraction system powered by Mistral-7B LLM.
Extracts structured information from technical documents.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from src.api.app import app
from src.core.unified_config import get_config

def check_model_availability():
    """Check if the Mistral model is available"""
    config = get_config()
    model_path = config.llm_model_path
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please ensure the Mistral model is downloaded.")
        return False
    
    file_size_gb = os.path.getsize(model_path) / (1024 * 1024 * 1024)
    print(f"SUCCESS: Mistral model found: {model_path}")
    print(f"   Size: {file_size_gb:.2f} GB")
    return True

def main():
    """Run the Explainium API server"""
    print("=" * 60)
    print("Explainium - Knowledge Extraction API")
    print("=" * 60)
    
    # Check model availability
    if not check_model_availability():
        print("\nERROR: Cannot start server without Mistral model")
        return 1
    
    config = get_config()
    
    print(f"\nStarting API server...")
    print(f"   Host: {config.api_host}")
    print(f"   Port: {config.api_port}")
    print(f"   Environment: {config.environment.value}")
    print(f"   LLM: Mistral-7B-Instruct-v0.2")
    print(f"\nAPI available at: http://{config.api_host}:{config.api_port}")
    print(f"Documentation: http://{config.api_host}:{config.api_port}/docs")
    print(f"\nUpload documents to /extract endpoint for knowledge extraction")
    
    try:
        uvicorn.run(
            app,
            host=config.api_host,
            port=config.api_port,
            log_level=config.log_level.lower(),
            reload=False
        )
        return 0
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return 0
    except Exception as e:
        print(f"ERROR: Server error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
