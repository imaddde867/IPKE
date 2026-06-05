#!/usr/bin/env python3
"""
Industrial Procedural Knowledge Extraction (IPKE) - API Server

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
from src.core.unified_config import get_config, UnifiedConfig

def check_model_availability():
    """Check if the Mistral model is available"""
    if os.getenv('SKIP_MODEL_CHECK', 'false').strip().lower() in {'1', 'true', 'yes'}:
        print("INFO: Skipping LLM model check (SKIP_MODEL_CHECK=true)")
        return True

    config = get_config()
    model_path = config.llm_model_path
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found: {model_path}")
        print("Mount your models volume at /app/models or set SKIP_MODEL_CHECK=true for chunking-only workloads.")
        return False
    
    file_size_gb = os.path.getsize(model_path) / (1024 * 1024 * 1024)
    print(f"SUCCESS: Mistral model found: {model_path}")
    print(f"   Size: {file_size_gb:.2f} GB")
    return True

def log_runtime_settings(config: UnifiedConfig):
    """Log runtime configuration useful for thesis experiments."""
    backend_requested = config.gpu_backend
    backend_resolved = config.detect_gpu_backend()
    llm_backend = getattr(config, "llm_backend", "transformers")
    if llm_backend.lower().startswith("llama"):
        llm_label = "llama.cpp"
    else:
        llm_label = llm_backend
    device_label = backend_resolved if backend_resolved in {"cuda", "metal"} else "cpu"
    print("\nRuntime configuration:")
    print(f"   Chunking method: {config.chunking_method}")
    print(f"   Requested GPU backend: {backend_requested}")
    print(f"   Resolved backend: {backend_resolved}")
    print(f"   LLM strategy: {llm_label}/{device_label}")
    print(f"   LLM model path: {config.llm_model_path}")
    print(f"   Embedding model path: {config.embedding_model_path}")
    print(f"   Upload directory: {config.upload_directory}")
    print(f"   Data mount: {Path('/app/data').resolve()}")
    print(f"   Results mount: {Path('/app/results').resolve()}")
    print("   Thesis metrics: chunk_count, avg_chunk_size, cohesion, and processing times will be logged per request.")

def main():
    """Run the IPKE API server"""
    print("=" * 60)
    print("Industrial Procedural Knowledge Extraction (IPKE) - API")
    print("=" * 60)
    
    # Check model availability
    if not check_model_availability():
        print("\nERROR: Cannot start server without Mistral model")
        return 1
    
    config = get_config()
    log_runtime_settings(config)
    
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
