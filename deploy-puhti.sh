#!/bin/bash
# EXPLAINIUM - CSC Puhti Deployment Script
# Run this script on Puhti to start your application

echo "🚀 EXPLAINIUM CSC Puhti Deployment"
echo "=================================="

# Load required modules
echo "📦 Loading modules..."
# module load python-data  # Commented out to avoid conflicts with venv_scratch

# Activate virtual environment
echo "🐍 Activating Python environment..."
source venv_scratch/bin/activate

# Set CSC-specific environment variables
export ENVIRONMENT=production
export PYTHONPATH="${PYTHONPATH}:/scratch/project_2015237/explainium-2.0"
# SECRET_KEY should be set via environment variable for security
# export SECRET_KEY=${SECRET_KEY:-"explainium_secret_key_2024_puhti_deployment"}
export DATABASE_URL=sqlite:///./explainium.db
export REDIS_URL=redis://localhost:6379
export UPLOAD_DIRECTORY=/projappl/project_2015237/explainium-2.0/uploaded_files
export MAX_FILE_SIZE_MB=500
export LOG_LEVEL=INFO

# Set AI processing environment variables
export EXPLAINIUM_LLM_CHUNK_TIMEOUT=300
export EXPLAINIUM_LLM_CHUNK_RETRIES=1
export EXPLAINIUM_LLM_CHUNK_BACKOFF=60
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# HPC-specific environment variables for compute node compatibility
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/hf_home
export TORCH_DISABLE_DYNAMO=1
export PYTORCH_DISABLE_DYNAMO=1
export TORCH_COMPILE_DISABLED=1

export TORCH_DISABLE_TRITON=1
export TORCH_DISABLE_CUDNN=1
export TORCH_DISABLE_CUDA=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
export HF_HOME=/tmp/hf_home
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# Enable AI engines with fast timeout
export AI_ENGINE_TIMEOUT=10

# Create cache directories
mkdir -p /tmp/transformers_cache /tmp/hf_home

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$UPLOAD_DIRECTORY" logs

# Note: PostgreSQL and Redis need to be installed separately on Puhti
# For now, we'll use SQLite for the database
echo "📝 Using SQLite database (no PostgreSQL setup required)"

# Initialize database (optional - skip if causing issues)
echo "🗄️ Initializing database..."
./venv_scratch/bin/python -c "from src.database.database import init_db; init_db()" || echo "⚠️ Database initialization skipped - continuing..."

# Test essential imports only
echo "🔍 Testing essential components..."
./venv_scratch/bin/python -c "
import sys
print(f'Python: {sys.version.split()[0]}')

# Test Streamlit (required for web interface)
try:
    import streamlit
    print('✅ Streamlit: OK')
except Exception as e:
    print(f'❌ Streamlit: FAILED - {e}')
    sys.exit(1)

# Test PyTorch (optional - skip if hanging)
print('⚠️ Skipping PyTorch test (known HPC compatibility issues)')
print('   - PyTorch will be imported when needed for processing')

print('✅ Essential components verified')
"

# Start the application
echo "🚀 Starting EXPLAINIUM..."
echo "Backend API: http://localhost:8000"
echo "Frontend Dashboard: http://localhost:8501"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

# Start the main application
echo "Starting EXPLAINIUM application..."
./venv_scratch/bin/python -m streamlit run src/frontend/knowledge_table.py --server.port 8501 --server.address 0.0.0.0
