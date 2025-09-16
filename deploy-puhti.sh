#!/bin/bash
# EXPLAINIUM - CSC Puhti Deployment Script

echo "EXPLAINIUM CSC Puhti Deployment"
echo "==============================="

echo "Loading modules..."
source venv_scratch/bin/activate

# Environment Configuration
export ENVIRONMENT=production
export PYTHONPATH="${PYTHONPATH}:/scratch/project_2015237/explainium-2.0"
export DATABASE_URL=sqlite:///./explainium.db
export UPLOAD_DIRECTORY=/projappl/project_2015237/explainium-2.0/uploaded_files
export MAX_FILE_SIZE_MB=500
export LOG_LEVEL=INFO

# AI Processing Configuration
export EXPLAINIUM_LLM_CHUNK_TIMEOUT=300
export EXPLAINIUM_LLM_CHUNK_RETRIES=1
export EXPLAINIUM_LLM_CHUNK_BACKOFF=60
export AI_ENGINE_TIMEOUT=10

# HPC Compatibility Settings
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# PyTorch Optimization
export TORCH_DISABLE_TRITON=1
export TORCH_DISABLE_CUDNN=1
export TORCH_DISABLE_CUDA=1
export CUDA_VISIBLE_DEVICES=""
export TORCH_DISABLE_DYNAMO=1
export PYTORCH_DISABLE_DYNAMO=1
export TORCH_COMPILE_DISABLED=1

# Thread Limiting
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Create directories
mkdir -p /tmp/transformers_cache /tmp/hf_home /tmp/hf_datasets_cache
mkdir -p "$UPLOAD_DIRECTORY" logs

# Initialize database
if [ ! -f "explainium.db" ]; then
    echo "Initializing database..."
    ./venv_scratch/bin/python -c "from src.database.database import init_db; init_db()" || echo "Database initialization skipped"
fi

# Test essential components
echo "Testing environment..."
./venv_scratch/bin/python -c "
import sys
print(f'Python: {sys.version.split()[0]}')

try:
    import streamlit
    print('Streamlit: OK')
except Exception as e:
    print(f'Streamlit: FAILED - {e}')
    sys.exit(1)

print('Essential components verified')
"

# Start application
echo "Starting EXPLAINIUM..."
echo "Frontend: http://localhost:8501"
echo "Press Ctrl+C to stop"

./venv_scratch/bin/python -m streamlit run src/frontend/knowledge_table.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false \
    --logger.level error \
    --server.runOnSave false \
    --server.fileWatcherType none \
    --server.enableWebsocketCompression false \
    --server.maxUploadSize 500
