#!/bin/bash
# EXPLAINIUM - CSC Puhti Deployment Script
# Run this script on Puhti to start your application

echo "🚀 EXPLAINIUM CSC Puhti Deployment"
echo "=================================="

# Load required modules
echo "📦 Loading modules..."
module load python-data
module load postgresql
module load redis

# Activate virtual environment
echo "🐍 Activating Python environment..."
source venv/bin/activate

# Set CSC-specific environment variables
export ENVIRONMENT=production
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=explainium
export DB_USER=$USER
export DB_PASSWORD=explainium_db_2024
export REDIS_HOST=localhost
export REDIS_PORT=6379
export UPLOAD_DIRECTORY=/scratch/project_2001234/explainium-2.0-1/uploaded_files
export MAX_FILE_SIZE_MB=500
export LOG_LEVEL=INFO

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploaded_files logs

# Start PostgreSQL (if not running)
echo "🐘 Starting PostgreSQL..."
pg_ctl -D $HOME/postgres_data -l $HOME/postgres.log start

# Start Redis (if not running)
echo "🔴 Starting Redis..."
redis-server --daemonize yes --port 6379

# Initialize database
echo "🗄️ Initializing database..."
python -c "from src.database.database import init_db; init_db()"

# Start the application
echo "🚀 Starting EXPLAINIUM..."
echo "Backend API: http://localhost:8000"
echo "Frontend Dashboard: http://localhost:8501"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

# Start backend in background
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
streamlit run src/frontend/knowledge_table.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

# Wait for interrupt
trap 'echo "Shutting down..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
