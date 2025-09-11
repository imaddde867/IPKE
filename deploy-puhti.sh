#!/bin/bash
# EXPLAINIUM - CSC Puhti Deployment Script
# Run this script on Puhti to start your application

echo "🚀 EXPLAINIUM CSC Puhti Deployment"
echo "=================================="

# Load required modules
echo "📦 Loading modules..."
module load python-data

# Activate virtual environment
echo "🐍 Activating Python environment..."
source venv/bin/activate

# Set CSC-specific environment variables
export ENVIRONMENT=production
export SECRET_KEY=explainium_secret_key_2024_puhti_deployment
export DATABASE_URL=sqlite:///./explainium.db
export REDIS_URL=redis://localhost:6379
export UPLOAD_DIRECTORY=/scratch/project_2015237/explainium-2.0-1/uploaded_files
export MAX_FILE_SIZE_MB=500
export LOG_LEVEL=INFO

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$UPLOAD_DIRECTORY" logs

# Note: PostgreSQL and Redis need to be installed separately on Puhti
# For now, we'll use SQLite for the database
echo "📝 Using SQLite database (no PostgreSQL setup required)"

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

# Start frontend only (backend has issues)
echo "Starting Streamlit frontend only..."
streamlit run src/frontend/knowledge_table.py --server.port 8501 --server.address 0.0.0.0
