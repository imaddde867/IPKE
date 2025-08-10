#!/bin/bash

# EXPLAINIUM - Application Startup Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=================================================================="
echo -e "${BLUE}🧠 EXPLAINIUM - Starting Application${NC}"
echo "=================================================================="

# Activate virtual environment
source venv/bin/activate

# Function to start backend
start_backend() {
    echo -e "${GREEN}🔧 Starting Backend API...${NC}"
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
}

# Function to start frontend
start_frontend() {
    echo -e "${GREEN}🎨 Starting Frontend Dashboard...${NC}"
    # Ensure we're in the project root directory
    cd "$(dirname "$0")"
    streamlit run src/frontend/knowledge_table.py --server.port 8501 --server.address localhost &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
}

# Start services
start_backend
sleep 3  # Give backend time to start
start_frontend

echo ""
echo "=================================================================="
echo -e "${GREEN}✅ EXPLAINIUM is now running!${NC}"
echo "=================================================================="
echo "🎨 Frontend Dashboard: http://localhost:8501"
echo "🔧 Backend API:        http://localhost:8000"
echo "📚 API Documentation:  http://localhost:8000/docs"
echo "=================================================================="
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
trap 'echo -e "\n${BLUE}🛑 Shutting down...${NC}"; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
