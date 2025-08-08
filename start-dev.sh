#!/bin/bash

# EXPLAINIUM Development Launcher
# Starts backend and frontend in development mode

set -e

echo "🚀 Starting EXPLAINIUM Development Environment..."

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    docker-compose down
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start backend services with Docker Compose
echo "📦 Starting backend services..."
docker-compose up -d db redis elasticsearch tika

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Start the FastAPI app
echo "🔧 Starting FastAPI backend..."
docker-compose up -d app celery_worker celery_beat

# Start the React frontend
echo "⚛️  Starting React frontend..."
cd src/frontend
npm start &
FRONTEND_PID=$!
cd ../..

echo "✅ Development environment is ready!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "❤️  Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for frontend process
wait $FRONTEND_PID