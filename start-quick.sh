#!/bin/bash

# EXPLAINIUM Quick Launcher
# Fastest way to start the full application

set -e

echo "⚡ Quick Starting EXPLAINIUM..."

# Start all services in background
echo "📦 Starting all services..."
docker-compose up -d

# Wait a moment for backend to be ready
echo "⏳ Waiting for backend..."
sleep 8

# Start frontend
echo "⚛️  Starting frontend..."
cd src/frontend
npm start &
FRONTEND_PID=$!
cd ../..

echo "✅ EXPLAINIUM is ready!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "To stop: ./stop.sh"
echo "Press Ctrl+C to stop frontend only"

# Wait for frontend
wait $FRONTEND_PID