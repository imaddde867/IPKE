#!/bin/bash

# EXPLAINIUM Stop Script
# Stops all services

echo "🛑 Stopping EXPLAINIUM services..."

# Stop Docker services
docker-compose down

# Kill any remaining frontend processes
pkill -f "react-scripts start" 2>/dev/null || true
pkill -f "npm start" 2>/dev/null || true

echo "✅ All services stopped!"