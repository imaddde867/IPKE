#!/bin/bash

# EXPLAINIUM - Stop All Services

echo "🛑 Stopping EXPLAINIUM services..."

# Kill streamlit processes
pkill -f "streamlit.*knowledge_table" || true

# Kill uvicorn processes
pkill -f "uvicorn.*src.api.app" || true

# Kill any remaining python processes related to the app
pkill -f "python.*src/" || true

echo "✅ All services stopped"
