.PHONY: dev start stop backend frontend services logs clean install help

# Default target
help:
	@echo "🧠 EXPLAINIUM - Quick Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev      - Start development environment (backend + frontend)"
	@echo "  make start    - Quick start all services"
	@echo "  make stop     - Stop all services"
	@echo ""
	@echo "Individual Services:"
	@echo "  make backend  - Start backend only"
	@echo "  make frontend - Start frontend only"
	@echo "  make services - Start supporting services only"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs     - View all service logs"
	@echo "  make clean    - Clean up containers and volumes"
	@echo "  make install  - Install frontend dependencies"

# Development mode - backend services + frontend
dev:
	@echo "🚀 Starting development environment..."
	@./start-dev.sh

# Quick start everything
start:
	@echo "⚡ Quick starting EXPLAINIUM..."
	@./start-quick.sh

# Stop all services
stop:
	@echo "🛑 Stopping all services..."
	@./stop.sh

# Backend only
backend:
	@echo "🔧 Starting backend services..."
	@docker-compose up app celery_worker celery_beat

# Frontend only
frontend:
	@echo "⚛️  Starting frontend..."
	@cd src/frontend && npm start

# Supporting services only
services:
	@echo "📦 Starting supporting services..."
	@docker-compose up -d db redis elasticsearch tika

# View logs
logs:
	@docker-compose logs -f

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose down -v
	@docker system prune -f

# Install frontend dependencies
install:
	@echo "📦 Installing frontend dependencies..."
	@cd src/frontend && npm install