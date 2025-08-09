# EXPLAINIUM - Clean Makefile
# Professional build automation with clear targets

.PHONY: help dev prod quick stop clean logs test install lint format check health

# Default target
help:
	@echo "🧠 EXPLAINIUM - Clean Build System"
	@echo ""
	@echo "Development:"
	@echo "  make dev      - Start development environment"
	@echo "  make quick    - Quick start (skip health checks)"
	@echo "  make prod     - Start production environment"
	@echo ""
	@echo "Management:"
	@echo "  make stop     - Stop all services"
	@echo "  make clean    - Clean deployment (removes all data)"
	@echo "  make logs     - View service logs"
	@echo "  make health   - Check system health"
	@echo ""
	@echo "Development Tools:"
	@echo "  make install  - Install dependencies"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Run code linting"
	@echo "  make format   - Format code"
	@echo "  make check    - Run all checks (lint + test)"
	@echo ""

# Development environment
dev:
	@echo "🚀 Starting development environment..."
	@./deploy.sh --dev

# Production environment
prod:
	@echo "⚡ Starting production environment..."
	@./deploy.sh --prod

# Quick start
quick:
	@echo "⚡ Quick start..."
	@./deploy.sh --quick

# Stop services
stop:
	@echo "🛑 Stopping all services..."
	@docker-compose down --remove-orphans

# Clean deployment
clean:
	@echo "🧹 Clean deployment..."
	@./deploy.sh --clean

# View logs
logs:
	@echo "📋 Viewing service logs..."
	@docker-compose logs -f

# Health check
health:
	@echo "🔍 Checking system health..."
	@curl -f http://localhost:8000/health || echo "❌ System not healthy"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@pip install -r requirements-clean.txt
	@python -m spacy download en_core_web_sm

# Run tests
test:
	@echo "🧪 Running tests..."
	@pytest tests/ -v --cov=src --cov-report=term-missing

# Lint code
lint:
	@echo "🔍 Running code linting..."
	@flake8 src/ --max-line-length=120 --ignore=E203,W503
	@mypy src/ --ignore-missing-imports

# Format code
format:
	@echo "✨ Formatting code..."
	@black src/ --line-length=120
	@isort src/ --profile=black

# Run all checks
check: lint test
	@echo "✅ All checks completed"

# Docker operations
docker-build:
	@echo "🐳 Building Docker images..."
	@docker-compose build --no-cache

docker-pull:
	@echo "📥 Pulling Docker images..."
	@docker-compose pull

# Database operations
db-migrate:
	@echo "🗄️  Running database migrations..."
	@docker-compose exec app alembic upgrade head

db-reset:
	@echo "🔄 Resetting database..."
	@docker-compose exec app python -c "from src.database.database import reset_db; reset_db()"

# Monitoring
monitor:
	@echo "📊 System monitoring..."
	@docker-compose exec app python -c "
from src.database.crud import get_knowledge_analytics
from src.database.database import get_db_session
with get_db_session() as db:
    analytics = get_knowledge_analytics(db)
    print('System Analytics:')
    for key, value in analytics.items():
        print(f'  {key}: {value}')
"

# Backup
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@docker-compose exec db pg_dump -U postgres explainium > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created in backups/ directory"

# Restore (usage: make restore BACKUP_FILE=backup_file.sql)
restore:
	@echo "🔄 Restoring from backup..."
	@if [ -z "$(BACKUP_FILE)" ]; then echo "❌ Please specify BACKUP_FILE=filename"; exit 1; fi
	@docker-compose exec -T db psql -U postgres -d explainium < $(BACKUP_FILE)
	@echo "✅ Restore completed"

# Performance testing
perf-test:
	@echo "⚡ Running performance tests..."
	@python -c "
import requests
import time
import statistics

# Test API response times
times = []
for i in range(10):
    start = time.time()
    response = requests.get('http://localhost:8000/health')
    end = time.time()
    if response.status_code == 200:
        times.append(end - start)

if times:
    print(f'Average response time: {statistics.mean(times):.3f}s')
    print(f'Min response time: {min(times):.3f}s')
    print(f'Max response time: {max(times):.3f}s')
else:
    print('❌ Performance test failed')
"

# Security scan
security-scan:
	@echo "🔒 Running security scan..."
	@pip install safety bandit
	@safety check -r requirements-clean.txt
	@bandit -r src/ -f json -o security-report.json || echo "⚠️  Security issues found, check security-report.json"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@pip install sphinx sphinx-rtd-theme
	@sphinx-quickstart docs --quiet --project="EXPLAINIUM" --author="Team" --release="2.0" --language="en"
	@echo "✅ Documentation template created in docs/"

# Environment info
env-info:
	@echo "🔧 Environment Information:"
	@echo "Python: $(shell python --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell docker-compose --version)"
	@echo "OS: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"

# Cleanup development environment
dev-clean:
	@echo "🧹 Cleaning development environment..."
	@docker system prune -f
	@docker volume prune -f
	@pip cache purge
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Development environment cleaned"