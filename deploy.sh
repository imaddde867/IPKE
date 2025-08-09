#!/bin/bash

# EXPLAINIUM - Clean Deployment Script
# Professional deployment automation with proper error handling

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="explainium"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handler
error_handler() {
    log_error "Deployment failed at line $1"
    log_info "Cleaning up..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    exit 1
}

trap 'error_handler $LINENO' ERR

# Help function
show_help() {
    cat << EOF
EXPLAINIUM Deployment Script

Usage: $0 [OPTIONS]

Options:
    --quick         Quick deployment (skip health checks and optimizations)
    --dev           Development mode (with debug settings)
    --prod          Production mode (default)
    --clean         Clean deployment (remove all data)
    --logs          Show logs after deployment
    --help          Show this help message

Examples:
    $0                  # Standard production deployment
    $0 --quick          # Quick deployment
    $0 --dev --logs     # Development with logs
    $0 --clean --prod   # Clean production deployment

EOF
}

# Parse command line arguments
QUICK_MODE=false
DEV_MODE=false
PROD_MODE=true
CLEAN_MODE=false
SHOW_LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            PROD_MODE=false
            shift
            ;;
        --prod)
            PROD_MODE=true
            DEV_MODE=false
            shift
            ;;
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Banner
echo "=================================================================="
echo "🧠 EXPLAINIUM - Knowledge Extraction System"
echo "=================================================================="
echo "Mode: $([ "$DEV_MODE" = true ] && echo "Development" || echo "Production")"
echo "Quick: $QUICK_MODE"
echo "Clean: $CLEAN_MODE"
echo "=================================================================="

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create environment file
create_env_file() {
    log_info "Creating environment configuration..."
    
    cat > "$ENV_FILE" << EOF
# EXPLAINIUM Environment Configuration
ENVIRONMENT=$([ "$DEV_MODE" = true ] && echo "development" || echo "production")

# Database Configuration
DB_HOST=db
DB_PORT=5432
DB_NAME=explainium
DB_USER=postgres
DB_PASSWORD=password
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Tika Configuration
TIKA_HOST=tika
TIKA_PORT=9998

# Celery Configuration
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
CELERY_CONCURRENCY=4

# Processing Configuration
UPLOAD_DIRECTORY=/app/uploads
MAX_FILE_SIZE_MB=100
ENABLE_OCR=true
ENABLE_AUDIO_PROCESSING=true
PARALLEL_PROCESSING=true
BATCH_SIZE=10

# AI Configuration
SPACY_MODEL=en_core_web_sm
CLASSIFICATION_MODEL=facebook/bart-large-mnli
NER_MODEL=dslim/bert-base-NER
WHISPER_MODEL=base
CONFIDENCE_THRESHOLD=0.7
ENABLE_GPU=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=$([ "$DEV_MODE" = true ] && echo "true" || echo "false")
CORS_ORIGINS=*
MAX_REQUEST_SIZE=104857600
RATE_LIMIT=100

# Logging Configuration
LOG_LEVEL=$([ "$DEV_MODE" = true ] && echo "DEBUG" || echo "INFO")
LOG_ENABLE_CONSOLE=true

# Security Configuration
SECRET_KEY=$(openssl rand -hex 32)
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
ENABLE_HTTPS=false

EOF
    
    log_success "Environment file created"
}

# Clean deployment
clean_deployment() {
    if [ "$CLEAN_MODE" = true ]; then
        log_warning "Performing clean deployment (all data will be lost)..."
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Stopping and removing all containers and volumes..."
            docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true
            docker system prune -f 2>/dev/null || true
            log_success "Clean deployment completed"
        else
            log_info "Clean deployment cancelled"
            exit 0
        fi
    fi
}

# Build and start services
deploy_services() {
    log_info "Building and starting services..."
    
    # Build images
    log_info "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    # Start services
    log_info "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_success "Services started"
}

# Wait for services
wait_for_services() {
    if [ "$QUICK_MODE" = false ]; then
        log_info "Waiting for services to be ready..."
        
        # Wait for database
        log_info "Waiting for database..."
        timeout=60
        while ! docker-compose -f "$COMPOSE_FILE" exec -T db pg_isready -U postgres &>/dev/null; do
            sleep 2
            timeout=$((timeout - 2))
            if [ $timeout -le 0 ]; then
                log_error "Database failed to start within 60 seconds"
                exit 1
            fi
        done
        log_success "Database is ready"
        
        # Wait for Redis
        log_info "Waiting for Redis..."
        timeout=30
        while ! docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping &>/dev/null; do
            sleep 2
            timeout=$((timeout - 2))
            if [ $timeout -le 0 ]; then
                log_error "Redis failed to start within 30 seconds"
                exit 1
            fi
        done
        log_success "Redis is ready"
        
        # Wait for Tika
        log_info "Waiting for Tika..."
        timeout=60
        while ! curl -f http://localhost:9998/tika &>/dev/null; do
            sleep 3
            timeout=$((timeout - 3))
            if [ $timeout -le 0 ]; then
                log_error "Tika failed to start within 60 seconds"
                exit 1
            fi
        done
        log_success "Tika is ready"
        
        # Wait for main application
        log_info "Waiting for main application..."
        timeout=120
        while ! curl -f http://localhost:8000/health &>/dev/null; do
            sleep 5
            timeout=$((timeout - 5))
            if [ $timeout -le 0 ]; then
                log_error "Main application failed to start within 120 seconds"
                exit 1
            fi
        done
        log_success "Main application is ready"
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Wait a bit more for the database to be fully ready
    sleep 5
    
    # Run Alembic migrations
    if docker-compose -f "$COMPOSE_FILE" exec -T app alembic upgrade head; then
        log_success "Database migrations completed"
    else
        log_warning "Alembic migrations failed, trying manual table creation..."
        docker-compose -f "$COMPOSE_FILE" exec -T app python -c "
from src.database.database import init_db
try:
    init_db()
    print('Manual database initialization completed')
except Exception as e:
    print(f'Manual database initialization failed: {e}')
    exit(1)
"
        log_success "Manual database initialization completed"
    fi
}

# Health check
health_check() {
    if [ "$QUICK_MODE" = false ]; then
        log_info "Performing health checks..."
        
        # Check main application
        if curl -f http://localhost:8000/health &>/dev/null; then
            log_success "Main application health check passed"
        else
            log_error "Main application health check failed"
            exit 1
        fi
        
        # Check if we can upload a test file (optional)
        log_info "System is ready for document processing"
    fi
}

# Show deployment summary
show_summary() {
    echo ""
    echo "=================================================================="
    echo "🎉 EXPLAINIUM Deployment Complete!"
    echo "=================================================================="
    echo "🌐 Web Interface:     http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo "🔍 Health Check:      http://localhost:8000/health"
    echo "=================================================================="
    echo ""
    echo "Services Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    
    if [ "$DEV_MODE" = true ]; then
        echo "Development Mode Commands:"
        echo "  View logs:           docker-compose -f $COMPOSE_FILE logs -f"
        echo "  Stop services:       docker-compose -f $COMPOSE_FILE down"
        echo "  Restart service:     docker-compose -f $COMPOSE_FILE restart <service>"
        echo ""
    fi
    
    echo "Useful Commands:"
    echo "  View logs:           ./deploy-clean.sh --logs"
    echo "  Stop all:            docker-compose -f $COMPOSE_FILE down"
    echo "  Clean restart:       ./deploy-clean.sh --clean"
    echo ""
}

# Show logs
show_logs() {
    if [ "$SHOW_LOGS" = true ]; then
        log_info "Showing service logs (Ctrl+C to exit)..."
        docker-compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Main deployment flow
main() {
    cd "$SCRIPT_DIR"
    
    check_prerequisites
    create_env_file
    clean_deployment
    deploy_services
    wait_for_services
    run_migrations
    health_check
    show_summary
    show_logs
}

# Run main function
main "$@"