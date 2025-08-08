#!/bin/bash

# Enhanced EXPLAINIUM 2.0 Deployment Script
# This script deploys the enhanced knowledge extraction system

set -e  # Exit on any error

echo "🧠 EXPLAINIUM 2.0 Enhanced Deployment Script"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to backup existing database
backup_database() {
    print_status "Creating database backup..."
    
    if docker-compose ps | grep -q "db.*Up"; then
        BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
        print_status "Backing up database to $BACKUP_FILE"
        
        docker-compose exec -T db pg_dump -U postgres explainium > "$BACKUP_FILE" 2>/dev/null || {
            print_warning "Could not create database backup (database might be empty)"
        }
    else
        print_status "Database not running, skipping backup"
    fi
}

# Function to stop existing services
stop_services() {
    print_status "Stopping existing services..."
    docker-compose down --remove-orphans
    print_success "Services stopped"
}

# Function to build and start services
start_services() {
    print_status "Building and starting enhanced services..."
    
    # Build with enhanced configuration
    docker-compose build --no-cache
    
    # Start core services first
    print_status "Starting core services (database, redis, elasticsearch)..."
    docker-compose up -d db redis elasticsearch tika
    
    # Wait for services to be ready
    print_status "Waiting for core services to be ready..."
    sleep 30
    
    # Check if services are healthy
    for i in {1..30}; do
        if docker-compose exec db pg_isready -U postgres >/dev/null 2>&1; then
            print_success "Database is ready"
            break
        fi
        print_status "Waiting for database... ($i/30)"
        sleep 2
    done
    
    # Start application services
    print_status "Starting application services..."
    docker-compose up -d app celery_worker celery_beat
    
    print_success "All services started"
}

# Function to run database migrations
run_migrations() {
    print_status "Running enhanced database migrations..."
    
    # Wait for application to be ready
    sleep 10
    
    # Run enhanced migrations
    docker-compose exec app python -m src.database.enhanced_migrations || {
        print_warning "Enhanced migrations failed, trying standard migrations..."
        docker-compose exec app alembic upgrade head
    }
    
    print_success "Database migrations completed"
}

# Function to initialize enhanced features
initialize_enhanced_features() {
    print_status "Initializing enhanced features..."
    
    # Download required AI models
    print_status "Downloading AI models..."
    docker-compose exec app python -c "
import spacy
import whisper
import nltk
try:
    spacy.load('en_core_web_sm')
    print('SpaCy model already available')
except OSError:
    print('Downloading SpaCy model...')
    spacy.cli.download('en_core_web_sm')

try:
    whisper.load_model('base')
    print('Whisper model loaded successfully')
except Exception as e:
    print(f'Whisper model loading: {e}')

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded')
except Exception as e:
    print(f'NLTK download: {e}')
"
    
    print_success "Enhanced features initialized"
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check if services are running
    if ! docker-compose ps | grep -q "app.*Up"; then
        print_error "Application service is not running"
        return 1
    fi
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            print_success "API health check passed"
            break
        fi
        print_status "Waiting for API to be ready... ($i/30)"
        sleep 2
    done
    
    # Check enhanced health endpoint
    if curl -f http://localhost:8000/health/detailed >/dev/null 2>&1; then
        print_success "Enhanced health check passed"
    else
        print_warning "Enhanced health check failed (some features may not be available)"
    fi
    
    print_success "Deployment verification completed"
}

# Function to display deployment summary
display_summary() {
    echo ""
    echo "🎉 EXPLAINIUM 2.0 Enhanced Deployment Complete!"
    echo "=============================================="
    echo ""
    echo "Access Points:"
    echo "• Web Interface: http://localhost:8000"
    echo "• API Documentation: http://localhost:8000/docs"
    echo "• Enhanced Health Check: http://localhost:8000/health/detailed"
    echo "• Process Hierarchy: http://localhost:8000/processes/hierarchy"
    echo "• Compliance Dashboard: http://localhost:8000/compliance/dashboard"
    echo "• Risk Dashboard: http://localhost:8000/risks/dashboard"
    echo ""
    echo "Enhanced Features:"
    echo "• ✅ Multi-format document processing (PDF, DOCX, images, videos, audio)"
    echo "• ✅ Advanced knowledge extraction with AI"
    echo "• ✅ Process hierarchy management"
    echo "• ✅ Compliance tracking and monitoring"
    echo "• ✅ Risk assessment and management"
    echo "• ✅ Semantic search and analytics"
    echo ""
    echo "Management Commands:"
    echo "• View logs: docker-compose logs -f"
    echo "• Stop services: docker-compose down"
    echo "• Restart services: docker-compose restart"
    echo "• Check status: docker-compose ps"
    echo ""
    echo "For support and documentation, visit: https://github.com/imaddde867/explainium-2.0"
    echo ""
}

# Function to handle cleanup on failure
cleanup_on_failure() {
    print_error "Deployment failed. Cleaning up..."
    docker-compose down --remove-orphans
    exit 1
}

# Set trap for cleanup on failure
trap cleanup_on_failure ERR

# Main deployment process
main() {
    echo "Starting enhanced deployment process..."
    echo ""
    
    # Parse command line arguments
    SKIP_BACKUP=false
    SKIP_MIGRATION=false
    QUICK_START=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-migration)
                SKIP_MIGRATION=true
                shift
                ;;
            --quick)
                QUICK_START=true
                SKIP_BACKUP=true
                shift
                ;;
            -h|--help)
                echo "Enhanced EXPLAINIUM 2.0 Deployment Script"
                echo ""
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-backup    Skip database backup"
                echo "  --skip-migration Skip database migrations"
                echo "  --quick          Quick start (skip backup and some checks)"
                echo "  -h, --help       Show this help message"
                echo ""
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_prerequisites
    
    if [ "$SKIP_BACKUP" = false ]; then
        backup_database
    fi
    
    stop_services
    start_services
    
    if [ "$SKIP_MIGRATION" = false ]; then
        run_migrations
    fi
    
    if [ "$QUICK_START" = false ]; then
        initialize_enhanced_features
        verify_deployment
    fi
    
    display_summary
}

# Run main function with all arguments
main "$@"
