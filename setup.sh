#!/bin/bash

# EXPLAINIUM - Automated Setup Script
# Sets up the entire system from scratch

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Banner
clear
echo "=================================================================="
log_header "🧠 EXPLAINIUM - Automated Setup"
echo "=================================================================="
echo "Setting up your AI-powered knowledge extraction system..."
echo "This will take 5-10 minutes depending on your internet connection."
echo "=================================================================="
echo ""

# Check system requirements
log_info "Checking system requirements..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_warning "This system is optimized for macOS with Apple Silicon"
    log_info "Continuing with generic setup..."
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required but not installed"
    log_info "Please install Python 3.9+ and run this script again"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log_success "Python $PYTHON_VERSION detected"

# Check available RAM
if command -v sysctl &> /dev/null; then
    RAM_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    log_info "Available RAM: ${RAM_GB}GB"
    if [ "$RAM_GB" -lt 16 ]; then
        log_warning "16GB+ RAM recommended for optimal performance"
    fi
fi

# Step 1: Create virtual environment
log_info "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
log_success "Virtual environment activated"

# Step 2: Upgrade pip and install dependencies
log_info "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    log_success "Dependencies installed successfully"
else
    log_error "requirements.txt not found"
    exit 1
fi

# Step 3: Create necessary directories
log_info "Creating application directories..."
mkdir -p uploaded_files
mkdir -p logs
mkdir -p models
log_success "Directories created"

# Step 4: Download AI models (in background)
log_info "Preparing AI models (this may take a few minutes)..."
python -c "
import sys
sys.path.append('src')
try:
    from core.config import AIConfig
    from ai.advanced_knowledge_engine import AdvancedKnowledgeEngine
    print('✅ AI components verified')
except ImportError as e:
    print(f'⚠️  AI components will be loaded on first use: {e}')
" || log_warning "AI models will be downloaded on first use"

# Step 5: Test basic functionality
log_info "Testing system components..."

# Test database
python -c "
import sys
sys.path.append('src')
try:
    from database.database import init_db
    init_db()
    print('✅ Database initialized')
except Exception as e:
    print(f'⚠️  Database will initialize on first run: {e}')
" || log_warning "Database will initialize on first run"

# Test frontend
python -c "
import streamlit
print('✅ Streamlit frontend ready')
" || log_error "Streamlit installation failed"

# Test backend
python -c "
import fastapi
import uvicorn
print('✅ FastAPI backend ready')
" || log_error "FastAPI installation failed"

# Step 6: Create startup scripts
log_info "Creating startup scripts..."

# Create start.sh
cat > start.sh << 'EOF'
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
EOF

chmod +x start.sh

# Create stop.sh
cat > stop.sh << 'EOF'
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
EOF

chmod +x stop.sh

log_success "Startup scripts created"

# Step 7: Final verification
log_info "Running final system verification..."

# Test imports
python -c "
import sys
sys.path.append('src')

# Test core imports
try:
    import streamlit
    import fastapi
    import pandas
    import plotly
    print('✅ Core dependencies verified')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)

# Test AI imports (optional)
try:
    from ai.advanced_knowledge_engine import AdvancedKnowledgeEngine
    from core.config import AIConfig
    print('✅ AI components available')
except ImportError:
    print('⚠️  AI components will load on first use')

print('✅ System verification complete')
"

if [ $? -eq 0 ]; then
    log_success "System verification passed"
else
    log_error "System verification failed"
    exit 1
fi

# Success message
echo ""
echo "=================================================================="
log_header "🎉 EXPLAINIUM Setup Complete!"
echo "=================================================================="
log_success "Your AI-powered knowledge extraction system is ready!"
echo ""
echo "🚀 To start the application:"
echo "   ./start.sh"
echo ""
echo "🛑 To stop the application:"
echo "   ./stop.sh"
echo ""
echo "📖 Quick Start Guide:"
echo "   1. Run: ./start.sh"
echo "   2. Open: http://localhost:8501"
echo "   3. Upload any document, image, video, or audio file"
echo "   4. Watch AI extract knowledge automatically!"
echo ""
echo "🔧 Troubleshooting:"
echo "   - Check logs in the 'logs' directory"
echo "   - Ensure 16GB+ RAM available"
echo "   - Restart with: ./stop.sh && ./start.sh"
echo ""
echo "=================================================================="
log_success "Setup completed successfully! 🎉"
echo "=================================================================="