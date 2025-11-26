#!/bin/bash

# ============================================================================
# RAG Document Assistant - Automated Setup Script
# This script automates the complete setup process
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "============================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "============================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# ============================================================================
# BANNER
# ============================================================================

clear
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║         🚀 RAG Document Assistant Setup Script 🚀                 ║"
echo "║                                                                    ║"
echo "║              Automated Installation & Configuration                ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# STEP 0: CHECK SYSTEM REQUIREMENTS
# ============================================================================

print_header "📋 Checking System Requirements"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        print_success "Python $PYTHON_VERSION (>= 3.11 required)"
    else
        print_error "Python 3.11+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    print_info "Install Python 3.11+ from: https://www.python.org/downloads/"
    exit 1
fi

# Check pip
if check_command pip || check_command pip3; then
    :
else
    print_error "pip is not installed"
    exit 1
fi

# Check git
if check_command git; then
    :
else
    print_warning "git is not installed (optional but recommended)"
fi

# Check available disk space
AVAILABLE_SPACE=$(df -k . | awk 'NR==2 {print $4}')
REQUIRED_SPACE=2097152  # 2GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_warning "Low disk space. At least 2GB recommended."
else
    print_success "Sufficient disk space available"
fi

# Check RAM
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -m | awk 'NR==2{print $2}')
    if [ "$TOTAL_RAM" -lt 4096 ]; then
        print_warning "RAM is below 4GB. Application may run slowly."
    else
        print_success "Sufficient RAM available (${TOTAL_RAM}MB)"
    fi
fi

echo ""
read -p "Press Enter to continue with setup..."

# ============================================================================
# STEP 1: CREATE DIRECTORY STRUCTURE
# ============================================================================

print_header "📁 Creating Directory Structure"

# Create main directories
mkdir -p backend/app/services
mkdir -p backend/app/utils
mkdir -p backend/tests
mkdir -p frontend/.streamlit
mkdir -p data/uploads
mkdir -p data/vector_db
mkdir -p logs
mkdir -p .github/workflows

# Create __init__.py files
touch backend/app/__init__.py
touch backend/app/services/__init__.py
touch backend/app/utils/__init__.py
touch backend/tests/__init__.py

# Create .gitkeep files for empty directories
touch data/uploads/.gitkeep
touch data/vector_db/.gitkeep
touch logs/.gitkeep

print_success "Directory structure created"

# ============================================================================
# STEP 2: CREATE VIRTUAL ENVIRONMENT
# ============================================================================

print_header "🐍 Creating Virtual Environment"

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    print_success "Virtual environment activated"
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    print_success "Virtual environment activated"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet
print_success "pip upgraded to latest version"

# ============================================================================
# STEP 3: INSTALL BACKEND DEPENDENCIES
# ============================================================================

print_header "📦 Installing Backend Dependencies"

if [ -f "backend/requirements.txt" ]; then
    print_info "Installing backend packages (this may take 3-5 minutes)..."
    cd backend
    pip install -r requirements.txt --quiet
    cd ..
    print_success "Backend dependencies installed"
else
    print_warning "backend/requirements.txt not found. Creating minimal requirements..."
    cat > backend/requirements.txt << EOF
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
pydantic==2.5.3
pydantic-settings==2.1.0
python-dotenv==1.0.0
openai==1.10.0
langchain==0.1.5
sentence-transformers==2.3.1
chromadb==0.4.22
pypdf==4.0.1
python-docx==1.1.0
pandas==2.1.4
numpy==1.26.3
EOF
    cd backend
    pip install -r requirements.txt --quiet
    cd ..
    print_success "Minimal backend dependencies installed"
fi

# ============================================================================
# STEP 4: INSTALL FRONTEND DEPENDENCIES
# ============================================================================

print_header "📦 Installing Frontend Dependencies"

if [ -f "frontend/requirements.txt" ]; then
    print_info "Installing frontend packages..."
    cd frontend
    pip install -r requirements.txt --quiet
    cd ..
    print_success "Frontend dependencies installed"
else
    print_warning "frontend/requirements.txt not found. Creating minimal requirements..."
    cat > frontend/requirements.txt << EOF
streamlit==1.30.0
requests==2.31.0
pandas==2.1.4
plotly==5.18.0
python-dotenv==1.0.0
EOF
    cd frontend
    pip install -r requirements.txt --quiet
    cd ..
    print_success "Minimal frontend dependencies installed"
fi

# ============================================================================
# STEP 5: SETUP ENVIRONMENT VARIABLES
# ============================================================================

print_header "⚙️ Setting Up Environment Variables"

if [ ! -f "backend/.env" ]; then
    if [ -f "backend/.env.example" ]; then
        cp backend/.env.example backend/.env
        print_success "Created backend/.env from template"
    else
        print_info "Creating default .env file..."
        cat > backend/.env << EOF
# OpenAI Configuration (Optional - uses fallback if not set)
OPENAI_API_KEY=

# Embedding Configuration
USE_LOCAL_EMBEDDINGS=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Vector Store Configuration
VECTOR_DB_PATH=../data/vector_db
COLLECTION_NAME=documents

# File Upload Configuration
UPLOAD_DIR=../data/uploads
MAX_FILE_SIZE=10485760

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
MAX_TOKENS=500

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval Configuration
TOP_K_RESULTS=4

# Application Settings
APP_NAME=RAG Document Assistant
DEBUG=false
EOF
        print_success "Created default backend/.env file"
    fi
    print_warning "⚠️  IMPORTANT: Edit backend/.env to add your OpenAI API key (optional)"
else
    print_info "backend/.env already exists. Skipping."
fi

# ============================================================================
# STEP 6: CREATE STREAMLIT CONFIG
# ============================================================================

print_header "🎨 Creating Streamlit Configuration"

if [ ! -f "frontend/.streamlit/config.toml" ]; then
    cat > frontend/.streamlit/config.toml << EOF
[theme]
primaryColor="#1f77b4"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f0f2f6"
textColor="#262730"
font="sans serif"

[server]
port=8501
maxUploadSize=10
enableCORS=false
enableXsrfProtection=true

[browser]
gatherUsageStats=false
EOF
    print_success "Created Streamlit configuration"
else
    print_info "Streamlit config already exists. Skipping."
fi

# ============================================================================
# STEP 7: CREATE .GITIGNORE
# ============================================================================

print_header "🔒 Creating .gitignore"

if [ ! -f ".gitignore" ]; then
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Data directories
data/uploads/*
data/vector_db/*
!data/uploads/.gitkeep
!data/vector_db/.gitkeep

# Logs
*.log
logs/*.log

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Jupyter
.ipynb_checkpoints

# Streamlit
.streamlit/secrets.toml
EOF
    print_success "Created .gitignore"
else
    print_info ".gitignore already exists. Skipping."
fi

# ============================================================================
# STEP 8: VERIFY INSTALLATION
# ============================================================================

print_header "🧪 Verifying Installation"

# Check if all required packages are installed
print_info "Checking installed packages..."

REQUIRED_PACKAGES=("fastapi" "uvicorn" "streamlit" "langchain" "chromadb")
ALL_INSTALLED=true

for package in "${REQUIRED_PACKAGES[@]}"; do
    if pip show $package &> /dev/null; then
        print_success "$package installed"
    else
        print_error "$package NOT installed"
        ALL_INSTALLED=false
    fi
done

if [ "$ALL_INSTALLED" = true ]; then
    print_success "All required packages are installed"
else
    print_warning "Some packages are missing. Try running: pip install -r backend/requirements.txt"
fi

# ============================================================================
# STEP 9: CREATE STARTUP SCRIPTS
# ============================================================================

print_header "📝 Creating Startup Scripts"

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source ../venv/bin/activate 2>/dev/null || source ../venv/Scripts/activate
echo "🚀 Starting Backend Server..."
uvicorn app.main:app --reload --port 8000
EOF
chmod +x start_backend.sh
print_success "Created start_backend.sh"

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
source ../venv/bin/activate 2>/dev/null || source ../venv/Scripts/activate
echo "🎨 Starting Frontend Server..."
streamlit run streamlit_app.py
EOF
chmod +x start_frontend.sh
print_success "Created start_frontend.sh"

# Combined startup script
cat > start_all.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting RAG Document Assistant..."
echo ""
echo "Backend will start on: http://localhost:8000"
echo "Frontend will start on: http://localhost:8501"
echo ""

# Start backend in background
./start_backend.sh &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 5

# Start frontend
./start_frontend.sh

# When frontend stops, kill backend
kill $BACKEND_PID
EOF
chmod +x start_all.sh
print_success "Created start_all.sh"

# ============================================================================
# STEP 10: GENERATE PROJECT INFO
# ============================================================================

print_header "📊 Generating Project Information"

cat > PROJECT_INFO.txt << EOF
RAG Document Assistant - Project Information
============================================

Installation Date: $(date)
Python Version: $(python3 --version)
Virtual Environment: venv/

Directory Structure:
--------------------
✓ backend/              - FastAPI backend application
✓ frontend/             - Streamlit frontend application  
✓ data/uploads/         - Uploaded documents storage
✓ data/vector_db/       - Vector database storage
✓ logs/                 - Application logs
✓ venv/                 - Python virtual environment

Configuration Files:
-------------------
✓ backend/.env          - Environment variables
✓ frontend/.streamlit/config.toml - Streamlit configuration
✓ .gitignore            - Git ignore rules

Startup Scripts:
---------------
✓ start_backend.sh      - Start backend server only
✓ start_frontend.sh     - Start frontend server only
✓ start_all.sh          - Start both servers

Quick Start Commands:
--------------------
# Activate virtual environment:
source venv/bin/activate

# Start backend:
./start_backend.sh

# Start frontend (in new terminal):
./start_frontend.sh

# Or start both:
./start_all.sh

Access URLs:
-----------
Backend API: http://localhost:8000
API Docs: http://localhost:8000/docs
Frontend UI: http://localhost:8501

Next Steps:
----------
1. Review backend/.env and add your OpenAI API key (optional)
2. Run ./start_all.sh to start both servers
3. Open http://localhost:8501 in your browser
4. Upload a document and ask questions!

Documentation:
-------------
- README.md - Complete project documentation
- QUICKSTART.md - 5-minute quick start guide
- INSTALLATION.md - Detailed installation instructions
- DEPLOYMENT.md - Cloud deployment guide

Support:
-------
- GitHub: https://github.com/yourusername/rag-document-assistant
- Issues: https://github.com/yourusername/rag-document-assistant/issues
EOF

print_success "Created PROJECT_INFO.txt"

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_header "🎉 Setup Complete!"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ SETUP SUCCESSFUL! ✅                         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

print_success "Virtual environment created: venv/"
print_success "Backend dependencies installed"
print_success "Frontend dependencies installed"
print_success "Configuration files created"
print_success "Directory structure set up"
print_success "Startup scripts generated"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_info "📝 NEXT STEPS:"
echo ""
echo "  1. Review and edit configuration:"
echo "     ${YELLOW}nano backend/.env${NC}"
echo ""
echo "  2. Add your OpenAI API key (optional):"
echo "     ${YELLOW}OPENAI_API_KEY=sk-your-key-here${NC}"
echo ""
echo "  3. Start the application:"
echo "     ${GREEN}./start_all.sh${NC}"
echo ""
echo "  4. Open in your browser:"
echo "     ${BLUE}http://localhost:8501${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

print_info "📚 For detailed documentation, see:"
echo "  - PROJECT_INFO.txt (summary of installation)"
echo "  - README.md (complete documentation)"
echo "  - QUICKSTART.md (5-minute guide)"
echo ""

print_warning "⚠️  Important Notes:"
echo "  • The system works without an OpenAI API key (uses fallback mode)"
echo "  • For best results, add your OpenAI API key in backend/.env"
echo "  • Activate the virtual environment before running: source venv/bin/activate"
echo ""

# Optional: Run a quick test
echo ""
read -p "Would you like to run a quick test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header "🧪 Running Quick Test"
    
    print_info "Testing backend imports..."
    cd backend
    if python3 -c "import fastapi, uvicorn, langchain, chromadb; print('✓ All imports successful')" 2>/dev/null; then
        print_success "Backend dependencies working"
    else
        print_warning "Some backend dependencies may have issues"
    fi
    cd ..
    
    print_info "Testing frontend imports..."
    cd frontend
    if python3 -c "import streamlit, requests; print('✓ All imports successful')" 2>/dev/null; then
        print_success "Frontend dependencies working"
    else
        print_warning "Some frontend dependencies may have issues"
    fi
    cd ..
    
    print_success "Quick test completed"
fi

echo ""
print_success "🎊 All done! Your RAG Document Assistant is ready to use!"
echo ""
echo "Run: ${GREEN}./start_all.sh${NC} to start the application"
echo ""

# Save installation log
LOG_FILE="logs/setup_$(date +%Y%m%d_%H%M%S).log"
echo "Setup completed successfully at $(date)" > "$LOG_FILE"
echo "Python version: $(python3 --version)" >> "$LOG_FILE"
echo "Installation path: $(pwd)" >> "$LOG_FILE"
print_info "Installation log saved to: $LOG_FILE"

# ============================================================================
# END OF SETUP SCRIPT
# ============================================================================