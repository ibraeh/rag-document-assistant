@echo off
REM ============================================================================
REM RAG Document Assistant - Windows Setup Script
REM Automated installation for Windows
REM ============================================================================

setlocal enabledelayedexpansion

REM Colors (limited in CMD)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "NC=[0m"

REM ============================================================================
REM BANNER
REM ============================================================================

cls
echo ========================================================================
echo.
echo          RAG Document Assistant Setup Script (Windows)
echo.
echo                 Automated Installation ^& Configuration
echo.
echo ========================================================================
echo.

REM ============================================================================
REM CHECK SYSTEM REQUIREMENTS
REM ============================================================================

echo [STEP 1/10] Checking System Requirements...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR] Python is not installed or not in PATH%NC%
    echo Please install Python 3.11+ from: https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo %GREEN%[OK] Python !PYTHON_VERSION! found%NC%
)

REM Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR] pip is not installed%NC%
    pause
    exit /b 1
) else (
    echo %GREEN%[OK] pip found%NC%
)

REM Check git (optional)
git --version >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[WARNING] git is not installed (optional but recommended)%NC%
) else (
    echo %GREEN%[OK] git found%NC%
)

echo.
pause

REM ============================================================================
REM CREATE DIRECTORY STRUCTURE
REM ============================================================================

echo.
echo [STEP 2/10] Creating Directory Structure...
echo.

mkdir backend\app\services 2>nul
mkdir backend\app\utils 2>nul
mkdir backend\tests 2>nul
mkdir frontend\.streamlit 2>nul
mkdir data\uploads 2>nul
mkdir data\vector_db 2>nul
mkdir logs 2>nul
mkdir .github\workflows 2>nul

REM Create __init__.py files
type nul > backend\app\__init__.py
type nul > backend\app\services\__init__.py
type nul > backend\app\utils\__init__.py
type nul > backend\tests\__init__.py

REM Create .gitkeep files
type nul > data\uploads\.gitkeep
type nul > data\vector_db\.gitkeep
type nul > logs\.gitkeep

echo %GREEN%[OK] Directory structure created%NC%

REM ============================================================================
REM CREATE VIRTUAL ENVIRONMENT
REM ============================================================================

echo.
echo [STEP 3/10] Creating Virtual Environment...
echo.

if exist venv (
    echo %YELLOW%[WARNING] Virtual environment already exists. Skipping.%NC%
) else (
    python -m venv venv
    echo %GREEN%[OK] Virtual environment created%NC%
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo %GREEN%[OK] pip upgraded%NC%

REM ============================================================================
REM INSTALL BACKEND DEPENDENCIES
REM ============================================================================

echo.
echo [STEP 4/10] Installing Backend Dependencies...
echo This may take 3-5 minutes...
echo.

if exist backend\requirements.txt (
    cd backend
    pip install -r requirements.txt --quiet
    cd ..
    echo %GREEN%[OK] Backend dependencies installed%NC%
) else (
    echo %YELLOW%[WARNING] Creating minimal requirements.txt%NC%
    (
        echo fastapi==0.109.0
        echo uvicorn[standard]==0.27.0
        echo python-multipart==0.0.6
        echo pydantic==2.5.3
        echo pydantic-settings==2.1.0
        echo python-dotenv==1.0.0
        echo openai==1.10.0
        echo langchain==0.1.5
        echo sentence-transformers==2.3.1
        echo chromadb==0.4.22
        echo pypdf==4.0.1
        echo python-docx==1.1.0
        echo pandas==2.1.4
        echo numpy==1.26.3
    ) > backend\requirements.txt
    cd backend
    pip install -r requirements.txt --quiet
    cd ..
    echo %GREEN%[OK] Minimal backend dependencies installed%NC%
)

REM ============================================================================
REM INSTALL FRONTEND DEPENDENCIES
REM ============================================================================

echo.
echo [STEP 5/10] Installing Frontend Dependencies...
echo.

if exist frontend\requirements.txt (
    cd frontend
    pip install -r requirements.txt --quiet
    cd ..
    echo %GREEN%[OK] Frontend dependencies installed%NC%
) else (
    echo %YELLOW%[WARNING] Creating minimal requirements.txt%NC%
    (
        echo streamlit==1.30.0
        echo requests==2.31.0
        echo pandas==2.1.4
        echo plotly==5.18.0
        echo python-dotenv==1.0.0
    ) > frontend\requirements.txt
    cd frontend
    pip install -r requirements.txt --quiet
    cd ..
    echo %GREEN%[OK] Minimal frontend dependencies installed%NC%
)

REM ============================================================================
REM SETUP ENVIRONMENT VARIABLES
REM ============================================================================

echo.
echo [STEP 6/10] Setting Up Environment Variables...
echo.

if not exist backend\.env (
    (
        echo # OpenAI Configuration (Optional^)
        echo OPENAI_API_KEY=
        echo.
        echo # Embedding Configuration
        echo USE_LOCAL_EMBEDDINGS=true
        echo EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
        echo.
        echo # Vector Store Configuration
        echo VECTOR_DB_PATH=../data/vector_db
        echo COLLECTION_NAME=documents
        echo.
        echo # File Upload Configuration
        echo UPLOAD_DIR=../data/uploads
        echo MAX_FILE_SIZE=10485760
        echo.
        echo # LLM Configuration
        echo LLM_MODEL=gpt-3.5-turbo
        echo LLM_TEMPERATURE=0.7
        echo MAX_TOKENS=500
        echo.
        echo # Document Processing
        echo CHUNK_SIZE=1000
        echo CHUNK_OVERLAP=200
        echo.
        echo # Application Settings
        echo APP_NAME=RAG Document Assistant
        echo DEBUG=false
    ) > backend\.env
    echo %GREEN%[OK] Created backend\.env file%NC%
    echo %YELLOW%[IMPORTANT] Edit backend\.env to add your OpenAI API key (optional^)%NC%
) else (
    echo %YELLOW%[INFO] backend\.env already exists%NC%
)

REM ============================================================================
REM CREATE STREAMLIT CONFIG
REM ============================================================================

echo.
echo [STEP 7/10] Creating Streamlit Configuration...
echo.

if not exist frontend\.streamlit\config.toml (
    (
        echo [theme]
        echo primaryColor="#1f77b4"
        echo backgroundColor="#ffffff"
        echo secondaryBackgroundColor="#f0f2f6"
        echo textColor="#262730"
        echo font="sans serif"
        echo.
        echo [server]
        echo port=8501
        echo maxUploadSize=10
        echo enableCORS=false
        echo.
        echo [browser]
        echo gatherUsageStats=false
    ) > frontend\.streamlit\config.toml
    echo %GREEN%[OK] Created Streamlit configuration%NC%
) else (
    echo %YELLOW%[INFO] Streamlit config already exists%NC%
)

REM ============================================================================
REM CREATE .GITIGNORE
REM ============================================================================

echo.
echo [STEP 8/10] Creating .gitignore...
echo.

if not exist .gitignore (
    (
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo venv/
        echo env/
        echo.
        echo # Environment
        echo .env
        echo .env.local
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo.
        echo # Data
        echo data/uploads/*
        echo data/vector_db/*
        echo !data/uploads/.gitkeep
        echo !data/vector_db/.gitkeep
        echo.
        echo # Logs
        echo *.log
        echo logs/*.log
        echo.
        echo # Testing
        echo .pytest_cache/
        echo htmlcov/
    ) > .gitignore
    echo %GREEN%[OK] Created .gitignore%NC%
) else (
    echo %YELLOW%[INFO] .gitignore already exists%NC%
)

REM ============================================================================
REM CREATE STARTUP SCRIPTS
REM ============================================================================

echo.
echo [STEP 9/10] Creating Startup Scripts...
echo.

REM Backend startup script
(
    echo @echo off
    echo cd backend
    echo call ..\venv\Scripts\activate.bat
    echo echo Starting Backend Server...
    echo uvicorn app.main:app --reload --port 8000
) > start_backend.bat
echo %GREEN%[OK] Created start_backend.bat%NC%

REM Frontend startup script
(
    echo @echo off
    echo cd frontend
    echo call ..\venv\Scripts\activate.bat
    echo echo Starting Frontend Server...
    echo streamlit run streamlit_app.py
) > start_frontend.bat
echo %GREEN%[OK] Created start_frontend.bat%NC%

REM Combined startup script
(
    echo @echo off
    echo echo Starting RAG Document Assistant...
    echo echo.
    echo echo Backend will start on: http://localhost:8000
    echo echo Frontend will start on: http://localhost:8501
    echo echo.
    echo echo Press Ctrl+C to stop servers
    echo echo.
    echo start /B call start_backend.bat
    echo timeout /t 5 /nobreak ^> nul
    echo call start_frontend.bat
) > start_all.bat
echo %GREEN%[OK] Created start_all.bat%NC%

REM ============================================================================
REM VERIFY INSTALLATION
REM ============================================================================

echo.
echo [STEP 10/10] Verifying Installation...
echo.

python -c "import fastapi, uvicorn, streamlit" 2>nul
if errorlevel 1 (
    echo %YELLOW%[WARNING] Some packages may not be installed correctly%NC%
) else (
    echo %GREEN%[OK] Core packages verified%NC%
)

REM ============================================================================
REM FINAL SUMMARY
REM ============================================================================

cls
echo ========================================================================
echo.
echo                    SETUP COMPLETE!
echo.
echo ========================================================================
echo.
echo %GREEN%[OK] Virtual environment created: venv\%NC%
echo %GREEN%[OK] Backend dependencies installed%NC%
echo %GREEN%[OK] Frontend dependencies installed%NC%
echo %GREEN%[OK] Configuration files created%NC%
echo %GREEN%[OK] Startup scripts generated%NC%
echo.
echo ========================================================================
echo.
echo NEXT STEPS:
echo.
echo   1. Edit configuration:
echo      notepad backend\.env
echo.
echo   2. Add your OpenAI API key (optional):
echo      OPENAI_API_KEY=sk-your-key-here
echo.
echo   3. Start the application:
echo      start_all.bat
echo.
echo   4. Open in your browser:
echo      http://localhost:8501
echo.
echo ========================================================================
echo.
echo IMPORTANT NOTES:
echo   - The system works without OpenAI API key (uses fallback mode)
echo   - For best results, add your OpenAI API key in backend\.env
echo   - Activate virtual environment: venv\Scripts\activate.bat
echo.
echo ========================================================================
echo.

REM Create PROJECT_INFO.txt
(
    echo RAG Document Assistant - Windows Installation
    echo =============================================
    echo.
    echo Installation Date: %date% %time%
    echo Python Version: !PYTHON_VERSION!
    echo Installation Path: %cd%
    echo.
    echo Quick Start:
    echo   1. Activate: venv\Scripts\activate.bat
    echo   2. Start all: start_all.bat
    echo   3. Open: http://localhost:8501
    echo.
    echo Access URLs:
    echo   Backend API: http://localhost:8000
    echo   API Docs: http://localhost:8000/docs
    echo   Frontend UI: http://localhost:8501
) > PROJECT_INFO.txt

echo %GREEN%[INFO] Installation log saved to PROJECT_INFO.txt%NC%
echo.
echo Press any key to finish setup...
pause >nul

endlocal