@echo off
REM Setup script for Pipeline Calculator v3.0 - Windows

echo Pipeline Calculator v3.0 - Setup Script
echo ========================================

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create directory structure
echo.
echo Creating directory structure...
if not exist "src" mkdir src
if not exist ".github\workflows" mkdir .github\workflows
if not exist "test_data" mkdir test_data

REM Move main script if needed
if exist "pipeline_calculator_v3.py" (
    if not exist "src\pipeline_calculator_v3.py" (
        echo Moving main script to src directory...
        move pipeline_calculator_v3.py src\
    )
)

REM Test installation
echo.
echo Testing installation...
python -c "import pyproj, pandas, numpy, scipy, customtkinter, tkinterdnd2; print('All dependencies installed successfully!')"

if %errorlevel% equ 0 (
    echo.
    echo ================================
    echo Setup completed successfully!
    echo ================================
    echo.
    echo To run the application:
    echo   python src\pipeline_calculator_v3.py
    echo.
    echo To build executable locally:
    echo   pyinstaller --onefile --windowed --name Pipeline_Calculator_v3 src\pipeline_calculator_v3.py
    echo.
    echo Virtual environment is activated. To deactivate:
    echo   deactivate
) else (
    echo.
    echo Setup failed. Please check error messages above.
)

pause