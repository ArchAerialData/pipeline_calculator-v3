#!/bin/bash
# Setup script for Pipeline Calculator v3.0

echo "Pipeline Calculator v3.0 - Setup Script"
echo "======================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✓ Python version $python_version meets requirements"
else
    echo "✗ Python version $python_version is below required $required_version"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p src
mkdir -p .github/workflows
mkdir -p test_data

# Move main script to src directory if needed
if [ -f "pipeline_calculator_v3.py" ] && [ ! -f "src/pipeline_calculator_v3.py" ]; then
    echo "Moving main script to src directory..."
    mv pipeline_calculator_v3.py src/
fi

# Create a simple test to verify installation
echo ""
echo "Testing installation..."
python3 -c "
import sys
print('Testing imports...')
try:
    import pyproj
    print('✓ pyproj')
    import pandas
    print('✓ pandas')
    import numpy
    print('✓ numpy')
    import scipy
    print('✓ scipy')
    import customtkinter
    print('✓ customtkinter')
    import tkinterdnd2
    print('✓ tkinterdnd2')
    print('')
    print('All dependencies installed successfully!')
except ImportError as e:
    print(f'✗ Failed to import: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "Setup completed successfully!"
    echo "================================"
    echo ""
    echo "To run the application:"
    echo "  python src/pipeline_calculator_v3.py"
    echo ""
    echo "To build executables locally:"
    echo "  pyinstaller --onefile --windowed --name Pipeline_Calculator_v3 src/pipeline_calculator_v3.py"
    echo ""
    echo "To deactivate virtual environment:"
    echo "  deactivate"
else
    echo ""
    echo "Setup failed. Please check error messages above."
    exit 1
fi