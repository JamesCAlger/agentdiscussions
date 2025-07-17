@echo off
REM Batch script to run example conversations on Windows
REM Usage: run_examples.bat

echo Starting Agentic Conversation System Examples
echo ============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if the package is installed
python -c "import agentic_conversation" >nul 2>&1
if errorlevel 1 (
    echo Error: agentic_conversation package is not installed
    echo Please run: pip install -e .
    pause
    exit /b 1
)

REM Run the examples
echo Running example conversations...
python examples/scripts/run_examples.py

echo.
echo Examples completed. Check the logs directory for results.
pause