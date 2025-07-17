@echo off
REM Batch script to run performance benchmarks on Windows
REM Usage: benchmark.bat

echo Starting Performance Benchmark
echo ==============================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if API keys are set
if "%OPENAI_API_KEY%"=="" if "%ANTHROPIC_API_KEY%"=="" (
    echo Error: No API keys found
    echo Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    pause
    exit /b 1
)

REM Run the benchmark
echo Running performance benchmarks...
python examples/scripts/performance_benchmark.py

echo.
echo Benchmark completed. Check the benchmark_results directory for detailed results.
pause