@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

set "PORT=8502"
if not "%~1"=="" set "PORT=%~1"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python or add it to PATH, then run this file again.
    pause
    exit /b 1
)

python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Streamlit is not installed in the current Python environment.
    echo Install it with:
    echo     python -m pip install streamlit
    pause
    exit /b 1
)

echo Starting Surrogate Modeling Workbench...
echo Working directory: %CD%
echo URL: http://localhost:%PORT%
echo.

start "" "http://localhost:%PORT%"
python -m streamlit run "%~dp0SurrogateModelingUI.py" --server.port %PORT% --server.headless true

echo.
echo Streamlit has stopped.
pause
