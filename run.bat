@echo off

echo Starting Eye Image Overlay Application...
echo.

:: Check if virtual environment exists
if not exist venv (
    echo Error: Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

:: Activate virtual environment and run the application
call venv\Scripts\activate.bat

echo.
echo Running application...
python src/main.py

if errorlevel 1 (
    echo.
    echo Error: Application exited with an error
    pause
    exit /b 1
)

deactivate 