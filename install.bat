@echo off

echo Installing Eye Image Overlay Application...
echo.

:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in the PATH
    echo Please install Python 3.8 or later from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies with better error handling
echo Installing dependencies...
echo Installing core dependencies...

:: Install setuptools first for Python 3.12 compatibility
pip install setuptools>=60.0.0
if errorlevel 1 (
    echo Warning: Failed to install setuptools. Trying without version constraint...
    pip install setuptools
)

:: Install each package individually for better error handling
pip install customtkinter==5.2.0
if errorlevel 1 (
    echo Warning: Failed to install customtkinter. Trying without version constraint...
    pip install customtkinter
)

pip install pillow==10.0.0
if errorlevel 1 (
    echo Warning: Failed to install pillow. Trying without version constraint...
    pip install pillow
)

pip install opencv-python==4.8.0.74
if errorlevel 1 (
    echo Warning: Failed to install opencv-python. Trying without version constraint...
    pip install opencv-python
)

pip install mediapipe==0.10.21
if errorlevel 1 (
    echo Warning: Failed to install mediapipe. Trying without version constraint...
    pip install mediapipe
)

pip install numpy>=1.25.2,<2.0.0
if errorlevel 1 (
    echo Warning: Failed to install numpy. Trying without version constraint...
    pip install numpy
)

pip install plexapi==4.15.1
if errorlevel 1 (
    echo Warning: Failed to install plexapi. Trying without version constraint...
    pip install plexapi
)

pip install requests==2.31.0
if errorlevel 1 (
    echo Warning: Failed to install requests. Trying without version constraint...
    pip install requests
)

echo.
echo Installation complete!
echo Run the application using run.bat
echo.
pause

deactivate 
