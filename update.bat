@echo off
echo Updating Eye Image Overlay Application...
echo.

:: Check if virtual environment exists
if not exist venv (
    echo Error: Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo Updating pip...
python -m pip install --upgrade pip

:: Update dependencies individually
echo Updating dependencies...

pip install customtkinter==5.2.0
if errorlevel 1 (
    echo Warning: Failed to update customtkinter. Trying without version constraint...
    pip install --upgrade customtkinter
)

pip install pillow==10.0.0
if errorlevel 1 (
    echo Warning: Failed to update pillow. Trying without version constraint...
    pip install --upgrade pillow
)

pip install opencv-python==4.8.0.74
if errorlevel 1 (
    echo Warning: Failed to update opencv-python. Trying without version constraint...
    pip install --upgrade opencv-python
)

pip install mediapipe==0.10.5
if errorlevel 1 (
    echo Warning: Failed to update mediapipe. Trying without version constraint...
    pip install --upgrade mediapipe
)

pip install numpy==1.25.2
if errorlevel 1 (
    echo Warning: Failed to update numpy. Trying without version constraint...
    pip install --upgrade numpy
)

pip install plexapi==4.15.1
if errorlevel 1 (
    echo Warning: Failed to update plexapi. Trying without version constraint...
    pip install --upgrade plexapi
)

pip install requests==2.31.0
if errorlevel 1 (
    echo Warning: Failed to update requests. Trying without version constraint...
    pip install --upgrade requests
)

echo.
echo Update complete!
echo Run the application using run.bat
echo.
pause

deactivate 