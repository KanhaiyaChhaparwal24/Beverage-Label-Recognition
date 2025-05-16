@echo off
REM Quick launcher for Cold Drink Detector UI

echo Cold Drink Detector - UI Launcher
echo ------------------------------

REM Initialize conda for batch file
call conda.bat activate base
if %ERRORLEVEL% NEQ 0 (
    echo Error: Conda initialization failed.
    echo Please make sure conda is installed and properly set up.
    pause
    exit /b 1
)

REM Activate Conda Environment
call conda.bat activate tf_env
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate tf_env conda environment.
    echo Please make sure the environment exists.
    echo You can create it with: conda create -n tf_env python=3.9
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking requirements...
pip freeze | findstr "ultralytics" > nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to install required packages.
        exit /b 1
    )
)

REM Check if at least one trained model exists in any of the expected directories
echo Looking for trained models...

echo.
echo Starting Cold Drink Detector UI...
python cold_drink_detector.py

exit /b 0
