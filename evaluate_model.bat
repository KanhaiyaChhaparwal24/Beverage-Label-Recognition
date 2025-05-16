@echo off
REM Cold Drink Model Evaluation

echo Cold Drink Detector - Model Evaluation
echo ------------------------------------

REM Activate Conda Environment
call conda activate tf_env
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate tf_env conda environment.
    echo Please make sure the environment exists.
    echo You can create it with: conda create -n tf_env python=3.9
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

REM Check if trained model exists
if not exist "runs\detect\quick_cold_drinks_detector2\weights\best.pt" (
    if not exist "runs\detect\cold_drinks_detector\weights\best.pt" (
        if not exist "runs\detect\quick_cold_drinks_detector\weights\best.pt" (
            echo No trained model found. You need to train a model first.
            exit /b 1
        )
    )
)

echo Running model evaluation...
python evaluate_model.py

echo.
echo Evaluation complete! Press any key to exit...
pause > nul
exit /b 0
