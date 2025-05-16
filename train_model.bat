@echo off
echo Setting up environment for training...

:: Activate the conda environment
call conda activate tf_env || (
    echo Failed to activate tf_env conda environment
    echo Please make sure the conda environment is properly set up
    pause
    exit /b 1
)

:: Set OpenMP environment variable to avoid conflicts
set KMP_DUPLICATE_LIB_OK=TRUE

echo Starting model training with YOLOv8...

:: Use train_model_fixed.py if it exists, otherwise use train_model.py
if exist "train_model_fixed.py" (
    python train_model_fixed.py
) else (
    python train_model.py
)

echo Training process completed.
pause
