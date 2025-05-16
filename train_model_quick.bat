@echo off
echo Setting up environment for quick training...

:: Activate the conda environment
call conda activate tf_env || (
    echo Failed to activate tf_env conda environment
    echo Please make sure the conda environment is properly set up
    pause
    exit /b 1
)

:: Set OpenMP environment variable to avoid conflicts
set KMP_DUPLICATE_LIB_OK=TRUE

echo Starting quick model training with YOLOv8 (5 epochs)...
python train_model_quick.py
echo Training process completed.
pause
