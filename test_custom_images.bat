@echo off
REM Cold Drink Custom Image Testing Script

echo Cold Drink Detector - Custom Image Testing
echo ---------------------------------------

REM Activate Conda Environment
call conda activate tf_env
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate tf_env conda environment.
    echo Please make sure the environment exists.
    echo You can create it with: conda create -n tf_env python=3.9
    exit /b 1
)

REM Check if test_images directory exists and is not empty
if not exist "test_images\" (
    echo Creating test_images directory...
    mkdir test_images
    echo.
    echo Please place some test images in the 'test_images' folder
    echo then run this script again.
    pause
    exit /b 0
)

REM Count files in the directory
set COUNT=0
for %%F in (test_images\*.*) do set /a COUNT+=1

if %COUNT% EQU 0 (
    echo The test_images folder is empty.
    echo Please add some image files (.jpg, .png, etc.) to the 'test_images' folder
    echo then run this script again.
    pause
    exit /b 0
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

REM Run the test script
echo Running detection on custom images...
python test_custom_images.py

echo.
echo Testing complete! Results are in the 'test_results' folder.
echo Press any key to exit...
pause > nul
exit /b 0
