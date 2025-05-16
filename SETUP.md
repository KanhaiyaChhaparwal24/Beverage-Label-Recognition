# Setup Guide for Cold Drink Detector

This guide will help you set up the Cold Drink Detector project after cloning from GitHub.

## Prerequisites

1. **Python Environment**:
   - Python 3.8 or higher
   - Conda (recommended for environment management)

2. **GPU Support** (Optional but recommended):
   - CUDA-compatible GPU
   - CUDA Toolkit 11.0+ and cuDNN

## Step 1: Setting Up the Environment

Create and activate a conda environment:

```bash
# Create a new conda environment
conda create -n tf_env python=3.9
conda activate tf_env

# Install required packages
pip install -r requirements.txt
```

## Step 2: Obtaining the Dataset

The dataset is not included in this repository due to its size. You have two options:

### Option 1: Download the pre-processed dataset

1. Download the "cold drinks.v1i.yolov8" dataset from [this link](https://universe.roboflow.com/cold-drinks-cftkw/cold-drinks-qki32/dataset/1)
2. Extract the downloaded zip file
3. Place the extracted "cold drinks.v1i.yolov8" folder in the project root directory

### Option 2: Use your own dataset

1. Prepare your dataset in YOLOv8 format with the following structure:
   ```
   cold drinks.v1i.yolov8/
   ├── data.yaml          # Dataset configuration
   ├── train/
   │   ├── images/        # Training images
   │   └── labels/        # Training annotations
   └── valid/
       ├── images/        # Validation images
       └── labels/        # Validation annotations
   ```
2. Update `data.yaml` with the appropriate class names and paths

## Step 3: Obtaining the Base Model

1. Download the YOLOv8 nano pre-trained model:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```
   Or download it manually from [Ultralytics](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)

2. Place the `yolov8n.pt` file in the project root directory

## Step 4: Train the Model

Run one of the training scripts:

```bash
# For quick training (5 epochs)
python train_model_quick.py

# For full training (30 epochs)
python train_model.py
```

## Step 5: Run the UI

After training is complete, you can run the UI:

```bash
python run_ui.py
```

## Troubleshooting

If you encounter any issues:

1. **Missing packages**: Make sure all required packages are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA errors**: If you have CUDA-related errors, try installing the CPU-only version of PyTorch
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Model not found**: Ensure you've trained the model or downloaded a pre-trained one

4. **UI not showing**: Check if you're using a virtual environment with display support
