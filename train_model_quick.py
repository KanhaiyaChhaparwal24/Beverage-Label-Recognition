"""
Cold Drink Detector - Quick Model Training
---------------------------------------
This script provides a faster training option for the YOLOv8 model.
"""

import os
import yaml
from ultralytics import YOLO

# Configuration for quick training
DATASET_PATH = r"c:\Users\LENOVO\Desktop\Coding\cold_drink_project\cold drinks.v1i.yolov8"
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")
MODEL_NAME = "yolov8n.pt"  # Smallest model
EPOCHS = 5               # Minimal epochs for quick training
IMG_SIZE = 416           # Smaller image size
BATCH_SIZE = 16

def quick_train():
    """Run a quick training session"""
    print("Cold Drink Detector - Quick Training Mode")
    print("----------------------------------------")
    
    # Load data configuration
    try:
        with open(DATA_YAML, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"Training on {len(data['names'])} cold drink classes")
    except Exception as e:
        print(f"Error loading data configuration: {str(e)}")
        return
    
    # Create a YOLOv8 model
    print(f"Loading {MODEL_NAME} model...")
    model = YOLO(MODEL_NAME)
    
    # Run quick training
    print(f"Starting quick training with {EPOCHS} epochs...")
    print("This will be much faster than the full training!")
      # Train the model with optimized settings for speed
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='quick_cold_drinks_detector',
        patience=3,           # Early stopping
        workers=4,            # Adjust based on your CPU
        cache=True,           # Cache images for faster training
        amp=False,            # No mixed precision on CPU
        device='cpu'          # Use CPU for training
    )
    
    # Run a quick validation
    print("\nRunning quick validation...")
    metrics = model.val()
    
    print(f"\nTraining completed!")
    print(f"Model saved at: {os.path.join('runs', 'detect', 'quick_cold_drinks_detector')}")
    print("\nTo use this model for detection, run the detector application.")

if __name__ == "__main__":
    quick_train()
