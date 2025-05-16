"""
Cold Drink Detector - Model Training
----------------------------------
This script trains a YOLOv8 model on the cold drinks dataset.
"""

import os
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Configuration
DATASET_PATH = r"c:\Users\LENOVO\Desktop\Coding\cold_drink_project\cold drinks.v1i.yolov8"
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")
MODEL_NAME = "yolov8n.pt"  # Using the nano model
EPOCHS = 30
IMG_SIZE = 640  # Standard image size
BATCH_SIZE = 16

def train_model():
    """Train the YOLOv8 model on the cold drinks dataset"""
    print("Cold Drink Detector - Model Training")
    print("-----------------------------------")
    
    # Load dataset info
    try:
        with open(DATA_YAML, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"Dataset Information:")
        print(f"Number of classes: {data['nc']}")
        print(f"Classes: {data['names']}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Create model
    print(f"Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    # Train the model
    print(f"Starting training for {EPOCHS} epochs...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='cold_drinks_detector',
        patience=10,          # Early stopping
        workers=4,            # Adjust based on your CPU
        cache=True,           # Cache images in RAM for faster training
        amp=False,            # Disable mixed precision on CPU
        device='cpu'          # Use CPU for training
    )
    
    # Evaluate the model
    print("Evaluating model on validation dataset...")
    metrics = model.val()
    
    print(f"\nValidation metrics:")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.p}")
    print(f"Recall: {metrics.box.r}")
    
    print("\nModel training and evaluation completed!")
    print(f"Model saved at: {os.path.join('runs', 'detect', 'cold_drinks_detector')}")

if __name__ == "__main__":
    train_model()
