"""
Model Evaluation Script for Cold Drink Detector
----------------------------------------------
This script evaluates the performance of the trained YOLOv8 model on the validation dataset.
It provides detailed metrics for each class and visualizes some example detections.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml
import random

# Set paths
project_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(project_path, "cold drinks.v1i.yolov8")
valid_images_path = os.path.join(dataset_path, "valid", "images")
yaml_path = os.path.join(dataset_path, "data.yaml")

# Find the best model
model_dirs = [
    os.path.join(project_path, "runs", "detect", "quick_cold_drinks_detector2", "weights", "best.pt"),
    os.path.join(project_path, "runs", "detect", "cold_drinks_detector", "weights", "best.pt"),
    os.path.join(project_path, "runs", "detect", "quick_cold_drinks_detector", "weights", "best.pt")
]

# Use the first model that exists
model_path = next((path for path in model_dirs if os.path.exists(path)), None)

if not model_path:
    print("No trained model found. Please train the model first.")
    exit(1)

print(f"Using model: {model_path}")

# Load class names from data.yaml
try:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        class_names = data.get('names', [])
    print(f"Loaded {len(class_names)} class names from data.yaml")
except Exception as e:
    print(f"Error loading class names: {str(e)}")
    class_names = []

# Load model
print("Loading YOLOv8 model...")
model = YOLO(model_path)

# Run validation
print("Running validation on the validation dataset...")
results = model.val(data=yaml_path, split='val', verbose=True)

# Print detailed results per class
print("\nDetailed Results per Class:")
print("-" * 80)
print(f"{'Class':30s} {'Precision':10s} {'Recall':10s} {'mAP50':10s} {'mAP50-95':10s}")
print("-" * 80)

# Check if class metrics are available
if hasattr(results, 'names') and hasattr(results, 'metrics') and hasattr(results.metrics, 'class_result'):
    for i, class_metric in enumerate(results.metrics.class_result):
        if i < len(class_names):
            precision, recall, ap50, ap = class_metric  # Extract metrics
            print(f"{class_names[i]:30s} {precision:.4f}     {recall:.4f}     {ap50:.4f}     {ap:.4f}")

print("-" * 80)

# Plot some examples from validation set
print("\nGenerating example detections...")

# Get a few random images from validation set
val_images = [f for f in os.listdir(valid_images_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if val_images:
    # Select 5 random images, or all if less than 5
    sample_size = min(5, len(val_images))
    sample_images = random.sample(val_images, sample_size)
    
    # Create output directory for visualizations
    vis_dir = os.path.join(project_path, "evaluation_results")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process each sample image
    for i, img_file in enumerate(sample_images):
        img_path = os.path.join(valid_images_path, img_file)
        
        # Run prediction
        results = model(img_path)
        result = results[0]  # first and only result
        
        # Get the annotated image
        annotated_img = result.plot()
        
        # Save the annotated image
        output_path = os.path.join(vis_dir, f"detection_{i+1}.jpg")
        cv2.imwrite(output_path, annotated_img)
        print(f"Saved annotated image to {output_path}")

    print(f"\nEvaluation complete. Check {vis_dir} for visualization examples.")
else:
    print("No validation images found.")

print("\nEvaluation completed!")
