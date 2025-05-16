"""
Image Testing Script for Cold Drink Detector
-------------------------------------------
This script allows testing the model on custom images placed in the test_images folder.
It reads images from test_images folder, runs detection, and saves results to test_results folder.
"""

import os
import sys
import cv2
from glob import glob
from ultralytics import YOLO
import yaml

def main():
    # Set up paths
    project_path = os.path.dirname(os.path.abspath(__file__))
    test_images_dir = os.path.join(project_path, "test_images")
    results_dir = os.path.join(project_path, "test_results")
    dataset_path = os.path.join(project_path, "cold drinks.v1i.yolov8")
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Find model file
    model_dirs = [
        os.path.join(project_path, "runs", "detect", "quick_cold_drinks_detector2", "weights", "best.pt"),
        os.path.join(project_path, "runs", "detect", "cold_drinks_detector", "weights", "best.pt"),
        os.path.join(project_path, "runs", "detect", "quick_cold_drinks_detector", "weights", "best.pt")
    ]
    
    model_path = next((path for path in model_dirs if os.path.exists(path)), None)
    
    if not model_path:
        print("Error: No trained model found. Please train the model first.")
        return
    
    # Load class names from data.yaml
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data.get('names', [])
        print(f"Loaded {len(class_names)} class names from data.yaml")
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        class_names = []
    
    # Check if test images directory exists
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory '{test_images_dir}' not found.")
        return
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(test_images_dir, ext)))
        image_files.extend(glob(os.path.join(test_images_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in '{test_images_dir}'.")
        print("Please add some image files to test.")
        return
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Process each image
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"Processing: {img_name}")
        
        # Run detection
        results = model.predict(img_path, conf=0.25)
        result = results[0]
        
        # Get annotated image
        annotated_img = result.plot()
        
        # Save the result
        output_path = os.path.join(results_dir, f"detected_{img_name}")
        cv2.imwrite(output_path, annotated_img)
        
        # Print results
        print(f"  Found {len(result.boxes)} objects:")
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            confidence = box.conf.item()
            print(f"  - {class_name}: {confidence:.2f}")
    
    print("\nDetection completed!")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
