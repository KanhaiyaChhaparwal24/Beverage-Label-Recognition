"""
Direct launcher for Cold Drink Detector
--------------------------------------
A simplified script to directly launch the Cold Drink Detector UI
without the complexities of batch files.
"""

import os
import sys
import subprocess

def main():
    print("Cold Drink Detector - Direct Launcher")
    print("------------------------------------")
    print()
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if model exists
    model_paths = [
        os.path.join(script_dir, "runs", "detect", "quick_cold_drinks_detector2", "weights", "best.pt"),
        os.path.join(script_dir, "runs", "detect", "cold_drinks_detector2", "weights", "best.pt"),
        os.path.join(script_dir, "runs", "detect", "quick_cold_drinks_detector", "weights", "best.pt"),
        os.path.join(script_dir, "runs", "detect", "cold_drinks_detector", "weights", "best.pt")
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            model_found = True
            break
    
    if not model_found:
        print("ERROR: No trained model found!")
        print("Please run train_model.py or train_model_quick.py first.")
        input("Press Enter to exit...")
        return
    
    # Run the detector
    print("Starting Cold Drink Detector UI...")
    try:
        detector_path = os.path.join(script_dir, "cold_drink_detector.py")
        subprocess.run([sys.executable, detector_path], check=True)
    except Exception as e:
        print(f"Error running detector: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
