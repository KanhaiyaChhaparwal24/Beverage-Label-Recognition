"""
Cold Drink Detector - Graphical User Interface
--------------------------------------------
This script provides a user-friendly graphical interface for detecting
cold drinks in images using a trained YOLOv8 model.

Features:
- Load custom images from your computer
- Load random images from the dataset
- Adjust detection confidence threshold with a slider
- View detection results with class names, confidence scores and locations
- Display annotated images with bounding boxes around detected drinks

Usage: Run this script after training the model with either train_model.py
       or train_model_quick.py.
"""

import os
import sys
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

class ColdDrinkDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cold Drink Detector")
        self.root.geometry("900x600")
        self.root.minsize(800, 500)
          
        # Set up model and dataset paths
        self.dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "cold drinks.v1i.yolov8")
          # Try different trained model paths in order of preference
        model_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "runs", "detect", "quick_cold_drinks_detector2", "weights", "best.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "runs", "detect", "cold_drinks_detector", "weights", "best.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "runs", "detect", "quick_cold_drinks_detector", "weights", "best.pt")
        ]
        
        # Find first available trained model
        self.model_path = next((path for path in model_paths if os.path.exists(path)), None)
        
        # Check if a trained model exists
        if self.model_path is None:
            messagebox.showerror("Error", "No trained model found. Please train the model first using train_model.py or train_model_quick.py")
            sys.exit(1)
            
        print(f"Using trained model: {self.model_path}")
        
        # Load class names from data.yaml
        self.class_names = self.load_class_names()
        
        # Load model
        self.model = None  # Will be loaded when needed
        
        # Create UI elements
        self.create_ui()
        
        # Track the current image
        self.current_image_path = None
        self.current_image = None
        self.detection_results = None
    
    def load_class_names(self):
        """Load class names from data.yaml"""
        data_yaml_path = os.path.join(self.dataset_path, "data.yaml")
        try:
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('names', [])
        except Exception as e:
            print(f"Error loading class names: {str(e)}")
            # Provide default class names based on the dataset
            return ['7up bottel', '7up can', 'Red Bull bottel', 'Red Bull can', 
                    'clemon bottel', 'clemon can', 'coca cola bottel', 'coca cola can', 
                    'coke bottel', 'coke can', 'fanta bottel', 'fanta can', 
                    'mirinda bottel', 'mirinda can', 'mojo bottel', 'mojo can', 
                    'mountain dew bottel', 'mountain dew bottle', 'mountain dew can', 
                    'pepsi bottel', 'pepsi can', 'speed bottel', 'speed can', 
                    'sprite bottel', 'sprite can', 'tango bottel', 'tango can']
    
    def create_ui(self):
        """Create the UI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X, side=tk.TOP, pady=5)
        
        # Buttons for loading images
        ttk.Button(control_frame, text="Load Image", command=self.load_custom_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Random Dataset Image", command=self.load_random_dataset_image).pack(side=tk.LEFT, padx=5)
          # Confidence threshold slider
        ttk.Label(control_frame, text="Confidence:").pack(side=tk.LEFT, padx=(20, 5))
        self.conf_var = tk.DoubleVar(value=0.25)
        conf_slider = ttk.Scale(control_frame, from_=0.1, to=0.9, variable=self.conf_var, 
                               orient=tk.HORIZONTAL, length=150)
        conf_slider.pack(side=tk.LEFT, padx=5)
        self.conf_label = ttk.Label(control_frame, text=f"{0.25:.2f}")
        self.conf_label.pack(side=tk.LEFT)
        
        # Update confidence label when slider moves
        def update_conf_label(*args):
            self.conf_label.config(text=f"{self.conf_var.get():.2f}")
        self.conf_var.trace("w", update_conf_label)
        
        # Detect button
        self.detect_btn = ttk.Button(control_frame, text="Detect Cold Drinks", command=self.detect_drinks, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=20)
        
        # Split view for image and results
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Image frame (left side)
        self.image_frame = ttk.Frame(paned_window, padding="5")
        paned_window.add(self.image_frame, weight=2)
        
        # Create a canvas for displaying the image
        self.canvas = tk.Canvas(self.image_frame, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results frame (right side)
        results_frame = ttk.Frame(paned_window, padding="5")
        paned_window.add(results_frame, weight=1)
        
        # Results area
        ttk.Label(results_frame, text="Detection Results:").pack(anchor=tk.W, pady=(0, 5))
        
        # Treeview for displaying detection results
        columns = ("Class", "Confidence", "Location")
        self.result_tree = ttk.Treeview(results_frame, columns=columns, show="headings")
          # Define column headings with optimized widths for smaller display
        column_widths = {
            "Class": 120,
            "Confidence": 70,
            "Location": 100
        }
        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=column_widths[col])
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack elements
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Please load an image.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model(self):
        """Load the YOLO model if it's not already loaded"""
        if self.model is None:
            try:
                self.status_var.set("Loading YOLOv8 model...")
                self.root.update_idletasks()
                self.model = YOLO(self.model_path)
                self.status_var.set("Model loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.status_var.set("Error loading model.")
                return False
        return True
    
    def load_custom_image(self):
        """Load a custom image from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_random_dataset_image(self):
        """Load a random image from the dataset"""
        # Look for images in train and valid directories
        image_paths = []
        for subset in ["train", "valid"]:
            img_dir = os.path.join(self.dataset_path, subset, "images")
            if os.path.exists(img_dir):
                for img_file in os.listdir(img_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_paths.append(os.path.join(img_dir, img_file))
        
        if image_paths:
            # Select a random image
            random_img = random.choice(image_paths)
            self.load_image(random_img)
        else:
            messagebox.showinfo("Notice", "No images found in the dataset.")
    
    def load_image(self, image_path):
        """Load and display an image"""
        try:
            # Read and display the image
            self.current_image_path = image_path
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Failed to read image")
            
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image(image_rgb)
            
            self.status_var.set(f"Loaded image: {os.path.basename(image_path)}")
            self.detect_btn.config(state=tk.NORMAL)
            
            # Clear previous results
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image.")

    def display_image(self, img_array):
        """Display an image on the canvas"""
        self.canvas.delete("all")
        
        # Resize image to fit the canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Ensure the canvas has been drawn
        if canvas_width <= 1:
            canvas_width = self.image_frame.winfo_width() or 500  # Provide default if not yet drawn
            canvas_height = self.image_frame.winfo_height() or 400  # Provide default if not yet drawn
        
        img_height, img_width = img_array.shape[:2]
        
        # Calculate scale to fit image in canvas
        width_scale = canvas_width / img_width
        height_scale = canvas_height / img_height
        scale = min(width_scale, height_scale)
        
        # Resize image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_img = cv2.resize(img_array, (new_width, new_height))
        
        # Convert to PhotoImage
        img = Image.fromarray(resized_img)
        self.photo_image = ImageTk.PhotoImage(img)
        
        # Display on canvas
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo_image, anchor=tk.CENTER
        )
    
    def detect_drinks(self):
        """Run detection on the current image"""
        if self.current_image is None:
            messagebox.showinfo("Notice", "Please load an image first.")
            return
        
        # Load model if not loaded
        if not self.load_model():
            return
        
        try:
            # Update status
            self.status_var.set("Detecting cold drinks...")
            self.root.update_idletasks()
            
            # Get confidence threshold
            conf_threshold = self.conf_var.get()
            
            # Run inference
            results = self.model.predict(self.current_image, conf=conf_threshold)[0]
            
            # Convert back to RGB for display
            annotated_img = results.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Display the annotated image
            self.display_image(annotated_img_rgb)
            
            # Update results in the treeview
            self.update_results_view(results)
            
            # Update status
            detection_count = len(results.boxes)
            self.status_var.set(f"Detection complete. Found {detection_count} object(s).")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.status_var.set("Error during detection.")
    
    def update_results_view(self, results):
        """Update the results treeview with detection info"""
        # Clear previous results
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Get detection data
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes.xyxy is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []
        cls_ids = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else []
        
        # Add results to treeview
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            cls_name = results.names[int(cls_id)] if int(cls_id) < len(results.names) else f"Class {int(cls_id)}"
            conf_str = f"{conf:.2f}"
            box_str = f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
            
            # Insert into treeview
            self.result_tree.insert("", tk.END, values=(cls_name, conf_str, box_str))

def main():
    # Create the tkinter application
    root = tk.Tk()
    app = ColdDrinkDetectorUI(root)
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')  # Use a more modern theme
    
    # Set window icon if available
    try:
        # Attempt to set an icon
        pass
    except:
        pass
        
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()
