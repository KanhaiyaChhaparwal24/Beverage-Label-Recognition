# BeverageVision Insights

A YOLOv8-based application for detecting various cold drinks (bottles and cans) in images using deep learning.

## Project Overview

This project uses YOLOv8 (You Only Look Once version 8) object detection to identify and classify different types of cold drinks in images. It implements a complete detection pipeline from model training to visual results presentation.

### Purpose

The Cold Drink Detector was developed to demonstrate practical applications of deep learning in product recognition. It can be used in various scenarios:
- Retail inventory management
- Automated checkout systems
- Consumer behavior analysis
- Marketing research for beverage companies

### Key Features

The application features a simple but powerful graphical user interface that allows you to:
- Load images from your computer or directly from the dataset
- Detect cold drinks with adjustable confidence threshold
- View detection results with class name, confidence score and precise location
- Analyze various beverage types and brands in a single image

## Dataset

The project uses a comprehensive dataset of cold drink images that captures the diversity in appearance, packaging, and brand designs.

### Dataset Source
The dataset is available on Roboflow Universe: [Cold Drinks Dataset](https://universe.roboflow.com/cold-drinks-cftkw/cold-drinks-qki32/dataset/1)

### Dataset Specifications
- **27 different classes** of cold drinks
- Contains both **bottles and cans** for major beverage brands
- Includes images with **varying angles, lighting conditions, and backgrounds**
- Organized in the **YOLOv8 format** with train and validation splits
- Images are preprocessed and augmented to improve model generalization

### Cold Drink Classes
- 7up (bottle and can)
- Red Bull (bottle and can)
- Coca Cola (bottle and can)
- Fanta (bottle and can)
- Pepsi (bottle and can)
- Sprite (bottle and can)
- Mirinda (bottle and can)
- Mountain Dew (bottle and can)
- Clemon (bottle and can)
- Mojo (bottle and can)
- Tango (bottle and can)
- Speed (bottle and can)
- And others

The dataset contains sufficient examples of each class to enable effective training and accurate detection.

## Project Structure

```
cold_drink_project/
├── cold drinks.v1i.yolov8/       # Dataset folder
│   ├── data.yaml                 # Dataset configuration
│   ├── train/                    # Training images and labels
│   └── valid/                    # Validation images and labels
├── cold_drink_detector.py        # Main detection application with UI
├── run_ui.py                     # Python script to directly launch the UI
├── train_model.py                # Full model training script (30 epochs)
├── train_model_quick.py          # Quick training script (5 epochs)
├── evaluate_model.py             # Model evaluation script
├── test_custom_images.py         # Script to test model on custom images
├── test_images/                  # Folder to place custom test images
├── requirements.txt              # Required packages
├── start_ui.bat                  # Script to launch the UI directly
├── train_model.bat               # Script to run full training
├── train_model_quick.bat         # Script to run quick training
├── evaluate_model.bat            # Script to evaluate model performance
├── test_custom_images.bat        # Script to test on custom images
└── README.md                     # This file
```

## Setup and Installation

1. Make sure you have Conda installed
2. Activate the TF environment:
   ```
   conda activate tf_env
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## How to Use

The easiest way to use this project is with the provided scripts:

```
run_ui.py            # Python script to directly launch the UI
start_ui.bat         # Batch file to directly start the UI
train_model.bat      # For full training (30 epochs)
train_model_quick.bat # For quick training (5 epochs)
evaluate_model.bat   # To evaluate model performance on validation data
test_custom_images.bat # To test the model on your own images
```

For normal usage, simply run either:
- `python run_ui.py` (Python script) 
- `start_ui.bat` (Windows batch file)

If you want to train a new model, you have two options:
1. **Full training** - `train_model.bat` (30 epochs, better accuracy)
2. **Quick training** - `train_model_quick.bat` (5 epochs, faster)

### Running the Detector

The detector UI allows you to:

1. Load custom images or random images from the dataset
2. Adjust the confidence threshold using the slider
3. Run detection and see results
4. View the detected objects with their class names, confidence scores, and locations

## Model Training

Two training options are provided:

1. **Full Training** (train_model.py):
   - 30 epochs
   - Standard 640px image size
   - Better accuracy but takes longer
   ```
   python train_model.py
   ```

2. **Quick Training** (train_model_quick.py):
   - 5 epochs only
   - Smaller 416px image size
   - Faster training but possibly lower accuracy
   ```
   python train_model_quick.py
   ```

The trained model will be saved in the `runs/detect/` directory.

## Technical Details

### Deep Learning Framework
- **YOLOv8** (You Only Look Once version 8) - state-of-the-art object detection model
- Uses the **nano variant** (yolov8n.pt) for optimal balance of speed and accuracy
- Leverages transfer learning from pretrained weights for faster convergence
- Training uses **PyTorch** backend via the Ultralytics implementation

### Model Architecture
- **Backbone**: CSPDarknet-based feature extractor
- **Neck**: PANet for feature aggregation across different scales
- **Head**: Specialized layers for object detection and classification

### Training Optimizations
- **Early stopping** to prevent overfitting and save training time
- **Mixed precision training** for better GPU utilization
- **Learning rate scheduling** to improve convergence
- Option for **smaller image size** (416px) in quick training mode
- **Data caching** to speed up training iterations

### User Interface
- Built with **Tkinter** for cross-platform compatibility
- **Responsive design** that adapts to window resizing
- **Split-pane interface** with image view and detection results
- **Adjustable confidence threshold** via slider control
- **Detailed results view** with class, confidence, and bounding box information

## Usage Examples

### Basic Workflow

```
# Activate your conda environment
conda activate tf_env

# Install requirements
pip install -r requirements.txt

# Directly run the UI (if model is already trained)
python run_ui.py

# Or use the batch file
start_ui.bat
```

### Training Only

If you want to train the model separately:

```
# For full training (better accuracy)
python train_model.py

# For quick training (faster results)
python train_model_quick.py
```

### Running the Detector

To run the detector after training has been completed:

```
python cold_drink_detector.py
```

Note: You must train the model before running the detector. The application will not run with a pretrained model.

## Model Evaluation and Testing

Once you've trained your model, you can evaluate its performance and test it on new images using the provided tools:

### Model Evaluation

The `evaluate_model.py` script performs a detailed evaluation of your model's performance on the validation dataset:

```
evaluate_model.bat
```

This will:
- Run the model on the validation dataset
- Display detailed metrics for each class
- Generate visualizations of detections on sample validation images
- Save the results in the `evaluation_results` directory

The evaluation provides insights into how well your model is performing on each class of cold drinks, helping you identify which categories might need more training data or attention.

### Testing on Custom Images

To test the model on your own images:

1. Place your test images in the `test_images` folder
2. Run the test script:
   ```
   test_custom_images.bat
   ```
3. Check the results in the `test_results` folder

This allows you to see how well your model performs on real-world images outside the training dataset.

### Direct UI Launch

If you've already trained a model and just want to run the detector UI:

```
start_ui.bat
```

This launches the graphical interface directly, allowing you to:
- Upload your own images
- Load random dataset images
- Adjust the detection confidence threshold
- View detailed detection results

## Future Enhancements

The Cold Drink Detector project has significant potential for expansion and improvement. Here are some planned enhancements for future versions:

### Real-time Detection
- Implement webcam support for real-time detection of cold drinks
- Add video file processing capability for analyzing recorded content
- Create frame-by-frame analysis with tracking of detected items

### Model Improvements
- Fine-tune the model on challenging cases like partial occlusion
- Expand the dataset with more diverse lighting conditions and backgrounds
- Experiment with YOLOv8-medium or YOLOv8-large models for improved accuracy
- Implement model quantization for faster inference without significant accuracy loss

### User Interface Enhancements
- Add dark mode and customizable themes
- Implement batch processing with progress indicators
- Create detection history to review past results
- Add export functionality for detection results (CSV, JSON)
- Develop visualization tools for detection statistics

### Integration Capabilities
- Create a REST API for remote access to detection features
- Develop mobile app integration for Android/iOS
- Build plugins for popular inventory management systems
- Implement cloud storage integration for saving results

### Deployment Options
- Create containerized version with Docker for easy deployment
- Add web interface option for browser-based detection
- Develop installation packages for Windows, macOS and Linux
- Create a lightweight version optimized for edge devices

### Additional Features
- Multi-language support for international users
- Barcode/QR code integration for product identification
- Price lookup capabilities based on detected products
- Inventory counting features for retail applications
- Customer behavior analysis with detection aggregation

## Contributors

- **Kanhaiya Chhaparwal** - Developer

## Help me Improve

<p> Hello readers, if you find any bugs, please consider raising issue so that I can address them asap and connect with me on
<a href="mailto:kanhaiyaac24@gmail.com">Email</a> or
<a href="https://www.linkedin.com/in/kanhaiya-chhaparwal/">Linkedin</a>
