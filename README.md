# Real-Time Facial Emotion Recognition

This project implements a real-time facial emotion recognition system using the `ResEmoteNet` model. It detects faces in a webcam feed, extracts face regions, and predicts emotions using a pre-trained PyTorch model. The system is built with Flask for the backend, providing a web interface for users to access the demo locally or externally via ngrok.

## Features

- Real-time face detection using OpenCV's Haar Cascade classifier.
- Emotion prediction using the `ResEmoteNet` model (trained on FER2013 dataset, accuracy ~79.79%).
- Supports multiple faces in a single frame.
- Displays processed video feed with bounding boxes and emotion labels (e.g., `happy`, `sad`, `angry`).
- GPU acceleration for faster inference (e.g., using NVIDIA RTX 3070).
- External access via ngrok for remote users.

## Directory Structure

Below is the structure of the project directory:

```
Recognition/
├── dataset/ # Directory for FER2013 dataset
│ ├── test/ # Test dataset
│ └── train/ # Training dataset
├── models/ # Directory for pre-trained model weights
│ ├── best_mobilenet_model.keras
│ ├── best_fer2013_ResEmoteNet.pth # Training ResEmoteNet model weights
│ ├── best_with_cnn_model1.keras
│ ├── best_with_cnn_model2_SGD_remove_augmentation.keras
│ ├── best_with_cnn_model2_SGD.keras
│ ├── best_with_cnn_model2.keras
│ ├── best_with_cnn_model3_SGD.keras
│ ├── best_with_cnn_model3.keras
│ ├── CNN_basic_model.keras
│ ├── CNN_model_augmentation_normalization_SGD.keras
│ └── CNN_model_with_batch_normalization.keras
├── ResEmoteNet/ # Directory for ResEmoteNet model implementation
│ └── approach/ # Subdirectory for model code
│ └── ResEmoteNet.py # ResEmoteNet model definition
├── templates/ # Directory for HTML templates
│ └── index.html # Frontend HTML file for webcam display
├── .gitignore # Git ignore file
├── app.py # Main Flask application script
├── facial_expression_pipeline.ipynb # Jupyter notebook for pipeline development
├── facial_recognition_MobileNetV2.ipynb # Notebook for MobileNetV2 experiments
├── facial_recognition_ResEmoteNet.ipynb # Notebook for ResEmoteNet experiments
├── facial_recognition_training.ipynb # Notebook for training experiments
├── facial_recognition_transformer.ipynb # Notebook for transformer experiments
├── README.md # Project documentation
├── temp.jpg # Temporary image file
└── test.py # Script for testing model loading
```
