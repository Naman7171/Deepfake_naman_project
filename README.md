Deepfake Detection Project

Overview

This repository contains the implementation of a deepfake detection system that processes both images and videos. The project includes training, preprocessing, image prediction, and video prediction functionalities, utilizing advanced techniques to identify manipulated media.

Features

Video Prediction: Analyze videos to detect deepfake content.
Image Prediction: Evaluate individual images for authenticity.
Preprocessing: Includes data preprocessing routines to prepare images and videos for analysis.
Model Training: Train models using various architectures to enhance detection accuracy.
Project Structure
bash
Copy code

Deepfake_Detection/
├── README.md
├── image_detection.py            # Script for image prediction
├── video_detection.py            # Script for video prediction
├── preprocessing                 # Data preprocessing script
├── training                      # Script for model training
├── weights/                      # Directory containing trained model weights
├── IMAGE/                        # Directory containing test images
│   ├── FAKE/
│   └── REAL/
└── VIDEO/                        # Directory containing test videos
Requirements

To run this project, you need the following packages:

Python 3.x
PyTorch
OpenCV
Pandas
Albumentations
You can install the required packages using:

bash
Copy code
pip install -r requirements.txt
Usage
1. Image Prediction
To run image predictions, use the following command:

bash
Copy code
python image_detection.py --image-dir "path/to/image/directory" --weights-dir "path/to/weights" --models "model1" "model2" --output "predictions.csv"
2. Video Prediction
To run video predictions, use the following command:

bash
Copy code
python video_detection.py --video-dir "path/to/video/directory" --weights-dir "path/to/weights" --models "model1" "model2" --output "predictions.csv"
3. Training the Model
To train the model, execute:

bash
Copy code
python train.py --train-dir "path/to/training/data" --weights-dir "path/to/save/weights"
Contributing
Contributions are welcome! If you have suggestions for improvements or find bugs, feel free to create an issue or submit a pull request.

Acknowledgments
EfficientNet for the architecture used in the model.
OpenCV for image and video processing.
Albumentations for data augmentation and preprocessing.
