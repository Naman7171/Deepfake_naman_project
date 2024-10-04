# Deepfake Detection Project

## Overview

This repository contains the implementation of a deepfake detection system that processes both images and videos. The project includes training, preprocessing, image prediction, and video prediction functionalities, utilizing advanced techniques to identify manipulated media.

## Features

- **Video Prediction**: Analyze videos to detect deepfake content.
- **Image Prediction**: Evaluate individual images for authenticity.
- **Preprocessing**: Includes data preprocessing routines to prepare images and videos for analysis.
- **Model Training**: Train models using various architectures to enhance detection accuracy.

## Project Structure

Deepfake_Detection/ ├── README.md ├── image_detection.py # Script for image prediction ├── video_detection.py # Script for video prediction ├── preprocess.py # Data preprocessing script ├── train.py # Script for model training ├── weights/ # Directory containing trained model weights ├── IMAGE/ # Directory containing test images │ ├── FAKE/ │ └── REAL/ └── VIDEO/ # Directory containing test videos


## Requirements

To run this project, you need the following packages:

- Python 3.x
- PyTorch
- OpenCV
- Pandas
- Albumentations

You can install the required packages using:

```bash
pip install -r requirements.txt
## Usage

### 1. Image Prediction
To run image predictions, use the following command:

```bash
python image_detection.py --image-dir "path/to/image/directory" --weights-dir "path/to/weights" --models "model1" "model2" --output "predictions.csv"

###2. Video Prediction
To run video predictions, use the following command:
python video_detection.py --video-dir "path/to/video/directory" --weights-dir "path/to/weights" --models "model1" "model2" --output "predictions.csv"

###3. Training the Model
To train the model, execute:
python train.py --train-dir "path/to/training/data" --weights-dir "path/to/save/weights"

###Contributing
Contributions are welcome! If you have suggestions for improvements or find bugs, feel free to create an issue or submit a pull request.

###Acknowledgments
EfficientNet for the architecture used in the model.
OpenCV for image and video processing.
Albumentations for data augmentation and preprocessing.


### Instructions for Usage
1. **Copy the Markdown content** provided above.
2. **Open your `README.md` file** in your preferred text editor.
3. **Paste the content** where you want it to appear in the file.
4. **Save the changes**.

After you save the `README.md` file, it should reflect the formatting correctly when viewed on GitHub. If you need any further assistance, feel free to ask!

