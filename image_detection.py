import os
import re
import torch
import cv2
import pandas as pd
from albumentations import Compose, Resize, Normalize
from training.zoo.classifiers import DeepFakeClassifier
import argparse


# Load multiple trained models
def load_models(model_checkpoint_paths):
    models = []
    for model_checkpoint_path in model_checkpoint_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        print(f"Loading state dict from {model_checkpoint_path}")

        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)

        model.eval()  # Set to evaluation mode
        models.append(model)
    return models


# Preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # Define desired dimensions for resizing
    desired_height = 224  # Set this to your preferred height
    desired_width = 224  # Set this to your preferred width

    # Define normalization parameters
    mean = [0.485, 0.456, 0.406]  # Mean for normalization
    std = [0.229, 0.224, 0.225]  # Standard deviation for normalization

    # Create the transformation
    transform = Compose([
        Resize(height=desired_height, width=desired_width),
        Normalize(mean=mean, std=std)
    ])

    transformed = transform(image=image)['image']
    transformed = torch.tensor(transformed).permute(2, 0, 1).float().unsqueeze(0).cuda()  # Shape: (1, C, H, W)
    return transformed


# Make predictions with multiple models
def predict_deepfake(models, image_tensor):
    all_probabilities = []
    with torch.no_grad():
        for model in models:
            output = model(image_tensor)
            probabilities = torch.sigmoid(output)  # Apply sigmoid for binary classification
            all_probabilities.append(probabilities.cpu().numpy())

    # Average probabilities from all models
    average_probabilities = sum(all_probabilities) / len(all_probabilities)
    return average_probabilities


# Full detection function that processes images (single or directory)
def detect_deepfake_images(image_path_or_dir, model_checkpoint_paths, output_csv):
    models = load_models(model_checkpoint_paths)

    predictions = []

    if os.path.isdir(image_path_or_dir):
        # If the input is a directory, process all images
        image_filenames = sorted([x for x in os.listdir(image_path_or_dir) if x.lower().endswith(('.jpg', '.png'))])
        image_paths = [os.path.join(image_path_or_dir, img) for img in image_filenames]
    else:
        # If the input is a single file
        image_filenames = [os.path.basename(image_path_or_dir)]
        image_paths = [image_path_or_dir]

    for image_path, image_filename in zip(image_paths, image_filenames):
        image_tensor = preprocess_image(image_path)
        probabilities = predict_deepfake(models, image_tensor)

        # Assuming a binary classification (0: real, 1: deepfake)
        label = "Deepfake" if probabilities[0][0] > 0.5 else "real"
        prediction = probabilities[0][0]

        predictions.append({
            "filename": image_filename,
            "label": label,
            "prediction": prediction
        })

    # Save results to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFake Detection on Images")
    parser.add_argument('--image-dir', type=str, required=True, help="Path to an image file or a directory with images")
    parser.add_argument('--weights-dir', type=str, default="weights", help="Path to directory with model weights")
    parser.add_argument('--models', nargs='+', required=True, help="List of model weight filenames")
    parser.add_argument('--output', type=str, required=True, help="Path to output CSV file")

    args = parser.parse_args()

    # Get full paths to the models
    model_checkpoint_paths = [os.path.join(args.weights_dir, model) for model in args.models]

    # Run the deepfake detection
    detect_deepfake_images(args.image_dir, model_checkpoint_paths, args.output)
