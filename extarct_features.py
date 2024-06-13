import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load a pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(model_url, output_shape=[1280], trainable=False)

# Function to preprocess an image for the model
def preprocess_image(image, target_size=(224, 224)):
    # Resize and normalize the image
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = image.astype(np.float32)
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Function to extract features from an image URL
def extract_features(image_url):
    try:
        # Download the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Ensure the request was successful
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Preprocess and extract features
        preprocessed_image = preprocess_image(image)
        features = feature_extractor(preprocessed_image)
        return features.numpy().flatten()  # Flatten the features into a 1D array
    except Exception as e:
        # Handle exceptions (e.g., timeout, invalid URL, etc.)
        print(f"Error extracting features from {image_url}: {e}")
        return None

# Read CSV with URLs (assuming the CSV has a column named 'image_url')
csv_path = "images.csv"  # Path to your CSV file
data = pd.read_csv(csv_path)

# Initialize a list to store the features for each image
feature_list = []

# Iterate over the rows in the CSV
for idx, row in data.iterrows():
    image_url = row["Image URLs"]
    features = extract_features(image_url)
    if features is not None:
        # Create a dictionary with the extracted features
        feature_dict = {"image_url": image_url}
        for i in range(len(features)):
            feature_dict[f"feature_{i}"] = features[i]
        feature_list.append(feature_dict)

# Create a DataFrame with the extracted features
features_df = pd.DataFrame(feature_list)

# Save to CSV
output_csv_path = "extracted_features.csv"
features_df.to_csv(output_csv_path, index=False)

print("Feature extraction completed and saved to:", output_csv_path)
