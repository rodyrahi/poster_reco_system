import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Load the pre-trained model for feature extraction
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(model_url, output_shape=[1280], trainable=False)

# Function to preprocess the image for the model
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

# Function to extract features from an image URL
def extract_features(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Check if the request is successful
        image = Image.open(BytesIO(response.content)).convert("RGB")
        preprocessed_image = preprocess_image(image)
        features = feature_extractor(preprocessed_image)
        return features.numpy().flatten()
    except Exception as e:
        print(f"Error extracting features from {image_url}: {e}")
        return None

# Load the existing CSV with features
csv_path = "extracted_features.csv"  # Path to the CSV with existing features
data = pd.read_csv(csv_path)

# Prepare the existing features for clustering
features = data.drop("image_url", axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train a K-means model (if you haven't already)
kmeans = KMeans(n_clusters=4, random_state=42)  # Using 5 clusters as an example
kmeans.fit(scaled_features)

# Perform PCA for visualization
pca = PCA(n_components=2)  # Reduce to 2D for plotting
pca_features = pca.fit_transform(scaled_features)

# Create a DataFrame with PCA components and cluster information
pca_df = pd.DataFrame(pca_features, columns=["PCA1", "PCA2"])
pca_df["cluster"] = kmeans.labels_

# Now let's add a new image to the plot
new_image_url = "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/corporate-poster-design-template-917be18af3ade307065a46e2200ae9bd_screen.jpg?ts=1637041777"  # Change to a valid image URL
new_features = extract_features(new_image_url)

if new_features is not None:
    # Standardize the new image's features
    new_features_scaled = scaler.transform([new_features])

    # Predict the cluster for the new image
    new_image_cluster = kmeans.predict(new_features_scaled)[0]

    # Transform new image's features with PCA
    new_image_pca = pca.transform(new_features_scaled)

    # Add the new image to the plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue="cluster",
        palette=sns.color_palette("hsv", 5),
        alpha=0.6,
        legend="full"
    )

    # Plot the new image's position and label its cluster
    plt.scatter(new_image_pca[0][0], new_image_pca[0][1], color="red", s=100, label="New Image")
    plt.title("K-means Clustering with PCA - New Image")
    plt.legend()
    plt.show()

    print("The new image belongs to cluster:", new_image_cluster)

 

# Compute the distances between the new image features and all other images
    distances = cdist(new_features_scaled, scaled_features, metric='euclidean')

    # Find the index of the image with the smallest distance
    nearest_index = np.argmin(distances)

    # Retrieve the URL of the nearest image
    nearest_image_url = data.loc[nearest_index, "image_url"]

    print("Nearest image URL:", nearest_image_url)
else:
    print("Error extracting features from the new image.")



