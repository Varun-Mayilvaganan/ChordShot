# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
from collections import Counter
import webcolors

# Function to find the closest color name from a given RGB value
def closest_color(rgb):
    """Find the closest CSS3 color name using Euclidean distance."""
    min_distance = float("inf")
    closest_name = None

    for color_name, hex_value in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        distance = np.sqrt((r_c - rgb[0]) ** 2 + (g_c - rgb[1]) ** 2 + (b_c - rgb[2]) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_name = color_name

    return closest_name

def get_color_name(rgb):
    """Try to get an exact match, otherwise find the closest color."""
    try:
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        return closest_color(rgb)

# Extract Two Dominant Colors
def extract_dominant_colors(image_path, k=4):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found!")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))

    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(image)

    # Find the two most common colors
    counter = Counter(kmeans.labels_)
    top_two_indices = [idx for idx, _ in counter.most_common(2)]
    top_two_colors = [kmeans.cluster_centers_[idx].astype(int) for idx in top_two_indices]

    # Convert to color names
    return [get_color_name(tuple(color)) for color in top_two_colors]

# Save Colors to JSON
def save_colors(image_path, json_file="image_features.json"):
    try:
        colors = extract_dominant_colors(image_path)

        # Load existing data or create a new dictionary
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        data["dominant_colors"] = colors

        # Save to JSON
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Dominant colors saved: {colors}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the Module
if __name__ == "__main__":
    image_path = "sample.jpg"  # Make sure this image exists
    save_colors(image_path)
