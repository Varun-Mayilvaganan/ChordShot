import numpy as np
import pandas as pd
import joblib
from skimage.feature import daisy, hog
from skimage import io
from skimage.color import rgb2gray
import skimage
import warnings
warnings.filterwarnings("ignore")

# Load models once
hybrid_classifier = joblib.load(r"I:\ChordShot\models\hybrid_classifier.pkl")
daisy_cluster_model = joblib.load(r"I:\ChordShot\models\daisy_cluster_model.pkl")

# Define category mapping
category_mapping = {
    0: "bedroom", 1: "CALsuburb", 2: "industrial", 3: "kitchen", 4: "livingroom",
    5: "MITcoast", 6: "MITforest", 7: "MIThighway", 8: "MITinsidecity",
    9: "MITmountain", 10: "MITopencountry", 11: "MITstreet", 12: "MITtallbuilding",
    13: "PARoffice", 14: "store"
}

def extract_daisy_and_hog_features_from_image(file_path, daisy_step_size=32, daisy_radius=32, hog_pixels_per_cell=16, hog_cells_per_block=1):
    """Extracts DAISY and HOG features from an image."""
    img = io.imread(file_path)

    # Convert to grayscale if it's an RGB image
    if len(img.shape) == 3:
        img = rgb2gray(img)

    # Resize image
    img = skimage.transform.resize(img, (300, 250))

    # Extract DAISY features
    descs = daisy(img, step=daisy_step_size, radius=daisy_radius, rings=2, histograms=8, orientations=8)
    
    # Reshape DAISY descriptors
    descs_num = descs.shape[0] * descs.shape[1]
    daisy_descriptors = descs.reshape(descs_num, descs.shape[2])

    # Extract HOG features
    hog_descriptor = hog(img, orientations=8, pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell),
                         cells_per_block=(hog_cells_per_block, hog_cells_per_block), feature_vector=True)

    return daisy_descriptors, hog_descriptor

def extract_daisy_hog_hybrid_feature_from_image(fname):
    """Extracts hybrid features from an image and returns a combined feature vector."""
    daisy_features, hog_feature = extract_daisy_and_hog_features_from_image(fname, daisy_step_size=8, daisy_radius=8)

    # Assign each DAISY feature to a cluster
    img_clusters = daisy_cluster_model.predict(daisy_features)
    cluster_freq_counts = pd.DataFrame(img_clusters, columns=['cnt'])['cnt'].value_counts()

    # Create BoVW vector
    bovw_vector = np.zeros(daisy_cluster_model.n_clusters)
    for key in cluster_freq_counts.keys():
        bovw_vector[key] = cluster_freq_counts[key]

    # Normalize features
    bovw_feature = bovw_vector / np.linalg.norm(bovw_vector)
    hog_feature = hog_feature / np.linalg.norm(hog_feature)

    return list(bovw_feature) + list(hog_feature)

def predict_image_category(image_path):
    """Predicts the category of an input image."""
    hybrid_feature_vector = extract_daisy_hog_hybrid_feature_from_image(image_path)
    predicted_index = hybrid_classifier.predict([hybrid_feature_vector])[0]
    return category_mapping[predicted_index]
