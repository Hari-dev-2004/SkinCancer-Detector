# image_matching.py
import cv2
import numpy as np
import os
import pickle

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

def compute_feature(image):
    feature = image.flatten()
    return feature

def cache_dataset_features(dataset_folder, cache_file='dataset_features.pkl'):
    features = {}
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                image = preprocess_image(image_path)
                feature = compute_feature(image)
                label = os.path.basename(os.path.dirname(image_path))  # The folder name is the label
                features[image_path] = {'feature': feature, 'label': label}

    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)

def load_dataset_features(cache_file='dataset_features.pkl'):
    with open(cache_file, 'rb') as f:
        features = pickle.load(f)
    return features

def compare_features(feature1, feature2):
    distance = np.linalg.norm(feature1 - feature2)
    return distance

def find_most_similar(image_path, dataset_features, top_n=5):
    uploaded_image = preprocess_image(image_path)
    uploaded_feature = compute_feature(uploaded_image)
    
    scores = []
    for image_path, data in dataset_features.items():
        feature = data['feature']
        label = data['label']
        score = compare_features(uploaded_feature, feature)
        scores.append((label, score))
    
    scores.sort(key=lambda x: x[1])
    
    return scores[:top_n]

def process_image(image_path, dataset_features):
    similar_images = find_most_similar(image_path, dataset_features)
    
    total_score = sum(score for _, score in similar_images)
    results = [[label, score, (total_score - score) / total_score * 100] for label, score in similar_images]
    
    if results[0][2]<=95:
        results[0][2]=None
        
    return results
