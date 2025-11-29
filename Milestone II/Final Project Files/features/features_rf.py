import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

def color_histogram(img, bins=(256, 256, 256)):
    """
    Compute a normalized 3-channel color histogram for the image.
    Returns a flattened 1D feature vector.
    """
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0,256, 0,256, 0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def glcm_texture(img):
    """
    Compute GLCM-based texture features (energy, homogeneity, correlation, contrast, dissimilarity)
    from a grayscale version of the image.
    Returns a tuple of 5 values.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    return energy, homogeneity, correlation, contrast, dissimilarity

def shape_features(img):
    """
    Compute simple shape-based features: circularity and aspect ratio,
    based on the largest contour in the grayscale image.
    Returns (circularity, aspect_ratio).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0, 0.0
    contour = contours[0]
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        circularity = 0.0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0.0
    return circularity, aspect_ratio

def extract_features(img):
    """
    Given a BGR image (as read by cv2), compute a feature vector:
    [color_histogram, GLCM texture features, shape features].
    """
    hist = color_histogram(img)
    energy, homogeneity, correlation, contrast, dissimilarity = glcm_texture(img)
    circularity, aspect_ratio = shape_features(img)
    return np.hstack([
        hist,
        energy,
        homogeneity,
        correlation,
        contrast,
        dissimilarity,
        circularity,
        aspect_ratio
    ])

def read_images_from_paths(paths, base_path=None):
    """
    Given a list of image file paths (or relative paths) and an optional base directory,
    read and return a list of images (as arrays). Uses cv2 to load images.
    Returns list of images.
    """
    imgs = []
    for p in tqdm(paths, total=len(paths), desc="Reading images"):
        fullpath = os.path.join(base_path, p) if base_path is not None else p
        img = cv2.imread(fullpath)
        imgs.append(img)
    return imgs

def store_features_labels(imgs, labels):
    """
    Given a list of images and corresponding labels, extract features for each image,
    filter out invalid images/features, and return (X, y):
      - X: 2D numpy array of shape (n_samples, n_features)
      - y: 1D numpy array of labels
    """
    features = []
    valid_labels = []
    for img, lab in tqdm(zip(imgs, labels), total=len(imgs), desc="Extracting features"):
        if img is None:
            continue
        feat = extract_features(img)
        if feat is None or np.any(np.isnan(feat)):
            continue
        features.append(feat)
        valid_labels.append(lab)
    if not features:
        return np.empty((0,0)), np.array([])
    return np.vstack(features), np.array(valid_labels)