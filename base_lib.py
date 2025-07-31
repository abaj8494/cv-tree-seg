import copy
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import re
from scipy.stats import randint
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
import time

IMAGE_SIZE = (128, 128)


dir_path = Path(f'../../USA_segmentation/') # Where the USA Segmentation folder is relative

masks_dir = dir_path / 'masks'
rgb_dir = dir_path / 'RGB_images'
nrg_dir = dir_path / 'NRG_images'

mask_prefix = str(masks_dir) + '/mask_'
rgb_prefix = str(rgb_dir) + '/RGB_'
nrg_prefix = str(nrg_dir) + '/NRG_'

filenames = [re.search(r'(?:^[^_]*_)(.*)', re.search(r'[^\/]*$', str(f)).group()).group(1) for f in list(masks_dir.iterdir())]

def plot_input_file(base_filename: str):
    rgb_filename = rgb_prefix + base_filename
    nrg_filename = nrg_prefix + base_filename
    mask_filename = mask_prefix + base_filename

    rgb_image = cv2.imread(rgb_filename)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    nrg_image = cv2.imread(nrg_filename)
    nrg_image = cv2.cvtColor(nrg_image, cv2.COLOR_BGR2RGB)

    mask_image = cv2.imread(mask_filename)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    fig.suptitle(base_filename)

    axes[0].imshow(rgb_image)
    axes[0].axis('off')
    axes[0].set_title('RGB')

    axes[1].imshow(nrg_image)
    axes[1].axis('off')
    axes[1].set_title('NRG')

    axes[2].set_title('mask')
    axes[2].imshow(mask_image, cmap='gray')
    axes[2].axis('off')

    # fig.tight_layout()
    fig.show()

"""
Given a base_filename, read in the RGB/NRG images and concantenate them into a 6-channel flattened image with the mask data (ground truth) as a separate vector
"""
def extract_basic_image_data(base_filename: str, image_size=IMAGE_SIZE):
    rgb_filename = rgb_prefix + base_filename
    nrg_filename = nrg_prefix + base_filename
    mask_filename = mask_prefix + base_filename

    mask = cv2.resize(cv2.imread(mask_filename, flags=cv2.IMREAD_GRAYSCALE), image_size)
    mask = mask.reshape(-1)
    y = (mask > 127).astype(np.uint8)
 
    rgb_image = cv2.cvtColor(cv2.resize(cv2.imread(rgb_filename), image_size), cv2.COLOR_BGR2RGB)
    nrg_image = cv2.cvtColor(cv2.resize(cv2.imread(nrg_filename), image_size), cv2.COLOR_BGR2RGB)

    rgb_image_lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    nrg_image_lab = cv2.cvtColor(nrg_image, cv2.COLOR_RGB2LAB)
    
    image = np.concatenate((rgb_image, nrg_image), axis=2)
    # image = np.concatenate((rgb_image_lab, nrg_image_lab), axis=2)
    X = image.reshape(-1, 6)

    return X, y
    
"""
Returns the features/labels given input base filenames
"""
def extract_multiple_images(train_files: list[str], image_size=IMAGE_SIZE):
    X = []
    y = []
    for base_filename in train_files:
        X_a, y_a = extract_basic_image_data(base_filename, image_size)
        X.append(X_a)
        y.append(y_a)
    
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

"""
Experiment with HoG (histogram of gradients) features
"""
def extract_hog(base_filename: str, image_size=IMAGE_SIZE):
    rgb_filename = rgb_prefix + base_filename
    nrg_filename = nrg_prefix + base_filename

    gray_rgb = cv2.resize(cv2.imread(rgb_filename, flags=cv2.IMREAD_GRAYSCALE), image_size)
    gray_nrg = cv2.resize(cv2.imread(nrg_filename, flags=cv2.IMREAD_GRAYSCALE), image_size)

    _, hog_rgb = hog(gray_rgb, visualize=True)    
    _, hog_nrg = hog(gray_nrg, visualize=True)

    hog_image = np.stack((hog_rgb, hog_nrg), axis=1)
    return hog_image.reshape(-1, 2)

def extract_multiple_hog(train_files: list[str], image_size=IMAGE_SIZE):
    X = []
    for base_filename in train_files:
        X.append(extract_hog(base_filename, image_size))
    X = np.vstack(X)
    return X

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return 1.0 if union == 0 else intersection / union
