Image Processing Utilities
This repository contains Python scripts for various image processing tasks, including histogram analysis, connected component labeling, and feature extraction.

Installation
To use these scripts, make sure you have Python installed on your system. Additionally, you'll need to install the following dependencies:

OpenCV (cv2)
NumPy (numpy)
Matplotlib (matplotlib)
You can install these dependencies using pip:
pip install opencv-python numpy matplotlib


Usage
1. Histogram Analysis
The get_histogram function in histogram_analysis.py computes and plots the histogram of an input image. It also performs histogram smoothing using a convolution mask and applies thresholding to convert the image to binary. Finally, it identifies connected components in the binary image and computes features for each component.

To use the get_histogram function:

import cv2
import matplotlib.pyplot as plt
from histogram_analysis import get_histogram

# Load the image
image = cv2.imread('path/to/your/image.jpg')

# Get the histogram
histogram = get_histogram(image)

# Plot the histogram
plt.plot(histogram)
plt.title('Histogram of Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()


2. Connected Component Labeling
The find_connected_components function in connected_components.py performs connected component labeling on a binary image and returns the labeled image along with the number of components found.

To use the find_connected_components function:

import cv2
from connected_components import find_connected_components

# Load a binary image (thresholded)
binary_image = cv2.imread('path/to/your/binary/image.jpg', cv2.IMREAD_GRAYSCALE)

# Find connected components
components, num_components = find_connected_components(binary_image)

# Output the labeled image and number of components
print("Number of components:", num_components)
cv2.imshow('Labeled Image', components)
cv2.waitKey(0)
cv2.destroyAllWindows()


3. Feature Extraction
The compute_component_features function in feature_extraction.py computes various features for connected components in a binary image, such as area, perimeter, centroid, and compactness measures.

To use the compute_component_features function:

import cv2
from feature_extraction import compute_component_features

# Load a binary image (thresholded)
binary_image = cv2.imread('path/to/your/binary/image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute component features
areas, perimeters, centroids, compactness_measures = compute_component_features(binary_image)

# Output the computed features
print("Areas:", areas)
print("Perimeters:", perimeters)
print("Centroids:", centroids)
print("Compactness Measures:", compactness_measures)

Feel free to modify and extend these scripts according to your specific image processing needs!
