import cv2
import numpy as np
import matplotlib.pyplot as plt


import numpy as np

# def find_valley_between_peaks(histogram):
#     for intensity in histogram:
#         print(intensity)
#     print("hist shape", histogram.shape)
#     # Convert histogram to integers
#     histogram_int = np.round(histogram).astype(int)

#     # Step 1: Find the frequencies of each value in the histogram
#     frequency = np.bincount(histogram_int)

#     # Step 2: Find the two highest frequencies
#     highest_indices = np.argsort(frequency)[-2:]
#     highest_values = frequency[highest_indices]

#     # Step 3: Find the lowest frequency between the two highest frequencies
#     start_index, end_index = sorted(highest_indices)
#     min_valley_frequency = min(frequency[start_index:end_index])

#     # Step 4: Find the value in the histogram corresponding to the lowest frequency
#     min_valley_value = np.where(frequency == min_valley_frequency)[0][0]

#     return min_valley_value

def find_connected_components(binary_image):
    rows, cols = binary_image.shape
    labels = np.zeros_like(binary_image, dtype=int)
    next_label = 1
    equivalence = {}

    # First pass
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] != 0:  # Check for non-zero values
                xU = labels[i-1, j] if i > 0 else 0
                xL = labels[i, j-1] if j > 0 else 0

                if xU == 0 and xL == 0:
                    labels[i, j] = next_label
                    next_label += 1
                elif xU != 0 and xL == 0:
                    labels[i, j] = xU
                elif xU == 0 and xL != 0:
                    labels[i, j] = xL
                else:
                    labels[i, j] = xL
                    if xU != xL:
                        equivalence[xU] = xL

    # Second pass
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] != 0:  # Check for non-zero values
                xC = labels[i, j]
                xU = labels[i-1, j] if i > 0 else 0
                xL = labels[i, j-1] if j > 0 else 0

                if xU != 0 and xC != 0:
                    if xU != xC:
                        equivalence[xU] = xC
                if xL != 0 and xC != 0:
                    if xL != xC:
                        equivalence[xL] = xC

    # Update equivalent labels
    for label in list(equivalence.keys()):
        while equivalence[label] in equivalence:
            equivalence[label] = equivalence[equivalence[label]]

    # Update labels
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:
                equivalent_label = equivalence.get(labels[i, j], labels[i, j])
                labels[i, j] = equivalent_label

    return labels, len(set(labels.flatten()) - {0})


def colorize_connected_components(labels):
    # Get unique labels excluding 0
    unique_labels = np.unique(labels)[1:]

    # Create a blank color image with the same shape as labels
    color_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    # Assign colors to each unique label
    for label in unique_labels:
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        color_image[labels == label] = color

    return color_image



def compute_component_features(labeled_image):
    num_labels = np.max(labeled_image)

    # Initialize lists to store features
    areas = []
    perimeters = []
    centroids = []
    compactness_measures = []

    for label in range(1, num_labels + 1):
        # Extract the binary mask for the current labeled component
        component_mask = (labeled_image == label).astype(np.uint8)

        # Compute area
        area = np.sum(component_mask)
        areas.append(area)

        # Compute perimeter
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum([cv2.arcLength(contour, True) for contour in contours])
        perimeters.append(perimeter)

        # Compute centroid
        M = cv2.moments(component_mask)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        centroids.append((cx, cy))

        # Compute compactness measure
        compactness = perimeter ** 2 / (4 * np.pi * area) if area != 0 else 0
        compactness_measures.append(compactness)

    return areas, perimeters, centroids, compactness_measures


def grayscale_to_binary(image, valley):
    # Create a new numpy array of the same size as the input image
    binary_image = np.zeros_like(image, dtype=np.uint8)

    # Loop through each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # If the intensity value is less than the threshold (valley), set it to 0, otherwise set it to 255
            if image[i, j] < valley:
                binary_image[i, j] = 0
            else:
                binary_image[i, j] = 255

    return binary_image

def get_histogram(image):
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # print(grayscale_image.shape)
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0,256])

    print(histogram.shape)

    # Define the convolution mask
    mask = np.array([1/9, 2/9, 3/9, 2/9, 1/9])

    # Perform 1D convolution on the histogram
    histogram_convolved = np.convolve(histogram.flatten(), mask, mode='same')

    # valley = find_valley_between_peaks(histogram_convolved)
    # print("valley",valley)
    # Threshold the grayscale image using the valley value
    threshold_intensity = 80
    binary_image = grayscale_to_binary(grayscale_image, threshold_intensity)

    print("threshold intensity", threshold_intensity)
    
    #perform connected components
    components, classes = find_connected_components(binary_image)
    print(components, classes)
    colored_image = colorize_connected_components(components)
    #display
    print("number of components id", classes)
    cv2.imshow('binary image', binary_image)
    cv2.imshow('grayscale image', grayscale_image)
    cv2.imshow('pseudo color components', colored_image)

   

    areas, perimeters, centroids, compactness_measures = compute_component_features(binary_image)
    print("Areas:", areas)
    print("Perimeters:", perimeters)
    print("Centroids:", centroids)
    print("Compactness Measures:", compactness_measures)

    return histogram_convolved


path = '../image_data/keys.jpg'
image = cv2.imread(path)  # Load the image
histogram = get_histogram(image)  # Get the histogram
# print(histogram)
plt.plot(histogram)  # Plot the histogram
plt.title('Histogram of Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()
