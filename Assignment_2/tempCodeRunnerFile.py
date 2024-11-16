import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def detect_edges(image):
    """
    Detect edges in the image using Sobel operators
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = gaussian_filter(gray, sigma=1)
    
    # Compute gradients using Sobel
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold to get binary edge image
    threshold = 0.5 * np.max(magnitude)
    edges = (magnitude > threshold).astype(np.uint8)
    
    return edges

def hough_circle_transform(edges, min_radius, max_radius, step_radius=1, threshold=0.4):
    """
    Implement Circle Hough Transform from scratch
    """
    height, width = edges.shape
    # Initialize the accumulator array for (x, y, r)
    radii = np.arange(min_radius, max_radius + 1, step_radius)
    accumulator = np.zeros((height, width, len(radii)))
    
    # Get edge pixel coordinates
    y_idxs, x_idxs = np.nonzero(edges)
    
    # For each edge pixel
    for i in range(len(x_idxs)):
        x, y = x_idxs[i], y_idxs[i]
        
        # For each possible radius
        for r_idx, r in enumerate(radii):
            # Generate circle points
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = x - r * np.cos(theta)
            circle_y = y - r * np.sin(theta)
            
            # Round to nearest pixel
            circle_x = np.round(circle_x).astype(int)
            circle_y = np.round(circle_y).astype(int)
            
            # Keep only valid pixel coordinates
            valid_pts = (circle_x >= 0) & (circle_x < width) & (circle_y >= 0) & (circle_y < height)
            circle_x = circle_x[valid_pts]
            circle_y = circle_y[valid_pts]
            
            # Vote in accumulator array
            accumulator[circle_y, circle_x, r_idx] += 1
    
    # Normalize accumulator
    accumulator = accumulator / accumulator.max()
    
    return accumulator, radii

def find_coin_centers(accumulator, radii, threshold=0.4):
    """
    Find circle centers from accumulator array
    """
    centers = []
    radii_detected = []
    
    # Find peaks in accumulator above threshold
    for r_idx, radius in enumerate(radii):
        acc_layer = accumulator[:, :, r_idx]
        peaks = np.argwhere(acc_layer > threshold)
        
        for peak in peaks:
            centers.append((peak[1], peak[0]))  # (x, y)
            radii_detected.append(radius)
    
    return centers, radii_detected

def create_coin_mask(image_shape, centers, radii):
    """
    Create binary mask highlighting detected coins
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for (x, y), r in zip(centers, radii):
        cv2.circle(mask, (x, y), r, 1, -1)
    
    return mask

def detect_coins(image, min_radius=20, max_radius=100):
    """
    Main function to detect coins in image
    """
    # Detect edges
    edges = detect_edges(image)
    
    # Apply Hough Transform
    accumulator, radii = hough_circle_transform(edges, min_radius, max_radius)
    
    # Find circle centers and radii
    centers, detected_radii = find_coin_centers(accumulator, radii)
    
    # Create binary mask
    mask = create_coin_mask(image.shape, centers, detected_radii)
    
    return mask


# Read image
image = cv2.imread('coins.jpg')

# Detect coins and get mask
coin_mask = detect_coins(image, min_radius=20, max_radius=100)

# Save or display result
cv2.imwrite('coin_mask.jpg', coin_mask * 255)

# Display the original image, edge image, and binary mask
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(coin_mask, cmap='gray')
plt.title('mask')
plt.axis('off')


plt.show()

