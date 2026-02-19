import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_rain_drops(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    
    # Apply median filter
    median = cv2.medianBlur(img, 23)
    
    # Subtract median from original to isolate potential rain drops
    diff = cv2.absdiff(img, median)
    
    # Convert difference to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # Threshold to identify rain drops
    _, rain_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Dilate to connect nearby rain drops
    kernel = np.ones((3,3), np.uint8)
    dilated_mask = cv2.dilate(rain_mask, kernel, iterations=1)
    
    # Inpaint only the areas identified as rain drops
    result = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_TELEA)
    
    return img, diff, result

def plot_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()

# Usage
input_image = "rain.png"
original, subtracted, processed = remove_rain_drops(input_image)

plot_image(original, "Original Image")
plot_image(subtracted, "Subracted Image")
plot_image(processed, "Final image")
