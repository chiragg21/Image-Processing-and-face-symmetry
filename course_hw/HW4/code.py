import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Create a white background image
img_size = (500, 500, 3)
bg = np.ones(img_size, dtype=np.uint8) * 255

centers = [(100, 100), (400, 100), (100, 400), (400, 400)]
radius = 50

# Define the RGB colors for Cyan, Magenta, Yellow, Black 
colors_rgb = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]  

# Create the RGB image with the colored circles
rgb_image = bg.copy()
for center, color in zip(centers, colors_rgb):
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                rgb_image[x, y] = color



# Convert the RGB image to CMYK

rgb = np.array(rgb_image) / 255.0
C = 1 - rgb[:, :, 0]
M = 1 - rgb[:, :, 1]
Y = 1 - rgb[:, :, 2]

K = np.minimum(C, np.minimum(M, Y))

C = (C - K) / (1 - K + 1e-5)
M = (M - K) / (1 - K + 1e-5)
Y = (Y - K) / (1 - K + 1e-5)

C[np.isnan(C)] = 0
M[np.isnan(M)] = 0
Y[np.isnan(Y)] = 0

cmyk_image = np.stack((C, M, Y, K), axis=-1)


# Split the CMYK image into channels
C, M, Y, K = [cmyk_image[:, :, i] for i in range(4)]

# Plotting the RGB and CMYK images using matplotlib
plt.figure(figsize=(6, 6))

# Plot the original RGB image
plt.imshow(rgb_image)
plt.title('Original RGB Image')
plt.axis('off')
plt.show()


# Plot the individual CMYK channels
plt.figure(figsize=(12,6))
plt.subplot(2, 2, 1)  
plt.imshow(C, cmap='gray')
plt.title('Cyan Channel')
plt.axis('off')

plt.subplot(2, 2, 2)  
plt.imshow(Y, cmap='gray')
plt.title('Yellow Channel')
plt.axis('off')

plt.subplot(2, 2, 3)  
plt.imshow(M, cmap='gray')
plt.title('Magenta Channel')
plt.axis('off')

plt.subplot(2, 2, 4)  
plt.imshow(K, cmap='gray')
plt.title('Black Channel')
plt.axis('off')

plt.tight_layout()
plt.show()
