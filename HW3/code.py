import numpy as np
import cv2
import matplotlib.pyplot as plt

def gauss_dist_mat(shape, freq):
    m, n = shape
    cy, cx = m // 2 , n // 2     # center coordinates
    x = np.arange(n)
    y = np.arange(m)
    X, Y = np.meshgrid(x, y)
    
    dist_mat = np.sqrt((X - cx)**2 + (Y - cy)**2) # distance matrix
    gauss_mat = 1 - np.exp(- (dist_mat**2) / (2 * (freq**2))) # gaussian matrix
    
    return gauss_mat

def highpass_func(img, freq):
    
    # Fourier Transform
    f_transform= np.fft.fft2(img)
    f_transform_shift = np.fft.fftshift(f_transform)
    
    #Gaussian Highpass Filter matrix
    hpass_filt = gauss_dist_mat(img.shape, freq)
    
    # highpass filter
    filt_f_transform= f_transform_shift * hpass_filt
    
    # Inverse Fourier Transform
    f_invshift = np.fft.ifftshift(filt_f_transform)
    final_img = np.fft.ifft2(f_invshift)
    final_img = np.abs(final_img)
    
    return img, final_img, hpass_filt


# load image in grayscale
image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)  

# cutoff frequency
cutoff_frequency = [20,40,80]  

for freq in cutoff_frequency:
    original_img, filtered_img, highpass_filter = highpass_func(image, freq)

    # Plots
    
    plt.figure(figsize=(18, 6))
    plt.suptitle(f'Highpass Filtering with Cutoff Frequency = {freq}', fontsize=16, weight='bold')

    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Grayscale Image', fontsize=12)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(highpass_filter, cmap='gray')
    plt.title('Gaussian Highpass Filter')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(filtered_img, cmap='gray')
    plt.title('Filtered Output Image', fontsize=12)
    plt.axis('off')

    plt.tight_layout(pad=2, rect=[0, 0, 1, 0.95])
    plt.show()