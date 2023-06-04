import cv2
import numpy as np
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

def create_hybrid_image(image1, image2, cutoff_frequency):
    # Perform Fourier Transform on the images
    image1_fft = fftshift(fft2(image1))
    image2_fft = fftshift(fft2(image2))

    # Compute the size of the images
    height, width = image1.shape[:2]

    # Create a Gaussian filter
    gaussian_filter = np.zeros((height, width), dtype=np.float32)
    radius = int(cutoff_frequency * min(height, width) / 2)
    gaussian_filter[height // 2 - radius:height // 2 + radius, width // 2 - radius:width // 2 + radius] = 1

    # Apply the Gaussian filter in the frequency domain
    image1_fft_filtered = image1_fft * gaussian_filter
    image2_fft_filtered = image2_fft * (1 - gaussian_filter)

    # Perform Inverse Fourier Transform to obtain the hybrid image
    hybrid_image_fft = ifft2(ifftshift(image1_fft_filtered + image2_fft_filtered))
    hybrid_image = np.abs(hybrid_image_fft).astype(np.uint8)

    return hybrid_image

# Read the two input images
image2 = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('marylin.jpg', cv2.IMREAD_GRAYSCALE)

# Set the cutoff frequency (proportion of high frequencies to keep)
cutoff_frequency = 0.1

# Create the hybrid image
hybrid_image = create_hybrid_image(image1, image2, cutoff_frequency)

# Display the hybrid image
cv2.imshow('Hybrid Image', hybrid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

