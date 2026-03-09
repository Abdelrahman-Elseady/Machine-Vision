import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_and_rezize_image(image_path, size=(64, 128)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, size)  # Resize to the specified size
    return image

import numpy as np

def compute_gradients(gray_image):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]])  

    h, w = gray_image.shape

    Gx = np.zeros((h, w))
    Gy = np.zeros((h, w))
    padded = np.pad(gray_image, 1, mode='edge')

    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]

            Gx[i, j] = np.sum(region * Kx)
            Gy[i, j] = np.sum(region * Ky)

    return Gx, Gy

def compute_magnitude_and_orientation(Gx, Gy):
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) * (180 / np.pi)  
    orientation[orientation < 0] += 180 
    return magnitude, orientation   