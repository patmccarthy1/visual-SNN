import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt

# read in input image
im = plt.imread('data/sample.png')                                                                                      # read in test image
plt.imshow(im)                                                                                                          # show image to check import worked

# =============================================================================
# Gabor filters using scikit-image module
# =============================================================================

# function to compute features using Gabor filter kernels
def compute_features(image, kernels):
    features = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        features[k, 0] = filtered.mean()
        features[k, 1] = filtered.var()
    return features

# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

# extract features
feats = 

# =============================================================================
# Gabor filters using OpenCV module
# =============================================================================
