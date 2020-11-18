import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# read in image to array
im = cv2.imread('data/sample.png', 0)                                                                 # read in image as greyscale (specified by second argument)                                                                  
plt.imshow(im,cmap='gray')                                                                            # show image to check import worked (using pyplot because opencv imshow function opens in new window)

#%% # Gabor filters using scikit-image module - computes mean and variance of features for each kernel
# ====================================================================================================

# function to compute features using Gabor filter kernels
def compute_features(image, kernels):
    features = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        features[k, 0] = filtered.mean()
        features[k, 1] = filtered.var()
    return features, filtered

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
feat, filt = compute_features(im,kernels)
plt.imshow(filt,cmap='gray')                                                                            # show image to check import worked (using pyplot because opencv imshow function opens in new window)
#%% Gabor filters using OpenCV module - computes full image for each kernel
# =========================================================================

# Gabor filter parameters
ksize = 5 # kernel size
sigma = 5 # SD of Gaussian in Gabor filters
theta = 1*np.pi/2 # filter orientation
lamda = 1*np.pi/4 # wavelength of sinusoid in Gabor filters
gamma = 0.1 # filter spatial aspect ratio
phi = 0.8 # phase offset of sinusoid in Gabor filters

kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
plt.imshow(kernel)

output = cv2.filter2D(im, cv2.CV_8UC3, kernel)
plt.imshow(output)