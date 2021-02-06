import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# read in image to array
im = cv2.imread('../data/n4p2/n4p2_01.png', 0)   # read in image as greyscale (specified by second argument)                                                                  
plt.imshow(im,cmap='gray')                # show image to check import worked (using pyplot because opencv imshow function opens in new window)


# Gabor filter parameters
ksize = 10 # kernel size
sigma = 5 # SD of Gaussian in Gabor filters
theta = 1*np.pi/2 # filter orientation
lamda = 1*np.pi/4 # wavelength of sinusoid in Gabor filters
gamma = 0.1 # filter spatial aspect ratio
phi = 0.8 # phase offset of sinusoid in Gabor filters

kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
plt.imshow(kernel)

output = cv2.filter2D(im, cv2.CV_8UC3, kernel)
plt.imshow(output)