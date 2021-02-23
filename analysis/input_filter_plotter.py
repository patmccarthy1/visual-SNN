import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% function definitions

def generate_gabor_filters():
     filters = []                                                                                                            
     ksize = 10 # kernel size
     phi_list = [0, np.pi/2, 3*np.pi/4, np.pi] # phase offset of sinusoid 
     lamda = 5 # wavelength of sinusoid 
     theta_list = [0, np.pi/2, np.pi, 3*np.pi/2] # filter orientation
     b = 1.1 # spatial bandwidth in octaves (will be used to determine SD)
     sigma = lamda*(2**b+1)/np.pi*(2**b-1) * np.sqrt(np.log(2)/2)
     gamma = 0.5 # filter aspect ratio
     for phi in phi_list:
         for theta in theta_list:
             filt = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
             filters.append(filt)
     return filters

def plot_kernel(image, kernels, centre_x, centre_y, filter_num):
    kernel = kernels[filter_num]
    image[int(np.floor(-len(kernel)/2+centre_x)):int(np.floor(len(kernel)/2+centre_x)),int(np.floor(-len(kernel)/2+centre_y)):int(np.floor(len(kernel)/2+centre_y))] = image[int(np.floor(-len(kernel)/2+centre_x)):int(np.floor(len(kernel)/2+centre_x)),int(np.floor(-len(kernel)/2+centre_y)):int(np.floor(len(kernel)/2+centre_y))]+kernel
    return image

#%% 

filters = generate_gabor_filters()
empty_image = np.zeros([256,256])

cent_x = np.random.randint(10,250,size=200)
cent_y = np.random.randint(10,250,size=200)
filt_n = np.random.randint(0,len(filters),size=200)

new_image = plot_kernel(empty_image, filters, cent_x[0], cent_y[0], filt_n[0])
for x in cent_x[1:]:
    for y in cent_y[1:]:
        for f in filt_n[1:]:
            new_image = plot_kernel(new_image, filters, x, y, f)
            
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) # this line adds sub-axes
im = ax.imshow(new_image,cmap='gray') # this line creates the image using the pre-defined sub axes
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])


