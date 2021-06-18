import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

#%% function definitions

def generate_gabor_filters():
     filters = []                                                                                                            
     ksize = 10 # kernel size
     phi_list = [np.pi/2, np.pi] # phase offset of sinusoid 
     lamda = 5.2 # wavelength of sinusoid 
     theta_list = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi] # filter orientation
     b = 1.1 # spatial bandwidth in octaves (will be used to determine SD)
     sigma = lamda*(2**b+1)/np.pi*(2**b-1) * np.sqrt(np.log(2)/2)
     gamma = 0.5 # filter aspect ratio
     for phi in phi_list:
         for theta in theta_list:
             filt = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
             filters.append(filt)
     return filters

def plot_kernel(image, kernels, centre_x, centre_y, filter_num, weight):
    kernel = kernels[filter_num]
    weighted_kernel = 2 * np.array(kernel) #[x * weight for x in kernel]
    if int(np.floor(-len(kernel)/2+centre_x)) >= 0 and int(np.floor(len(kernel)/2+centre_x)) <= 255 and int(np.floor(-len(kernel)/2+centre_y)) >= 0 and int(np.floor(len(kernel)/2+centre_y)) <= 255:
        image[int(np.floor(-len(kernel)/2+centre_x)):int(np.floor(len(kernel)/2+centre_x)),
              int(np.floor(-len(kernel)/2+centre_y)):int(np.floor(len(kernel)/2+centre_y))] = image[int(np.floor(-len(kernel)/2+centre_x)):int(np.floor(len(kernel)/2+centre_x)),int(np.floor(-len(kernel)/2+centre_y)):int(np.floor(len(kernel)/2+centre_y))] + weighted_kernel
    return image
#%% read in data
def read_images(img_dir):
    images = [cv2.imread(file, 0) for file in glob.glob(img_dir+"/*.png")]
    # for image_idx, image in enumerate(images):
    #     fig, ax = plt.subplots(1,1)
    #     ax.imshow(image, cmap='gray')
    #     ax.set_title('Stimulus {}'.format(image_idx+1))
    #     plt.axis('off')
    #     plt.show()
    return images
ims = read_images('../input_data/n3p2')
im = ims[5]

data = pd.read_table('simulation_9/weighted_inputs_l3.csv', sep=',')
multi_weights =  np.array(data.loc[0])
# multi_weights = [x*10e60 for x in multi_weights]
cent_x = np.array(data.loc[1]).astype(int)
cent_y = np.array(data.loc[2]).astype(int)
filt_n = np.array(data.loc[3]).astype(int)

# crude plot

plt.figure(1,[50,50])
plt.scatter(cent_y,cent_x,marker='s',color='cyan')
plt.xlim([0,256])
plt.ylim([256,0])
plt.imshow(im,cmap='plasma')

#%%

x_locs = []
y_locs = []

for x in range(256):
    for y in range(256):
        if im[x,y] == 0:
            x_locs.append(x)
            y_locs.append(y)
            
plt.figure(1,[20,20])
plt.scatter(y_locs,x_locs)
plt.xlim([0,256])
plt.ylim([256,0])

        
#%% plot

filters = generate_gabor_filters()
empty_image = np.zeros([256,256])
# empty_image = im

i=0
new_image = plot_kernel(empty_image, filters, cent_x[0], cent_y[0], filt_n[0], multi_weights[0])
for x in cent_x[1:]:
    i = i+1
    for y in cent_y[1:]:
        for f in filt_n[1:]:
            new_image = plot_kernel(new_image, filters, x, y, f, multi_weights[i])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) # this line adds sub-axes
im = ax.imshow(new_image,cmap='plasma') # this line creates the image using the pre-defined sub axes
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

# from pylab import figure, cm
# from matplotlib.colors import LogNorm
# # C = some matrix
# f = figure(figsize=(6.2,5.6))
# ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
# axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])
# im = ax.matshow(C, cmap=cm.gray_r, norm=LogNorm(vmin=0.01, vmax=1))
# t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
# f.colorbar(im, cax=axcolor, ticks=t, format='$%.2f$')
# f.show()

