import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def read_images(img_dir):
    images = [cv2.imread(file, 0) for file in glob.glob(img_dir+"/*.png")]
    # for image_idx, image in enumerate(images):
    #     fig, ax = plt.subplots(1,1)
    #     ax.imshow(image, cmap='gray')
    #     ax.set_title('Stimulus {}'.format(image_idx+1))
    #     plt.axis('off')
    #     plt.show()
    return images

ims = read_images('../input_data/n4p2_resized')
ims_reordered = []
new_order = [0,6,4,11,13,10,7,14,15,8,9,12,3,2,1,5]

for i in range(16):
    im_num = new_order[i]
    ims_reordered.append(ims[im_num])

fig = plt.figure(figsize=[8,2])
fig.subplots_adjust(hspace = .1, wspace=.1)
for idx in range(len(ims)):
    ax = fig.add_subplot(2, 8, idx+1) # this line adds sub-axes
    filtim = ax.imshow(ims_reordered[idx],cmap='gray',vmin=0, vmax=255) # this line creates the image using the pre-defined sub axes
    # ax.set_title(str(idx+1), y=1, x=.15,pad=-10,color='white',fontsize=8)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.83, 0.15, 0.025, 0.71])
fig.colorbar(filtim, cax=cbar_ax1)
plt.savefig('input_images.eps')