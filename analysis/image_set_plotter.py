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

ims = read_images('../input_data/n3p2')
set_1 = [ims[5],ims[0],ims[4],ims[2]]
set_2 = [ims[5],ims[0],ims[4],ims[1]]
set_3 = [ims[5],ims[0],ims[1],ims[6]]
set_4 = [ims[5],ims[1],ims[6],ims[3]]

# create plot of filtered images
fig = plt.figure(figsize=[4,4])
fig.subplots_adjust(hspace = .1, wspace=.1)
ax = fig.add_subplot(4, 4, 1) 
im = ax.imshow(set_1[0],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 2) 
im = ax.imshow(set_2[0],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 3) 
im = ax.imshow(set_3[0],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 4) 
im = ax.imshow(set_4[0],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 5) 
im = ax.imshow(set_1[1],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 6) 
im = ax.imshow(set_2[1],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 7) 
im = ax.imshow(set_3[1],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 8) 
im = ax.imshow(set_4[1],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 9) 
im = ax.imshow(set_1[2],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 10) 
im = ax.imshow(set_2[2],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 11) 
im = ax.imshow(set_3[2],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 12) 
im = ax.imshow(set_4[2],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 13) 
im = ax.imshow(set_1[3],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 14) 
im = ax.imshow(set_2[3],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 15) 
im = ax.imshow(set_3[3],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = fig.add_subplot(4, 4, 16) 
im = ax.imshow(set_4[3],cmap='gray') 
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
# rect_1 = plt.Rectangle(
#     # (lower-left corner), width, height
#     (0.12, 0.118), 0.19, 0.77, fill=False, color="r", lw=1, 
#     zorder=1000, transform=fig.transFigure, figure=fig
# )
# fig.patches.extend([rect_1])
# rect_2 = plt.Rectangle(
#     # (lower-left corner), width, height
#     (0.318, 0.31), 0.19, 0.577, fill=False, color="r", lw=1, 
#     zorder=1000, transform=fig.transFigure, figure=fig
# )
# fig.patches.extend([rect_2])
# rect_3 = plt.Rectangle(
#     # (lower-left corner), width, height
#     (0.5158, 0.503), 0.19, 0.385, fill=False, color="r", lw=1, 
#     zorder=1000, transform=fig.transFigure, figure=fig
# )
# fig.patches.extend([rect_3])
# rect_4 = plt.Rectangle(
#     # (lower-left corner), width, height
#     (0.7158, 0.698), 0.19, 0.19, fill=False, color="r", lw=1, 
#     zorder=1000, transform=fig.transFigure, figure=fig
# )
# fig.patches.extend([rect_4])
plt.savefig('image_subsets_no_border.eps',dpi=500)



