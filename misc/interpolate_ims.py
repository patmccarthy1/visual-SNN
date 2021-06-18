import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

# function to read images from file and store as arrays which can be passed into model
def read_images(img_dir):
    images = [cv2.imread(file, 0) for file in glob.glob(img_dir+"/*.png")]
    return images

def resize_images(imgs,lower,upper):
    resized = []
    imgs = [im[lower:upper,lower:upper] for im in ims]
    resized  = [np.reshape(np.array([cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)]),[256,256]) for img in imgs]
    return resized
    
ims = read_images('input_data/n3p2')


resized = resize_images(ims,30,225)

for i in range(8):
    plt.figure()
    plt.subplot(121)
    plt.imshow(ims[i])
    plt.title('image {} original'.format(i+1))
    plt.subplot(122)
    plt.imshow(resized[i])
    plt.title('image {} resized'.format(i+1))

