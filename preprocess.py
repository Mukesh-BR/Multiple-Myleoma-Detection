import os
import random
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import numpy as np
import keras
from scipy import ndimage, misc

def preprocess_image(img):
    img = img.astype(np.uint8)
    (channel_b, channel_g, channel_r) = cv2.split(img)

    result = ndimage.maximum_filter(channel_g, size=5)
    # ret3,result = cv2.threshold(result,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,result = cv2.threshold(channel_g,120,255,cv2.THRESH_BINARY_INV)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11, 11))
    clahe_g = clahe.apply(channel_g)

    image = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    image[:,:,0] = channel_g
    image[:,:,1] = clahe_g
    image[:,:,2] = result

    image = image.astype(np.uint8)

    image = img_to_array(image)

    return image

def preprocess_mask(img):
    img = img.astype(np.uint8)
    return img[:,:,0].reshape((256,256,1))


# img=cv2.imread("/home/team6/Project/MiMM_SBILab/patches/train/images/0/1015.jpg")
# img_result=preprocess_image(img)
# cv2.imwrite("preprocess.jpg",img_result)
