import numpy as np
import scipy.misc as misc
import skimage.transform as sktf

# TODO : 1. Change the order : resize -> Transform -> create 3 channels

# For a single slice 2-D (width x height) image -------------------------------------------
# Reduce dicom size from 512 x 512 to 224 x 224
def resize_2d(dcmimage, resize_shape=(224,224)):
    if dcmimage.shape != (512,512):
        print("The size of DICOM is not 512 X 512. Please check again!")
        return

    dcmimage = np.float64(dcmimage)
    return sktf.resize(dcmimage, resize_shape, mode="constant")


# Crop dicom size from 512 x 512 to 227 x 227
# initial_point is upper left corner of the cropped image (i.e. (0,0) would be the first crop from upper left)
# initial_point (144,144) from (512/2) - (224/2) = 144, so the cropped image is at the center of the original image
def crop_2d(dcmimage, crop_shape=(224,224), initial_point=(144,144)):
    if dcmimage.shape != (512,512):
        print("The size of DICOM is not 512 X 512. Please check again!")
        return

    dcmimage = dcmimage[initial_point[0]:(initial_point[0]+crop_shape[0]-1), initial_point[1]:(initial_point[1]+crop_shape[1]-1)]
    return dcmimage


# Create 3 channel image from 1 channel gray scale image by repeating them
def create3channel_2d(dcmimage, num_channel=3):
    return np.repeat(dcmimage[:,:,np.newaxis],3,axis=2)

# For 3-D (N x width x height) image batch ------------------------------------------------
# Reduce dicom size from N x 512 x 512 to N x 224 x 224
def resize_3d(dcmimage, resize_shape=(224,224), num_channel=3):
    if dcmimage.shape[1] != 512 or dcmimage.shape[2] != 512:
        print(dcmimage.shape[1], dcmimage.shape[2], " is the shape of your input, not 512 X 512. Please check again!")
        return

    for i in range(dcmimage.shape[0]):
        if i==0:
            img = sktf.resize(dcmimage[i], resize_shape, mode="constant")
            img = img[np.newaxis,:,:]
        else:
            temp = sktf.resize(dcmimage[i], resize_shape, mode="constant")
            temp = temp[np.newaxis,:,:]
            img = np.append(img,temp,axis=0)

    return img

# initial_point is upper left corner of the cropped image (i.e. (0,0) would be the first crop from upper left)
# initial_point (144,144) from (512/2) - (224/2) = 144, so the cropped image is at the center of the original image
def crop_3d(dcmimage, crop_shape=(224,224), initial_point=(144,144), num_channel=3):
    if dcmimage.shape[1] != 512 or dcmimage.shape[2] != 512:
        print(dcmimage.shape[1], dcmimage.shape[2], " is the shape of your input, not 512 X 512. Please check again!")
        return

    for i in range(dcmimage.shape[0]):
        if i==0:
            img = dcmimage[i, initial_point[0]:(initial_point[0]+crop_shape[0]), initial_point[1]:(initial_point[1]+crop_shape[1])]
            img = img[np.newaxis,:,:]
        else:
            temp = dcmimage[i, initial_point[0]:(initial_point[0]+crop_shape[0]), initial_point[1]:(initial_point[1]+crop_shape[1])]
            temp = temp[np.newaxis,:,:]
            img = np.append(img,temp,axis=0)

    return img




# Create 3 channel image from 1 channel gray scale image by repeating them for batch image (N x width x height)
def create3channel_3d(dcmimage, num_channel=3):
    return np.repeat(dcmimage[:,:,:,np.newaxis],3,axis=3)

# Functions for Data Augmentation ----------------------------------------------------------
### bit sampling ### for both 2-D and 3-D
def bitsampling(dcmimage, bit=8):
    dcmimage = np.matrix.round(dcmimage / bit)
    dcmimage = dcmimage * bit

    return dcmimage

### Geometric rotation & flip ###
# Rotate dicom by given degree in clockwise
def rotate(dcmimage, angle):
    return sktf.rotate(dcmimage,angle=angle)

# Flip left to right
def fliplr(dcmimage):
    return np.fliplr(dcmimage)

# Flip upside down
def flipud(dcmimage):
    return np.flipud(dcmimage)
