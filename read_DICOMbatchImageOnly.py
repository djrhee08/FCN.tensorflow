import dicom_utils.dicom_parse as dp
import dicom_utils.dicomPreprocess as dpp
import dicom_utils.mat2numpy as m2n

import matplotlib.pyplot as plt
import numpy as np
import glob, os
import re
import random

# Class for batch images only (with no mask)

class read_DICOMbatchImage:

    def __init__(self, dir_name='../H&N_CTONLY/', opt_resize=True, resize_shape=(224,224), opt_crop=False, crop_shape=(224,224)):
        self.voxel_range = np.array([0,3000]) # any voxel bigger than 3000 would be 3000, less than 0 would be 0
        self.normalization_range = np.array([0,255])
        self.dir_name = dir_name

        self.file_index = 0
        self.slice_index = 0

        self.opt_resize = opt_resize
        self.resize_shape = resize_shape  # for VGG16/VGG19, (224,224) used. For inception, (299,299) used
        self.opt_crop = opt_crop
        self.crop_shape = crop_shape

        self.num_channel = 3              # 3 channel input
        self.num_files = 0

        self.image_fname = []
        for file in glob.glob(self.dir_name + "/matfiles/*.mat"):
            self.image_fname.append(os.path.basename(file))
            self.num_files += 1

        print("num file : ", self.num_files)
        # Randomize the order of data
        self.random_batch = random.sample(range(0,self.num_files),self.num_files)
        print("random_batch: ", self.random_batch)

        self.image = self.read_file()

    # read dicom file with the next iterator (resized and 3 channeled)
    def read_file(self):
        print(self.image_fname[self.random_batch[self.file_index]])
        image = m2n.loadimage(self.dir_name + '/matfiles/' + self.image_fname[self.random_batch[self.file_index]])

        # Broadcasting from int32 to float64/32(?) is mandatory for resizing images without pixel value change!
        image = np.float64(image)

        self.file_index += 1

        if self.opt_resize == True and self.opt_crop == True:
            print("Both resize and crop are selected. Please choose either one")

        # resize 2-D N number of image (N x width x height)
        if self.opt_resize == True:
            print("image is resized")
            image = dpp.resize_3d(dcmimage=image, resize_shape=self.resize_shape)

        # crop 2-D N number of image (N x width x height)
        if self.opt_crop == True:
            print("image is cropped")
            image = dpp.crop_3d(dcmimage=image, crop_shape=self.crop_shape)

        return image


    def read_slice(self):
        if self.slice_index >= self.image.shape[0]: # not '>' but '>=' because shape slice_index starts from 0
            self.file_index = self.file_index + 1

            # When the images are all used, go back to the first image again!
            if self.file_index >= self.num_files:
                print('Files are all used. Move back to the first image')
                self.file_index = 0

            self.slice_index = 0
            self.image = self.read_file()
            image_slice = self.image[self.slice_index]
        else:
            image_slice = self.image[self.slice_index]
            self.slice_index = self.slice_index + 1

        return image_slice


    # need to rewrite this function to be generalized. + Need to study about CT number a bit. Is CT Number = 0  water for every cases?
    def normalize(self, image):
        bin_size = (self.voxel_range.max() - self.voxel_range.min())/(self.normalization_range.max() - self.normalization_range.min())
        image = image - self.voxel_range.min()
        image = np.matrix.round(image / bin_size)

        return image


    # Returns the batch image with given batch_size
    def next_batch(self, batch_size):
        index = 0
        batch_img = 0

        while index < batch_size:
            image_slice = self.read_slice()

            # Any voxel bigger than 3000 would be 3000, less than 0 would be 0
            image_slice[image_slice > self.voxel_range.max()] = self.voxel_range.max()
            image_slice[image_slice < self.voxel_range.min()] = self.voxel_range.min()
            # Normalization of pixels from the given range (0,255)
            image_slice = self.normalize(image_slice)
            image_slice = image_slice[np.newaxis,:,:]

            if index == 0:
                batch_img = image_slice
            else:
                batch_img = np.append(batch_img, image_slice, axis=0)

            index += 1

        #print(image_slice.shape, batch_img.shape)

        batch_img = dpp.create3channel_3d(dcmimage=batch_img,num_channel=3) # (N, Width, Height, 3)

        # return float32 format as its the input of tensorflow (?) CHECK!
        return np.float32(batch_img)