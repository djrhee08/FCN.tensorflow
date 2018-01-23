import dicom_utils.dicom_parse as dp
import dicom_utils.dicomPreprocess as dpp
import dicom_utils.mat2numpy as m2n

import matplotlib.pyplot as plt
import numpy as np
import glob, os
import re

# TODO : 1. finish next_batch function.
# TODO : 2. finish zero_slice function.

class read_DICOM:

    def __init__(self, dir_name='DICOM_data', contour_name='GTVp', remove_option=False, rotation=True, bitsampling=True):
        self.dir_name = dir_name
        self.contour_name = contour_name
        self.remove_option = remove_option # Not taking account of the slices with no mask pixel

        self.batch_offset = 0
        self.file_index = 0

        self.resize_shape = (227, 227)  # for VGG16 and VGG19. for inception, (299,299) used
        self.num_channel = 3            # 3 channel input

        # Data Augmentation options
        self.rotation = rotation
        self.bitsampling = bitsampling

        mask_fname = []
        mask_index = []

        # Only choose the dataset with given contour name
        for file in glob.glob(self.dir_name + "/mask/*.mat"):
            num_str, mask, name = m2n.parsemask(file)

            for i in range(num_str):
                if name[i] == self.contour_name:
                    mask_index.append(i)
                    mask_fname.append(os.path.basename(file))
                    break

        # Matching mask with img
        image_fname = [0]*len(mask_fname)

        for file in glob.glob(self.dir_name + "/image/*.mat"):
            img_name = os.path.basename(file)
            for masks in mask_fname:
                if re.split("[._]", img_name)[1] == re.split("[._]",masks)[1]:
                    idx = mask_fname.index(masks)
                    image_fname[idx] = img_name

        # To check if mask & image matched in order
        # for i in range(len(mask_fname)):
        #    print(image_fname[i], mask_fname[i], mask_index[i])

        self.image_fname = image_fname
        self.mask_fname = mask_fname
        self.mask_index = mask_index

        self.image, self.mask = self.read_file()

    # read dicom file with the next iterator (resized and 3 channeled)
    def read_file(self):
        image = m2n.loadimage(self.dir_name + '/image/' + self.image_fname[self.file_index])
        _, mask, str_name = m2n.parsemask(self.dir_name + '/mask/' + self.mask_fname[self.file_index])
        mask = mask[self.mask_index[self.file_index]]

        # Broadcasting from int32 to float64 is mandatory for resizing images!
        image = np.float64(image)
        mask = np.float64(mask)

        print(str_name[self.mask_index[self.file_index]])

        self.file_index += 1

        # resize and make it 3 channel
        image = dpp.resize_create3channel_3d(dcmimage=image,resize_shape=self.resize_shape,num_channel=3)
        mask = dpp.resize_create3channel_3d(dcmimage=mask, resize_shape=self.resize_shape,num_channel=3)

        return image, mask

    # Data Augmentation function
    def augment_img(self,img):
        aug_img = img
        # Rotation of 90, 180, 270
        if self.rotation == True:
            aug_img = np.append(aug_img, dpp.rotate(img,90),  axis=0)
            aug_img = np.append(aug_img, dpp.rotate(img,180), axis=0)
            aug_img = np.append(aug_img, dpp.rotate(img,270), axis=0)
        # 4, 8 bit sampling
        if self.bitsampling == True:
            aug_img = np.append(aug_img, dpp.bitsampling(img, 4), axis=0)
            aug_img = np.append(aug_img, dpp.bitsampling(img, 8), axis=0)

        # bit sampling and Rotation at the same time
        if (self.rotation == True and self.bitsampling == True):
            aug_img = np.append(aug_img, dpp.rotate(dpp.bitsampling(img,4),90),  axis=0)
            aug_img = np.append(aug_img, dpp.rotate(dpp.bitsampling(img,4),180), axis=0)
            aug_img = np.append(aug_img, dpp.rotate(dpp.bitsampling(img,4),270), axis=0)

            aug_img = np.append(aug_img, dpp.rotate(dpp.bitsampling(img,8),90),  axis=0)
            aug_img = np.append(aug_img, dpp.rotate(dpp.bitsampling(img,8),180), axis=0)
            aug_img = np.append(aug_img, dpp.rotate(dpp.bitsampling(img,8),270), axis=0)

        return aug_img


    # Remove the slices with no mask entries (optional)
    def zero_slices(self,slice_mask):
        if sum(slice_mask.flatten()) == 0: # as its binary mask with 0 and 1, sum = 0 means there are no 'masked' pixels
            return True
        else:
            return False

    # Set batch offset to be 0
    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset



"""

    def next_batch(self, batch_size):
        batch_img = []
        batch_mask = []

        if self.batch_offset > self.image.shape[0]:


        while len(batch_img) < batch_size:
            slice_mask = self.mask_fname[self.slice_num]
            if self.remove_option = True:
                while zero_slices(slice_mask) == True:
                    self.slice_num += 1
                else:


            slice_img = self.image_fname[self.slice_num]
            batch_img.append(slice_img)
            batch_mask.append(slice_mask)

        return batch_img, batch_mask

"""
    # Return images and masks with the given batch size
    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.batch_img.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]