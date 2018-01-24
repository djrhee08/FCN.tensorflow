import dicom_utils.dicom_parse as dp
import dicom_utils.dicomPreprocess as dpp
import dicom_utils.mat2numpy as m2n

import matplotlib.pyplot as plt
import numpy as np
import glob, os
import re

# TODO : 2. finish adapting zero_slice function in next_batch function.
# TODO : 5. Consider Normalization of pixel values (when broadcasting int32 -> float64)

class read_DICOM:

    def __init__(self, dir_name='DICOM_data', contour_name='GTVp', zero_slices=False, resize_shape=(227,227),
                 rotation=True, rotation_angle=[90], bitsampling=True, bitsampling_bit=[4]):
        self.dir_name = dir_name
        self.contour_name = contour_name
        self.zero_slices = zero_slices # Not taking account of the slices with no mask pixel

        self.file_index = 0
        self.slice_index = 0

        self.resize_shape = resize_shape  # for VGG16 and VGG19. for inception, (299,299) used
        self.num_channel = 3              # 3 channel input

        # Data Augmentation options
        self.rotation = rotation
        self.rotation_angle = rotation_angle
        self.bitsampling = bitsampling
        self.bitsampling_bit = bitsampling_bit

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

        # Broadcasting from int32 to float64/32(?) is mandatory for resizing images without pixel value change!
        image = np.float64(image)
        mask = np.float64(mask)

        print(str_name[self.mask_index[self.file_index]])

        self.file_index += 1

        # resize 2-D N number of image (N x width x height)
        image = dpp.resize_3d(dcmimage=image, resize_shape=self.resize_shape)
        mask = dpp.resize_3d(dcmimage=mask, resize_shape=self.resize_shape)

        return image, mask

    def read_slice(self):
        if self.slice_index > self.image.shape[1]:
            self.file_index = self.file_index + 1
            self.slice_index = 0
            self.image, self.mask = self.read_file()
            image_slice = self.image[self.slice_index]
            mask_slice = self.mask[self.slice_index]
        else:
            image_slice = self.image[self.slice_index]
            mask_slice = self.mask[self.slice_index]
            self.slice_index = self.slice_index + 1

        return image_slice, mask_slice


    # Data Augmentation function
    def augment_img(self,img,type='image'):
        if not (type == 'image' or type == 'mask'):
            print('Type should be either image or mask. The undefined type is not acceptable : ', type)
            return


        aug_img = img[np.newaxis,:,:]
        # Rotation with given angles, apply both image and mask equally
        if self.rotation == True:
            for i in range(len(self.rotation_angle)):
                rotate_img = dpp.rotate(img,self.rotation_angle[i])
                aug_img = np.append(aug_img, rotate_img[np.newaxis,:,:], axis=0)

        # Bitsampling with given bits, apply only for image
        if self.bitsampling == True:
            if type == 'image':
                for i in range(len(self.bitsampling_bit)):
                    bitsample_img = dpp.bitsampling(img, self.bitsampling_bit[i])
                    aug_img = np.append(aug_img, bitsample_img[np.newaxis,:,:], axis=0)
            elif type == 'mask':
                for i in range(len(self.bitsampling_bit)):
                    aug_img = np.append(aug_img, img[np.newaxis, :, :], axis=0)

        # Bitsampling and Rotation at the same time
        if (self.rotation == True and self.bitsampling == True):
            for i in range(len(self.rotation_angle)):
                if type == 'image':
                    for j in range(len(self.bitsampling_bit)):
                        rotate_bitsample_img = dpp.rotate(dpp.bitsampling(img, self.bitsampling_bit[j]),
                                                          self.rotation_angle[i])
                        aug_img = np.append(aug_img, rotate_bitsample_img[np.newaxis,:,:], axis=0)
                elif type == 'mask':
                    for j in range(len(self.bitsampling_bit)):
                        rotate_bitsample_img = dpp.rotate(img,self.rotation_angle[i])
                        aug_img = np.append(aug_img, rotate_bitsample_img[np.newaxis, :, :], axis=0)

        return aug_img


    # Remove the slices with no mask entries, NOT WORKING SOMEHOW!!!
    def zero_slices(self,slice_mask):
        t = 0
        if sum(slice_mask.flatten()) == 0.0: # as its binary mask with 0 and 1, sum = 0 means there are no 'masked' pixels
            t = 1
        else:
            t = 0

        return t

    # Returns the batch image and mask with given batch_size, as batch_size is defined as the number of original image before augmentation
    def next_batch(self, batch_size):
        index = 0
        batch_img = 0
        batch_mask = 0

        while index < batch_size:
            image_slice, mask_slice = self.read_slice()
            zero_eval = sum(mask_slice.flatten())

            # Zero slice adaptation
            if zero_eval > 0:
                image_aug = self.augment_img(img=image_slice,type='image')
                mask_aug = self.augment_img(img=mask_slice,type='mask')

                if index == 0:
                    batch_img = image_aug
                    batch_mask = mask_aug
                else:
                    batch_img = np.append(batch_img, image_aug, axis=0)
                    batch_mask = np.append(batch_mask, mask_aug, axis=0)

                index += 1

        batch_img = dpp.create3channel_3d(dcmimage=batch_img,num_channel=3)
        batch_mask = dpp.create3channel_3d(dcmimage=batch_mask, num_channel=3)

        return batch_img, batch_mask
