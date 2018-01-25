import numpy as np
import read_DICOMbatch as batch
import dicom_utils.mat2numpy as m2n
import dicom_utils.dicomPreprocess as dpp
import matplotlib.pyplot as plt

rotation_angle = [90, 180, 270]
bitsampling_bit = [4,8]

resize_shape = (227,227)
batch_size = 1

total_batch_size = batch_size * (len(rotation_angle) + len(bitsampling_bit) + len(rotation_angle)*len(bitsampling_bit) + 1)
#b1 = batch.read_DICOM(dir_name="DICOM_data", contour_name='GTVp', resize_shape=resize_shape, rotation=True,
#                      rotation_angle=rotation_angle, bitsampling=True, bitsampling_bit=bitsampling_bit)

b1 = batch.read_DICOM(dir_name="AQA", contour_name='External', resize_shape=resize_shape, rotation=True,
                      rotation_angle=rotation_angle, bitsampling=True, bitsampling_bit=bitsampling_bit)

img, mask = b1.next_batch(batch_size=batch_size)
img2, mask2 = b1.next_batch(batch_size=batch_size)
img3, mask3 = b1.next_batch(batch_size=batch_size)


print(img.shape, mask.shape)
print(img.flatten().max(), img.flatten().min())

plt.figure(2)
plt.subplot(231)
plt.imshow(img[0,:,:,0],cmap='gray')
plt.subplot(232)
plt.imshow(img2[0,:,:,1],cmap='gray')
plt.subplot(233)
plt.imshow(img3[0,:,:,2],cmap='gray')
plt.subplot(234)
plt.imshow(mask[0,:,:,0],cmap='gray')
plt.subplot(235)
plt.imshow(mask2[0,:,:,1],cmap='gray')
plt.subplot(236)
plt.imshow(mask3[0,:,:,2],cmap='gray')
plt.show()

"""
a = mask3[36,:,:,0]
b = mask3[40,:,:,0]
c = mask3[41,:,:,0]

print(a.max(), b.max(), c.max())
print(a.min(), b.min(), c.min())

print(np.sum((np.squeeze(a) - np.squeeze(b)).flatten()))
print(np.sum((np.squeeze(a) - np.squeeze(c)).flatten()))
"""



#train_dataset = dcm.read_DICOM(dirname="DICOM_data", contourname='GTVp')
#test_dataset = dcm.read_DICOM(dirname="DICOM_test", contourname='GTVp')

#train_dataset.nextbatch(batch_size)
#validation_dataset.nextbatch(batch_size)

"""
# training_set = 70% of dataset, test_set = 30% of dataset
num_training_set = round(0.7 * num_dataset)
num_test_set = num_dataset - num_training_set
"""