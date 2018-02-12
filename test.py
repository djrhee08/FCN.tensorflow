import numpy as np
import read_DICOMbatchImageOnly as dicom_batchImage
import dicom_utils.mat2numpy as m2n
import dicom_utils.dicomPreprocess as dpp
import matplotlib.pyplot as plt

rotation_angle = [90, 180, 270]
bitsampling_bit = [4,8]

resize_shape = (224,224)
crop_shape = resize_shape
batch_size = 1


img_dir_name = '..\H&N_CTONLY'
dicom_records = dicom_batchImage.read_DICOMbatchImage(dir_name=img_dir_name, opt_resize=True, resize_shape=(224,224),
                                            opt_crop=False, crop_shape=(224,224))


valid_images = dicom_records.next_batch(batch_size=4)

print(valid_images.shape)
plt.subplot(2,2,1)
plt.imshow(valid_images[0,:,:,0],cmap="gray")

plt.subplot(2,2,2)
plt.imshow(valid_images[1,:,:,0],cmap="gray")

plt.subplot(2,2,3)
plt.imshow(valid_images[2,:,:,0],cmap="gray")

plt.subplot(2,2,4)
plt.imshow(valid_images[3,:,:,0],cmap="gray")
plt.show()


"""
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