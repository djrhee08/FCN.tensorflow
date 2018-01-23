import numpy as np
import read_DICOMbatch as dcm
import dicom_utils.mat2numpy as m2n
import dicom_utils.dicomPreprocess as dpp
import matplotlib.pyplot as plt

test = dcm.read_DICOM(dir_name="DICOM_data", contour_name='GTVp',rotation=False)

"""
plt.figure(2)
plt.subplot(231)
plt.imshow(img2[10,:,:,0],cmap='gray')
plt.subplot(232)
plt.imshow(img2[10,:,:,1],cmap='gray')
plt.subplot(233)
plt.imshow(img2[10,:,:,2],cmap='gray')
plt.subplot(234)
plt.imshow(mask2[10,:,:,0],cmap='gray')
plt.subplot(235)
plt.imshow(mask2[10,:,:,1],cmap='gray')
plt.subplot(236)
plt.imshow(mask2[10,:,:,2],cmap='gray')
plt.show()
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