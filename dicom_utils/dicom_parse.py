from os import listdir
from os.path import isfile, join
import sys
import glob
import dicom
import numpy as np

# Get dicom files from the given directory and returns image in 3d array
def getdicomdir(dicomdir="test_dataset"):
    dcmFiles = glob.glob(dicomdir+"/*.dcm")

    print(len(dcmFiles))
    dcmImages = []
    dcmStrFiles = []
    sliceNum = []
    imagePos = []

    for i in range(len(dcmFiles)):
        file = dicom.read_file(dcmFiles[i])
        mod = file.Modality
        if (mod == "MR") or (mod == "CT"):
            if hasattr(file,'RescaleSlope'):
                slope = file.RescaleSlope
            else:
                slope = 1
            if hasattr(file,'RescaleIntercept'):
                intercept = file.RescaleIntercept
            else:
                intercept = 0

            dcmImages.append(file.pixel_array * slope + intercept)
            sliceNum.append(file[0x20, 0x13].value)  # InstanceNumber
            imagePos.append(file[0x20, 0x32].value)  # ImagePosition
        elif mod == "RTSTRUCT":
            dcmStrFiles.append(dcmFiles[i])

    sliceNum = list(map(int, sliceNum)) # Cast sliceNum in integer

    # To check if any file has been missed
    sliceNumSorted = sorted(sliceNum)
    if not np.array_equal(np.asarray(sliceNumSorted),np.arange(1,len(sliceNumSorted)+1)):
        print("DICOM image file slices are incomplete!")
        print(np.asarray(sliceNumSorted))
        sys.exit()

    data_sorted = zip(sliceNum, imagePos, dcmImages)
    data_sorted = sorted(data_sorted)

    sn, imagePosSorted, dcmImagesSorted = zip(*data_sorted)

    return np.asarray(dcmImagesSorted)



