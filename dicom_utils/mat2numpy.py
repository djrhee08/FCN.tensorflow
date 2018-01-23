import scipy.io as sio
import numpy as np

def loadimage(fname='image.mat'):
    a = sio.loadmat(fname)
    img = a['img']
    return img

def loadmask(fname='mask.mat'):
    a = sio.loadmat(fname)
    mask = a['mask']
    return mask

def parsemask(fname='mask.mat'):
    name = []
    mask = []
    data = loadmask(fname)
    num_str = data.shape[1]
    for i in range(num_str):
        name.append(data[0,i]['name'].item(0))
        mask.append(data[0,i]['data'].item(0))

    return num_str, mask, name

"""
num_str, mask, name = parsemask('test_dataset/1/mask.mat')

print(type(mask))
print(len(mask))
for i in range(num_str):
    if name[i] == 'GTVp':
        print(name[i])
        print(len(mask))
        print(mask[i].shape)
"""