from ctypes import *
import numpy as np
import os

width = c_int()
height = c_int()
nComponents = c_int()

libPNM = cdll.LoadLibrary('./libPNM.so') # Change to the absolute path where libPNM.so is saved

libPNM.loadPPM.restype = POINTER(c_ubyte)
libPNM.writePPM.restype = None

libPNM.loadPFM.restype = POINTER(c_float)
libPNM.writePFM.restype = None

def loadPPM(fileName):
    if not os.path.exists(fileName):
        raise IOError('No such file or directory: ' + fileName)
    data_ptr = libPNM.loadPPM(fileName, byref(width), byref(height), byref(nComponents))
    return np.ctypeslib.as_array(data_ptr, shape=(height.value, width.value, nComponents.value))

def writePPM(fileName, im):
    if not im.dtype == np.uint8:
        raise TypeError('PPM images must be of type uint8: ' + str(im.dtype) + ' found instead')
    height,width,nComponents = im.shape
    data_ptr = np.ctypeslib.as_ctypes(im)
    libPNM.writePPM(fileName, width, height, nComponents, data_ptr)

def loadPFM(fileName):
    if not os.path.exists(fileName):
        raise IOError('No such file or directory: ' + fileName)
    data_ptr = libPNM.loadPFM(fileName, byref(width), byref(height), byref(nComponents))
    return np.ctypeslib.as_array(data_ptr, shape=(height.value, width.value, nComponents.value))

def writePFM(fileName, im):
    if not (im.dtype == np.float32 or im.dtype ==np.float64):
        raise TypeError('PFM images must be of type float32 or float64: ' + str(im.dtype) + ' found instead')
    if not len(im.shape) == 3:
        h,w = im.shape
        tmp = im
        im = np.empty(shape=(h,w,3), dtype=np.float32)
        im[:,:,0] = im[:,:,1] = im[:,:,2] = tmp
        print(im.shape)
    height,width,nComponents = im.shape
    data_ptr = np.ctypeslib.as_ctypes(np.float32(im))
    libPNM.writePFM(fileName, width, height, nComponents, data_ptr)
