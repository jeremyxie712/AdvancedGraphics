import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
from PNM import *

def w_function(x):
	return 16*x**2*(1-x)**2

def tone_map(F, stops, gamma):
	# x should be an hsv image
	# scales with exposure
	x = F.copy()
	x[:,:,2] *= 2**stops
	# Clip
	x[x[:,:,2] > 1, 2] = 1
	x[:,:,2] **= (1/gamma)
	return x


def main(sample_num,flag=True):
    filename = '../GraceCathedral/grace_latlong.pfm'
    img = loadPFM(filename)

    ################
    print(img.shape) ##Checking the dimensions, (512,1024,3)
    ################

    height,width,channel = img.shape

    F = img.copy()

    intensity = np.sum(img,axis=2) ##Averaging the luminance
    intensity = intensity / 3

    for h in range(height):
        for w in range(width):
            for c in range(channel):
                if F[h][w][c] < 0:
                    F[h][w][c] = 0
                elif F[h][w][c] > 255:
                    F[h][w][c] = 255
            intensity[h][w] = intensity[h][w] * math.sin((h / (height - 1.)) * np.pi)
    #######################
    print(intensity.shape) ##Checking the dimensions, should be (512,1024)
    #######################

    cdf_1d = np.sum(intensity,axis=1) / np.sum(intensity)
    cdf_2d = np.zeros((height,width))  ##(512,1024)

    for idx_height in range(1,height):
        temp_cdf = intensity[idx_height] / np.sum(intensity[idx_height])
        cdf_1d[idx_height] = cdf_1d[idx_height] + cdf_1d[idx_height - 1]
        for idx_width in range(1,width):
            temp_cdf[idx_width] = temp_cdf[idx_width] + temp_cdf[idx_width - 1]
        cdf_2d[idx_height] = temp_cdf
    ################
    print('1d cdf shape: ')
    print(cdf_1d.shape)    ##Checking the dimensions, should be (512.)
    print('2d cdf shape: ')
    print(cdf_2d.shape)    ##Checking the dimensions, should be (512,1024)
    ################
    sampler = np.zeros((height,width,channel))
    for i in range(sample_num):
        idx_row = -1
        idx_col = -1

        # mu = np.mean(intensity)

        sampling_distribution_row = np.random.uniform(0,1)
        for h in range(height):
            if cdf_1d[h] >= sampling_distribution_row:
                idx_row = h
                sampling_distribution_col = np.random.uniform(0,1)
                for w in range(width):
                    if cdf_2d[idx_row][w] >= sampling_distribution_col:
                        idx_col = w
                        break
                break
        for window in [[idx_row-2, idx_col-2], [idx_row-2, idx_col-1], [idx_row-2, idx_col], [idx_row-2, idx_col+1],
                      [idx_row+2, idx_col-1], [idx_row+2, idx_col], [idx_row+2, idx_col+1], [idx_row+2, idx_col+2],
                      [idx_row-2, idx_col+2], [idx_row-1, idx_col+2], [idx_row, idx_col+2], [idx_row+1, idx_col+2],
                      [idx_row-1, idx_col-2], [idx_row, idx_col-2], [idx_row+1, idx_col-2], [idx_row+2, idx_col-2],
                      [idx_row, idx_col]]:
                      if -1 < window[0] < 512 and -1 < window[1] < 1024:
                          if flag == True:
                              sampler[window[0]][window[1]] = img[idx_row][idx_col]
                          F[window[0]][window[1]] = [0,0,10]


    for h in range(height):
        for w in range(width):
            for c in range(channel):
                F[h,w,c] = ((F[h,w,c]/255) ** (1./1.5)*255)
                for s in range(6):
                    F[h,w,c] = F[h,w,c]*2
                if F[h,w,c] > 255:
                    F[h,w,c] = 255
                elif F[h,w,c] < 0:
                    F[h,w,c] = 0

    writePPM('../Sample_Num: {}.ppm'.format(sample_num),F.astype(np.uint8))

    if sample_num == 256:
        for h in range(height):
            for w in range(width):
                for c in range(channel):
                    sampler[h,w,c] = ((sampler[h,w,c]/255) ** (1./1.5)*255)
                    for s in range(6):
                        sampler[h,w,c] = sampler[h,w,c]*2
                    if sampler[h,w,c] > 255:
                        sampler[h,w,c] = 255
                    elif sampler[h,w,c] < 0:
                        sampler[h,w,c] = 0

    writePPM('../Sampler_{}.ppm'.format(sample_num),sampler.astype(np.uint8))



main(1024,flag=True)
