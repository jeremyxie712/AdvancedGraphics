import sys
import os
import numpy as np
from PNM import *

import matplotlib.pyplot as pyplot
import matplotlib.colors as colors


# Implement a tone mapper
def tone_map(F, stops = 0, gamma = 2.1):
	# directly work on r, g, b image
	F *= 2**stops
	# Clip all values that are larger than 1
	F[F>1] = 1
	# implement gamma correction
	F **= (1/gamma)
	return F

# function for showing images
def PFMplot(F_, stops = 0, gamma=2.1, save_img = True, filename = 'Default'):
	F = F_.copy()
	# Process image to be viewable
	F = tone_map(F, stops = stops, gamma = gamma)
	F = (F*255.0).astype('uint8')
	pyplot.imshow(F)
	pyplot.show()
	if save_img:
		writePFM(filename+'.pfm', F_)
		writePPM(filename+'.ppm', F)

# Get Centroid
def GetCentroid(image):
	# Get the Median centroid's y axis(axis 0)
	axis_sum = image.sum(axis=1)
	weight = np.linspace(1, len(axis_sum), len(axis_sum))
	momentum_y = weight*axis_sum #Here we calculate the momentum and use it to find the centroid
	centroid_y = int(sum(momentum_y)/sum(axis_sum))
	# Get the centroid's x axis(axis 1)
	axis_sum = image.sum(axis=0)
	weight = np.linspace(1, len(axis_sum), len(axis_sum))
	momentum_x = weight*axis_sum
	centroid_x = int(sum(momentum_x)/sum(axis_sum))
	# Return the centroid
	return np.array([centroid_x, centroid_y])

# Get Median
def GetMedian(image):
	# Get the Median centroid's y axis(axis 0)
	axis_sum = image.sum(axis=1)
	if len(axis_sum) > 1:
		cum_sum = np.cumsum(axis_sum)
		median_y = np.argmax(cum_sum > cum_sum[-1]/2.0)
	else:
		median_y = 0
	# Get the centroid's x axis(axis 1)
	if len(axis_sum) > 1:
		axis_sum = image.sum(axis=0)
		cum_sum = np.cumsum(axis_sum)
		median_x = np.argmax(cum_sum > cum_sum[-1]/2.0)
	else:
		median_x = 0
	# Return the centroid
	return np.array([median_x, median_y])
# Median Cut
def GetMedianCut(sub_image, map_, median = None, curr_depth = 0, target_depth = 1):
	# GetMedianCut(sub_image):
	# input: sub_image: np.array
	# input: map_: np.array
	# input: centroid: None or the precalculated centroid for the sub_image
	# input: curr_depth: int
	# input: target_depth: int
	# output: centroids: np.array with shape (num_of_centroids, 2)
	if median is None:
		c_x, c_y = GetMedian(sub_image)
	else:
		c_x, c_y = median

	y_len, x_len = sub_image.shape
	cut_axis = y_len < x_len # Find the axis to perform median cut
	# partition the sub_image
	if cut_axis:
		img1 = sub_image[:,:c_x]
		map_1 = map_[:,:c_x,:]
		img2 = sub_image[:,c_x:]
		map_2 = map_[:,c_x:,:]
		line_of_cut = c_x
		#draw the line of cut
		map_[:,c_x-1:c_x+2,:] = 1
	else:
		img1 = sub_image[:c_y, :]
		map_1 = map_[:c_y,:,:]
		img2 = sub_image[c_y:, :]
		map_2 = map_[c_y:,:,:]
		line_of_cut = c_y
		#draw the line of cut
		map_[c_y-1:c_y+2,:,:]= 1
	# For the two sub_sub_image calculate its center_pixel position
	median_1 = GetMedian(img1)
	median_2 = GetMedian(img2)
	#centroid_1 = GetCentroid(img1)
	#centroid_2 = GetCentroid(img2)
	if curr_depth+1 == target_depth:
		#We add an offset to the sub_sub_image to recover the real position \
		# of its centroid in the image frame.
		median_2[int(not cut_axis)] += line_of_cut
		return np.vstack([median_1, median_2])
		#centroid_2[int(not cut_axis)] += line_of_cut
		#return np.vstack([centroid_1, centroid_2])
	else:
		medians_1 = GetMedianCut(img1, map_1, median=median_1, curr_depth = curr_depth+1, target_depth = target_depth)
		#centroids_1 = GetMedianCut(img1, map_1, median=centroid_1, curr_depth = curr_depth+1, target_depth = target_depth)
		medians_2 = GetMedianCut(img2, map_2, median=median_2, curr_depth = curr_depth+1, target_depth = target_depth)
		#centroids_2 = GetMedianCut(img2, map_2, median=centroid_2, curr_depth = curr_depth+1, target_depth = target_depth)
		# Add the shift to centroids_2
		medians_2[:,int(not cut_axis)] += line_of_cut
		return np.vstack([medians_1, medians_2])
		#return np.vstack([centroids_1, centroids_2])

# Plot Centroids on map_
def GetBoundingBox(centroid, max_x, max_y, size = 9):
	# This function returns the bounding box with the given size of a centroid
	top_y = max(0, centroid[1]-(size-1)//2)
	bottom_y = min(max_y-1, centroid[1]+(size-1)//2)
	leftmost_x = max(0, centroid[0]-(size-1)//2)
	rightmost_x = min(max_x-1, centroid[0]+(size-1)//2)
	return top_y, bottom_y, leftmost_x, rightmost_x
def PlotCentroids(map_, dict_, size = 9):
	# Get the boundaries of the map_
	max_y, max_x,_ = map_.shape
	for centroid in dict_:
		top_y, bottom_y, leftmost_x, rightmost_x \
		 			= GetBoundingBox(centroid, max_x, max_y, size = size)
		# Draw the top vertex
		map_[top_y:top_y+2,leftmost_x:rightmost_x, :] = np.array([0,0,1])
		# Draw the bottom vertex
		map_[bottom_y-2:bottom_y,leftmost_x:rightmost_x, :] = np.array([0,0,1])
		# Similarly draw the left and right vertices
		map_[top_y:bottom_y,leftmost_x:leftmost_x+2, :] = np.array([0,0,1])
		map_[top_y:bottom_y,rightmost_x-2:rightmost_x, :] = np.array([0,0,1])
def GetLightSources(map_, dict_, size = 5):
	background_map = np.zeros(map_temp.shape, dtype = np.float32)
	max_y, max_x,_ = map_.shape
	for centroid in dict_:
		top_y, bottom_y, leftmost_x, rightmost_x \
		 			= GetBoundingBox(centroid, max_x, max_y, size = size)
		background_map[top_y:bottom_y, leftmost_x:rightmost_x, :] \
					= map_[top_y:bottom_y, leftmost_x:rightmost_x, :]
	return background_map

if '__main__' == __name__:
	# Create directory for storing images
	if not os.path.exists('../part3img'):
		os.makedirs('../part3img')

	# Load the lat long map, size = 512x1024x3
	map_ =  loadPFM('../GraceCathedral/grace_latlong.pfm')
	# Get the intensity of each pixel, size = 512x1024
	unscaled_I = np.average(map_, axis=2)
	# Now create the sin matrix for scaling
	scale_sin = np.sin(np.linspace(0, np.pi, 512).reshape(-1, 1))
	scale_sin_matrix = np.repeat(scale_sin, 1024, axis=1) # Now scale_sin_matrix have size 512x1024
	# Get the intensity
	I = unscaled_I*scale_sin_matrix
	######################################
	#Define the log2 number of partitions
	log_2_partitions = list(range(1,7))
	for depth in log_2_partitions:
		map_temp = map_.copy()
		dict_ = GetMedianCut(I, map_temp, target_depth = depth)
		PlotCentroids(map_temp, dict_, size = 9)
		### Now draw the results
		PFMplot(map_temp, save_img = True, filename = '../part3img/EM{}'.format(2**depth))
		#dict_
	# Now specifically draw the pixels representing the lightsource
	map_temp = map_.copy()
	light_source = GetLightSources(map_temp, dict_, size = 5)
	PFMplot(light_source, save_img = True, filename = '../part3img/lightsource')
