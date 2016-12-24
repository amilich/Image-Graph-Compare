"""
Convert image to graph 

Andrew Milich 
December 2016 
"""

import struct
import scipy.misc
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def arr_to_img(arr, filename, mode, size): 
	new_im = Image.new(mode, size)
	new_im.putdata(arr)

def img_to_arr(img): 
	return list(im.getdata())

def get_neighbors(pt_index, pt_arr, width, height): 
	neighbors = [] 
	# TODO 
	return neighbors 

def hex_to_rgb(value):
	value = str(value)
	lv = len(value)
	return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def col_diff(rgb_1, rgb_2): 
	r1 = rgb_1[0]
	r2 = rgb_2[0]

	g1 = rgb_1[1]
	g2 = rgb_2[1]

	b1 = rgb_1[2]
	b2 = rgb_2[2]	
	return max(abs(r1 - r2), max(abs(b1 - b2), abs(g1 - g2)))

def set_edge(pt_arr, edge_arr, cent_rgb, x, y, edge_threshold): 
	for i in range(x - 1, x + 2): 
		for j in range(y - 1, y + 2): 
			curr_rgb = hex_to_rgb(pt_arr[i, j])
			if col_diff(cent_rgb, curr_rgb) > edge_threshold: 
				edge_arr[x, y] = 0xffffff
				return 
	return

def to_edges(pt_arr, edge_threshold = 50): 
	edge_arr = np.array(pt_arr.shape)

	for (x, y), value in np.ndenumerate(pt_arr): 
		cent_rgb = hex_to_rgb(pt_arr[x, y])
		set_edge(pt_arr, edge_arr, cent_rgb, x, y, edge_threshold)

	return 

if __name__ == '__main__':
	im = Image.open('res/cat.jpg')

	pixel_arr = img_to_arr(im)
	width, height = im.size
	pixel_arr = [pixel_arr[i * width:(i + 1) * width] for i in xrange(height)]
	hex_arr = np.zeros((height, width), dtype=np.int64) # rows, cols

	for i in range(len(pixel_arr)): 
		for j in range(len(pixel_arr[i])): 
			rgb = pixel_arr[i][j]
			hex_arr[i, j] = (rgb[0] << 16 & 0xff0000) + (rgb[1] << 8 & 0x00ff00) 
			+ (rgb[2] & 0x0000ff) & 0xffffff	
	hex_arr = to_edges(hex_arr)

	kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_arr)

	# scipy.misc.imshow(pixel_arr)
	scipy.misc.imsave("res/new.png", pixel_arr)

