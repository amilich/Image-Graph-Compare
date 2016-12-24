"""
Convert image to graph 

Andrew Milich 
December 2016 
"""

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

def to_edges(pt_arr, edge_threshold = 50): 
	for tup in pt_arr: 
		for pt in get_neighbors(tup): 
			pass
	return 

if __name__ == '__main__':
	im = Image.open('res/cat.jpg')

	pixel_arr = img_to_arr(im)
	width, height = im.size
	pixel_arr = [pixel_arr[i * width:(i + 1) * width] for i in xrange(height)]
	hex_arr = np.zeros((height, width)) # rows, cols
	
	print hex_arr
	print hex_arr.shape

	for i in pixel_arr: 
		print len(i)
		for j in i: 
			hex_arr[i, j] = (j[0] << 16 & 0xff0000) + (j[1] << 8 & 0x00ff00) 
			+ (j[2] & 0x0000ff) & 0xffffff	
	pixel_arr = to_edges(pixel_arr)

	kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_arr)

	# scipy.misc.imshow(pixel_arr)
	scipy.misc.imsave("res/new.png", pixel_arr)

