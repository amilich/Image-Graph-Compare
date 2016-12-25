"""
Convert image to graph 

Andrew Milich 
December 2016 
"""

import struct
import scipy.misc
import numpy as np
import math
from sklearn.cluster import KMeans
from PIL import Image

def arr_to_img(arr, filename, mode, size): 
	new_im = Image.new(mode, size)
	new_im.putdata(arr)

def img_to_arr(im): 
	return list(im.getdata())

def hex_to_rgb(value):
	value = "%06x" % value
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
			if i >= pt_arr.shape[0] or j >= pt_arr.shape[1]:
				continue
			curr_rgb = hex_to_rgb(pt_arr[i, j])
			if col_diff(cent_rgb, curr_rgb) > edge_threshold: 
				edge_arr[x, y] = 0xffffff
				return 
	return

def to_edges(pt_arr, edge_threshold = 70): 
	edge_arr = np.zeros(pt_arr.shape)

	for (x, y), value in np.ndenumerate(pt_arr): 
		cent_rgb = hex_to_rgb(pt_arr[x, y])
		set_edge(pt_arr, edge_arr, cent_rgb, x, y, edge_threshold)

	return edge_arr

def pt_dist(x1, y1, x2, y2): 
	return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def create_img_adj_mat(filename):
	im = Image.open(filename)
	pixel_arr = img_to_arr(im)
	width, height = im.size
	pixel_arr = [pixel_arr[i * width:(i + 1) * width] for i in xrange(height)]
	hex_arr = np.zeros((height, width), dtype=np.int64) # rows, cols

	for i in range(len(pixel_arr)): 
		for j in range(len(pixel_arr[i])): 
			rgb = pixel_arr[i][j]
			hex_arr[i, j] = (rgb[0] << 16 & 0xff0000) + (rgb[1] << 8 & 0x00ff00) 
			+ (rgb[2] & 0x0000ff) & 0xffffff	

	print("Creating edges for {}".format(filename))
	hex_arr = to_edges(hex_arr)
	print("Clustering")
	cluster_pts = []
	for (x, y), value in np.ndenumerate(hex_arr): 
		if value is not 0: 
			cluster_pts.append((x, y))

	num_clusters = 10
	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(cluster_pts)
	centers = kmeans.cluster_centers_
	print(centers)
	adj_mat = np.zeros((num_clusters, num_clusters))
	for x in range(num_clusters): 
		for y in range(num_clusters): 
			adj_mat[x, y] = pt_dist(centers[x][0], centers[x][1], 
									centers[y][0], centers[y][1])
	return adj_mat

def euc_distance(mat1, mat2): 
	mat = (np.asmatrix((mat1 - mat2)) * np.asmatrix((mat1 - mat2)).H)
	return math.sqrt(float(np.trace(mat)))

if __name__ == '__main__':
	adj_mat_1 = create_img_adj_mat('res/k1.jpg')
	adj_mat_2 = create_img_adj_mat('res/k2.jpg')
	adj_mat_3 = create_img_adj_mat('res/t1.jpg')
	print euc_distance(adj_mat_1, adj_mat_2)
	print euc_distance(adj_mat_2, adj_mat_3)
	print euc_distance(adj_mat_1, adj_mat_3)
	# adj_mat_2 = create_img_adj_mat('res/cat2.jpg')
	# scipy.misc.imsave("res/new.jpg", hex_arr.tolist())

