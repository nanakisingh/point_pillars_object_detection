# Import libraries 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import KDTree
import numpy as np
np.set_printoptions(suppress=True)

'''
    Performs data preprocessing by removing point clouds outside a given 
    FOV range. Extracts first three coordinates of points only 
     
    @param [in] pointcloud, minradius, maxradius
    @param [out] processed_pointcloud
'''

def fov_range(pointcloud, minradius=0, maxradius=20):
    # Calculate radial distance of each point (straight line distance from origin) and removes if outside range
    pointcloud = pointcloud[:,:3]
    points_radius = np.sqrt(np.sum(pointcloud[:,:2] ** 2, axis=1))

    # Uses mask to remove points to only store in-range points
    radius_mask = np.logical_and(
        minradius <= points_radius, 
        points_radius <= maxradius
    )
    processed_pointcloud = pointcloud[radius_mask]
    
    processed_pointcloud = processed_pointcloud[:, [1,0,2]]

    return processed_pointcloud


'''
    Generates a psuedoimage of spatial dimensions (h_bins, w_bins), where 
    each bin has size (h_len, w_len)
    
    Splits elements into bins using radial distance, converts each points into
    9 dimensional vector, and performs max poling so each bin is represented as
    a 9 dimensional vector
    
    @param [in] points, h_bins, w_bins, h_len, w_len
    @param [out] (H,W,9) dimensions psuedoimage 
'''

# First version: 9 dim vector 
def generate_psuedoimage(points, h_bins, w_bins, h_len, w_len):
    
    
    x_vals = [point[0] for point in points]
    y_vals = [point[1] for point in points]
    z_vals = [point[2] for point in points]
        
    # Split points by (x,y) into grids 
    h_binsize = (h_len/h_bins)
    w_binsize = (w_len/w_bins)
    
    mid = (w_len//2)

    h_bin_ind = np.arange(0, h_bins)
    w_bin_ind = np.arange(0, w_bins)
    
    dict_bin = dict()

    # # Initialise dict_bin 

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            dict_bin[(i,j)] = []

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            lowerbound_x = i*w_binsize - mid
            lowerbound_y = j*h_binsize
            
            upperbound_x = i*w_binsize + w_binsize - mid
            upperbound_y = j*h_binsize + h_binsize
            
            for k in range(len(x_vals)):
                x_new = x_vals[k]
                y_new = y_vals[k]
                z_new = z_vals[k]
                
                if (lowerbound_x < x_new <= upperbound_x) and (lowerbound_y < y_new <= upperbound_y):
                    dict_bin[(i,j)] = dict_bin.get((i,j), [])+[[x_new, y_new, z_new]]

    # Iterate through each bin 
    for bin in dict_bin:
        elements = np.array(dict_bin[bin])
        if elements.size > 0:
        
            # First: find mean 
            mean_x = np.round(np.sum(elements[:, 0])/elements.shape[0], 2)
            mean_y = np.round(np.sum(elements[:, 1])/elements.shape[0], 2)
            mean_z = np.round(np.sum(elements[:, 2])/elements.shape[0], 2)
            mean_array = np.array([mean_x, mean_y, mean_z])
            
            # Convert point into x,y,z, xp, yp, zp,xc, yc, zc
            diff_p = abs(elements-mean_array)
            output = np.hstack((elements, diff_p))

            # Max pooling across rows 
            max_values = np.round(np.max(output, axis=0), 2)
            
            # Elements order: x, y, z, xp, yp, zp, xc, yc, zc
            all_values = np.hstack((max_values, mean_array))
            
            dict_bin[bin] = all_values
        else: 
            dict_bin[bin] = np.zeros(9)
    
    print(dict_bin)
    
    # Generate Psuedoimage
    psuedoimage = np.zeros((h_bins, w_bins, 9))

    for i in range(psuedoimage.shape[0]):
        for j in range(psuedoimage.shape[1]):
            psuedoimage[i,j] = dict_bin[(i,j)]
 
    return psuedoimage

# Second version: mean_x, mean_y, mean_z, max_z, min_z, variance
def generate_psuedoimage_2(points, h_bins, w_bins, h_len, w_len):

    x_vals = [point[0] for point in points]
    y_vals = [point[1] for point in points]
    z_vals = [point[2] for point in points]
        
    # Split points by (x,y) into grids 
    h_binsize = (h_len/h_bins)
    w_binsize = (w_len/w_bins)
    
    mid = (w_len//2)

    h_bin_ind = np.arange(0, h_bins)
    w_bin_ind = np.arange(0, w_bins)
    
    dict_bin = dict()

    # # Initialise dict_bin 

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            dict_bin[(i,j)] = []

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            lowerbound_x = i*w_binsize - mid
            lowerbound_y = j*h_binsize
            
            upperbound_x = i*w_binsize + w_binsize - mid
            upperbound_y = j*h_binsize + h_binsize
            
            for k in range(len(x_vals)):
                x_new = x_vals[k]
                y_new = y_vals[k]
                z_new = z_vals[k]
                
                if (lowerbound_x < x_new <= upperbound_x) and (lowerbound_y < y_new <= upperbound_y):
                    dict_bin[(i,j)] = dict_bin.get((i,j), [])+[[x_new, y_new, z_new]]

    # Iterate through each bin 
    for bin in dict_bin:
        elements = np.array(dict_bin[bin])
        if elements.size > 0:
            
            # First: find mean 
            mean_x = np.round(np.sum(elements[:, 0])/elements.shape[0], 2)
            mean_y = np.round(np.sum(elements[:, 1])/elements.shape[0], 2)
            mean_z = np.round(np.sum(elements[:, 2])/elements.shape[0], 2)
            # Second: lowest and highest z 
            min_z = np.round(np.min(elements[:, 2], axis=0), 2)
            max_z = np.round(np.max(elements[:, 2], axis=0), 2)
            
            # Third: variance 
            variance =  np.round(np.var(elements[:, 2], axis=0),5)
            
            all_values = np.array([mean_x, mean_y, mean_z, max_z, min_z, variance])
            
            dict_bin[bin] = all_values
        else: 
            dict_bin[bin] = np.zeros(6)
    
    print(dict_bin)
    
    # Generate Psuedoimage
    psuedoimage = np.zeros((h_bins, w_bins, 6))

    for i in range(psuedoimage.shape[0]):
        for j in range(psuedoimage.shape[1]):
            psuedoimage[i,j] = dict_bin[(i,j)]
            
    # Fill pusedoimage with nearest neighbor points 
    # psuedoimage = fill_psuedoimage(psuedoimage, h_bins, w_bins)
    # print(psuedoimage)
            
    return psuedoimage

# Third version: 9 dim vector + variance 
def generate_psuedoimage_3(points, h_bins, w_bins, h_len, w_len):
    
    
    x_vals = [point[0] for point in points]
    y_vals = [point[1] for point in points]
    z_vals = [point[2] for point in points]
        
    # Split points by (x,y) into grids 
    h_binsize = (h_len/h_bins)
    w_binsize = (w_len/w_bins)
    
    mid = (w_len//2)

    h_bin_ind = np.arange(0, h_bins)
    w_bin_ind = np.arange(0, w_bins)
    
    dict_bin = dict()

    # # Initialise dict_bin 

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            dict_bin[(i,j)] = []

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            lowerbound_x = i*w_binsize - mid
            lowerbound_y = j*h_binsize
            
            upperbound_x = i*w_binsize + w_binsize - mid
            upperbound_y = j*h_binsize + h_binsize
            
            for k in range(len(x_vals)):
                x_new = x_vals[k]
                y_new = y_vals[k]
                z_new = z_vals[k]
                
                if (lowerbound_x < x_new <= upperbound_x) and (lowerbound_y < y_new <= upperbound_y):
                    dict_bin[(i,j)] = dict_bin.get((i,j), [])+[[x_new, y_new, z_new]]

    # Iterate through each bin 
    for bin in dict_bin:
        elements = np.array(dict_bin[bin])
        if elements.size > 0:
        
            # First: find mean 
            mean_x = np.round(np.sum(elements[:, 0])/elements.shape[0], 2)
            mean_y = np.round(np.sum(elements[:, 1])/elements.shape[0], 2)
            mean_z = np.round(np.sum(elements[:, 2])/elements.shape[0], 2)
            mean_array = np.array([mean_x, mean_y, mean_z])
            
            # Convert point into x,y,z, xp, yp, zp,xc, yc, zc
            diff_p = abs(elements-mean_array)
            output = np.hstack((elements, diff_p))

            # Max pooling across rows 
            max_values = np.round(np.max(output, axis=0), 2)
            
            # Variance 
            variance =  np.round(np.var(elements[:, 2], axis=0),5)
            
            # Elements order: x, y, z, xp, yp, zp, xc, yc, zc, var
            all_values = np.hstack((max_values, mean_array))
            all_values = np.hstack((all_values, variance))
            
            dict_bin[bin] = all_values
        else: 
            dict_bin[bin] = np.zeros(10)
    
    
    # Generate Psuedoimage
    psuedoimage = np.zeros((h_bins, w_bins, 10))

    for i in range(psuedoimage.shape[0]):
        for j in range(psuedoimage.shape[1]):
            psuedoimage[i,j] = dict_bin[(i,j)]
            
 
            
    return psuedoimage

# Fourth version: mean_z, max_z, min_z, variance
def generate_psuedoimage_4(points, h_bins, w_bins, h_len, w_len):

    x_vals = [point[0] for point in points]
    y_vals = [point[1] for point in points]
    z_vals = [point[2] for point in points]
        
    # Split points by (x,y) into grids 
    h_binsize = (h_len/h_bins)
    w_binsize = (w_len/w_bins)
    
    mid = (w_len//2)

    h_bin_ind = np.arange(0, h_bins)
    w_bin_ind = np.arange(0, w_bins)
    
    dict_bin = dict()

    # # Initialise dict_bin 

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            dict_bin[(i,j)] = []

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            lowerbound_x = i*w_binsize - mid
            lowerbound_y = j*h_binsize
            
            upperbound_x = i*w_binsize + w_binsize - mid
            upperbound_y = j*h_binsize + h_binsize
            
            for k in range(len(x_vals)):
                x_new = x_vals[k]
                y_new = y_vals[k]
                z_new = z_vals[k]
                
                if (lowerbound_x < x_new <= upperbound_x) and (lowerbound_y < y_new <= upperbound_y):
                    dict_bin[(i,j)] = dict_bin.get((i,j), [])+[[x_new, y_new, z_new]]

    # Iterate through each bin 
    for bin in dict_bin:
        elements = np.array(dict_bin[bin])
        if elements.size > 0:
            
            # First: find mean 
            mean_z = np.round(np.sum(elements[:, 2])/elements.shape[0], 2)
            # Second: lowest and highest z 
            min_z = np.round(np.min(elements[:, 2], axis=0), 2)
            max_z = np.round(np.max(elements[:, 2], axis=0), 2)
            
            # Third: variance 
            variance =  np.round(np.var(elements[:, 2], axis=0),5)
            
            all_values = np.array([mean_z, max_z, min_z, variance])
            
            dict_bin[bin] = all_values
        else: 
            dict_bin[bin] = np.zeros(4)
        
    # Generate Psuedoimage
    psuedoimage = np.zeros((h_bins, w_bins, 4))

    for i in range(psuedoimage.shape[0]):
        for j in range(psuedoimage.shape[1]):
            psuedoimage[i,j] = dict_bin[(i,j)]
            
    # Fill pusedoimage with nearest neighbor points 
    # psuedoimage = fill_psuedoimage(psuedoimage, h_bins, w_bins)
    # print(psuedoimage)
            
    return psuedoimage

# Fifth version: 5 random points 
def generate_psuedoimage_5(points, h_bins, w_bins, h_len, w_len):
    
    
    x_vals = [point[0] for point in points]
    y_vals = [point[1] for point in points]
    z_vals = [point[2] for point in points]
        
    # Split points by (x,y) into grids 
    h_binsize = (h_len/h_bins)
    w_binsize = (w_len/w_bins)
    
    mid = (w_len//2)

    h_bin_ind = np.arange(0, h_bins)
    w_bin_ind = np.arange(0, w_bins)
    
    dict_bin = dict()

    # # Initialise dict_bin 

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            dict_bin[(i,j)] = []

    for i in range(len(h_bin_ind)):
        for j in range(len(w_bin_ind)):
            lowerbound_x = i*w_binsize - mid
            lowerbound_y = j*h_binsize
            
            upperbound_x = i*w_binsize + w_binsize - mid
            upperbound_y = j*h_binsize + h_binsize
            
            for k in range(len(x_vals)):
                x_new = x_vals[k]
                y_new = y_vals[k]
                z_new = z_vals[k]
                
                if (lowerbound_x < x_new <= upperbound_x) and (lowerbound_y < y_new <= upperbound_y):
                    dict_bin[(i,j)] = dict_bin.get((i,j), [])+[[x_new, y_new, z_new]]

    # number_of_points
    num_points = 10
    # Iterate through each bin 
    for bin in dict_bin:
        elements = np.array(dict_bin[bin])
        if elements.shape[0] > num_points:
            random_indices = np.random.randint(low=0, high=(elements.shape[0]-1), size=num_points)
            random_elements = elements[random_indices]
            
            dict_bin[bin] = np.ndarray.flatten(random_elements)
        else: 
            
            dict_bin[bin] = np.ndarray.flatten(np.zeros((num_points, 3)))
    
    
    # Generate Psuedoimage
    psuedoimage = np.zeros((h_bins, w_bins, num_points*3))

    for i in range(psuedoimage.shape[0]):
        for j in range(psuedoimage.shape[1]):
            psuedoimage[i,j] = dict_bin[(i,j)]

    return psuedoimage
