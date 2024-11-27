'''
Evaluation code

    Base algorithm: Filtering algorithm + Clustering using DBScan 
    10617 project: Filtering - Grace and Conrad, Intensity Based evaluation 
    
    Evaluated based on accuracy and runtime

'''
import math
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import cluster
import time

### Filtering Algorithm - G&C ###

'''
Data preprocessing - remove points clouds outside given FOV range
    - PC is only PC[:, :3] 
    - Remove radial point 0 - 40 
'''

def fov_range(pointcloud, minradius=0, maxradius=40):
    # Calculate radial distance of each point (straight line distance from origin) and removes if outside range
    pointcloud = pointcloud[:,:3]
    points_radius = np.sqrt(np.sum(pointcloud[:,:2] ** 2, axis=1))

    # Uses mask to remove points to only store in-range points
    radius_mask = np.logical_and(
        minradius <= points_radius, 
        points_radius <= maxradius
    )
    pointcloud = pointcloud[radius_mask]

    return pointcloud


'''
G&C Filtering Algorithm 
    - Split point into M segments based on angle 
    - Split each segment into rbins based on radial distance of each point 
    - Fit a line to each segment 
    - Any point below the line is the 'ground' and is filtered
'''

alpha = 0.1
num_bins = 10
height_threshold = 0.13

def grace_and_conrad_filtering(points, alpha, num_bins, height_threshold):

    angles = np.arctan2(points[:, 1], points[:, 0])  # Calculate angle for each point
    bangles = np.where(angles < 0, angles + 2 * np.pi, angles)

    # NOTE: making gangles from min to max to avoid iterating over empty regions
    if (bangles.size > 0): 
        
        gangles = np.arange(np.min(bangles), np.max(bangles), alpha)

        # Map angles to segments
        segments = np.digitize(bangles, gangles) - 1 
        # Calculate range for each point
        ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2) 

        rmax = np.max(ranges)
        rmin = np.min(ranges)
        bin_size = (rmax - rmin) / num_bins
        rbins = np.arange(rmin, rmax, bin_size)
        regments = np.digitize(ranges, rbins) - 1

        M, N = len(gangles), len(rbins)
        grid_cell_indices = segments * N + regments

        gracebrace = []
        for seg_idx in range(M):
            Bines = []
            min_zs = []
            prev_z = None
            for range_idx in range(N):
                bin_idx = seg_idx * N + range_idx
                idxs = np.where(grid_cell_indices == bin_idx)
                bin = points[idxs, :][0]
                if bin.size > 0:
                    min_z = np.min(bin[:, 2])
                    binLP = bin[bin[:, 2] == min_z][0].tolist()
                    min_zs.append(min_z)
                    Bines.append([np.sqrt(binLP[0]**2 + binLP[1]**2), binLP[2]])
                    prev_z = min_z

            if Bines:
                i = 0
                while i < len(min_zs):
                    good_before = i == 0 or min_zs[i] - min_zs[i - 1] < 0.1
                    good_after = i == len(min_zs) - 1 or min_zs[i] - min_zs[i + 1] < 0.1
                    if not (good_before and good_after):
                        Bines.pop(i)
                        min_zs.pop(i)
                        i -= 1
                    i += 1

                seg = segments == seg_idx
                X = [p[0] for p in Bines]
                Y = [p[1] for p in Bines]
                
                X = np.array(X)
                Y = np.array(Y)

                x_bar = np.mean(X)
                y_bar = np.mean(Y)
                x_dev = X - x_bar
                y_dev = Y - y_bar
                ss = np.sum(x_dev * x_dev)

                slope = np.sum(x_dev * y_dev) / np.sum(x_dev * x_dev) if ss != 0 else 0
                intercept = y_bar - slope * x_bar
                
                points_seg = points[seg]
                pc_compare = slope * np.sqrt(points_seg[:, 0]**2 + points_seg[:, 1]**2) + intercept
                pc_mask = (pc_compare + height_threshold) < points_seg[:, 2]
                conradbonrad = points_seg[pc_mask]
                if conradbonrad.tolist(): gracebrace.extend(conradbonrad.tolist())
     
        gracebrace = np.array(gracebrace)
        return gracebrace.reshape((-1, 3))


### Filtering Algorithm - Intensity Based ###

# NOTE: Need x,y,z,r data 
def intensity_filtering(points, max_dist=40):
    
    # Calculate radial distance
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2) 
    
    # Thresholding for intensity
    mask = (points[:, 3]*(r**2) > 800) & (r < max_dist)
    filtered_points = points[mask]
    print(mask)
    
    return np.array(filtered_points)


### Clustering Algorithm - HDBSCAN ###

'''
HDBSCAN Clustering Algorithm 
    - Performs clustering using HDBSCAN 
    - Predicts cone centers 
    - Outputs (x,y,z) center position of each cone
'''

'''
    Runs HDBSCAN on points
    clusterer.labels_ - each element is the cluster label. Noisy points assigned -1
'''
def run_dbscan(points, eps=0.5, min_samples=1):
    clusterer = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(points)

    return clusterer

'''
    Returns the centroid for each cluster 
    Note: no additional filtering performed on size of cluster - i.e. using radial distance or ground plane
'''
def get_centroids_z(points, labels, scalar=1):
  
    points = np.zeros(points.shape) + points[:, :3]
    # Scales z-axis by scalar 
    points[:, 2] *= scalar
    
    
    n_clusters = np.max(labels) + 1
    centroids = []

    # Default probability of 1 to each point 
    probs = np.ones((points.shape[0], 1))

    # Iterate over each cluster 
    for i in range(n_clusters):
      
        # Extract points that belong to n_clusters[i]
        idxs = np.where(labels == i)[0]
        cluster_points = points[idxs]

        # Weighted average center for each cluster of points 
        cluster_probs = probs[idxs]
        scale = np.sum(cluster_probs)
        center = np.sum(cluster_points * cluster_probs, axis=0) / scale

        centroids.append(center)
        
        # NOTE: No additional filtering performed on size of cluster - i.e. using radial distance or ground plane
       

    return np.array(centroids)

'''
    Main code for cone clustering
'''
def predict_cones_z(points): 
    
    if points.shape[0] == 0:
        return np.zeros((0, 3))

    points = points[:, :3]
    zmax = (points[:, 2] + 1).max(axis=0)
    endscal = (abs(zmax))
    points[:, 2] /= endscal

    # Run DBSCAN - returns probabilities 
    clusterer = run_dbscan(points, min_samples=2, eps=0.3)
    labels = clusterer.labels_.reshape((-1, 1))

    # Extracts centroids for each cluster 
    centroids = get_centroids_z(points, labels)

    return centroids.reshape((-1, 3))


### Utils ###

'''
Timer class
'''
class Timer:
    def __init__(self):
        self.data = dict()

    def start(self, timer_name):
        self.data[timer_name] = [time.time(), 0]

    def end(self, timer_name, msg="", ret=False):
        self.data[timer_name][1] = time.time()

        if not ret:
            print(f"{timer_name}: {(self.data[timer_name][1] - self.data[timer_name][0]) * 1000:.2f} ms {msg}")
        else:
            return round((self.data[timer_name][1] - self.data[timer_name][0]) * 1000, 3)
    
    def total(self):
        total = 0.0
        for section in self.data: 
            total += (self.data[section][1] - self.data[section][0])
        total = round(total*1000, 3)
        print(f"Total time: {total} ms")
        return total

'''
Visualisation Code
'''
def visualise(points_unfiltered, point_filtered):
    fig = plt.figure()

    # First frame
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim([0, 40])
    ax1.set_ylim([-20, 20])
    ax1.set_zlim([-3, 1])
    ax1.scatter(points_unfiltered[:, 0], points_unfiltered[:, 1], points_unfiltered[:, 2], s=1)
    # ax1.view_init(elev=90, azim=0)
    ax1.set_title("Points Pre-processed")

    # Second frame
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim([0, 40])
    ax2.set_ylim([-20, 20])
    ax2.set_zlim([-3, 1])
    ax2.scatter(point_filtered[:, 0], point_filtered[:, 1], point_filtered[:, 2], s=1)
    # ax2.view_init(elev=90, azim=0)
    ax2.set_title("Points Post-processed")

    plt.show()
        


### MAIN RUN ###

'''
Base Algorithm - G&C
'''
def run_g_and_c(): 
    
    # Load in points from real data frame
    folder_path = "instance-1501.npz"

    with np.load(folder_path, allow_pickle=True) as data:
        single_frame = data['points']
    
    print()
    timer = Timer()
    # Process data 
    timer.start("data_processing")
    processed_data_frame = fov_range(single_frame, 0, 40)
    timer.end("data_processing", "Completed Data Processing")
    print("Pre-processed - Num of points:", len(single_frame))
    print("Post-processed - Num of points:", len(processed_data_frame))
    print()
    # visualise(single_frame, processed_data_frame)

    # Ground Filtering 
    timer.start("ground_filtering")
    ground_filtered_points = grace_and_conrad_filtering(processed_data_frame, alpha, num_bins, height_threshold)
    timer.end("ground_filtering", "Completed Ground Filtering")
    print("Ground Filtered - Num of points:", len(ground_filtered_points))
    print()
    # visualise(single_frame, ground_filtered_points)

    # Clustering 
    timer.start("clustering")
    clusters = predict_cones_z(ground_filtered_points)
    timer.end("clustering", "Completed Clustering")
    print("Clustered - Num of points:", len(clusters))
    print()
    # visualise(ground_filtered_points, clusters)
    timer.total()
    print()

'''
Comparative Algorithm - Intensity Based
'''
def run_intensity():
    
    # Load in points from real data frame
    folder_path = "instance-1501.npz"

    with np.load(folder_path, allow_pickle=True) as data:
        single_frame = data['points']
        
    print()
    timer = Timer()
    # Process data 
    timer.start("data_processing")
    processed_data_frame = fov_range(single_frame, 0, 40)
    timer.end("data_processing", "Completed Data Processing")
    print("Pre-processed - Num of points:", len(single_frame))
    print("Post-processed - Num of points:", len(processed_data_frame))
    print()
    # visualise(single_frame, processed_data_frame)

    # Ground Filtering 
    timer.start("ground_filtering")
    ground_filtered_points = intensity_filtering(processed_data_frame, 40)
    timer.end("ground_filtering", "Completed Ground Filtering")
    print("Ground Filtered - Num of points:", len(ground_filtered_points))
    print()
    # visualise(single_frame, ground_filtered_points)

    # Clustering 
    timer.start("clustering")
    clusters = predict_cones_z(ground_filtered_points)
    timer.end("clustering", "Completed Clustering")
    print("Clustered - Num of points:", len(clusters))
    print()
    # visualise(ground_filtered_points, clusters)
    timer.total()
    print()


# NOTE: Need x,y,z,r data
# Main run
# run_intensity()