'''
    Evaluating Baseline Model
'''

import numpy as np 
import os 
import json
import math
from sklearn import cluster
import matplotlib.pyplot as plt

'''
    Intensity Algorithm
'''

def itensity_filtering(points):
    
    points = points[~np.all(points[:, :3] == [0, 0, 0], axis=1)]
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    r_mask = (points[:, 3] * r  >= 150)
    # np.log(r)
    
    points[~r_mask, :3] = 0
    points = points[:, :3]
    points = points[:, [1, 0, 2]] 
    points[:, 0] = -points[:, 0] 
    points = points[~np.all(points == 0, axis=1)]
    
    points = points[:, [1, 0, 2]]
    points[:, 1] = -points[:, 1]
    
    return points
    

'''
    Data preprocessing - remove points clouds outside given FOV range
'''

def fov_range(pointcloud, minradius=0, maxradius=20):
    pointcloud = pointcloud[:,:4]
    points_radius = np.sqrt(np.sum(pointcloud[:,:2] ** 2, axis=1))

    radius_mask = np.logical_and(
        minradius <= points_radius, 
        points_radius <= maxradius
    )
    pointcloud = pointcloud[radius_mask]
    return pointcloud

''' 
    HDBSCAN Clustering Algorithm 
'''

def run_dbscan(points, eps=0.5, min_samples=1):
    clusterer = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(points)

    return clusterer

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

'''
    Baseline Filtering Algorithm (i.e. Ground Truth)
'''

def grace_and_conrad_filtering(points):
    alpha = 0.1
    num_bins = 10
    height_threshold = 0.13
    
    # change so that take lowest x points - averga eand fit a plane across all segments 

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

def baseline_intensity_filtering(points, max_dist=20,  g_and_c=True):
    

    points = points[~np.all(points[:, :3] == [0, 0, 0], axis=1)]

    #array of r
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    #range mask
    mask_close = (r < 10)
    mask_far = (r > 10) & (r < 30)

    # Split the data into 2 arrays based on the masks
    close_points = points[mask_close]     # Points with 0 < r < 10
    far_points = points[mask_far]   # Points with 10 < r < 50
    far_r = np.sqrt(far_points[:, 0]**2 + far_points[:, 1]**2)

    #far points ground removal & transformation
    r_mask = (far_points[:, 3] * far_r * np.log(far_r) >= 200)
    far_points[~r_mask, :3] = 0
    far_points = far_points[:, :3]
    far_points = far_points[:, [1, 0, 2]] 
    far_points[:, 0] = -far_points[:, 0] 
    far_points = far_points[~np.all(far_points == 0, axis=1)]
    
    if g_and_c: 
        
        # print("CLOSE ", close_points.shape)
        close_points_t = grace_and_conrad_filtering(close_points[:, :3])
        
        
        close_points_t = close_points_t[:, [1, 0, 2]]
        close_points_t[:, 0] = -close_points_t[:, 0] 
        points = np.concatenate((close_points_t , far_points), axis=0)
        points = points[:, [1, 0, 2]]
        points[:, 1] = -points[:, 1]
        return points 
    
    #close points removal & transformation
    threshold = .1
    close_points = close_points[:, :3]
    close_points = close_points[:, [1, 0, 2]] 
    close_points[:, 0] = -close_points[:, 0] 
    random_selection = close_points[np.random.choice(close_points.shape[0], 300, replace=False)]
    sorted_selection = random_selection[random_selection[:, 2].argsort()]
    remaining_points = sorted_selection[:-5]
    lowest_z_points = remaining_points
    X = lowest_z_points[:, 0]  
    Y = lowest_z_points[:, 1]  
    Z = lowest_z_points[:, 2] 
    A = np.vstack([X, Y, np.ones_like(X)]).T  #
    B = Z  
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, d = coeffs

    plane_z_values = close_points[:, 0] * a + close_points[:, 1] * b + d
    plane_mask = close_points[:, 2] >= plane_z_values + threshold #
    close_points = close_points[plane_mask]

    points = np.concatenate((close_points, far_points), axis=0)
    points = points[:, [1, 0, 2]]
    points[:, 1] = -points[:, 1]
    return points  
    

'''
    Determine accuracy of predictions
'''

def evaluate_clusters(clusters, predicted_points, MoE=1.2):
    # print("lenghts", len(clusters), len(predicted_points))
    results = []
    successful_predictions = 0
    
    # clusters = clusters[:, :2]
    # predicted_points = predicted_points[:, :2]
    
    for cluster in clusters: 
        distances = np.linalg.norm(predicted_points - cluster, axis=1)
        within_moe = distances <= MoE
        
        if np.any(distances <= MoE):
            successful_predictions += 1
            
            results.append(np.any(within_moe))
    
    accuracy = (successful_predictions / len(clusters)) * 100
    return accuracy, results

'''
    Visualise
'''
def visualise_3(true_points):
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 30])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-3, 3])
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], s=1)

    plt.show()
    
    
'''
    Main
'''

folder_path = '/Users/nanaki/Desktop/10617/Final project/evaluation_baseline/actual_data'
data_folder = os.listdir(folder_path)
data_folder.sort()
data_folder = data_folder[1:]

model_folder_path = '/Users/nanaki/Desktop/10617/Final project/evaluation_baseline/model_predictions'
model_predictions = os.listdir(model_folder_path)
model_predictions.sort()
model_predictions = model_predictions[1:]

basline_acc = []
model_acc = []

moe_values = np.arange(0.4, 1.2, 0.05)
baseline_acc_mean = 0.0
model_acc_mean = 0.0

for j in range(len(moe_values)):
    for i in range(len(data_folder)):
            
        # Create file paths 
        data_instance = data_folder[i]
        model_pred_instance = model_predictions[i]
        
        # print(model_pred_instance)
        # print(data_instance)
        
        file_path_data = os.path.join(folder_path, data_instance)
        file_path_model_pred = os.path.join(model_folder_path, model_pred_instance)

        # Load in model predictions
        with open(file_path_model_pred, 'r') as f: 
            file_contents = json.load(f)
            file_contents = np.array(file_contents)
            
            # visualise_3(file_contents)
            
            # Load in data 
        with np.load(file_path_data, allow_pickle=True) as data:
            single_frame = data['points']
            intensity_pred_instance = predict_cones_z(itensity_filtering(fov_range(single_frame, maxradius=10)))
            
        # Generate ground Truth 
        ground_truth = predict_cones_z(baseline_intensity_filtering(fov_range(single_frame, maxradius=10))) 
        
        # Evaluate both 
        acc1 = round(evaluate_clusters(ground_truth, intensity_pred_instance, MoE=moe_values[j])[0], 2)
        # print(acc1)
        acc2 = round(evaluate_clusters(ground_truth, file_contents, MoE=moe_values[j])[0], 2)
        # print(acc2, "%")
        
        baseline_acc_mean += acc1
        model_acc_mean += acc2

    baseline_acc_mean /= 21
    model_acc_mean /= 21
    
    basline_acc.append(baseline_acc_mean)
    model_acc.append(model_acc_mean)

moe_values = np.arange(0.4, 1.2, 0.05)

# Plot the data
plt.plot(moe_values, basline_acc, label='Baseline Accuracy', marker='o')
plt.plot(moe_values, model_acc, label='Model Accuracy', marker='o')

# Add labels, title, and legend
plt.xlabel('Margin of Error (metres)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Margin of Error')
plt.legend()

# Show the plot
plt.grid(True)  # Optional: Add a grid for better readability
plt.show()