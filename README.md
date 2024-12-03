## PointPillars Point Cloud 3D Cone Detection for Autonomous Vehicles 

Code for psuedoimage generation, SSD head, model training, and data generation

# Data Files:    

1. 20 raw point cloud frames (taken from Carnegie Mellon Racing's dataset) - actual_data
2. 20 txt files with bounding box coordinates for corresponding model predictions 

# Code Files: 

1. pseudoimage_generator: 
    - 5 different ways to create the pseudoimage from the raw point cloud data
    - Primary steps: 
        1. Filter the point cloud based on field of view and y-axis range (default max_radius = 20 metres)
        2. Seperate point cloud into vertical pillars based on x and y coordinates 
        3. Per pillar: perform feature extraction to create 3-10 dimensional vectors 
        4. Output (H,W,C) pseudoimage

2. cnn_architecture:
    - Classes for Backbone and SSD Head 
    - Includes MSE and Matched losses 
    - Loads in data and save model and predictions 
    - Model architecture details can be found in associated paper 

3. training_script: 
    -  Trains model for 300 epochs

4. evaluate_baseline: 
    - Accuracy evaluator that compares the position of model predictions relative to CMR's baseline intensity filtering and DBSCAN clustering algorithm and ground truth data 

5. save_pseudoimages and generate_ground_truth_data: 
    - Extract x,y,z coordinates from CMR's ground truth, labelled point cloud data 

