import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json

from pseudoimage_generator import *

save_dir = "./pseudos"

max_radius = 20
H = 40
W = 40

for i in range(0,1502):
    print("\nimage:", i)
    folder_path = f"./tt-4-18-eleventh/instance-{i}.npz"
    with np.load(folder_path, allow_pickle=True) as data:
        points = data['points']
    processed_points = fov_range(points, maxradius=max_radius)
    psuedoimage = generate_psuedoimage(processed_points, H, W, 20, 20) #CHANGE?
    psuedoimage_tensor = torch.from_numpy(psuedoimage).float()
    reshaped_pseudo = psuedoimage_tensor.permute(2, 0, 1).unsqueeze(0)

    tensor_list = reshaped_pseudo.tolist()
    
    # Save as a JSON file
    file_path = os.path.join(save_dir, f"tensor_{i}.json")
    with open(file_path, "w") as f:
        json.dump(tensor_list, f)

