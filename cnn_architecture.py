import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
import random
import ast
import numpy as np
import json

from pseudoimage_generator import *

class BackboneCNN(nn.Module):
    def __init__(self, in_channels=9):
        super(BackboneCNN, self).__init__()

        # 9 --> 64
        self.input_linear = nn.Conv2d(in_channels, 64, kernel_size=1)

        # S=2, L=4, F=C
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # S=4, L=6, F=2C
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # S=8, L=6, F=4C
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # upsampling layers -- init
        self.upsample0 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=4, padding=1, output_padding=3)

    def forward(self, x):
        x = self.input_linear(x)

        x1 = self.block1(x)  # s 2
        x2 = self.block2(x1) # s 4
        x3 = self.block3(x2) # s 8
        #up
        up0 = F.relu(self.upsample0(x1))
        up1 = F.relu(self.upsample1(x2))  # up --> s 2
        up2 = F.relu(self.upsample2(x3))  # up --> s 2

        out = torch.cat((up0, up1, up2), dim=1)  # along channel dim
        return out

## SSD ##################
class YOLOLocalisationHead(nn.Module):
    def __init__(self, numclasses=1, numbboxes=2, input_channels=64*6):
        super(YOLOLocalisationHead, self).__init__()
        self.num_classes = numclasses # cone, no cone 
        self.num_boxes = numbboxes # num boxes / cell
        self.input_channels = input_channels
        
        inter_layers = 512
        out_channels = self.num_boxes * (3 + self.num_classes) # 4 = x,y,z,c (x2)
        # for forward
        self.fc1 = nn.Conv2d(input_channels, inter_layers, kernel_size=1)
        self.fc2 = nn.Conv2d(inter_layers, out_channels, kernel_size=1)
        
    def forward(self, x):
        '''
            Input: x = (B, C, H, W) = (1,384,20,20)
            Output: out = (B, H, W, B*(3+C))
        '''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        x = x.permute(0,2,3,1)
        
        return x 

# SSD&Backbone Combo
class BackboneSSD(nn.Module):
    def __init__(self, num_anchors=2):
        super(BackboneSSD, self).__init__()
        self.backbone = BackboneCNN(in_channels=9)
        # Initialise model 
        num_classes = 1
        num_boxes = 2
        input_channels = 64*6
        self.ssd_head = YOLOLocalisationHead(numclasses=num_classes, numbboxes=num_boxes, input_channels=input_channels)

    def forward(self, x):
        features = self.backbone(x)
        output = self.ssd_head(features)
        return output

############################################################################# TRAIN

def load_boxes(file_name):
    with open(file_name, 'r') as file:
        centers = ast.literal_eval(file.read())
    centers = np.array(centers)

    return centers

def loss_fn(center_targets, preds_array, testingBool):
    preds_array = preds_array.view(-1, 8)  # --> (400, 8)

    pred1 = preds_array[:, :4]  # (x1, y1, z1, c1)
    pred2 = preds_array[:, 4:]  # (x2, y2, z2, c2)
    
    valid_pred1 = pred1[torch.abs(pred1[:, 3]) > 0.1, :3]
    valid_pred2 = pred2[torch.abs(pred2[:, 3]) > 0.1, :3]

    valid_preds = torch.cat([valid_pred1, valid_pred2], dim=0)  # (N_valid, 3)
    print("Number of preds with high confidence:", len(valid_preds)) 
    
    # NO VALID
    if valid_preds.size(0) == 0:
        return torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

    distances = torch.cdist(valid_preds, center_targets)  # (N_valid, N_targets)
    # closest
    min_distances, _ = distances.min(dim=1)  # N_preds size

    squared_errors = min_distances ** 2
    mse_loss = squared_errors.mean()

    ######################### best matches error (nearest)
    
    # Detach only for scipy stuff
    valid_preds_np = valid_preds.detach().cpu().numpy()
    center_targets_np = center_targets.detach().cpu().numpy()

    cost_matrix = np.linalg.norm(center_targets_np[:, None, :] - valid_preds_np[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # back to tensors
    matched_targets = center_targets[row_ind]
    matched_preds = valid_preds[col_ind]
    distances = torch.norm(matched_targets - matched_preds, dim=1)
    nearest_loss = torch.mean(distances)

    return (mse_loss, nearest_loss, 0)

def train_one_epoch(model, optimizer, last, train_is, test_is):
    model.train()
    total_train_mse_loss = 0
    total_train_nearest_loss = 0
    total_test_mse_loss = 0
    total_test_nearest_loss = 0

    max_radius = 20
    H = 40
    W = 40

    # TRAIN
    random.shuffle(train_is)
    for i in train_is:

        file_path = f"./pseudos/tensor_{i}.json"
        with open(file_path, "r") as f:
            loaded_list = json.load(f)
        reshaped_pseudo = torch.tensor(loaded_list)

        np_center_targets = load_boxes(f"./cmr_base_centroids/instance-{i}.txt")
        center_targets = torch.from_numpy(np_center_targets).float()

        preds_array = model(reshaped_pseudo)
        mse_loss, nearest_loss, l2_loss = loss_fn(center_targets, preds_array, False)

        loss = mse_loss + nearest_loss #+ l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_mse_loss += mse_loss.item()
        total_train_nearest_loss += nearest_loss.item()

        if last:
            save_predictions(preds_array, f"./preds/cone_predictions_train_{i}.txt")

    # TEST
    # model.eval()
    with torch.no_grad():
        random.shuffle(test_is)
        for i in test_is:

            file_path = f"./pseudos/tensor_{i}.json"
            with open(file_path, "r") as f:
                loaded_list = json.load(f)
            reshaped_pseudo = torch.tensor(loaded_list)

            np_center_targets = load_boxes(f"./cmr_base_centroids/instance-{i}.txt")
            center_targets = torch.from_numpy(np_center_targets).float()

            preds_array = model(reshaped_pseudo)
            mse_loss, nearest_loss, l2_loss = loss_fn(center_targets, preds_array,True)

            total_test_mse_loss += mse_loss.item()
            total_test_nearest_loss += nearest_loss.item()

            if last:
                save_predictions(preds_array, f"./preds/cone_predictions_test_{i}.txt")

    avg_train_mse_loss = total_train_mse_loss / len(train_is)
    avg_train_nearest_loss = total_train_nearest_loss / len(train_is)
    avg_test_mse_loss = total_test_mse_loss / len(test_is)
    avg_test_nearest_loss = total_test_nearest_loss / len(test_is)

    print(f"Train Avg MSE Loss: {avg_train_mse_loss:.4f}, Train Avg Nearest Loss: {avg_train_nearest_loss:.4f}")
    print(f"Test Avg MSE Loss: {avg_test_mse_loss:.4f}, Test Avg Nearest Loss: {avg_test_nearest_loss:.4f}")
    return avg_train_mse_loss, avg_train_nearest_loss, avg_test_mse_loss, avg_test_nearest_loss

def save_predictions(preds_array, file_path):
    preds_array = preds_array.view(-1, 8)  # 400X8
    pred1 = preds_array[:, :4]
    pred2 = preds_array[:, 4:]
    valid_pred1 = pred1[torch.abs(pred1[:, 3]) > 0.1, :3]
    valid_pred2 = pred2[torch.abs(pred2[:, 3]) > 0.1, :3]
    valid_preds = torch.cat([valid_pred1, valid_pred2], dim=0) #(N_valid, 3)

    valid_preds_list = valid_preds.cpu().detach().tolist()

    with open(file_path, "w") as f:
        f.write("[")
        for pred in valid_preds_list[:-1]:
            f.write(f"    {pred},\n")
        if len(valid_preds_list) > 0:
            f.write(f"    {valid_preds_list[-1]}\n")
        f.write("]\n")
