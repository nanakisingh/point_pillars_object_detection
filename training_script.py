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
from cnn_architecture import *


model = BackboneSSD()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100
epochs = np.zeros(num_epochs)
losses = np.zeros(num_epochs)
mse_losses = np.zeros(num_epochs)
nearest_losses = np.zeros(num_epochs)

losses_test = np.zeros(num_epochs)
mse_losses_test = np.zeros(num_epochs)
nearest_losses_test = np.zeros(num_epochs)

all_indices = list(range(0, 100, 1))
random.shuffle(all_indices)
print("Number of frames used:", len(all_indices))
train_indices = all_indices[:80]
test_indices = all_indices[80:]

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    mse_loss, nearest_loss, mse_loss_test, nearest_loss_test = train_one_epoch(model, optimizer,(epoch == num_epochs-1),train_indices,test_indices)
    epochs[epoch] = epoch+1
    losses[epoch] = mse_loss+nearest_loss
    mse_losses[epoch] = mse_loss
    nearest_losses[epoch] = nearest_loss
    losses_test[epoch] = mse_loss_test+nearest_loss_test
    mse_losses_test[epoch] = mse_loss_test
    nearest_losses_test[epoch] = nearest_loss_test

import matplotlib.pyplot as plt
plt.plot(epochs,losses,label="total")
plt.plot(epochs,mse_losses,label="mse")
plt.plot(epochs,nearest_losses,label="nearest")
plt.plot(epochs,losses_test,label="total_test")
plt.plot(epochs,mse_losses_test,label="mse_test")
plt.plot(epochs,nearest_losses_test,label="nearest_test")
plt.title("Loss over Training")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend()
plt.savefig("lossGraph.png")
plt.show()

