"""Deep Q Network architecture as implemented in: https://github.com/hardlyrichie/pytorch-flappy-bird/tree/master
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()

        torch.manual_seed(22)
        
        # Use same network architecture as DeepMind
        # Input is 4 frames stacked to infer velocity
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.fc1 = nn.Linear(3136, 512)
        
        # Output 2 values: fly up and do nothing
        self.fc2 = nn.Linear(512, 2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    