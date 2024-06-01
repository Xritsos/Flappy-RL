"""Deep Q Network architecture as implemented in: https://github.com/hardlyrichie/pytorch-flappy-bird/tree/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, input_dim):
        super(QNetwork, self).__init__()

        torch.manual_seed(22)
        
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512) 
        self.layer5 = nn.Linear(512, 512)
        
        self.output = nn.Linear(512, 2)

    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu6(self.layer5(x))
        
        x = self.output(x)

        return x