"""This architecture is based on:
@author: Viet Nguyen <nhviet1009@gmail.com>
@repo: https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch/tree/master

but is modified by adding SELU as activation and changing kernel sizes and strides.
Also, default pytorch weights are used instead of the custom _create_weights().
"""

import torch
import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self, activation_function):
        super(DeepQNetwork, self).__init__()
        
        torch.manual_seed(22)
        torch.cuda.manual_seed(22)
        
        if activation_function == "selu":
            activation = nn.SELU(inplace=True)
        elif activation_function == "silu":
            activation = nn.SiLU(inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=3, stride=3), 
                                   activation)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=2, stride=2), 
                                   activation)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=2, stride=2), 
                                   activation)

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), activation)
        self.fc2 = nn.Linear(512, 2)
    #     self._create_weights()

    # def _create_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.uniform_(m.weight, -0.01, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)

        return output
    