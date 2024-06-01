"""Train Flappy bird game with Deep Q Nets
"""

import os
import sys
import time

import math
import torch
import pygame
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from itertools import count
import matplotlib.pyplot as plt

sys.path.append('./')
from source.utils import Image
from source.models.linear import QNetwork
from source.utils.Plot import plot_durations
from source.utils.Buffer import ReplayMemory
from source.game import wrapped_flappy_bird as game


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.manual_seed(22)
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
        m.bias.data.fill_(0.01)

def train(test_id):
    torch.manual_seed(22)
    np.random.seed(22)
    random.seed(22)
    
    # ===================== initializations =========================
    device = "cuda"
    
    # Discount factor
    GAMMA = 0.99
    
    # batch size to read from replay memory
    BATCH_SIZE = 64
    
    # size of buffer (replay memory)
    MEMORY_SIZE = 50000
 
    # Epsilon values for ϵ greedy exploration
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.999995
    
    # update rate of the target network
    TAU = 0.005

    # total number of episodes to run training for
    EPISODES = 20000
    
    # number of steps before updating target network from Q network
    C_STEPS = 1000
    
    STEPS_DONE = 1
    
    # initialize networks
    Q_net = QNetwork(28224)  # policy network
    Q_net.to(device)
    # Q_net.apply(init_weights)
    
    target_net = QNetwork(28224)
    target_net.to(device)
    
    # no grad needed for target network for which the weights are copied from 
    # the policy net
    for param in target_net.parameters():
        param.requires_grad = False
        
    # init weights from policy net
    target_net.load_state_dict(Q_net.state_dict())
    
    # Initialize optimizer (another source suggests RMSProp for reproducibility)
    optimizer = optim.AdamW(Q_net.parameters(), lr=1e-4, amsgrad=False)
    # optimizer = optim.RMSprop(Q_net.parameters(), lr=1e-6, weight_decay=0.9, momentum=0.95)
    
    # Initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    episode_durations = []
    
    for episode in tqdm(range(EPISODES)):
        # Initialize game
        game_state = game.GameState()
        
        # Initial action is do nothing
        action = torch.zeros(2, dtype=torch.float32)
        action[0] = 1
        
        # [1, 0] is do nothing, [0, 1] is fly up
        image_data, reward, terminal = game_state.frame_step(action)
        
        # Image Preprocessing
        image_data = Image.resize_and_bgr2gray(image_data)
        image_data = Image.image_to_tensor(image_data)
        # image_data = image_data.to(device)
        
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)
        state_input = state.view(state.size()[0], -1).to(device)
        
        for c in count():
            action = torch.zeros(2, dtype=torch.float32).to(device)
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1.0 * STEPS_DONE / EPS_DECAY)
            
            if sample > eps_threshold:
                with torch.no_grad():
                    # action_index = Q_net(state).max(1).indices[0]
                    action_index = torch.argmax(Q_net(state_input))
            else:
                action_index = torch.randint(2, torch.Size([]), dtype=torch.int).to(device)
                
            action[action_index] = 1
            
            STEPS_DONE += 1
            
            # Get next state and reward
            image_data_next, reward, terminal = game_state.frame_step(action)
            image_data_next = Image.resize_and_bgr2gray(image_data_next)
            image_data_next = Image.image_to_tensor(image_data_next)
            # image_data_next = image_data_next.to(device)
            
            state_next = torch.cat((state.squeeze(0)[1:, :, :], image_data_next)).unsqueeze(0)
            # state_next = state_next.to(device)

            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
            #reward = reward.to(device)

            # Save transition to replay memory
            memory.push(state, action, reward, state_next, terminal)
            
            state = state_next
            
            if len(memory) >= 3000:
                # Sample random minibatch
                minibatch = memory.sample(min(len(memory), BATCH_SIZE))

                # Unpack minibatch
                state_batch = torch.cat(tuple(d[0] for d in minibatch))
                action_batch = torch.cat(tuple(d[1] for d in minibatch))
                reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(device)
                state_next_batch = torch.cat(tuple(d[3] for d in minibatch))
                
                state_next_batch_input = state_next_batch.view(state_next_batch.size()[0], -1)
                state_next_batch_input = state_next_batch_input.to(device)
                
                # Get output for the next state
                with torch.no_grad():
                    output_next_batch = target_net(state_next_batch_input)
                
                # Set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
                y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                          else reward_batch[i] + \
                                              GAMMA * torch.max(output_next_batch[i])
                                          for i in range(len(minibatch))))
                
                # Extract Q-value (this part i don't understand)
                state_batch_input = state_batch.view(state_batch.size()[0], -1).to(device)
                action_batch_input = action_batch.view(action_batch.size()[0], -1).to(device)
                q_value = torch.sum(Q_net(state_batch_input) * action_batch_input, dim=1).to(device)
                
                # Compute Huber loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(q_value, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(Q_net.parameters(), 100)
                optimizer.step()
                
            # if iterations reached the number update target network
            if STEPS_DONE % C_STEPS == 0:
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = Q_net.state_dict()
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU\
                                                + target_net_state_dict[key]*(1-TAU)
                                                
                target_net.load_state_dict(target_net_state_dict)
                
            if terminal:
                episode_durations.append(c + 1)
                plot_durations(episode_durations)
                break
                

if __name__ == "__main__":
    
    for test_id in [0]:
        plt.ion()
        
        train(test_id)

        plt.ioff()
        plt.show()
        