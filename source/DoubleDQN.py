"""Train Flappy bird game with Double Deep Q Nets.
"""

import os
import sys
import time

import torch
import pygame
from random import random, randint
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append('./')
from source.utils.process_image import pre_processing
from source.models.dqnet import DeepQNetwork
from source.utils.Plot import plot_durations
from source.utils.Buffer import ReplayMemory
from source.game.flappy_bird import FlappyBird

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


def train(test_id):
    torch.manual_seed(22)
    torch.cuda.manual_seed(22)
    np.random.seed(22)
    
    # ===================== initializations =========================
    device = "cuda"
    
    # Discount factor
    GAMMA = 0.99
    
    # batch size to read from replay memory
    BATCH_SIZE = 32
    
    # size of buffer (replay memory)
    MEMORY_SIZE = 50000
 
    # Epsilon values for ϵ greedy exploration
    EPS_START = 0.1
    EPS_END = 1e-6
    EPS_DECAY = 0.999995
    
    # update rate of the target network
    TAU = 0.005

    # total number iterations for each experiment
    ITERATIONS = 20000
    
    # number of steps before updating target network from Q network
    C_STEPS = 1000
    
    STEPS_DONE = 0
    
    IMAGE_SIZE = 84
    
    LR = 1e-6
    
    # initialize networks
    Q_net = DeepQNetwork()  # policy network
    Q_net.to(device)
    
    target_net = DeepQNetwork()
    target_net.to(device)
    
    # no grad needed for target network for which the weights are copied from 
    # the policy net
    for param in target_net.parameters():
        param.requires_grad = False
        
    # init weights from policy net
    target_net.load_state_dict(Q_net.state_dict())
    
    # Initialize optimizer (another source suggests RMSProp for reproducibility)
    optimizer = optim.Adam(Q_net.parameters(), lr=LR, amsgrad=False)
    # optimizer = optim.RMSprop(Q_net.parameters(), lr=LR, weight_decay=0.9, momentum=0.95)
    
    # set loss function
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    
    # Initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], 
                           IMAGE_SIZE, IMAGE_SIZE)
    
    image = torch.from_numpy(image).to(device)
    
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    episode_durations = []
    all_loss = []
    duration = 0
    min_loss = torch.tensor(1000, dtype=torch.float32).to(device)
    while STEPS_DONE < ITERATIONS: 
        prediction = Q_net(state)[0]
        
        # Exploration or exploitation
        epsilon = EPS_END + ((ITERATIONS - STEPS_DONE) * (EPS_START - EPS_END) / ITERATIONS)
        u = random()
        random_action = u <= epsilon
        if random_action:
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction).item()
            
        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], 
                                    IMAGE_SIZE, IMAGE_SIZE)
        
        next_image = torch.from_numpy(next_image).to(device)
        
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        memory.push(state, action, next_state, reward, terminal)
        
        if len(memory) > 5 * BATCH_SIZE:
            # Sample random batch
            batch = memory.sample(min(len(memory), BATCH_SIZE))
            
            state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = zip(*batch)
            
            state_batch = torch.cat(tuple(state for state in state_batch))
            action_batch = torch.from_numpy(np.array([[1, 0] if action == 0 else [0, 1] 
                                                      for action in action_batch], dtype=np.float32))
            
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = torch.cat(tuple(state for state in next_state_batch))
            
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            reward_batch = reward_batch.to(device)
            next_state_batch = next_state_batch.to(device)
            
            current_prediction_batch = Q_net(state_batch)
            
            with torch.no_grad():
                next_prediction_batch = target_net(next_state_batch)
                
            # perhaps if terminal is True it should append 0
            y_batch = torch.cat(tuple(reward if terminal else reward + \
                GAMMA * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

            q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
            optimizer.zero_grad()
            loss = criterion(q_value, y_batch)
            loss.backward()
            optimizer.step()
            
            print(f"Step {STEPS_DONE}/{ITERATIONS} --- Loss: {loss}")
            print(LINE_UP, end=LINE_CLEAR)

            state = next_state
            STEPS_DONE += 1
            duration += 1
            all_loss.append(loss)
            
            # if min_loss == 0.0:
            #     min_loss = loss
            
            if loss < min_loss:
                    min_loss = loss
                    torch.save(Q_net, f'./model_ckpts/{test_id}_model.pt')
            
            # if iterations reached the number update target network
            if STEPS_DONE % C_STEPS == 0:
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = Q_net.state_dict()
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU\
                                                + target_net_state_dict[key]*(1-TAU)
                                                
                target_net.load_state_dict(target_net_state_dict)
            
            if terminal:
                episode_durations.append(duration)
                plot_durations(episode_durations)
                duration = 0
                
    log_episodes = {'episode': [i for i in range(len(episode_durations))], 
                    'duration': episode_durations}

    log_losses = {'iteration': [i for i in range(ITERATIONS)],
                  'loss': [float(loss.detach().cpu()) for loss in all_loss]}

    df_episode = pd.DataFrame(log_episodes)
    df_episode.to_csv(f'./logs/episodes/{test_id}.csv', index=False)

    df_losses = pd.DataFrame(log_losses)
    df_losses.to_csv(f'./logs/losses/{test_id}.csv', index=False)
    
    pygame.quit()
    exit()

if __name__ == "__main__":
    
    for test_id in [0]:
        plt.ion()
        
        train(test_id)

        plt.ioff()
        plt.show()