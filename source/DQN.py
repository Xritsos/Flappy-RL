"""Train Flappy bird game with Deep Q Nets
"""

import os
import sys
import time

import torch
import pygame
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append('./')
from source.utils import Image
from source.models.dqnet import QNetwork
from source.utils.Plot import plot_durations
from source.utils.Buffer import ReplayMemory
from source.game import wrapped_flappy_bird as game


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.manual_seed(22)
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(test_id):
    torch.manual_seed(22)
    np.random.seed(22)
    random.seed(22)
    
    device = "cuda"
    
    # Here should read from the csv file to get params
    
    # Discount factor
    gamma = 0.99
    
    # Epsilon values for ϵ greedy exploration
    initial_epsilon = 0.1
    final_epsilon = 0.0001
    
    replay_memory_size = 10000
    num_iterations = 2000
    
    minibatch_size = 64
    episode_durations = []
    
    Q_net = QNetwork()
    Q_net.to(device)
    Q_net.apply(init_weights)
    
    start = time.time()
    
    # Initialize optimizer (another source suggests RMSProp for reproducibility)
    optimizer = optim.Adam(Q_net.parameters(), lr=1e-4)
    
    # Initialize loss function
    loss_func = nn.MSELoss()
    
    # Initialize game
    game_state = game.GameState()

    # Initialize replay memory
    memory = ReplayMemory(replay_memory_size)

    # Initial action is do nothing
    action = torch.zeros(2, dtype=torch.float32).to(device)
    action[0] = 1

    # [1, 0] is do nothing, [0, 1] is fly up
    image_data, reward, terminal = game_state.frame_step(action)

    # Image Preprocessing
    image_data = Image.resize_and_bgr2gray(image_data)
    image_data = Image.image_to_tensor(image_data)
    image_data = image_data.to(device)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)
    
    # Initialize epsilon value
    epsilon = initial_epsilon

    # Epsilon annealing
    epsilon_decrements = np.linspace(initial_epsilon, final_epsilon, num_iterations)
    
    t = 0
    
    track_loss = torch.tensor(0).to(device)
    all_loss = []
    count_time = time.time()
    
    # Train Loop
    print("Start Episode", 0)
    for iteration in range(num_iterations):
        # Get output from the neural network
        output = Q_net(state)[0]

        # Initialize action
        action = torch.zeros(2, dtype=torch.float32).to(device)

        # Epsilon greedy exploration
        random_action = random.random() <= epsilon
        
        if random_action:
            print("Performed random action!")
            
        action_index = [torch.randint(2, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        action_index = action_index.to(device)

        action[action_index] = 1

        # Get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = Image.resize_and_bgr2gray(image_data_1)
        image_data_1 = Image.image_to_tensor(image_data_1)
        image_data_1 = image_data_1.to(device)
        
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        #reward = reward.to(device)

        # Save transition to replay memory
        memory.push(state, action, state_1, reward, terminal)

        # Epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # Sample random minibatch
        minibatch = memory.sample(min(len(memory), minibatch_size))

        # Unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch)).to(device)
        action_batch = torch.cat(tuple(d[1] for d in minibatch)).to(device)
        state_1_batch = torch.cat(tuple(d[2] for d in minibatch)).to(device)
        reward_batch = torch.cat(tuple(d[3] for d in minibatch)).to(device)

        # Get output for the next state
        output_1_batch = Q_net(state_1_batch)

        # Set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        y_batch = y_batch.to(device)
        
        # Extract Q-value (this part i don't understand)
        q_value = torch.sum(Q_net(state_batch) * action_batch, dim=1).to(device)

        optimizer.zero_grad()

        # Returns a new Tensor, detached from the current graph, 
        # the result will never require gradient
        y_batch = y_batch.detach()

        # Calculate loss
        loss = loss_func(q_value, y_batch)
        all_loss.append(loss)

        # Do backward pass
        loss.backward()
        optimizer.step()

        # Set state to be state_1
        state = state_1
        
        if iteration == 0:
            track_loss = loss

        if iteration % 10 == 0:
            # torch.save(net, "model_weights/current_model_" + str(iteration) + ".pth")
            # if loss < track_loss:
            torch.save(Q_net, f'./model_ckpts/{test_id}_model.pt')
            track_loss = loss

        if iteration % 100 == 0:
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                np.max(output.cpu().detach().numpy()))

        t += 1

        # Plot duration
        if terminal:
            print("Start Episode", len(episode_durations) + 1)
            episode_durations.append(t)
            plot_durations(episode_durations)
            t = 0
            
    log_episodes = {'episode': [i for i in range(len(episode_durations))], 
           'duration': episode_durations}
    
    log_losses = {'iteration': [i for i in range(num_iterations)],
                  'loss': [loss.detach().cpu() for loss in all_loss]}
    
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
        