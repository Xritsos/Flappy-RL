"""Various functions for plotting."""

import torch
import pandas as pd
import matplotlib.pyplot as plt


def plot_durations(episode_durations, loss=[], offline_plot=False, test_id=None):
    """Plot durations of episodes and average over last 100 episodes"""
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if offline_plot == True:
        plt.title(f'Test {test_id}')
    else:
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), label="Episode duration")
    
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label="Mean (100) duration")
        
    if len(loss) != 0:
        plt.plot(loss, label="loss per episode")

    plt.legend(loc="upper left")
    plt.yscale('log')
    if offline_plot == True:
        plt.show()
    else:
        plt.pause(0.001)
    
    
if __name__ == "__main__":
    test_id = 21
    
    df = pd.read_csv(f'./logs/episodes/{test_id}.csv')
    
    df_2 = pd.read_csv(f'./logs/losses/{test_id}.csv')
    
    durations = df['duration']
    loss = df_2['loss']
    
    print()
    print(f"Average duration: {durations.sum() / len(durations)}")
    print(f"Average loss: {loss.sum() / len(loss)}")
    
    plot_durations(durations, offline_plot=True, test_id=test_id, loss=loss)
    