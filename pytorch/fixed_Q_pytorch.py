from IPython.core.debugger import set_trace
import os.path
from os import path
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import pandas as pd

window_size = 10
agent = Agent(state_size = 10, action_size=3, seed=0)

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))


def state_creator(data, timestep, window_size):
  
  starting_id = timestep - window_size + 1
      
  if starting_id >= 0:
    windowed_data = data[starting_id:timestep+1]
  else:
    windowed_data = - starting_id * [data[0]] + list(data[0:timestep+1])

  state = []
  for i in range(window_size - 1):
    state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))
    
  state = np.array([state])
  state = np.reshape(state, (-1))
  return state

dataset_2 = pd.read_csv('GRAE Historical Data 2018 practice.csv')
data = list(dataset_2['Price'])
data_samples = len(data)-1

def dqn(n_episodes=2000, max_t=len(data)-1, eps_start=1.0, eps_end=0.001, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_window = deque(maxlen=1500)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = state_creator(data, 0, window_size + 1)
        total_profit = 0
        inventory_gp = []
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state = state_creator(data, t+1, window_size + 1)
            reward = 0
            if action == 1: #Buying gp
                inventory_gp.append(data[t])
                print("AI Trader bought: ", stocks_price_format(data[t]))

            if action == 2 and len(inventory_gp) > 0: #Selling gp
                buy_price = inventory_gp.pop(0)
                total_profit += (data[t] - buy_price)
                if data[t] - buy_price>0:
                    reward = 1
                elif data[t] - buy_price==0:
                    reward = 0
                else:
                    reward = -1
                print("AI Trader sold: ", stocks_price_format(data[t]), " Profit: " + stocks_price_format(data[t] - buy_price))
                
            if t == data_samples - 1:
                done = True
            else:
                done = False
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("########################")
            
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
    return total_profit

scores = dqn()
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_mountain_car.pth')

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_mountain_car.pth'))

for i in range(3):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()