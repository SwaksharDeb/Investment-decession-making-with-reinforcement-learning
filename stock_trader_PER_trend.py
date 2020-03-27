from IPython.core.debugger import set_trace
import os.path
from os import path
import random
import torch
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from DQN_PER import Agent
import pandas as pd
import statistics
import time

window_size = 10
portfolio_size = 1
investment_size = 1
trend_size = 1
input_size = window_size + portfolio_size +investment_size +trend_size

Name = "runs/DQN with PER lr=4.8e-2 tau=1e-2 gamma 0.995 {}".format(int(time.time()))
tb = SummaryWriter(log_dir=Name) #initialize tensorboard object
agent = Agent(state_size = input_size, action_size=3, seed=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

def state_creator(data_gp, timestep, window_size, inventory_gp, investment,episode,trend_gp):
  gp_assest = (len(inventory_gp) * data_gp[timestep]) #portfolio value for gp at current timestep
  
  state = []
  for i in range(timestep-window_size+1, timestep+1):
     state.append(data_gp[i]-data_gp[i-1])
  state.append(0.0001*gp_assest)
  state.append(0.0001*investment)
  state.append(trend_gp[timestep])
  state = np.array([state])   # normalized input(state)
  #state = sigmoid(state)
  return state

#loading the training data
dataset_gp = pd.read_csv('datasets/WMT Historical Data 2018.csv')
data_gp = list(dataset_gp['Price'])
trend_gp = list(dataset_gp['trend'])
data_samples = len(data_gp)-1
scores = []

#loading the validation data
dataset_gp_evaluation = pd.read_csv('datasets/WMT Historical Data 2019.csv')
data_gp_evaluation = list(dataset_gp_evaluation['Price'])
trend_gp_evaluation = list(dataset_gp_evaluation['trend'])

# storing average rewards and Q-values
rewards_avg = deque(maxlen=200)  # average reward in each epoach
state_action_value_averages = deque(maxlen=200) # store avg Q value for each epoach
 
#evaluation loop
def evaluation():
    #setting up the parameter
    data_samples = len(data_gp_evaluation)-1
    inventory_gp_evaluation = []
    total_profit = 0
    investment_evaluation = 10000

    episode = 1
    state = state_creator(data_gp_evaluation,10,window_size,inventory_gp_evaluation,investment_evaluation,episode,trend_gp_evaluation)

    for t in range(10,data_samples-1):
        action, state_action_value = agent.act(state) # return action and corresponding Q value
        if action == 1 and int(investment_evaluation/data_gp_evaluation[t])>0:  #buy
            no_buy = int(investment_evaluation/data_gp_evaluation[t])
            for i in range(no_buy):
                investment_evaluation -= data_gp_evaluation[t]
                inventory_gp_evaluation.append(data_gp_evaluation[t])
            print("AI Trader bought: ", stocks_price_format(no_buy*data_gp_evaluation[t]))
                 
        if action == 2 and len(inventory_gp_evaluation)>0: #Selling
            buy_prices_gp = []
                
            for i in range(len(inventory_gp_evaluation)):
                buy_prices_gp.append(inventory_gp_evaluation[i])
            buy_price = sum(buy_prices_gp)  #buying price of gp stocks
                
                
            total_profit += (len(inventory_gp_evaluation)*data_gp_evaluation[t]) - buy_price
            profit = (len(inventory_gp_evaluation)*data_gp_evaluation[t]) - buy_price
            investment_evaluation = investment_evaluation + (len(inventory_gp_evaluation)*data_gp_evaluation[t])  #total investment or cash in hand
            print("AI Trader sold gp: ", stocks_price_format(len(inventory_gp_evaluation)*data_gp_evaluation[t])," Profit: " + stocks_price_format(profit))
            
            for i in range(len(inventory_gp_evaluation)):
                inventory_gp_evaluation.pop(0)   # empty the gp inventory after selling all of them
            
        if action == 0: #hold
            print("AI Trader is holding........")                
                                                                            
        next_state = state_creator(data_gp_evaluation,t+1,window_size,inventory_gp_evaluation,investment_evaluation,episode,trend_gp_evaluation)
        if investment_evaluation<=0 and len(inventory_gp_evaluation)==0: #checking for bankcrapcy
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("AI Trader is bankcrapted")
            scores.append(total_profit)
            print("########################")            
            break  # if bankcrapted end the seassion
        
        if t == data_samples - 11:
            done = True
        else:
            done = False
                       
        state = next_state  #assin next state to present state
    
    print("########################")
    print("TOTAL VALIDATION PROFIT: {}".format(total_profit))
    print("########################")


def dqn(n_episodes=5000, max_t=len(data_gp)-20, eps_start=1, eps_end=0.001, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    sell_date = deque(maxlen=1)
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        print("Episode: {}/{}".format(episode, n_episodes))
        investment = 10000
        inventory_gp = []
        state_action_values = []
        rewards = []
        state = state_creator(data_gp,10,window_size,inventory_gp,investment,episode,trend_gp)
        total_profit = 0
        total_reward = 0
    
        for t in range(10,max_t):
            action, state_action_value = agent.act(state, eps) # return action and corresponding Q value            
            state_action_values.append(state_action_value)
            reward = -100
            if action == 1 and int(investment/data_gp[t])>0: #Buying gp
                no_buy = int(investment/data_gp[t])
                for i in range(no_buy):
                    investment -= data_gp[t]
                    inventory_gp.append(data_gp[t])
                fifteen_days_min = min(data_gp[t+1:t+16])
                if data_gp[t] < fifteen_days_min:
                    reward = 1                  
                else:
                    reward = -1
                rewards.append(reward)
                print("AI Trader bought: ", stocks_price_format(no_buy*data_gp[t])," Reward: " + stocks_price_format(reward))

            if action == 2 and len(inventory_gp) > 0: #Selling gp
                buy_prices_gp = []
                for i in range(len(inventory_gp)):
                    buy_prices_gp.append(inventory_gp[i])
                buy_price = sum(buy_prices_gp)  #buying price of gp stocks                
                fifteen_days_max = max(data_gp[t+1:t+16])
                if data_gp[t] > fifteen_days_max:
                    reward = 1
                else:
                    reward = -1
                total_profit += (len(inventory_gp)*data_gp[t]) - buy_price
                profit = (len(inventory_gp)*data_gp[t]) - buy_price
                investment = investment + (len(inventory_gp)*data_gp[t])  #total investment or cash in hand
                
                rewards.append(reward)
                sell_date.append(data_gp[t])
                print("AI Trader sold: ", stocks_price_format(len(inventory_gp)*data_gp[t]), " Profit: " + stocks_price_format(profit)," Reward: " , stocks_price_format(reward))
                for i in range(len(inventory_gp)):
                    inventory_gp.pop(0)
                
            if action == 0:
                reward = 0
                print("AT Trader is holding.................","Reward: ",stocks_price_format(reward))
                
            next_state = state_creator(data_gp,t+1,window_size,inventory_gp,investment,episode,trend_gp)
            
            if investment<=0 and len(inventory_gp)==0: #checking for bankcrapcy
                reward = -1000
                done = True
                agent.step(state, action, reward, next_state, done)
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("AI Trader is bankcrapted")
                scores.append(total_profit)
                print("########################")            
                break  # if bankcrapted end the seassion
            
            if t == max_t - 1:
                done = True
            else:
                done = False
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                #scores.append(total_profit)
                rewards_avg.append(statistics.mean(rewards))
                state_action_value_averages.append(statistics.mean(state_action_values))
                tb.add_scalar("average reward", statistics.mean(rewards), episode) #add average reward to tensorboard
                tb.add_scalar("average Q-value", statistics.mean(state_action_values), episode) #add average Q_value to tensorboard
                tb.add_histogram("local network fc1 layer bias", agent.qnetwork_local.fc1.bias, episode) # add first hidden layer bias to tensorboard 
                tb.add_histogram("local network fc2 layer bias", agent.qnetwork_local.fc2.bias, episode) # add second hidden layer bias to tensorboard 
                tb.add_histogram("local network fc3 layer bias", agent.qnetwork_local.fc3.bias, episode)
                tb.add_histogram("local network fc4 layer bias", agent.qnetwork_local.fc4.bias, episode)
                tb.add_histogram("local network fc5 layer bias", agent.qnetwork_local.fc5.bias, episode)
                tb.add_histogram("local network fc1 layer weight", agent.qnetwork_local.fc1.weight, episode) # add first hidden layer weights to tensorboard
                tb.add_histogram("local network fc2 layer weight", agent.qnetwork_local.fc2.weight, episode) # add second hidden layer weights to tensorboard
                tb.add_histogram("local network fc3 layer weight", agent.qnetwork_local.fc3.weight, episode)
                tb.add_histogram("local network fc4 layer weight", agent.qnetwork_local.fc4.weight, episode)
                tb.add_histogram("local network fc5 layer weight", agent.qnetwork_local.fc5.weight, episode)
                print("########################")
            
        evaluation()
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        """
        #save the model weights
        if episode == 100:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_local_WMT_V5_.pth')
            torch.save(agent.qnetwork_target.state_dict(), 'checkpoint_qnetwork_target_WMT_V5_.pth')
        """            
    return rewards_avg

scores = dqn()
tb.close() # close the tensorboard object

################################################################################################
#save the model weights
#torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/checkpoint_qnetwork_local_WMT_PER.pth')
#torch.save(agent.qnetwork_target.state_dict(), 'checkpoints/checkpoint_qnetwork_target_WMT_V5_PER.pth')
################################################################################################

# plot the average_reward for each epoach
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(rewards_avg)), rewards_avg)
plt.ylabel('average rewards of WMT')
plt.xlabel('Episode #')
plt.show()

# plot the average_Q values for each epoach
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(state_action_value_averages)), state_action_value_averages)
plt.ylabel('average Q-values of WMT')
plt.xlabel('Episode #')
plt.show()

"""Test the agent over training set"""
agent = Agent(state_size = input_size, action_size=3, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoints/checkpoint_qnetwork_local_WMT_PER.pth'))
agent.qnetwork_target.load_state_dict(torch.load('checkpoints/checkpoint_qnetwork_target_WMT_V5_PER.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#import the test data
dataset_gp_test = pd.read_csv('datasets/WMT Historical Data 2016.csv')
data_gp_test = list(dataset_gp_test['Price'])
trend_gp_test = list(dataset_gp_test['trend'])

#test loop
timesteps_sells = []
sells = []
timesteps_buys = []
buys = []

def test():
    #setting up the parameter
    data_samples = len(data_gp_test)-1
    inventory_gp_test = []
    total_profit = 0
    investment_test = 10000
    episode = 1
    state = state_creator(data_gp_test,10,window_size,inventory_gp_test,investment_test,episode,trend_gp_test)

    for t in range(10,data_samples-1):
        action, state_action_value = agent.act(state)
        if action == 1 and int(investment_test/data_gp_test[t])>0:  #buy
            no_buy = int(investment_test/data_gp_test[t])
            for i in range(no_buy):
                investment_test -= data_gp_test[t]
                inventory_gp_test.append(data_gp_test[t])
            timesteps_buys.append(t-10)
            buys.append(data_gp_test[t])
            print("AI Trader bought: ", stocks_price_format(no_buy*data_gp_test[t]), "Investment= ",stocks_price_format(investment_test))
                 
        if action == 2 and len(inventory_gp_test)>0: #Selling
            buy_prices_gp = []
                
            for i in range(len(inventory_gp_test)):
                buy_prices_gp.append(inventory_gp_test[i])
            buy_price = sum(buy_prices_gp)  #buying price of gp stocks                
            total_profit += (len(inventory_gp_test)*data_gp_test[t]) - buy_price
            profit = (len(inventory_gp_test)*data_gp_test[t]) - buy_price
            investment_test = investment_test + (len(inventory_gp_test)*data_gp_test[t])  #total investment or cash in hand
            timesteps_sells.append(t-10)
            sells.append(data_gp_test[t])
            print("AI Trader sold gp: ", stocks_price_format(len(inventory_gp_test)*data_gp_test[t])," Profit: " + stocks_price_format(profit),"Investment= ",stocks_price_format(investment_test))
            
            for i in range(len(inventory_gp_test)):
                inventory_gp_test.pop(0)   # empty the gp inventory after selling all of them
            
        if action == 0: #hold
            print("AI Trader is holding........","Investment= ",stocks_price_format(investment_test))                
                                                                            
        next_state = state_creator(data_gp_test,t+1,window_size,inventory_gp_test,investment_test,episode,trend_gp_test)
                
        if investment_test<=0 and len(inventory_gp_test)==0: #checking for bankcrapcy
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("AI Trader is bankcrapted")
            print("########################")            
            break  # if bankcrapted end the seassion
                
        state = next_state  #assin next state to present state
    
    print("########################")
    print("TOTAL TEST PROFIT: {}".format(total_profit))
    print("########################")

test()

#plot the graph
stock_price = data_gp_test[10:]
plt.plot(stock_price, color = 'blue', label = 'WMT')
plt.scatter(timesteps_sells,sells,color='red',label='sell')
plt.scatter(timesteps_buys,buys,color='black',label='buy')
plt.title('WMT stock')
plt.xlabel('Time')
plt.ylabel('WMT Price')
plt.legend()
plt.show()

