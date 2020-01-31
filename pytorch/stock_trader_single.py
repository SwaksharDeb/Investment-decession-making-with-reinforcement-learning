from IPython.core.debugger import set_trace
import os.path
from os import path
import random
import torch
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import pandas as pd
import statistics

window_size = 10
portfolio_size = 1
investment_size = 1
input_size = window_size + portfolio_size +investment_size

agent = Agent(state_size = input_size, action_size=3, seed=0)

#agent.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local_gp_3.pth'))
#agent.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target_gp_3.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

#sc = MinMaxScaler()
def state_creator(data_sqph, timestep, window_size, inventory_sqph, investment,episode):
  sqph_assest = (len(inventory_sqph) * data_sqph[timestep]) #portfolio value for gp at current timestep
  
  state = []
  for i in range(timestep-window_size+1, timestep+1):
     state.append(data_sqph[i]-data_sqph[i-1])
              
  state.append(0.0001*sqph_assest)
  state.append(0.0001*investment)
  state = np.array([state])   # normalized input(state)
  """state = np.reshape(state, (-1,1))  
  
  if timestep == 10 and episode == 1:
      state = sc.fit_transform(state)
      state = np.reshape(state, (-1))  #converting state into vector(1D array)
      #writer.add_graph(agent,state)
  else:
      state = sc.transform(state)
      state = np.reshape(state, (-1))  #converting state into vector(1D array)
  """
  state = sigmoid(state)
  return state

#loading the training data
dataset_sqph = pd.read_csv('SQPH Historical Data2018 practice.csv')
data_sqph = list(dataset_sqph['Price'])
data_samples = len(data_sqph)-1
scores = []

#loading the validation data
dataset_sqph_validation = pd.read_csv('SQPH Historical Data 2019.csv')
data_sqph_validation = list(dataset_sqph_validation['Price'])

#validation loop
def validation():
    #setting up the parameter
    data_samples = len(data_sqph_validation)-1
    inventory_sqph_validation = []
    total_profit = 0
    investment_validation = 100000

    #f = open("state.txt","w")
    #testing loop
    episode = 1
    state = state_creator(data_sqph_validation,10,window_size,inventory_sqph_validation,investment_validation,episode)

    for t in range(10,data_samples-1):
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #f.write(str(sc.inverse_transform(state.cpu().data.numpy())))
        #action = np.argmax(agent.qnetwork_local(state).cpu().data.numpy())
        action = agent.act(state)
        reward = 0
        if action == 1 and int(investment_validation/data_sqph_validation[t])>0:  #buy
            no_buy = int(investment_validation/data_sqph_validation[t])
            for i in range(no_buy):
                investment_validation -= data_sqph_validation[t]
                inventory_sqph_validation.append(data_sqph_validation[t])
            #f.write("BUY")            
            print("AI Trader bought: ", stocks_price_format(no_buy*data_sqph_validation[t]))
                 
        if action == 2 and len(inventory_sqph_validation)>0: #Selling
            buy_prices_sqph = []
                
            for i in range(len(inventory_sqph_validation)):
                buy_prices_sqph.append(inventory_sqph_validation[i])
            buy_price = sum(buy_prices_sqph)  #buying price of gp stocks
                
                
            total_profit += (len(inventory_sqph_validation)*data_sqph_validation[t]) - buy_price
            profit = (len(inventory_sqph_validation)*data_sqph_validation[t]) - buy_price
            investment_validation = investment_validation + (len(inventory_sqph_validation)*data_sqph_validation[t])  #total investment or cash in hand
            reward = data_sqph_validation[t] - buy_prices_sqph[0]
            #stock_return = (w_gp*returns_gp_validation[t]) + (w_sqph*returns_sqph_validation[t])               
            #f.write("SELL")
            print("AI Trader sold gp: ", stocks_price_format(len(inventory_sqph_validation)*data_sqph_validation[t])," Profit: " + stocks_price_format(profit))
            
            for i in range(len(inventory_sqph_validation)):
                inventory_sqph_validation.pop(0)   # empty the gp inventory after selling all of them
            
        if action == 0: #hold
            #f.write("HOLD")
            print("AI Trader is holding........")                
                                                                            
        next_state = state_creator(data_sqph_validation,t+1,window_size,inventory_sqph_validation,investment_validation,episode)
        if investment_validation<=0 and len(inventory_sqph_validation)==0: #checking for bankcrapcy
                reward = -10
                done = True
                agent.memory.add(state, action, reward, next_state, done)
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("AI Trader is bankcrapted")
                scores.append(total_profit)
                print("########################")            
                break  # if bankcrapted end the seassion
            
        if t == data_samples - 1:
            done = True
        else:
            done = False
        agent.memory.add(state, action, reward, next_state, done)      
        if investment_validation<=0 and len(inventory_sqph_validation)==0: #checking for bankcrapcy
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("AI Trader is bankcrapted")
            scores.append(total_profit)
            print("########################")            
            break  # if bankcrapted end the seassion
                
        state = next_state  #assin next state to present state
    
    print("########################")
    print("TOTAL VALIDATION PROFIT: {}".format(total_profit))
    print("########################")
    #f.close()


def dqn(n_episodes=5000, max_t=len(data_sqph)-1, eps_start=1, eps_end=0.001, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_window = deque(maxlen=500)  # last 1500 scores
    rewards = deque(maxlen=200)
    rewards.append(10000)
    rewards.append(0)
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        print("Episode: {}/{}".format(episode, n_episodes))
        investment = 10000
        portfolio_value_ = investment
        inventory_sqph = []
        state = state_creator(data_sqph,10,window_size,inventory_sqph,investment,episode)
        total_profit = 0
        total_reward = 0
    
        for t in range(10,max_t):
            action = agent.act(state, eps)
            reward = 0
            if action == 1 and int(investment/data_sqph[t])>0: #Buying gp
                no_buy = int(investment/data_sqph[t])
                for i in range(no_buy):
                    investment -= data_sqph[t]
                    inventory_sqph.append(data_sqph[t])
                portfolio_value = (len(inventory_sqph)*data_sqph[t]) + investment
                rewards.append(reward)
                print("AI Trader bought: ", stocks_price_format(no_buy*data_sqph[t])," Reward: " + stocks_price_format(reward))

            if action == 2 and len(inventory_sqph) > 0: #Selling gp
                buy_prices_sqph = []
                for i in range(len(inventory_sqph)):
                    buy_prices_sqph.append(inventory_sqph[i])
                buy_price = sum(buy_prices_sqph)  #buying price of gp stocks
                
                total_profit += (len(inventory_sqph)*data_sqph[t]) - buy_price
                profit = (len(inventory_sqph)*data_sqph[t]) - buy_price
                investment = investment + (len(inventory_sqph)*data_sqph[t])  #total investment or cash in hand
                portfolio_value = investment
                reward = data_sqph[t] - buy_prices_sqph[0]
                rewards.append(reward)
                print("AI Trader sold: ", stocks_price_format(len(inventory_sqph)*data_sqph[t]), " Profit: " + stocks_price_format(profit)," Reward: " , stocks_price_format(reward))
                for i in range(len(inventory_sqph)):
                    inventory_sqph.pop(0)
                
            if action == 0:
                print("AT Trader is holding.................","Reward: ",stocks_price_format(reward))
                porfolio_value = (len(inventory_sqph)*data_sqph[t]) + investment
               
            next_state = state_creator(data_sqph,t+1,window_size,inventory_sqph,investment,episode)
            portfolio_value_ = portfolio_value
            
            if investment<=0 and len(inventory_sqph)==0: #checking for bankcrapcy
                reward = -10
                done = True
                agent.step(state, action, reward, next_state, done)
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("AI Trader is bankcrapted")
                scores.append(total_profit)
                print("########################")            
                break  # if bankcrapted end the seassion
            
            if t == data_samples - 1:
                done = True
            else:
                done = False
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                scores.append(total_profit)
                print("########################")
            
        validation()
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
    return total_profit

scores = dqn()

################################################################################################
#save the model weights
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_local_sqph_p.pth')
torch.save(agent.qnetwork_target.state_dict(), 'checkpoint_qnetwork_target_sqph_p.pth')
################################################################################################

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score sqph')
plt.xlabel('Episode #')
plt.show()

"""Test the agent over training set"""

agent = Agent(state_size = input_size, action_size=3, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local_sqph_p.pth'))
agent.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target_sqph_p.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#import the test data
dataset_gp_test = pd.read_csv('SQPH Historical Data 2010.csv')
data_gp_test = list(dataset_gp_test['Price'])

#test loop
def test():
    #setting up the parameter
    data_samples = len(data_gp_test)-1
    inventory_gp_test = []
    total_profit = 0
    investment_test = 100000

    f = open("state.txt","w")
    #testing loop
    episode = 1
    state = state_creator(data_gp_test,10,window_size,inventory_gp_test,investment_test,episode)

    for t in range(10,data_samples-1):
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #f.write(str(sc.inverse_transform(state.cpu().data.numpy())))
        #action = np.argmax(agent.qnetwork_local(state).cpu().data.numpy())
        action = agent.act(state)
        f.write(str(state))        
        if action == 1 and int(investment_test/data_gp_test[t])>0:  #buy
            no_buy = int(investment_test/data_gp_test[t])
            for i in range(no_buy):
                investment_test -= data_gp_test[t]
                inventory_gp_test.append(data_gp_test[t])
            f.write("BUY")            
            print("AI Trader bought: ", stocks_price_format(no_buy*data_gp_test[t]), "Investment= ",stocks_price_format(investment_test))
                 
        if action == 2 and len(inventory_gp_test)>0: #Selling
            buy_prices_gp = []
                
            for i in range(len(inventory_gp_test)):
                buy_prices_gp.append(inventory_gp_test[i])
            buy_price = sum(buy_prices_gp)  #buying price of gp stocks
                
                
            total_profit += (len(inventory_gp_test)*data_gp_test[t]) - buy_price
            profit = (len(inventory_gp_test)*data_gp_test[t]) - buy_price
            investment_test = investment_test + (len(inventory_gp_test)*data_gp_test[t])  #total investment or cash in hand
        
            #stock_return = (w_gp*returns_gp_validation[t]) + (w_sqph*returns_sqph_validation[t])               
            f.write("SELL")
            print("AI Trader sold gp: ", stocks_price_format(len(inventory_gp_test)*data_gp_test[t])," Profit: " + stocks_price_format(profit),"Investment= ",stocks_price_format(investment_test))
            
            for i in range(len(inventory_gp_test)):
                inventory_gp_test.pop(0)   # empty the gp inventory after selling all of them
            
        if action == 0: #hold
            f.write("HOLD")
            print("AI Trader is holding........","Investment= ",stocks_price_format(investment_test))                
                                                                            
        next_state = state_creator(data_gp_test,t+1,window_size,inventory_gp_test,investment_test,episode)
                
        if investment_test<=0 and len(inventory_gp_test)==0: #checking for bankcrapcy
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("AI Trader is bankcrapted")
            #scores.append(total_profit)
            print("########################")            
            break  # if bankcrapted end the seassion
                
        state = next_state  #assin next state to present state
    
    print("########################")
    print("TOTAL TEST PROFIT: {}".format(total_profit))
    print("########################")
    #f.close()

test()