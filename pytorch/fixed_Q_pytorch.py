from IPython.core.debugger import set_trace
import os.path
from os import path
import gym
import random
import torch
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import pandas as pd
import statistics
import math
from scipy.stats import pearsonr
import maximize_return

window_size = 10
portfolio_size = 2
investment_size = 1

agent = Agent(state_size = 2*window_size+portfolio_size+investment_size, action_size=3, seed=0)

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

def state_creator(data_gp, data_sqph, timestep, window_size, inventory_gp, inventory_sqph, investment):
  
  starting_id = timestep - window_size + 1
  
  gp_assest = len(inventory_gp) * data_gp[timestep]  #portfolio value for gp at current timestep
  sqph_assest = len(inventory_sqph) * data_sqph[timestep]  #portfolio value for sqph for sqph at current timestep
  
  if starting_id >= 0:
    windowed_data_gp = data_gp[starting_id:timestep+1]
    windowed_data_sqph = data_sqph[starting_id:timestep+1]
  else:
    windowed_data_gp = - starting_id * [data_gp[0]] + list(data_gp[0:timestep+1])
    windowed_data_sqph = - starting_id * [data_sqph[0]] + list(data_sqph[0:timestep+1])
  state = []
  for i in range(window_size - 1):
    state.append(sigmoid(windowed_data_gp[i+1] - windowed_data_gp[i]))  # getting consequent price diffrences for gp 
    state.append(sigmoid(windowed_data_sqph[i+1] - windowed_data_sqph[i]))  #getting consequent price diffrences for sqph
  state.append(sigmoid(gp_assest))
  state.append(sigmoid(sqph_assest))
  state.append(sigmoid(investment))
  state = np.array([state])   # normalized input(state)
  state = np.reshape(state, (-1))  #converting state into vector(1D array)
  return state

dataset_gp = pd.read_csv('GRAE Historical Data 2018.csv')
dataset_sqph = pd.read_csv('SQPH Historical Data2018 practice.csv')
data_gp = list(dataset_gp['Price'])
data_sqph = list(dataset_sqph['Price'])
data_samples = len(data_gp)-1

scores = []

def dqn(n_episodes=5000, max_t=len(data_gp)-1, eps_start=1, eps_end=0.001, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    max_num = 5   # buy or sell maximum 5 stocks
    mul_factor = max_num/100
    scores_window = deque(maxlen=100)  # last 100 scores
    returns_gp = deque(maxlen=100)    #last 100 return of gp
    returns_sqph = deque(maxlen=100)  #last 100 return of sqph
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        print("Episode: {}/{}".format(episode, n_episodes))
        investment = 100000
        inventory_gp = []
        inventory_sqph = []
        state = state_creator(data_gp, data_sqph, 1, window_size + 1, inventory_gp, inventory_sqph, investment)
        total_profit = 0
        stock_return_ = 0
        return_gp_ = 0
        return_sqph_ = 0
        for t in range(1,max_t):
            action = agent.act(state, eps)  #return action(0=hold, 1=buy, 2=sell)          
            reward = 0
            if len(returns_gp)<100 and len(returns_sqph)<100: 
                w_gp, w_sqph = 0.5, 0.5
                    
            if action == 0:
                print("AI Trader is holding........")
            
            if action == 1: #Buying
                if investment >= data_gp[t] and investment >= data_sqph[t]:
                    if len(returns_gp)==100 and len(returns_sqph)==100:
                        cov = np.cov(np.array(returns_gp),np.array(returns_sqph))
                        w = maximize_return.markwitz_portpolio([statistics.mean(returns_gp),statistics.mean(returns_sqph)],cov)
                        w_gp = w[0]
                        w_sqph = w[1]
                        
                    investment_gp = (investment*w_gp)
                    if investment_gp >= data_gp[t]:
                        no_stock_gp = int(round(investment_gp/data_gp[t])+1) #no of stock to buy
                    else:
                        no_stock_gp = 0
                        
                    investment_sqph = (investment*w_sqph)
                    if investment_sqph >= data_sqph[t]:
                        no_stock_sqph = int(round(investment_sqph/data_sqph[t])+1)  #no of stock to buy
                    else:
                        no_stock_sqph = 0
                    
                    for i in range(no_stock_gp):
                        investment = investment - data_gp[t]  #decrease the investment after buying each stock
                        inventory_gp.append(data_gp[t])
                    for i in range(no_stock_sqph):
                        investment = investment - data_sqph[t]  #decrease the investment after buying each stock
                        inventory_sqph.append(data_sqph[t])
                        
                    if no_stock_sqph == 0:
                        return_sqph = 0
                    else:
                        return_sqph = (((len(inventory_sqph)*data_sqph[t])-(len(inventory_sqph)*data_sqph[t-1]))/len(inventory_sqph)*data_sqph[t-1]) #return of sqph
                        returns_sqph.append(return_sqph)
                    if no_stock_gp == 0:
                        return_gp = 0                        
                    else:
                        return_gp = (((len(inventory_gp)*data_gp[t])-(len(inventory_gp)*data_gp[t-1]))/len(inventory_gp)*data_gp[t-1])   #return of gp
                        returns_gp.append(return_gp)                
                    
                    stock_return = (w_gp*return_gp) + (w_sqph*return_sqph)   #return of gp and sqph together
                    reward = stock_return - stock_return_  #reward is (present stock return - yesterday stock return) 
                    print("AI Trader bought gp: ", stocks_price_format(investment_gp),"AI Trader bought sqph: ", stocks_price_format(investment_sqph))
                 
            if action == 2 and (len(inventory_gp) > 0 or len( inventory_sqph) > 0): #Selling
                buy_prices_gp = []
                buy_prices_sqph = []
                
                for i in range(len(inventory_gp)):
                    buy_prices_gp.append(inventory_gp[i])
                buy_price_gp = sum(buy_prices_gp)  #buying price of gp stocks
                
                for i in range(len(inventory_sqph)):
                    buy_prices_sqph.append(inventory_sqph[i]) 
                buy_price_sqph = sum(buy_prices_sqph)  #buying price of sqph stocks
                
                buy_price = buy_price_gp + buy_price_sqph
                
                total_profit += (len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t]) - buy_price
                profit = (len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t]) - buy_price
                #if profit == 0:
                    #print("nothing")
                investment = investment + (len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t])  #total investment or cash in hand
                
                if len(inventory_sqph) > 0:
                    return_sqph = (((len(inventory_sqph)*data_sqph[t])-(len(inventory_sqph)*data_sqph[t-1])+profit)/len(inventory_sqph)*data_sqph[t-1])  #return of sqph
                    returns_sqph.append(return_sqph)
                else:
                    return_sqph = 0
                if len(inventory_gp) > 0:
                    return_gp = (((len(inventory_gp)*data_gp[t])-(len(inventory_gp)*data_gp[t-1])+profit)/len(inventory_gp)*data_gp[t-1])   #return of gp
                    returns_gp.append(return_gp)                
                else:
                    return_gp = 0
                    
                stock_return = (w_gp*return_gp) + (w_sqph*return_sqph)  #return of gp and sqph together
                reward = stock_return - stock_return_    #reward is (present stock return - yesterday stock return)           
                            
                for i in range(len(inventory_gp)):
                    inventory_gp.pop(0)   # empty the gp inventory after selling all of them
                for i in range(len(inventory_sqph)):
                    inventory_sqph.pop(0)  #empy the sqph inventory after selling all of them
                
                print("AI Trader sold gp: ", stocks_price_format(buy_price_gp),"AI Trader sold sqph: ", stocks_price_format(buy_price_sqph), " Profit: " + stocks_price_format(profit), )

            if action == 0 and len(inventory_gp)>0 or len(inventory_sqph)>0: #hold
                if len(inventory_sqph) > 0:                    
                    return_sqph = (((len(inventory_sqph)*data_sqph[t])-(len(inventory_sqph)*data_sqph[t-1]))/len(inventory_sqph)*data_sqph[t-1])  #return of sqph
                    returns_sqph.append(return_sqph)
                else:
                    return_sqph = 0
                
                if len(inventory_gp) > 0:    
                    return_gp = (((len(inventory_gp)*data_gp[t])-(len(inventory_gp)*data_gp[t-1]))/len(inventory_gp)*data_gp[t-1]) #return of gp
                    returns_gp.append(return_gp)                
                else:
                    return_gp = 0
                
                stock_return = (w_gp*return_gp) + (w_sqph*return_sqph) #return of gp and sqph together
                reward = stock_return - stock_return_    #reward is (present stock return - yesterday stock return)            
                                                                                        
            next_state = state_creator(data_gp, data_sqph, t+1, window_size + 1, inventory_gp, inventory_sqph, investment)
            
            if investment<=0 and len(inventory_gp)==0 and len(inventory_sqph)==0: #checking for bankcrapcy
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
            state = next_state  #assin next state to present state
            stock_return_ = stock_return  #assin present's stock return to yesterday's stock return
            if done:
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                scores.append(total_profit)
                print("########################")
            
        eps = max(eps_end, eps_decay*eps) # decrease epsilon after finishing each seassion
    return total_profit

scores = dqn()

############################################################################################################

torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_local.pth')
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_target.pth')
############################################################################################################

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
