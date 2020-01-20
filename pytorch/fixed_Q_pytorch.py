from IPython.core.debugger import set_trace
import os.path
from os import path
import random
import torch
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
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
input_size = 2*window_size+portfolio_size+investment_size

agent = Agent(state_size = input_size, action_size=3, seed=0)

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

sc = StandardScaler()
fit = sc.fit_transform(np.zeros([input_size,1]))

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
    state.append(windowed_data_gp[i+1] - windowed_data_gp[i])  # getting consequent price diffrences for gp 
    state.append(windowed_data_sqph[i+1] - windowed_data_sqph[i])  #getting consequent price diffrences for sqph
  state.append(gp_assest)
  state.append(sqph_assest)
  state.append(investment)
  state = np.array([state])   # normalized input(state)
  state = np.reshape(state, (-1,1))  
  state = sc.transform(state)
  state = np.reshape(state, (-1))  #converting state into vector(1D array)
  return state

dataset_gp = pd.read_csv('GRAE Historical Data 2010-2018.csv')
dataset_sqph = pd.read_csv('SQPH Historical Data 2010-2018.csv')
data_gp = list(dataset_gp['Price'])
data_sqph = list(dataset_sqph['Price'])
#returns_gp = list(dataset_gp['return'])
#returns_sqph = list(dataset_sqph['return'])
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
    scores_window = deque(maxlen=100)  # last 100 scores
    returns_gp = deque(maxlen=500)    #last 100 return of gp
    returns_sqph = deque(maxlen=500)  #last 100 return of sqph
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        print("Episode: {}/{}".format(episode, n_episodes))
        investment = 100000
        inventory_gp = []
        inventory_sqph = []
        rewards = []
        state = state_creator(data_gp, data_sqph, 0, window_size + 1, inventory_gp, inventory_sqph, investment)
        total_profit = 0
        stock_return_ = 0
        return_gp_ = 0
        return_sqph_ = 0
        portfolio_value_ = 0
        portfolio_value = 0
        len_gp_assest_ = 0
        len_sqph_assest_ = 0
        assest_gp_ = 0
        assest_sqph_ = 0
        for t in range(0,max_t):
            action = agent.act(state, eps)  #return action(0=hold, 1=buy, 2=sell)          
            reward = 0
            stock_return = 0
            if len(returns_gp)<500 and len(returns_sqph)<500:
                
                w_gp, w_sqph = np.random.dirichlet(np.ones(2),size=1).squeeze()
                
            #if action == 0 and len(inventory_gp)==0 and len(inventory_sqph)==0:
                #print("AI Trader is holding.........", "Reward is: ",stocks_price_format(reward))
            
            if action == 1: #Buying
                if investment >= data_gp[t] and investment >= data_sqph[t]:
                    if len(returns_gp)==500 and len(returns_sqph)==500:
                        cov = np.cov(np.array(returns_gp),np.array(returns_sqph))
                        w = maximize_return.markwitz_portpolio([statistics.mean(returns_gp),statistics.mean(returns_sqph)],cov)
                        w_gp = w[0]
                        w_sqph = w[1]
                            
                    investment_gp = (investment*w_gp)
                    if investment_gp >= data_gp[t]:
                        no_stock_gp = int(investment_gp/data_gp[t]) #no of stock to buy
                    else:
                        no_stock_gp = 0
                        
                    investment_sqph = (investment*w_sqph)
                    if investment_sqph >= data_sqph[t]:
                        no_stock_sqph = int(investment_sqph/data_sqph[t])  #no of stock to buy
                    else:
                        no_stock_sqph = 0
                    
                    for i in range(no_stock_gp):
                        investment = investment - data_gp[t]  #decrease the investment after buying each stock
                        inventory_gp.append(data_gp[t])
                    for i in range(no_stock_sqph):
                        investment = investment - data_sqph[t]  #decrease the investment after buying each stock
                        inventory_sqph.append(data_sqph[t])

                    if len_sqph_assest_>0:
                        return_sqph = ((len(inventory_sqph)*data_sqph[t])-(len_sqph_assest_*data_sqph[t-1]))/(len_sqph_assest_*data_sqph[t-1]) #return of sqph
                        #if return_sqph>=1:
                            #set_trace()
                        returns_sqph.append(return_sqph)
                    else:
                        return_sqph = 0
                        #returns_sqph.append(return_sqph)
                        
                    if len_gp_assest_>0:
                        return_gp = ((len(inventory_gp)*data_gp[t])-(len_gp_assest_*data_gp[t-1]))/(len_gp_assest_*data_gp[t-1])   #return of gp
                        #if return_gp>=1:
                            #set_trace()
                        returns_gp.append(return_gp)                
                    else:
                        return_gp = 0
                        #returns_sqph.append(return_gp)
                        
                    stock_return = (w_gp*return_gp) + (w_sqph*return_sqph)   #return of gp and sqph together
                    portfolio_value = len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t]
                    reward = stock_return #reward is (present stock return) 
                    rewards.append(reward)
                    print("AI Trader bought gp: ", stocks_price_format(investment_gp),"AI Trader bought sqph: ", stocks_price_format(investment_sqph),"Reward is: ",  stocks_price_format(reward))
                 
            if action == 2 and (int(len(inventory_gp)*0.8)>0 or int(len(inventory_sqph)*0.8)>0): #Selling
                buy_prices_gp = []
                buy_prices_sqph = []
                no_sell_gp = int(len(inventory_gp)*0.8)
                no_sell_sqph = int(len(inventory_sqph)*0.8)
                
                for i in range(no_sell_gp):
                    buy_prices_gp.append(inventory_gp[i])
                buy_price_gp = sum(buy_prices_gp)  #buying price of gp stocks
                
                for i in range(no_sell_sqph):
                    buy_prices_sqph.append(inventory_sqph[i]) 
                buy_price_sqph = sum(buy_prices_sqph)  #buying price of sqph stocks
                
                buy_price = buy_price_gp + buy_price_sqph
                
                assest_gp_ = len(inventory_gp)*data_gp[t]
                assest_sqph_ = len(inventory_sqph)*data_sqph[t]
                total_profit += (no_sell_gp*data_gp[t] + no_sell_sqph*data_sqph[t]) - buy_price
                profit = (no_sell_gp*data_gp[t] + no_sell_sqph*data_sqph[t]) - buy_price
                #if profit == 0:
                    #print("nothing")
                investment = investment + (no_sell_gp*data_gp[t] + no_sell_sqph*data_sqph[t])  #total investment or cash in hand
                revenue = no_sell_gp*data_gp[t] + no_sell_sqph*data_sqph[t]
                if len(inventory_sqph) > 0:
                    return_sqph = ((len(inventory_sqph)*data_sqph[t])-(len_sqph_assest_*data_sqph[t-1]))/(len_sqph_assest_*data_sqph[t-1])  #return of sqph
                    #if return_sqph>=1:
                            #set_trace()
                    returns_sqph.append(return_sqph)
                    
                if len(inventory_gp) > 0:
                    return_gp = ((len(inventory_gp)*data_gp[t])-(len_gp_assest_*data_gp[t-1]))/(len_gp_assest_*data_gp[t-1])   #return of gp
                    #if return_gp>=1:
                            #set_trace()
                    returns_gp.append(return_gp)                
                    
                for i in range(no_sell_gp):
                    inventory_gp.pop(0)   # empty the gp inventory after selling all of them
                for i in range(no_sell_sqph):
                    inventory_sqph.pop(0)  #empy the sqph inventory after selling all of them

                stock_return = (w_gp*return_gp) + (w_sqph*return_sqph)  #return of gp and sqph together
                #portfolio_value = len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t] + revenue
                reward = stock_return    #reward is (present stock return)           
                rewards.append(reward)
                                
                print("AI Trader sold gp: ", stocks_price_format(buy_price_gp),"AI Trader sold sqph: ", stocks_price_format(buy_price_sqph), " Profit: " + stocks_price_format(profit),"Reward is: ",  stocks_price_format(reward))

            if action == 0: #hold
                if len(inventory_sqph) > 0:                    
                    return_sqph = ((len(inventory_sqph)*data_sqph[t])-(len_sqph_assest_*data_sqph[t-1]))/(len_sqph_assest_*data_sqph[t-1])  #return of sqph
                    #if return_sqph>=1:
                            #set_trace()
                    returns_sqph.append(return_sqph)
                else:
                    return_sqph = 0
                    #returns_sqph.append(return_sqph)
                    
                
                if len(inventory_gp) > 0:    
                    return_gp = ((len(inventory_gp)*data_gp[t])-(len_gp_assest_*data_gp[t-1]))/(len_gp_assest_*data_gp[t-1]) #return of gp
                    #if return_gp>=1:
                            #set_trace()
                    returns_gp.append(return_gp)                
                else:
                    return_gp = 0
                    
                stock_return = (w_gp*return_gp) + (w_sqph*return_sqph) #return of gp and sqph together
                portfolio_value = len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t]
                reward = stock_return  #reward is (present stock return)           
                rewards.append(reward)
                print("AI Trader is holding........", "Reward is: ",stocks_price_format(reward))                
                                                                        
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
            portfolio_value_ = portfolio_value
            len_gp_assest_ = len(inventory_gp)
            len_sqph_assest_ = len(inventory_sqph)
            if done:
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                scores.append(total_profit)
                print("########################")
            
        eps = max(eps_end, eps_decay*eps) # decrease epsilon after finishing each seassion
    return total_profit

#scores = dqn()
############################################################################################################

"""torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_local.pth')
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_target.pth')
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
"""
############################################################################################################
"""Test the agent over training set"""
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local_good.pth'))
agent.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target_good.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load the dataset
dataset_gp_test = pd.read_csv('GRAE Historical Data 2019.csv')
data_gp_test = list(dataset_gp_test['Price'])
dataset_sqph_test = pd.read_csv('SQPH Historical Data 2019.csv')
data_sqph_test = list(dataset_sqph_test['Price'])

#setting up the parameter
data_samples = len(data_gp_test)-1
inventory_gp = []
inventory_sqph = []
returns_gp = deque(maxlen=100)
returns_sqph = deque(maxlen=100)
total_profit = 0
investment = 100000
stock_return_ = 0
return_gp_ = 0
return_sqph_ = 0
portfolio_value_ = 0
portfolio_value = 0
len_gp_assest_ = 0
len_sqph_assest_ = 0
assest_gp_ = 0
assest_sqph_ = 0

f = open("state.txt","w")
#testing loop
state = state_creator(data_gp_test,data_sqph_test,0,window_size + 1,inventory_gp,inventory_sqph,investment)

for t in range(0,data_samples-1):
    #next_state = state_creator(data_gp_test,data_sqph_test,t+1,window_size + 1,inventory_gp,inventory_sqph,investment)
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    f.write(str(sc.inverse_transform(state.cpu().data.numpy())))
    action = np.argmax(agent.qnetwork_local(state).cpu().data.numpy())
    
    if len(returns_gp)<100 or len(returns_sqph)<100:
        w_gp, w_sqph = 0.5, 0.5
    
    if action == 1:  #buy
        if investment >= data_gp[t] and investment >= data_sqph[t]:
            if len(returns_gp)==100 and len(returns_sqph)==100:
                cov = np.cov(np.array(returns_gp),np.array(returns_sqph))
                w = maximize_return.markwitz_portpolio([statistics.mean(returns_gp),statistics.mean(returns_sqph)],cov)
                w_gp = w[0]
                w_sqph = w[1]
                            
            investment_gp = (investment*w_gp)
            if investment_gp >= data_gp_test[t]:
                no_stock_gp = int(investment_gp/data_gp_test[t]) #no of stock to buy
            else:
                no_stock_gp = 0
                        
            investment_sqph = (investment*w_sqph)
            if investment_sqph >= data_sqph[t]:
                no_stock_sqph = int(investment_sqph/data_sqph_test[t])  #no of stock to buy
            else:
                no_stock_sqph = 0
                    
            for i in range(no_stock_gp):
                investment = investment - data_gp_test[t]  #decrease the investment after buying each stock
                inventory_gp.append(data_gp_test[t])
            for i in range(no_stock_sqph):
                investment = investment - data_sqph_test[t]  #decrease the investment after buying each stock
                inventory_sqph.append(data_sqph_test[t])
            
            if len_sqph_assest_>0:
                return_sqph = ((len(inventory_sqph)*data_sqph_test[t])-(len_sqph_assest_*data_sqph_test[t-1]))/(len_sqph_assest_*data_sqph_test[t-1]) #return of sqph
                returns_sqph.append(return_sqph)
            else:
                return_sqph = 0
                        
            if len_gp_assest_>0:
                return_gp = ((len(inventory_gp)*data_gp_test[t])-(len_gp_assest_*data_gp_test[t-1]))/(len_gp_assest_*data_gp_test[t-1])   #return of gp
                returns_gp.append(return_gp)                
            else:
                return_gp = 0
                    
            f.write("BUY")
            print("AI Trader bought gp: ", stocks_price_format(investment_gp),"AI Trader bought sqph: ", stocks_price_format(investment_sqph))
                 
    if action == 2 and (int(len(inventory_gp)*0.8)>0 or int(len(inventory_sqph)*0.8)>0): #Selling
        buy_prices_gp = []
        buy_prices_sqph = []
        no_sell_gp = int(len(inventory_gp)*0.8)
        no_sell_sqph = int(len(inventory_sqph)*0.8)
                
        for i in range(no_sell_gp):
            buy_prices_gp.append(inventory_gp[i])
            buy_price_gp = sum(buy_prices_gp)  #buying price of gp stocks
                
        for i in range(no_sell_sqph):
            buy_prices_sqph.append(inventory_sqph[i]) 
            buy_price_sqph = sum(buy_prices_sqph)  #buying price of sqph stocks
                
        buy_price = buy_price_gp + buy_price_sqph
                
        assest_gp_ = len(inventory_gp)*data_gp_test[t]
        assest_sqph_ = len(inventory_sqph)*data_sqph_test[t]
        total_profit += (no_sell_gp*data_gp_test[t] + no_sell_sqph*data_sqph_test[t]) - buy_price
        profit = (no_sell_gp*data_gp_test[t] + no_sell_sqph*data_sqph_test[t]) - buy_price
        investment = investment + (no_sell_gp*data_gp_test[t] + no_sell_sqph*data_sqph_test[t])  #total investment or cash in hand
        revenue = no_sell_gp*data_gp_test[t] + no_sell_sqph*data_sqph_test[t]
        
        if len(inventory_sqph) > 0:
            return_sqph = ((len(inventory_sqph)*data_sqph_test[t])-(len_sqph_assest_*data_sqph_test[t-1]))/(len_sqph_assest_*data_sqph_test[t-1])  #return of sqph
            returns_sqph.append(return_sqph)
                    
        if len(inventory_gp) > 0:
            return_gp = ((len(inventory_gp)*data_gp_test[t])-(len_gp_assest_*data_gp_test[t-1]))/(len_gp_assest_*data_gp_test[t-1])   #return of gp
            returns_gp.append(return_gp)                
                    
        for i in range(no_sell_gp):
            inventory_gp.pop(0)   # empty the gp inventory after selling all of them
        for i in range(no_sell_sqph):
            inventory_sqph.pop(0)  #empy the sqph inventory after selling all of them
                                
        f.write("SELL")
        print("AI Trader sold gp: ", stocks_price_format(buy_price_gp),"AI Trader sold sqph: ", stocks_price_format(buy_price_sqph), " Profit: " + stocks_price_format(profit))

    if action == 0: #hold
        if len(inventory_sqph) > 0:                    
            return_sqph = ((len(inventory_sqph)*data_sqph_test[t])-(len_sqph_assest_*data_sqph_test[t-1]))/(len_sqph_assest_*data_sqph_test[t-1])  #return of sqph
            returns_sqph.append(return_sqph)
        else:
            return_sqph = 0
                                    
        if len(inventory_gp) > 0:    
            return_gp = ((len(inventory_gp)*data_gp_test[t])-(len_gp_assest_*data_gp_test[t-1]))/(len_gp_assest_*data_gp_test[t-1]) #return of gp
            returns_gp.append(return_gp)                
        else:
            return_gp = 0
                    
        f.write("HOLD")
        print("AI Trader is holding........")                
                                                                        
    next_state = state_creator(data_gp_test,data_sqph_test,t+1,window_size + 1,inventory_gp,inventory_sqph,investment)
    #print("state is: ", sc.inverse_transform(state.cpu().data.numpy()))
            
    if investment<=0 and len(inventory_gp)==0 and len(inventory_sqph)==0: #checking for bankcrapcy
        print("########################")
        print("TOTAL PROFIT: {}".format(total_profit))
        print("AI Trader is bankcrapted")
        scores.append(total_profit)
        print("########################")            
        break  # if bankcrapted end the seassion
            
    state = next_state  #assin next state to present state
    len_gp_assest_ = len(inventory_gp)
    len_sqph_assest_ = len(inventory_sqph)

print("########################")
print("TOTAL PROFIT: {}".format(total_profit))
print("########################")
f.close()