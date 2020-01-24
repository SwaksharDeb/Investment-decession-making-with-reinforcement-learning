from IPython.core.debugger import set_trace
import os.path
from os import path
import random
import torch
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import pandas as pd
import statistics
import math
from scipy.stats import pearsonr
import math
import maximize_return

window_size = 10
portfolio_size = 2
investment_size = 1
input_size = 2*window_size+portfolio_size+investment_size

agent = Agent(state_size = input_size, action_size=3, seed=0)
#agent = Agent(state_size = 5, action_size=3, seed=0)
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

"""##Loading ANN_gp for trend analysis(classification)"""

#Load the pretrained model
"""with open('GP classification with factor.json', 'r') as f:
    model_json = f.read()

classifier = tf.keras.models.model_from_json(model_json)
# load weights into new model
classifier.load_weights("GP classification with factor.json.h5")

#loading the dataset
dataset_2 = pd.read_csv('GRAE Historical Data 2018 -2019practice.csv')
dataset_1 = pd.read_csv('GRAE Historical Data 2009-2017.csv')
sentiment = dataset_2.iloc[:, 16:17].values
X_classifier = dataset_2.iloc[:, [7,11,12,13,14]].values
y_classifier = dataset_2.iloc[:, 15:16].values

#feature scaling
sc_1 = MinMaxScaler()
X_classifier = sc_1.fit_transform(X_classifier)
"""

"""Loading LSTM_sqph for trend analaysis(regression)"""

#Load the pretrained model
"""with open('gp prediction with factor.json', 'r') as f:
    modelgp_json = f.read()

regressor = tf.keras.models.model_from_json(modelgp_json)

# load weights into new model
regressor.load_weights("gp prediction with factor.json.h5")

#preprocessing

dataset_test_1 = dataset_1.iloc[:,[1,7,11,12,13,14]]
dataset_test_1 = dataset_test_1.iloc[-60:,:] 
dataset_test_2 = dataset_2.iloc[:,[1,7,11,12,13,14]]
dataset_test = pd.concat([dataset_test_1, dataset_test_2], axis = 0, ignore_index=True, sort=False)
test_set = dataset_test.iloc[:,1:].values
test_set_y = dataset_test.iloc[:, 0:1].values

inputs = test_set[:,:]
sc_2 = MinMaxScaler(feature_range = (0, 1))
inputs = sc_1.transform(inputs)
test_set_scaled_y = sc_2.fit_transform(test_set_y)

X_regressor = []
for i in range(60, len(test_set)):
    X_regressor.append(inputs[i-60:i, :])

X_regressor[0] = np.reshape(X_regressor[0], (1,-1))
array = np.reshape(X_regressor[0],(1,60,-1))
 
for i in range(1,len(X_regressor)):
    X_regressor[i] = np.reshape(X_regressor[i],(1,-1))
    X_regressor[i] = np.reshape(X_regressor[i],(1,60,-1))
    array = np.vstack((array,X_regressor[i]))

X_regressor = array

y_regressor = []
for i in range(60,len(test_set_scaled_y)):
    y_regressor.append(test_set_scaled_y[i,0])

y_regressor = np.array(y_regressor)
y_regressor = np.reshape(y_regressor, (-1,1))
"""
sc = MinMaxScaler()
def state_creator(data_gp, data_sqph, timestep, window_size, inventory_gp, inventory_sqph, investment):
  
  starting_id = timestep - window_size + 1
  
  """y_pred = classifier.predict(np.reshape(X_classifier[timestep],(1,-1)))
  
  if y_pred>0.5:
    y_pred = 1
  else:
    y_pred = 0

  #predicting the price
  predicted_next_price = sc_2.inverse_transform(regressor.predict(np.reshape(X_regressor[timestep],(1,X_regressor[timestep].shape[0],X_regressor[timestep].shape[1]))))[0,0]
  if timestep > 0:
    predicted_present_price = sc_2.inverse_transform(regressor.predict(np.reshape(X_regressor[timestep-1],(1,X_regressor[timestep-1].shape[0],X_regressor[timestep-1].shape[1]))))[0,0]
  else:
      predicted_present_price = predicted_next_price
      
  diffrence = predicted_next_price - predicted_present_price
  if diffrence>0:
      diffrence = 1
  else:
      diffrence = 0
  """    
  gp_assest = len(inventory_gp) * data_gp[timestep]  #portfolio value for gp at current timestep
  sqph_assest = len(inventory_sqph) * data_sqph[timestep]  #portfolio value for sqph for sqph at current timestep
  
  """if starting_id >= 0:
    windowed_data_gp = data_gp[starting_id:timestep+1]
    windowed_data_sqph = data_sqph[starting_id:timestep+1]
  else:
    windowed_data_gp = - starting_id * [data_gp[0]] + list(data_gp[0:timestep+1])
    windowed_data_sqph = - starting_id * [data_sqph[0]] + list(data_sqph[0:timestep+1])

  state = []
  for i in range(window_size - 1):
    #state.append(windowed_data_gp[i+1] - windowed_data_gp[i])  # getting consequent price diffrences for gp 
    #state.append(windowed_data_sqph[i+1] - windowed_data_sqph[i])  #getting consequent price diffrences for sqph

  score = sentiment[timestep,0]
  state.append(y_pred)
  state.append(diffrence)
  state.append(score)
  """
  state = []
  for i in range(timestep-window_size, timestep):
     state.append(data_gp[i])
     state.append(data_sqph[i])
              
  state.append(gp_assest)
  state.append(sqph_assest)
  state.append(investment)
  state = np.array([state])   # normalized input(state)
  state = np.reshape(state, (-1,1))  
  
  if timestep == 0:
      state = sc.fit_transform(state)
  else:
      state = sc.transform(state)
  
  state = np.reshape(state, (-1))  #converting state into vector(1D array)
  return state

dataset_gp = pd.read_csv('GRAE Historical Data 2018.csv')
dataset_sqph = pd.read_csv('SQPH Historical Data2018 practice.csv')
dataset_return_gp  = pd.read_csv("GRAE Historical Data 2017.csv")
dataset_return_sqph  = pd.read_csv("SQPH Historical Data 2017.csv")
data_gp = list(dataset_gp['Price'][10:])
data_sqph = list(dataset_sqph['Price'][10:])
returns_gp = list(dataset_gp['return'][10:])
returns_gp_ = list(dataset_return_gp['return'][-200:])
returns_sqph = list(dataset_sqph['return'][10:])
returns_sqph_ = list(dataset_return_sqph['return'][-200:])
data_samples = len(data_gp)-1

scores = []

def dqn(n_episodes=5000000, max_t=len(data_gp)-1, eps_start=1, eps_end=0, eps_decay=0.995):
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
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        print("Episode: {}/{}".format(episode, n_episodes))
        
        returns_gp_list = deque(maxlen=210)    #last 100 return of gp
        for i in range(len(returns_gp_)):
            returns_gp_list.append(returns_gp_[i])
        for i in range(10):
            returns_gp_list.append(dataset_gp['return'][i])
        
        returns_sqph_list = deque(maxlen=210)  #last 100 return of sqph        
        for i in range(len(returns_sqph_)):
            returns_sqph_list.append(returns_sqph_[i])
        for i in range(10):
            returns_sqph_list.append(dataset_sqph['return'][i])
            
        investment = 100000
        inventory_gp = []
        inventory_sqph = []
        rewards = []
        state = state_creator(data_gp, data_sqph, 0, window_size, inventory_gp, inventory_sqph, investment)
        total_profit = 0
        total_reward = 0
        for t in range(0,max_t):
            action = agent.act(state, eps)  #return action(0=hold, 1=buy, 2=sell)          
            reward = 0
            stock_return = 0
            if action == 1: #Buying
                if investment >= data_gp[t] and investment >= data_sqph[t]:
                    if len(returns_gp_list)==210 and len(returns_sqph_list)==210:
                        cov = np.cov(np.array(returns_gp_list),np.array(returns_sqph_list))
                        #cov_p = np.cov(np.array(price_lookback_gp),np.array(price_lookback_sqph))
                        w = maximize_return.markwitz_portpolio([statistics.mean(returns_gp_list),statistics.mean(returns_sqph_list)],cov)
                        sharpe_ratio = maximize_return.sharpe_ratio(w,[statistics.mean(returns_gp_list),statistics.mean(returns_sqph_list)],cov)
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

                    returns_sqph_list.append(returns_sqph[t])
                    returns_gp_list.append(returns_gp[t])
                    #price_lookback_gp.append(data_gp[t])
                    #price_lookback_sqph.append(data_sqph[t])
                    stock_return = (w_gp*returns_gp[t]) + (w_sqph*returns_sqph[t])   #return of gp and sqph together
                    #portfolio_value = len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t]
                    #reward = sharpe_ratio
                    total_reward += reward 
                    rewards.append(reward)
                    investment_ = investment
                    print("AI Trader bought gp: ", stocks_price_format(investment_gp),"AI Trader bought sqph: ", stocks_price_format(investment_sqph),"Reward is: ",  stocks_price_format(reward))
                 
            if action == 2 and (len(inventory_gp) > 0 or len( inventory_sqph)) > 0: #Selling
                buy_prices_gp = []
                buy_prices_sqph = []
                for i in range(len(inventory_gp)):
                    buy_prices_gp.append(inventory_gp[i])
                buy_price_gp = sum(buy_prices_gp)  #buying price of gp stocks
                
                for i in range(len(inventory_sqph)):
                    buy_prices_sqph.append(inventory_sqph[i]) 
                buy_price_sqph = sum(buy_prices_sqph)  #buying price of sqph stocks
                
                buy_price = buy_price_gp + buy_price_sqph
                
                #assest_gp_ = len(inventory_gp)*data_gp[t]
                #assest_sqph_ = len(inventory_sqph)*data_sqph[t]
                total_profit += (len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t]) - buy_price
                profit = (len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t]) - buy_price
                investment = investment + (len(inventory_gp)*data_gp[t] + len(inventory_sqph)*data_sqph[t])  #total investment or cash in hand
                returns_sqph_list.append(returns_sqph[t])
                returns_gp_list.append(returns_gp[t])
                #price_lookback_gp.append(data_gp[t])
                #price_lookback_sqph.append(data_sqph[t])
                cov = np.cov(np.array(returns_gp_list),np.array(returns_sqph_list))
                #cov_p = cov_p = np.cov(np.array(price_lookback_gp),np.array(price_lookback_sqph))
                w =  maximize_return.markwitz_portpolio([statistics.mean(returns_gp_list),statistics.mean(returns_sqph_list)],cov)
                sharpe_ratio = maximize_return.sharpe_ratio(w,[statistics.mean(returns_gp_list),statistics.mean(returns_sqph_list)],cov) 
                w_gp = w[0]
                w_sqph = w[1]                
                stock_return = (w_gp*returns_gp[t]) + (w_sqph*returns_sqph[t])  #return of gp and sqph together
                if len(inventory_gp) > 0 and len( inventory_sqph) > 0:
                    reward = max(((data_gp[t]-inventory_gp[0])+(data_sqph[t]-inventory_sqph[0])),-1)    #reward is (present stock return - yesterday stock return)           
                elif len(inventory_gp)>0:
                    reward = max((data_gp[t]-inventory_gp[0]),-1)
                else:
                    reward = max((data_sqph[t]-inventory_sqph[0]),-1)
                
                for i in range(len(inventory_gp)):
                    inventory_gp.pop(0)   # empty the gp inventory after selling all of them
                for i in range(len(inventory_sqph)):
                    inventory_sqph.pop(0)  #empy the sqph inventory after selling all of them
                total_reward += reward
                rewards.append(reward)
                investment_ = investment
                print("AI Trader sold gp: ", stocks_price_format(buy_price_gp),"AI Trader sold sqph: ", stocks_price_format(buy_price_sqph), " Profit: " + stocks_price_format(profit),"Reward is: ",  stocks_price_format(reward))

            if action == 0: #hold
                returns_gp_list.append(returns_gp[t])
                returns_sqph_list.append(returns_sqph[t])
                cov = np.cov(np.array(returns_gp_list),np.array(returns_sqph_list))
                #cov_p = cov_p = np.cov(np.array(price_lookback_gp),np.array(price_lookback_sqph))
                w = maximize_return.markwitz_portpolio([statistics.mean(returns_gp_list),statistics.mean(returns_sqph_list)],cov)
                sharpe_ratio = maximize_return.sharpe_ratio(w,[statistics.mean(returns_gp_list),statistics.mean(returns_sqph_list)],cov) 
                w_gp = w[0]
                w_sqph = w[1]
                stock_return = (w_gp*returns_gp[t]) + (w_sqph*returns_sqph[t]) #return of gp and sqph together
                #reward = sharpe_ratio
                total_reward += reward
                rewards.append(reward)
                print("AI Trader is holding........", "Reward is: ",stocks_price_format(reward))                
                                                                        
            next_state = state_creator(data_gp, data_sqph, t+1, window_size, inventory_gp, inventory_sqph, investment)
            
            if investment<=0 and len(inventory_gp)==0 and len(inventory_sqph)==0: #checking for bankcrapcy
                reward = -10
                done = True
                agent.step(state, action, reward, next_state, done)
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("TOTAL REWARD: {}".format(total_reward))
                print("AI Trader is bankcrapted")
                scores.append(total_profit)
                print("########################")            
                break  # if bankcrapted end the seassion
            
            if t == data_samples - 1:
                done = True
            else:
                done = False
                
            agent.step(state, action, reward, next_state, done)  #save the sample and learn from it
            state = next_state  #assin next state to present state
            if done:
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("TOTAL REWARD: {}".format(total_reward))
                scores.append(total_profit)
                print("########################")
            
        eps = max(eps_end, eps_decay*eps) # decrease epsilon after finishing each seassion
    return total_profit

scores = dqn()
############################################################################################################
    
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_local.pth')
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_target.pth')
# plot the scores
fig = plt.figure()
#ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
"""
"""Test the agent over training set"""
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local.pth'))
agent.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load the dataset
dataset_gp_test = pd.read_csv('GRAE Historical Data 2019.csv')
data_gp_test = list(dataset_gp_test['Price'][10:])
dataset_sqph_test = pd.read_csv('SQPH Historical Data 2019.csv')
data_sqph_test = list(dataset_sqph_test['Price'][10:])
returns_gp_ = list(dataset_gp['return'][-200:])
returns_sqph_ = list(dataset_sqph['return'][-200:])
returns_gp = list(dataset_sqph_test['return'])
returns_sqph = list(dataset_sqph_test['return'])
#initializing return deque
returns_gp_list = deque(maxlen=210)    #last 200 return of gp
for i in range(len(returns_gp_)):
    returns_gp_list.append(returns_gp_[i])
for i in range(10):
    returns_gp_list.append(dataset_gp_test['return'][i])
returns_sqph_list = deque(maxlen=210)  #last 200 return of sqph
for i in range(len(returns_sqph_)):
    returns_sqph_list.append(returns_sqph_[i])
for i in range(10):
    returns_sqph_list.append(dataset_sqph_test['return'][i])
        
#setting up the parameter
data_samples = len(data_gp_test)-1
inventory_gp = []
inventory_sqph = []
total_profit = 0
investment = 100000

f = open("state.txt","w")
#testing loop
state = state_creator(data_gp_test,data_sqph_test,0,window_size,inventory_gp,inventory_sqph,investment)

for t in range(0,data_samples-1):
    #next_state = state_creator(data_gp_test,data_sqph_test,t+1,window_size + 1,inventory_gp,inventory_sqph,investment)
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    f.write(str(sc.inverse_transform(state.cpu().data.numpy())))
    action = np.argmax(agent.qnetwork_local(state).cpu().data.numpy())
        
    if action == 1:  #buy
        if investment >= data_gp[t] and investment >= data_sqph[t]:
            if len(returns_gp_list)==210 and len(returns_sqph_list)==210:
                cov = np.cov(np.array(returns_gp_list),np.array(returns_sqph_list))
                w = maximize_return.markwitz_portpolio([statistics.mean(returns_gp_list),statistics.mean(returns_sqph_list)],cov)
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
            
            returns_sqph_list.append(returns_sqph[t])
            returns_gp_list.append(returns_gp[t])
            stock_return = (w_gp*returns_gp[t]) + (w_sqph*returns_sqph[t])               
            f.write("BUY")
            print("AI Trader bought gp: ", stocks_price_format(investment_gp),"AI Trader bought sqph: ", stocks_price_format(investment_sqph))
                 
    if action == 2 and (len(inventory_gp)>0 or len(inventory_sqph)>0): #Selling
        buy_prices_gp = []
        buy_prices_sqph = []
                
        for i in range(len(inventory_gp)):
            buy_prices_gp.append(inventory_gp[i])
            buy_price_gp = sum(buy_prices_gp)  #buying price of gp stocks
                
        for i in range(len(inventory_sqph)):
            buy_prices_sqph.append(inventory_sqph[i]) 
            buy_price_sqph = sum(buy_prices_sqph)  #buying price of sqph stocks
                
        buy_price = buy_price_gp + buy_price_sqph
                
        total_profit += (len(inventory_gp)*data_gp_test[t] + len(inventory_sqph)*data_sqph_test[t]) - buy_price
        profit = (len(inventory_gp)*data_gp_test[t] + len(inventory_sqph)*data_sqph_test[t]) - buy_price
        investment = investment + (len(inventory_gp)*data_gp_test[t] + len(inventory_sqph)*data_sqph_test[t])  #total investment or cash in hand
        
        returns_sqph_list.append(returns_sqph[t])
        returns_gp_list.append(returns_gp[t])
        stock_return = (w_gp*returns_gp[t]) + (w_sqph*returns_sqph[t])               
            
        for i in range(len(inventory_gp)):
            inventory_gp.pop(0)   # empty the gp inventory after selling all of them
        for i in range(len(inventory_sqph)):
            inventory_sqph.pop(0)  #empy the sqph inventory after selling all of them
                                
        f.write("SELL")
        print("AI Trader sold gp: ", stocks_price_format(buy_price_gp),"AI Trader sold sqph: ", stocks_price_format(buy_price_sqph), " Profit: " + stocks_price_format(profit))

    if action == 0: #hold
        returns_sqph_list.append(returns_sqph[t])
        returns_gp_list.append(returns_gp[t])
        stock_return = (w_gp*returns_gp[t]) + (w_sqph*returns_sqph[t])               
        
        f.write("HOLD")
        print("AI Trader is holding........")                
                                                                        
    next_state = state_creator(data_gp_test,data_sqph_test,t+1,window_size,inventory_gp,inventory_sqph,investment)
    #print("state is: ", sc.inverse_transform(state.cpu().data.numpy()))
            
    if investment<=0 and len(inventory_gp)==0 and len(inventory_sqph)==0: #checking for bankcrapcy
        print("########################")
        print("TOTAL PROFIT: {}".format(total_profit))
        print("AI Trader is bankcrapted")
        scores.append(total_profit)
        print("########################")            
        break  # if bankcrapted end the seassion
            
    state = next_state  #assin next state to present state

print("########################")
print("TOTAL PROFIT: {}".format(total_profit))
print("########################")
f.close()