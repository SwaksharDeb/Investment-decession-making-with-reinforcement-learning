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

window_size = 10
trend_regression = 1
trend_classification = 1
sentiment_score = 1

agent = Agent(state_size = 2*window_size, action_size=6, seed=0)

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

def state_creator(data_gp, data_sqph, timestep, window_size):
  
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
  if starting_id >= 0:
    windowed_data_gp = data_gp[starting_id:timestep+1]
    windowed_data_sqph = data_sqph[starting_id:timestep+1]
  else:
    windowed_data_gp = - starting_id * [data_gp[0]] + list(data_gp[0:timestep+1])
    windowed_data_sqph = - starting_id * [data_sqph[0]] + list(data_sqph[0:timestep+1])
  state = []
  for i in range(window_size - 1):
    state.append(sigmoid(windowed_data_gp[i+1] - windowed_data_gp[i]))
    state.append(sigmoid(windowed_data_sqph[i+1] - windowed_data_sqph[i]))
  """score = sentiment[timestep,0]
  state.append(y_pred)
  state.append(diffrence)
  state.append(score)"""
  state = np.array([state])
  state = np.reshape(state, (-1))
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
    scores_window = deque(maxlen=500)  # last 1500 scores
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        print("Episode: {}/{}".format(episode, n_episodes))
        state = state_creator(data_gp, data_sqph, 0, window_size + 1)
        total_profit = 0        
        inventory_gp = []
        inventory_sqph = []
        portfolio_returns_gp = []
        portfolio_returns_sqph = []
        for t in range(max_t):
            action_gp, action_sqph = agent.act(state, eps)
            #print("action_gp is: ", action_gp)
            #print("action_sqph is: ", action_sqph)
            #action_values = np.array(action_values)
            #prob_array = softmax(action_values)
            #prob = (prob_array[0][action])*100
            #no_units = int(round(mul_factor*prob))
            next_state = state_creator(data_gp, data_sqph, t+1, window_size + 1)
            reward = 0
            
            if action_gp == 1: #Buying gp
                #if no_units <=0:
                    #no_units = 1
                #for i in range(no_units):                    
                inventory_gp.append(data_gp[t])
                print("AI Trader bought gp: ", stocks_price_format(data_gp[t]))
                
                if len(inventory_gp)>=2 and len(inventory_sqph)>=2:
                    portfolio_returns_gp.append(((len(inventory_gp)*data_gp[t])-(len(inventory_gp)*data_gp[t-1]))/len(inventory_gp)*data_gp[t])
                    portfolio_returns_sqph.append(((len(inventory_sqph)*data_sqph[t])-(len(inventory_sqph)*data_sqph[t-1]))/len(inventory_sqph)*data_sqph[t])
                    stock_return = (0.5*portfolio_returns_gp[len(portfolio_returns_gp)-1]) + ((0.5*portfolio_returns_sqph[len(portfolio_returns_sqph)-1]))
                    if len(portfolio_returns_gp)>=2 and len(portfolio_returns_sqph)>=2:
                        standard_deviation_gp = statistics.stdev(portfolio_returns_gp)
                        standard_deviation_sqph = statistics.stdev(portfolio_returns_sqph)
                        correlation, _ = pearsonr(np.array(portfolio_returns_gp), np.array(portfolio_returns_sqph))
                        standard_deviation = math.sqrt((0.5**2*standard_deviation_gp**2)+(0.5**2*standard_deviation_sqph**2)+(2*0.5*0.5*correlation*standard_deviation_gp*standard_deviation_sqph))
                        sharpe_ratio = stock_return / standard_deviation
                        reward = sharpe_ratio
                        #print("reward is : ",reward)

            if action_gp == 2 and len(inventory_gp) > 0: #Selling gp
                """buy_prices = []
                no_units = min(no_units, len(inventory_gp))
                for i in range(no_units):
                    buy_prices.append(inventory_gp.pop(0))
                buy_price = 0 
                for i in range(len(buy_prices)):
                    buy_price += buy_prices[i]
                total_profit += (no_units*data[t] - buy_price)
                if (no_units*data[t] - buy_price) > 0:
                    reward = 1
                elif (no_units*data[t] - buy_price) == 0:
                    reward = 0
                else:
                    reward = -1
                #reward = data[t] - buy_price
                """
                buy_price = inventory_gp.pop(0)
                total_profit += data_gp[t] - buy_price
                profit = data_gp[t] - buy_price
                print("AI Trader sold gp: ", stocks_price_format(data_gp[t]), " Profit: " + stocks_price_format(data_gp[t] - buy_price))
                
                if len(inventory_gp)>=1 and len(inventory_sqph)>=1:
                    portfolio_returns_gp.append(((len(inventory_gp)*data_gp[t])-(len(inventory_gp)*data_gp[t-1])+profit)/len(inventory_gp)*data_gp[t])
                    portfolio_returns_sqph.append(((len(inventory_sqph)*data_sqph[t])-(len(inventory_sqph)*data_sqph[t-1]))/len(inventory_sqph)*data_sqph[t])
                    stock_return = (0.5*portfolio_returns_gp[len(portfolio_returns_gp)-1]) + ((0.5*portfolio_returns_sqph[len(portfolio_returns_sqph)-1]))
                    if len(portfolio_returns_gp)>=2 and len(portfolio_returns_sqph)>=2:
                        standard_deviation_gp = statistics.stdev(portfolio_returns_gp)
                        standard_deviation_sqph = statistics.stdev(portfolio_returns_sqph)
                        correlation, _ = pearsonr(np.array(portfolio_returns_gp), np.array(portfolio_returns_sqph))
                        standard_deviation = math.sqrt((0.5**2*standard_deviation_gp**2)+(0.5**2*standard_deviation_sqph**2)+(2*0.5*0.5*correlation*standard_deviation_gp*standard_deviation_sqph))
                        sharpe_ratio = stock_return / standard_deviation
                        reward = sharpe_ratio
                        #print("reward is : ",reward)
                                    
            if action_sqph == 1: #Buying sqph
                #if no_units <=0:
                    #no_units = 1
                #for i in range(no_units):                    
                inventory_sqph.append(data_sqph[t])
                print("AI Trader bought sqph: ", stocks_price_format(data_sqph[t]))
                
                if len(inventory_gp)>=1 and len(inventory_sqph)>=1:
                    portfolio_returns_gp.append(((len(inventory_gp)*data_gp[t])-(len(inventory_gp)*data_gp[t-1]))/len(inventory_gp)*data_gp[t])
                    portfolio_returns_sqph.append(((len(inventory_sqph)*data_sqph[t])-(len(inventory_sqph)*data_sqph[t-1]))/len(inventory_sqph)*data_sqph[t])
                    stock_return = (0.5*portfolio_returns_gp[len(portfolio_returns_gp)-1]) + ((0.5*portfolio_returns_sqph[len(portfolio_returns_sqph)-1]))
                    if len(portfolio_returns_gp)>=2 and len(portfolio_returns_sqph)>=2:
                        standard_deviation_gp = statistics.stdev(portfolio_returns_gp)
                        standard_deviation_sqph = statistics.stdev(portfolio_returns_sqph)
                        correlation, _ = pearsonr(np.array(portfolio_returns_gp), np.array(portfolio_returns_sqph))
                        standard_deviation = math.sqrt((0.5**2*standard_deviation_gp**2)+(0.5**2*standard_deviation_sqph**2)+(2*0.5*0.5*correlation*standard_deviation_gp*standard_deviation_sqph))
                        sharpe_ratio = stock_return / standard_deviation
                        reward = sharpe_ratio
                        #print("reward is : ",reward)

            if action_sqph == 2 and len(inventory_sqph) > 0: #Selling gp
                buy_price = inventory_sqph.pop(0)
                total_profit += data_sqph[t] - buy_price
                profit = data_sqph[t] - buy_price
                print("AI Trader sold sqph: ", stocks_price_format(data_sqph[t]), " Profit: " + stocks_price_format(data_sqph[t] - buy_price))
                
                if len(inventory_gp)>=1 and len(inventory_sqph)>=1:
                    portfolio_returns_gp.append(((len(inventory_gp)*data_gp[t])-(len(inventory_gp)*data_gp[t-1]))/len(inventory_gp)*data_gp[t])
                    portfolio_returns_sqph.append(((len(inventory_sqph)*data_sqph[t])-(len(inventory_sqph)*data_sqph[t-1])+profit)/len(inventory_sqph)*data_sqph[t])
                    stock_return = (0.5*portfolio_returns_gp[len(portfolio_returns_gp)-1]) + ((0.5*portfolio_returns_sqph[len(portfolio_returns_sqph)-1]))
                    if len(portfolio_returns_gp)>=2 and len(portfolio_returns_sqph)>=2:
                        standard_deviation_gp = statistics.stdev(portfolio_returns_gp)
                        standard_deviation_sqph = statistics.stdev(portfolio_returns_sqph)
                        correlation, _ = pearsonr(np.array(portfolio_returns_gp), np.array(portfolio_returns_sqph))
                        standard_deviation = math.sqrt((0.5**2*standard_deviation_gp**2)+(0.5**2*standard_deviation_sqph**2)+(2*0.5*0.5*correlation*standard_deviation_gp*standard_deviation_sqph))
                        sharpe_ratio = stock_return / standard_deviation
                        reward = sharpe_ratio
                        #print("reward is : ",reward)
                    
            if t == data_samples - 1:
                done = True
            else:
                done = False
                
            agent.step(state, action_gp, action_sqph, reward, next_state, done)
            state = next_state
            if done:
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                scores.append(total_profit)
                print("########################")
            
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
    return total_profit

scores = dqn()
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_local.pth')
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_target.pth')
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
"""Test the agent over training set"""
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local.pth'))
agent.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load the dataset
dataset_test = pd.read_csv('GRAE Historical Data 2018 -2019practice.csv')
data = list(dataset_test['Price'])

#setting up the parameter
data_samples = len(data)-1
inventory_gp = []
return_list = []
total_profit = 0

#testing loop
state = state_creator(data, 0, window_size + 1)

for t in range(data_samples):
    next_state = state_creator(data, t+1, window_size + 1)
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    action = np.argmax(agent.qnetwork_local(state).cpu().data.numpy())
    action_values = agent.qnetwork_local(state)
    no_buy = np.max(action_values.cpu().data.numpy())
    if action == 1:
        if no_buy <= 0:
            no_buy = 1
        inventory_gp.append(no_buy*data[t])
        print("AI Trader bought: ", stocks_price_format(no_buy*data[t]))
        
    if action == 2 and len(inventory_gp)>0:
        buy_prices = []
        no_sell = len(inventory_gp)
        for i in range(len(inventory_gp)):
            buy_prices.append(inventory_gp.pop(0))
        buy_price = sum(buy_prices)
        total_profit += (no_sell*data[t] - buy_price)
        print("AI Trader sold: ", stocks_price_format(no_sell*data[t]), " Profit: " + stocks_price_format(no_sell*data[t] - buy_price))
    
    state = next_state
    
print("########################")
print("TOTAL PROFIT: {}".format(total_profit))
print("########################")