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

window_size = 10
trend_regression = 1
trend_classification = 1
sentiment_score = 1

agent = Agent(state_size = window_size+trend_regression+trend_classification+sentiment_score, action_size=3, seed=0)

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

"""##Loading ANN for trend analysis(classification)"""

#Load the pretrained model
with open('GP classification with factor.json', 'r') as f:
    model_json = f.read()

classifier = tf.keras.models.model_from_json(model_json)
# load weights into new model
classifier.load_weights("GP classification with factor.json.h5")

#loading the dataset
dataset_2 = pd.read_csv('GRAE Historical Data 2018 practice.csv')
dataset_1 = pd.read_csv('GRAE Historical Data 2009-2017.csv')
sentiment = dataset_2.iloc[:, 16:17].values
X_classifier = dataset_2.iloc[:, [7,11,12,13,14]].values
y_classifier = dataset_2.iloc[:, 15:16].values

#feature scaling
sc_1 = MinMaxScaler()
X_classifier = sc_1.fit_transform(X_classifier)


"""Loading LSTM for trend analaysis(regression)"""

#Load the pretrained model
with open('gp prediction with factor.json', 'r') as f:
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


def state_creator(data, timestep, window_size):
  
  starting_id = timestep - window_size + 1
  
  y_pred = classifier.predict(np.reshape(X_classifier[timestep],(1,-1)))
  
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
      
  if starting_id >= 0:
    windowed_data = data[starting_id:timestep+1]
  else:
    windowed_data = - starting_id * [data[0]] + list(data[0:timestep+1])

  state = []
  for i in range(window_size - 1):
    state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))
    
  score = sentiment[timestep,0]
  state.append(y_pred)
  state.append(diffrence)
  state.append(score)
  state = np.array([state])
  state = np.reshape(state, (-1))
  return state

dataset_2 = pd.read_csv('GRAE Historical Data 2018 practice.csv')
data = list(dataset_2['Price'])
data_samples = len(data)-1

scores = []

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
    scores_window = deque(maxlen=500)  # last 1500 scores
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        print("Episode: {}/{}".format(episode, n_episodes))
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

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local.pth'))

for i in range(3):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()