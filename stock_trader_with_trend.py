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
from dqn_agent import Agent
import pandas as pd
import statistics

window_size = 10
portfolio_size = 1
investment_size = 1
trend_size = 1
input_size = window_size + portfolio_size +investment_size +trend_size

agent = Agent(state_size = input_size, action_size=3, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local_WMT_V5_.pth'))
agent.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target_WMT_V5_.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

#sc = MinMaxScaler()
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
dataset_gp = pd.read_csv('WMT Historical Data 2018.csv')
data_gp = list(dataset_gp['Price'])
trend_gp = list(dataset_gp['trend'])
data_samples = len(data_gp)-1
scores = []

#loading the validation data
dataset_gp_evaluation = pd.read_csv('WMT Historical Data 2019.csv')
data_gp_evaluation = list(dataset_gp_evaluation['Price'])
trend_gp_evaluation = list(dataset_gp_evaluation['trend'])

#evaluation loop
def evaluation():
    #setting up the parameter
    data_samples = len(data_gp_evaluation)-1
    inventory_gp_evaluation = []
    total_profit = 0
    investment_evaluation = 10000

    #f = open("state.txt","w")
    #testing loop
    episode = 1
    state = state_creator(data_gp_evaluation,10,window_size,inventory_gp_evaluation,investment_evaluation,episode,trend_gp_evaluation)

    for t in range(10,data_samples-1):
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #f.write(str(sc.inverse_transform(state.cpu().data.numpy())))
        #action = np.argmax(agent.qnetwork_local(state).cpu().data.numpy())
        action = agent.act(state)
        #reward = -1
        if action == 1 and int(investment_evaluation/data_gp_evaluation[t])>0:  #buy
            no_buy = int(investment_evaluation/data_gp_evaluation[t])
            for i in range(no_buy):
                investment_evaluation -= data_gp_evaluation[t]
                inventory_gp_evaluation.append(data_gp_evaluation[t])
            #f.write("BUY")            
            #reward = (len(inventory_gp_evaluation)*data_gp_evaluation[t-1]) - (len(inventory_gp_evaluation)*data_gp_evaluation[t])
            print("AI Trader bought: ", stocks_price_format(no_buy*data_gp_evaluation[t]))
                 
        if action == 2 and len(inventory_gp_evaluation)>0: #Selling
            buy_prices_gp = []
                
            for i in range(len(inventory_gp_evaluation)):
                buy_prices_gp.append(inventory_gp_evaluation[i])
            buy_price = sum(buy_prices_gp)  #buying price of gp stocks
                
                
            total_profit += (len(inventory_gp_evaluation)*data_gp_evaluation[t]) - buy_price
            profit = (len(inventory_gp_evaluation)*data_gp_evaluation[t]) - buy_price
            investment_evaluation = investment_evaluation + (len(inventory_gp_evaluation)*data_gp_evaluation[t])  #total investment or cash in hand
            #reward = (len(inventory_gp_evaluation)*data_gp_evaluation[t]) - buy_price
            #stock_return = (w_gp*returns_gp_validation[t]) + (w_sqph*returns_sqph_validation[t])               
            #f.write("SELL")
            print("AI Trader sold gp: ", stocks_price_format(len(inventory_gp_evaluation)*data_gp_evaluation[t])," Profit: " + stocks_price_format(profit))
            
            for i in range(len(inventory_gp_evaluation)):
                inventory_gp_evaluation.pop(0)   # empty the gp inventory after selling all of them
            
        if action == 0: #hold
            #f.write("HOLD")
            #reward = 0
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
        
        #agent.memory.add(state, action, reward, next_state, done)      
                
        state = next_state  #assin next state to present state
    
    print("########################")
    print("TOTAL VALIDATION PROFIT: {}".format(total_profit))
    print("########################")
    #f.close()


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
    scores_window = deque(maxlen=500)  # last 1500 scores
    rewards = deque(maxlen=200)
    sell_date = deque(maxlen=1)
    rewards.append(10000)
    rewards.append(0)
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        print("Episode: {}/{}".format(episode, n_episodes))
        investment = 10000
        #portfolio_value_ = investment
        inventory_gp = []
        state = state_creator(data_gp,10,window_size,inventory_gp,investment,episode,trend_gp)
        total_profit = 0
        total_reward = 0
    
        for t in range(10,max_t):
            action = agent.act(state, eps)
            reward = -100
            #change_percent_stock = ((data_gp[t]-statistics.mean(data_gp[t-10:t]))/statistics.mean(data_gp[t-10:t]))*100
            if action == 1 and int(investment/data_gp[t])>0: #Buying gp
                no_buy = int(investment/data_gp[t])
                for i in range(no_buy):
                    investment -= data_gp[t]
                    inventory_gp.append(data_gp[t])
                #portfolio_value = (len(inventory_gp)*data_gp[t]) + investment
                #reward = (len(inventory_gp)*sell_date[0])-(len(inventory_gp)*data_gp[t])
                #if trend_gp[t]==0:
                    #reward = -1000    
                fifteen_days_min = min(data_gp[t+1:t+16])
                if data_gp[t] < fifteen_days_min:
                    #reward = -(change_percent_stock)*100
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
                #if trend_gp[t]==1:
                    #reward = -1000
                #if data_gp[t]<=max(data_gp[t:t+15]):
                    #reward = -1000
                #else:
                    #reward = change_percent_stock * 100
                    #reward = (len(inventory_gp)*data_gp[t]) - buy_price
                fifteen_days_max = max(data_gp[t+1:t+16])
                if data_gp[t] > fifteen_days_max:
                    #reward = (len(inventory_gp)*data_gp[t]) - buy_price
                    reward = 1
                else:
                    reward = -1
                total_profit += (len(inventory_gp)*data_gp[t]) - buy_price
                profit = (len(inventory_gp)*data_gp[t]) - buy_price
                investment = investment + (len(inventory_gp)*data_gp[t])  #total investment or cash in hand
                #portfolio_value = investment
                #reward =(len(inventory_gp)*data_gp[t]) - buy_price
                
                rewards.append(reward)
                sell_date.append(data_gp[t])
                print("AI Trader sold: ", stocks_price_format(len(inventory_gp)*data_gp[t]), " Profit: " + stocks_price_format(profit)," Reward: " , stocks_price_format(reward))
                for i in range(len(inventory_gp)):
                    inventory_gp.pop(0)
                
            if action == 0:
                #if abs(change_percent_stock)<=0.2:
                    #reward = 500
                #else:
                    #reward = 0
                """prev_days_max = max(data_gp[t-10:t])
                fifteen_days_min = min(data_gp[t+1:t+16])
                prev_days_min = min(data_gp[t-10:t])
                fifteen_days_max = max(data_gp[t+1:t+16])
                if (data_gp[t] > prev_days_max) and (data_gp[t] < fifteen_days_min):
                    reward = 1
                elif(data_gp[t] < prev_days_min) and (data_gp[t] > fifteen_days_max):
                    reward = 1
                else:
                    reward = -1"""
                reward = 0
                print("AT Trader is holding.................","Reward: ",stocks_price_format(reward))
                #porfolio_value = (len(inventory_gp)*data_gp[t]) + investment
               
            next_state = state_creator(data_gp,t+1,window_size,inventory_gp,investment,episode,trend_gp)
            #portfolio_value_ = portfolio_value
            
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
                scores.append(total_profit)
                print("########################")
            
        evaluation()
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        #save the model weights
        if episode == 100:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_local_WMT_V5_.pth')
            torch.save(agent.qnetwork_target.state_dict(), 'checkpoint_qnetwork_target_WMT_V5_.pth')            
    return total_profit

scores = dqn()

################################################################################################
#save the model weights
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_qnetwork_local_WMT_V5_.pth')
torch.save(agent.qnetwork_target.state_dict(), 'checkpoint_qnetwork_target_WMT_V5_.pth')
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

agent.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local_WMT_V5_.pth'))
agent.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target_WMT_V5_.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#import the test data
dataset_gp_test = pd.read_csv('WMT Historical Data 2016.csv')
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
    f = open("state.txt","w")
    #testing loop
    episode = 1
    state = state_creator(data_gp_test,10,window_size,inventory_gp_test,investment_test,episode,trend_gp_test)

    for t in range(10,data_samples-1):
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #f.write(str(sc.inverse_transform(state.cpu().data.numpy())))
        #action = np.argmax(agent.qnetwork_local(state).cpu().data.numpy())
        action = agent.act(state)
        f.write(str(state))
        change_percent_stock = ((data_gp_test[t]-statistics.mean(data_gp_test[t-10:t]))/statistics.mean(data_gp_test[t-10:t]))*100        
        if action == 1 and int(investment_test/data_gp_test[t])>0:  #buy
            no_buy = int(investment_test/data_gp_test[t])
            for i in range(no_buy):
                investment_test -= data_gp_test[t]
                inventory_gp_test.append(data_gp_test[t])
            f.write("BUY")
            if trend_gp_test[t]==0:
                reward = -1000    
            else:
                reward = -(change_percent_stock)*100
            #if reward >0:
            timesteps_buys.append(t-10)
            buys.append(data_gp_test[t])
            print("AI Trader bought: ", stocks_price_format(no_buy*data_gp_test[t]), "Investment= ",stocks_price_format(investment_test),"Reward: ",reward,"Timestep = ",t+2)
                 
        if action == 2 and len(inventory_gp_test)>0: #Selling
            buy_prices_gp = []
                
            for i in range(len(inventory_gp_test)):
                buy_prices_gp.append(inventory_gp_test[i])
            buy_price = sum(buy_prices_gp)  #buying price of gp stocks
                
            if trend_gp_test[t]==1:
                reward = -1000
            else:
                #reward = change_percent_stock * 100
                reward = (len(inventory_gp_test)*data_gp_test[t]) - buy_price
            total_profit += (len(inventory_gp_test)*data_gp_test[t]) - buy_price
            profit = (len(inventory_gp_test)*data_gp_test[t]) - buy_price
            investment_test = investment_test + (len(inventory_gp_test)*data_gp_test[t])  #total investment or cash in hand
            #if reward>0:
            timesteps_sells.append(t-10)
            sells.append(data_gp_test[t])
            #stock_return = (w_gp*returns_gp_validation[t]) + (w_sqph*returns_sqph_validation[t])               
            f.write("SELL")
            print("AI Trader sold gp: ", stocks_price_format(len(inventory_gp_test)*data_gp_test[t])," Profit: " + stocks_price_format(profit),"Investment= ",stocks_price_format(investment_test),"Reward: ",reward,"Timestep = ",t+2)
            
            for i in range(len(inventory_gp_test)):
                inventory_gp_test.pop(0)   # empty the gp inventory after selling all of them
            
        if action == 0: #hold
            f.write("HOLD")
            if abs(change_percent_stock)<=0.2:
                reward = 500
            else:
                reward = 0
            """if reward>0:
                timesteps_sells.append(t-10)
                sells.append(data_gp_test[t])"""                
            print("AI Trader is holding........","Investment= ",stocks_price_format(investment_test),"Timestep = ",t+2, "Reward: ",reward)                
                                                                            
        next_state = state_creator(data_gp_test,t+1,window_size,inventory_gp_test,investment_test,episode,trend_gp_test)
                
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

