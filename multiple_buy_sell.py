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
#import stock_trader_DOW_single #import sigmoid, state_creator, stocks_price_format
import maximize_return

window_size = 10
portfolio_size = 1
investment_size = 1
input_size = window_size + portfolio_size +investment_size

#load the pretrain model
model_unh = Agent(state_size = input_size, action_size=3, seed=0)
model_wmt = Agent(state_size = input_size, action_size=3, seed=0)

model_unh.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local_UNH.pth'))
model_unh.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target_UNH.pth'))
model_wmt.qnetwork_local.load_state_dict(torch.load('checkpoint_qnetwork_local_WMT.pth'))
model_wmt.qnetwork_target.load_state_dict(torch.load('checkpoint_qnetwork_target_WMT.pth'))

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

#sc = MinMaxScaler()
def state_creator(data_gp, timestep, window_size, inventory_gp, investment,episode):
  gp_assest = (len(inventory_gp) * data_gp[timestep]) #portfolio value for gp at current timestep
  
  state = []
  for i in range(timestep-window_size+1, timestep+1):
     state.append(data_gp[i]-data_gp[i-1])
              
  state.append(0.0001*gp_assest)
  state.append(0.0001*investment)
  state = np.array([state])   # normalized input(state)
  state = sigmoid(state)
  return state

#load the dataset
dataset_unh = pd.read_csv('UNH Historical Data 2016.csv')
data_unh = list(dataset_unh['Price'])
returns_unh_list = list(dataset_unh['return'])
dataset_wmt = pd.read_csv('WMT Historical Data 2016.csv')
data_wmt = list(dataset_wmt['Price'])
returns_wmt_list = list(dataset_wmt['return'])
data_samples = len(data_wmt)-1

#load the return list
data_return_unh = pd.read_csv('UNH Historical Data 2015.csv')
returns_unh_list_ = list(data_return_unh['return'][-200:])
data_return_wmt = pd.read_csv('WMT Historical Data 2015.csv')
returns_wmt_list_ = list(data_return_wmt['return'][-200:])

def main():
    investment = 10000
    total_profit = 0
    episode = 1
    inventory_unh =[]
    inventory_wmt = []
    
    returns_unh = deque(maxlen=210)
    for i in range(len(returns_unh_list_)):
        returns_unh.append(returns_unh_list_[i])
    for i in range(10):
        returns_unh.append(returns_unh_list[i])
    
    returns_wmt = deque(maxlen=210)
    for i in range(len(returns_wmt_list_)):
        returns_wmt.append(returns_wmt_list_[i])
    for i in range(10):
        returns_wmt.append(returns_wmt_list[i])
        
    state_unh = state_creator(data_unh,10,window_size,inventory_unh,investment,episode)
    state_wmt = state_creator(data_wmt,10,window_size,inventory_wmt,investment,episode)
    for t in range(10,data_samples):
        action_unh = model_unh.act(state_unh)
        action_wmt = model_wmt.act(state_wmt)
        
        
        if action_unh == 1 and action_wmt==1 and (investment>data_unh[t] or investment>data_wmt[t]):
            cov = np.cov(np.array(returns_unh),np.array(returns_wmt))
            w = maximize_return. markwitz_portpolio([statistics.mean(returns_unh),statistics.mean(returns_wmt)],cov)                
            w_unh = w[0]
            w_wmt = w[1]
                
            investment_unh = (investment*w_unh)
            if investment_unh >= data_unh[t]:
                no_stock_unh = int(investment_unh/data_unh[t]) #no of stock to buy
            else:
                no_stock_unh = 0
                        
            investment_wmt = (investment*w_wmt)
            if investment_wmt >= data_wmt[t]:
                no_stock_wmt = int(investment_wmt/data_wmt[t])  #no of stock to buy
            else:
                no_stock_wmt = 0
                    
            for i in range(no_stock_unh):
                investment = investment - data_unh[t]  #decrease the investment after buying each stock
                inventory_unh.append(data_unh[t])
            for i in range(no_stock_wmt):
                investment = investment - data_wmt[t]  #decrease the investment after buying each stock
                inventory_wmt.append(data_wmt[t])
            
            returns_wmt.append(returns_wmt_list[t])
            returns_unh.append(returns_unh_list[t])
                
            print("AI Trader bought UNH: ", stocks_price_format(no_stock_unh*data_unh[t]),"AI Trader bought WMT: ", stocks_price_format(no_stock_wmt*data_wmt[t]),"Investment=",stocks_price_format(investment))
                
        else:
            if action_unh == 1 and investment>data_unh[t]:
                no_buy = int(investment/data_unh[t])
                for i in range(no_buy):
                    investment -= data_unh[t]
                    inventory_unh.append(data_unh[t])
                        
                print("AI Trader bought UNH: ", stocks_price_format(no_buy*data_unh[t]),"Investment=",stocks_price_format(investment))
                
            if action_wmt == 1 and investment>data_wmt[t]:
                no_buy = int(investment/data_wmt[t])
                for i in range(no_buy):
                    investment -= data_wmt[t]
                    inventory_wmt.append(data_wmt[t])
                            
                print("AI Trader bought WMT: ", stocks_price_format(no_buy*data_wmt[t]),"Investment=",stocks_price_format(investment))
                                
            if action_unh == 2 and len(inventory_unh)>0:  #2=sell
                buy_prices_unh = []
                
                for i in range(len(inventory_unh)):
                    buy_prices_unh.append(inventory_unh[i])
                buy_price = sum(buy_prices_unh)  #buying price of gp stocks
                    
                    
                total_profit += (len(inventory_unh)*data_unh[t]) - buy_price
                profit = (len(inventory_unh)*data_unh[t]) - buy_price
                investment = investment + (len(inventory_unh)*data_unh[t])  #total investment or cash in hand
                print("AI Trader sold UNH: ", stocks_price_format(len(inventory_unh)*data_unh[t])," Profit: " + stocks_price_format(profit),"Investment=",stocks_price_format(investment))                
                for i in range(len(inventory_unh)):
                    inventory_unh.pop(0)   # empty the gp inventory after selling all of them
                    
            if action_wmt == 2 and len(inventory_wmt)>0:  #sell
                buy_prices_wmt = []
                
                for i in range(len(inventory_wmt)):
                    buy_prices_wmt.append(inventory_wmt[i])
                buy_price = sum(buy_prices_wmt)  #buying price of gp stocks
                    
                    
                total_profit += (len(inventory_wmt)*data_wmt[t]) - buy_price
                profit = (len(inventory_wmt)*data_wmt[t]) - buy_price
                investment = investment + (len(inventory_wmt)*data_wmt[t])  #total investment or cash in hand
                print("AI Trader sold WMT: ", stocks_price_format(len(inventory_wmt)*data_wmt[t])," Profit: " + stocks_price_format(profit),"Investment=",stocks_price_format(investment))                
                for i in range(len(inventory_wmt)):
                    inventory_wmt.pop(0)   # empty the gp inventory after selling all of them
                    
            if action_unh == 0:
                 print("AI Trader is holding for UNH........","Investment=",stocks_price_format(investment))
                 
            if action_wmt == 0:
                print("AI Trader is holding for WMT........","Investment=",stocks_price_format(investment))
                
        next_state_unh = state_creator(data_unh,t+1,window_size,inventory_unh,investment,episode)
        next_state_wmt = state_creator(data_wmt,t+1,window_size,inventory_wmt,investment,episode)
        if investment<=0 and len(inventory_unh)==0 and len(inventory_wmt)==0: #checking for bankcrapcy
                done = True
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("AI Trader is bankcrapted")
                print("########################")            
                break  # if bankcrapted end the seassion
            
        if t == data_samples - 1:
            done = True
        else:
            done = False                
        
        state_unh = next_state_unh  #assin next state to present state
        state_wmt = next_state_wmt
    print("########################")
    print("TOTAL PROFIT: {}".format(total_profit))
    print("########################")
    
main()
