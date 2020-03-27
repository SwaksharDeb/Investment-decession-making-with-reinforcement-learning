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

dataset_gp_test = pd.read_csv('datasets/UNH Historical Data 2017.csv')
data_gp_test = list(dataset_gp_test['Price'])

timesteps_holds = []
holds = []

for t in range(10,len(data_gp_test)-1):
    change_percent_stock = ((data_gp_test[t]-statistics.mean(data_gp_test[t-10:t]))/statistics.mean(data_gp_test[t-10:t]))*100        
    if change_percent_stock<0.2:
        timesteps_holds.append(t-10)
        holds.append(data_gp_test[t])

#plot the graph
stock_price = data_gp_test[10:]
plt.plot(stock_price, color = 'blue', label = 'WMT')
plt.scatter(timesteps_holds,holds,color='red',label='hold')
plt.title('WMT stock')
plt.xlabel('Time')
plt.ylabel('WMT Price')
plt.legend()
plt.show()


