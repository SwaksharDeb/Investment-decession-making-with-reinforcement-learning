# stock-trader
an automated trading bot using reinforcement learning

**DDQN_PER.py** = double deep Q learning with prioratized experience replay

**DQN_PER.py** = deep Q learning with prioratized experienced replay

**dqn_agent.py** = deep Q learning with experience replay

**model.py** = Architecture of the neural network

**stock_trader_PER_trend.py** = stock trader with DQN and prioritized experienced replay

**stock_trader_trend_DDQN_PER.py** = stock trader with DDQN and prioritized experienced replay

**stock_trader_with_trend.py** = stock trader with DQN and experienced replay

# Results
Currently, I am working on **stock_trader_DDQN_PER.py** (stock trader using deep double Q learning with prioratized experience replay) file. So, the results are shown using deep double Q learning with prioratized experience replay. 

Result in training set. Training set is 2018 walmert stock market.
![](https://github.com/SwaksharDeb/stock-trader/blob/master/photos/DDQN_PER%20with%20market%20factors%20training%20set.png)

Result in test set. Test set is 2019 walmert stock market.
![DDQN_PER with market factors training set.png](https://github.com/SwaksharDeb/stock-trader/blob/master/photos/DDQN_PER%20with%20market%20factors%20test%20set.png)


# Usage
1. Run any of the stock trader named file according to your chosen algorithm, module will be automatically imported.

# Note
Do not change the folder structure. you can also observ the average Q value and average reward at each episode in tensorboard and those tensorboard files will be stored at runs folder.
