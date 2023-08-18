## Investment Decision Making with Deep Reinforcement Learning
This is a repository for an automated trading bot for the stock market using reinforcement learning algorithms. Specifically, here we implemented **Deep Q Learning**, **Double Deep Q Learning** algorithms with **Prioritized Experience Replay**.

## How to Run

`DDQN_PER.py` is the code for the double deep Q learning with the prioritized experience replay algorithm. `DQN_PER.py` is the implementation for deep Q learning with prioritized experienced replay. `dqn_agent.py` is the deep Q learning with a vanilla experience replay algorithm. `model.py` is the architecture of the neural network that has been utilized in this project.

To train the stock trader bot using **Deep Q Learning with Prioritized Experience Replay** run this command in the terminal  ``python stock_trader_PER_trend.py ``. To train the stock trader bot using **Double Deep Q Learning with Prioritized Experience Replay** run this command in the terminal  ``stock_trader_trend_DDQN_PER.py``. To train the stock trader bot using **Deep Q Learning with Vanilla Experience Replay** run this command in the terminal  ``stock_trader_with_trend.py``. 
## Results
The results are shown using deep double Q learning with prioritized experience replay. 

Result in the training set. The training set is the 2018 Walmart stock market.
![](https://github.com/SwaksharDeb/stock-trader/blob/master/photos/DDQN_PER%20with%20market%20factors%20training%20set.png)

Result in the test set. The test set is the 2019 Walmart stock market.
![DDQN_PER with market factors training set.png](https://github.com/SwaksharDeb/stock-trader/blob/master/photos/DDQN_PER%20with%20market%20factors%20test%20set.png)
