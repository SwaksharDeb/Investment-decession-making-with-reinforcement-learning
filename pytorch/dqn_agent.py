from IPython.core.debugger import set_trace
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 5000      # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 0.001              # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)   #only update the local network parameters

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action_gp, action_sqph, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action_gp, action_sqph, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
        # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_num = 3
        self.qnetwork_local.eval()
        with torch.no_grad():
            #set_trace()
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action_gp = np.argmax(action_values[0][:3].cpu().data.numpy())
            action_sqph = np.argmax(action_values[0][-3:].cpu().data.numpy())
            return action_gp, action_sqph 
        else:
            #set_trace()
            random_action_gp = random.choice(np.arange(action_num))
            random_action_sqph = random.choice(np.arange(action_num))
            #action_values_gp = action_values[0][:3].cpu().data.numpy()
            #action_gp = action_values_gp[random_action_gp]
            #action_values_sqph = action_values[0][-3:].cpu().data.numpy()
            #action_sqph = action_values_sqph[random_action_sqph]
            return random_action_gp, random_action_sqph

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #self.qnetwork_local.train()
        #self.qnetwork_target.eval()
        states, actions_gp, actions_sqph, rewards, next_states, dones = experiences
            
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).unsqueeze(1)
        Q_targets_next_gp = (Q_targets_next)[:,:,:3]
        Q_targets_next_gp = Q_targets_next_gp.max(2)[0]
        Q_targets_next_sqph = (Q_targets_next)[:,:,-3:]
        Q_targets_next_sqph = Q_targets_next_sqph.max(2)[0]
        # Compute Q targets for current states 
        Q_targets_gp = rewards + (gamma * Q_targets_next_gp * (1 - dones))
        Q_targets_sqph = rewards + (gamma * Q_targets_next_sqph * (1 - dones))
        # Get expected Q values from local model
        Q_expected_gp = self.qnetwork_local(states)[:,:3].gather(1, actions_gp)
        Q_expected_sqph = self.qnetwork_local(states)[:,-3:].gather(1, actions_sqph)
        
        # Compute loss
        loss_gp = F.mse_loss(Q_expected_gp, Q_targets_gp)
        loss_sqph = F.mse_loss(Q_expected_sqph, Q_targets_sqph)
        loss = loss_gp + loss_sqph
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)    #updating the target network parameters                    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        #self.qnetwork_local.eval()
        #self.qnetwork_target.eval()
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)        #deque means double ended queue 
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action_gp", "action_sqph", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action_gp, action_sqph, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action_gp, action_sqph, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        """
        experience = deque(maxlen=int(10))
        memory = namedtuple("memory", field_names=["state","action"])
        e = memory(1,4)
        print(e.state)
        experience.append(e)
        e = memory(2,10)
        experience.append(e)
        e = memory(5,11)
        experience.append(e)
        experiences = random.sample(experience, 2)
        states = torch.from_numpy(np.vstack([i.state for i in experiences])).float()
        s = torch.from_numpy(np.array([1,2,3]))
        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #set_trace()
        actions_gp = torch.from_numpy(np.vstack([e.action_gp for e in experiences if e is not None])).long().to(device)
        actions_sqph = torch.from_numpy(np.vstack([e.action_sqph for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions_gp, actions_sqph, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)