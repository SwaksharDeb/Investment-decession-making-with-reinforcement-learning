import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=256, fc3_units=128,fc4_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        #self.dropout_1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        #self.dropout_2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        #self.dropout_3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        #x = self.dropout_1(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout_2(x)
        x = F.relu(self.fc3(x))
        #x = self.dropout_3(x)
        x = F.relu(self.fc4(x))
        return self.fc5(x)

