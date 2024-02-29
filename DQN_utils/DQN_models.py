"""
Models for DQN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # fully connected network with three linear layers
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, action_dim)


    # map states to a Q-value for each action
    def forward(self, state):
        x = F.relu(self.linear1(state))
        #x = F.relu(self.linear2(x))
        return self.linear5(x)
