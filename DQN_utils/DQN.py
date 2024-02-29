"""
DQN Implementation

"""

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, ASGD
from DQN_utils.DQN_models import QNetwork
import numpy as np

class DQN(object):
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9999
        self.tau = 0.001
        self.LR = 5e-2
        self.hidden_dim = 64

        # Q-Network
        # in my notes Q = Q_local and Q' = Q_target
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.hidden_dim)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.hidden_dim)

        self.optimizer = Adam(self.qnetwork_local.parameters(), lr=self.LR)

    # use the policy to select an action
    def select_action(self, state):
        # take action with highest Q value
        state = torch.FloatTensor(state)
        Q_state = self.qnetwork_local(state).detach().numpy()
        return Q_state

    # train the Q-functions
    def update_parameters(self, memory, batch_size):

        # sample a batch from memory
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(1).long()
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).detach()
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + self.gamma * Q_targets_next * (1.0 - dones)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return loss.item()

    # helper function for updating the weights of Q_target (Q')
    def soft_update(self, local_model, target_model, tau):
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
