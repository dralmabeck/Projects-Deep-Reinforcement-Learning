import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """Set boundaries of uniform distribution for initialization of weights of neural network.
    Params
    ======
        layer (nn.Layer): One hidden layer of a neural network
    """
        
    fan_in = layer.weight.data.size()[0] # Get size of hidden layer
    lim = 1. / np.sqrt(fan_in) # Calculate min/max
    
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, config):
        """Initialize parameters and build model.
        Params
        ======
            config (Config): Configuration class containing the following used parameters
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed_actor (int): Random seed
            fc_size (int): Base number of neurons
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(config.state_size, 8*config.fc_size)
        self.fc2 = nn.Linear(8*config.fc_size, 4*config.fc_size)
        self.fc3 = nn.Linear(4*config.fc_size, config.action_size)

        self.bn = nn.BatchNorm1d(config.state_size)

        random.seed(config.seed_actor)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights in the actor (policy) network"""

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        #self.fc1.bias.data.fill_(np.random.uniform())
        #self.fc2.bias.data.fill_(np.random.uniform())
        #self.fc3.bias.data.fill_(np.random.uniform())

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        x = self.bn(state)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        result = torch.tanh(self.fc3(x)) # tanh to yield continous value between -1.0 and 1.0

        return result


class Critic(nn.Module):
    """Critic (value) Model."""

    def __init__(self, config):
        """Initialize parameters and build model.
        Params
        ======
            config (Config): Configuration class containing the following used parameters
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed_critic (int): Random seed
            fc_size (int): Base number of neurons
        """

        super(Critic, self).__init__()

        self.fc1 = nn.Linear(config.state_size, 8*config.fc_size)
        self.fc2 = nn.Linear(8*config.fc_size + config.action_size, 4*config.fc_size)
        self.fc3 = nn.Linear(4*config.fc_size, 1)

        self.bn = nn.BatchNorm1d(config.state_size)

        random.seed(config.seed_critic)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights in the critic (value) network"""

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        #self.fc1.bias.data.fill_(np.random.uniform())
        #self.fc2.bias.data.fill_(np.random.uniform())
        #self.fc3.bias.data.fill_(np.random.uniform())

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        x = self.bn(state)
        x = F.leaky_relu(self.fc1(x))

        x = torch.cat((x, action), dim=1) # merge action vector into network
        x = F.leaky_relu(self.fc2(x))

        result = self.fc3(x) # no activation as we need a real Q value

        return result