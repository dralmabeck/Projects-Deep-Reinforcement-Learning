import copy
import numpy as np
import random

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, config):
        """Initialize parameters and noise process.
        Params
        ======
            config (Config): Configuration class containing the following used parameters
            theta (float): Theta for OU process
            sigma (float): Sigma for OU process
            mu (float): Mu for OU process
            gauss_mu (float): Mu for OU process normal distribution
            gauss_sigma (float): Sigma for OU process normal distribution
            n_agents (int): Number of agents
            action_size (int): Dimension of each action
            seed_noise (int): random seed
        """

        self.theta = config.theta
        self.sigma = config.sigma
        self.size = (config.n_agents, config.action_size)
        self.mu = config.mu * np.ones(self.size)
        self.gauss_mu = config.gauss_mu
        self.gauss_sigma = config.gauss_sigma
        random.seed(config.seed_noise)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""

        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.gauss(self.gauss_mu, self.gauss_sigma) for _ in range(len(x))])
        self.state = x + dx

        return self.state