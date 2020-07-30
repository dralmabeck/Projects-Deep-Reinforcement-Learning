import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from source.network import Actor, Critic
from source.noise_generator import OUNoise
from source.replay_buffer import ReplayBuffer


class ActorCritic():
    """Interacts with and learns from the environment."""

    def __init__(self, config):
        """Initialize an Agent object.

        Params
        ======
            config (Config): Configuration class containing the following used parameters
            device (torch.device): device (either CPU or GPU)
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            num_passes (int): number of learning passes each step
            tau (float): value for soft update of network weights
            gamma (float): discount factor for future rewards
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            seed_agent (int): random seed

        """

        self.config = config
        
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.batch_size = config.batch_size
        self.num_passes = config.num_passes
        self.tau = config.tau
        self.gamma = config.gamma
        self.device = config.device
        
        random.seed(config.seed_agent)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(config).to(self.device)
        self.actor_target = Actor(config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(config).to(self.device)
        self.critic_target = Critic(config).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic)

        # Noise process
        self.noise = OUNoise(config)

        # Replay memory
        self.memory = ReplayBuffer(config)


    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            
            # Learn several times per step
            for _ in range(self.num_passes):
                
                experiences = self.memory.sample()
                
                # ---------------------------- update critic ---------------------------- #
                self.update_critic(experiences, self.gamma)
                
                # ---------------------------- update actor ---------------------------- #
                self.update_actor(experiences, self.gamma)
                
                # ----------------------- update target networks ----------------------- #
                self.update_soft(self.critic_local, self.critic_target, self.tau)
                self.update_soft(self.actor_local, self.actor_target, self.tau)

           
    def reset(self):
        """Returns noise during training."""
        self.noise.reset()


    def act(self, state):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        action += self.noise.sample()

        self.actor_local.train()

        return np.clip(action, -1, 1)


    # Learn from experiences for Critic
    def update_critic(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        
        # Get next actions from actor and corresponding Q from critic
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Calculate target Q
        Q_targets = rewards + (self.gamma * Q_targets_next * (1.0 - dones))

        # Get expected Q
        Q_expected = self.critic_local(states, actions)

        # Standard mean-sqaured-error loss function
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Reset the gradients
        self.critic_optimizer.zero_grad()
        # Perform backpropagation
        critic_loss.backward()
        # Clip gradients
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        # Optimize weights
        self.critic_optimizer.step()

       
    # Learn from experiences for Actor
    def update_actor(self, experiences, gamma):
        """Update policy and value parameters using given batch of state tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        
        # Compute loss function for actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Reset the gradients
        self.actor_optimizer.zero_grad()
        # Perform backpropagation
        actor_loss.backward()
        # Optimize weights
        self.actor_optimizer.step()


    def update_soft(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)