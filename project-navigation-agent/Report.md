# Navigation - Introduction by Udacity

---

In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### 1. Start the Environment

We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).


```python
!pip -q install ./python
```

    [31mtensorflow 1.7.1 has requirement numpy>=1.13.3, but you'll have numpy 1.12.1 which is incompatible.[0m
    [31mipython 6.5.0 has requirement prompt-toolkit<2.0.0,>=1.0.15, but you'll have prompt-toolkit 3.0.5 which is incompatible.[0m



```python
from unityagents import UnityEnvironment
import numpy as np
```

Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Banana.app"`
- **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
- **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
- **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
- **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
- **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
- **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`

For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Banana.app")
```


```python
#env = UnityEnvironment(file_name="Banana.app")
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
```

    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		
    Unity brain name: BananaBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 37
            Number of stacked Vector Observation: 1
            Vector Action space type: discrete
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 


Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

### 2. Examine the State and Action Spaces

The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 

Run the code cell below to print some information about the environment.


```python
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
```

    Number of agents: 1
    Number of actions: 4
    States look like: [ 1.          0.          0.          0.          0.84408134  0.          0.
      1.          0.          0.0748472   0.          1.          0.          0.
      0.25755     1.          0.          0.          0.          0.74177343
      0.          1.          0.          0.          0.25854847  0.          0.
      1.          0.          0.09355672  0.          1.          0.          0.
      0.31969345  0.          0.        ]
    States have length: 37


### 3. Take Random Actions in the Environment

In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.

Once this cell is executed, you will watch the agent's performance, if it selects an action (uniformly) at random with each time step.  A window should pop up that allows you to observe the agent, as it moves through the environment.  

Of course, as part of the project, you'll have to change the code so that the agent is able to use its experience to gradually choose better actions when interacting with the environment!


```python
#env_info = env.reset(train_mode=False)[brain_name] # reset the environment
#state = env_info.vector_observations[0]            # get the current state
#score = 0                                          # initialize the score
#while True:
#    action = np.random.randint(action_size)        # select an action
#    env_info = env.step(action)[brain_name]        # send the action to the environment
#    next_state = env_info.vector_observations[0]   # get the next state
#    reward = env_info.rewards[0]                   # get the reward
#    done = env_info.local_done[0]                  # see if episode has finished
#    score += reward                                # update the score
#    state = next_state                             # roll over the state to next time step
#    if done:                                       # exit loop if episode finished
#        break
#    
#print("Score: {}".format(score))
```

When finished, you can close the environment.


```python
#env.close()
```

### 4. It's Your Turn!

Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
```python
env_info = env.reset(train_mode=True)[brain_name]
```

# Navigation - Student's project work

The following sections show my code and results from my work on this project.

I have implemented two different neural network architectures:
- Standard neural network with three fully connected layers with sizes of: 37 (states) -> 256 (hidden) -> 256 (hidden) -> 4 (actions)
- Duelling neural network (https://arxiv.org/abs/1511.06581) with two shared fully connected layers of size: 37 (states) -> 256 (hidden) -> 256 (hidden) and two seperated fully connected layers with one of sizes 256 (hidden) -> 128 (hidden) -> 1 (value branch) and one with sizes 256 (hidden) -> 128 (hidden) -> 4 (advantage branch)

Both neural network architectures implement:
- Linear fully connected layers: https://pytorch.org/docs/master/generated/torch.nn.Linear.html
- Exponential linear unit activation function: https://pytorch.org/docs/stable/nn.html#torch.nn.ELU
- Dropout between fully connected layersfor regularization: https://pytorch.org/docs/master/generated/torch.nn.Dropout.html

I have implemented four different agents interacting with the environment:
- SingleAgent:

 - Standard implementation of Agent using:
 - Standard neural QNetwork with three fully connected layers, exponential linear unit and dropout regularization
 - Standard Q-learning algorithm
 - Standard epsilon greedy policy
 - Standard experience replay with random sampling
 - Standard Mean-squared-error loss function
 - Adam optimizer: https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
 

- DoubleAgent:

 - Same as SingleAgent with two changes:
 - Double Q-Learning algorithm instead of single Q-Learning algorithm: https://arxiv.org/abs/1509.06461
 - Duelling neural DuelQNetwork instead of standard QNetwork

- Triple Agent:

 - Same as SingleAgent with two changes:
 - Prioritized experience replay: https://arxiv.org/abs/1511.05952
 - Weighted mean-squared-error loss function

- QuadrupleAgent:

 - Same as TripleAgent with one change:
 - Duelling neural DuelQNetwork instead of standard QNetwork

Note that the QuadrupleAgent features many of the improvements used in the Rainbow implementation (https://arxiv.org/abs/1710.02298), which is a state-of-the-art list of improvements and combined implementation for Deep Reinforcement learning.

### Import all necessary packages at once


```python
# Import packages for the data processing before and after simulations
import time
import pickle
import math
import datetime
import random
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
%matplotlib inline

# Import from PyTorch for neural network implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import from Unity Environment for the agents
from unityagents import UnityEnvironment
```

### Setting of variables for the neural network, training algorithms and unity environments


```python
BUFFER_SIZE = int(1e5)     # replay buffer size
BATCH_SIZE = 256           # minibatch size
GAMMA = 0.99               # discount factor
TAU = 0.001                # for soft update of target parameters
LR = 0.0005                # learning rate 
UPDATE_EVERY = 4           # how often to update the network

FC_SIZE = 128              # number of neurons in layer of neural network
P_DROPOUT = 0.0            # dropout probability for neurons and network regularization

ACTION_SIZE = brain.vector_action_space_size # number of actions
STATE_SIZE = len(state)    # number of states

N_EPISODES = 2000          # number of maximum episodes for training
MAX_T = 1000               # maximum time agent is in the environment in each episode
EPS_START = 1.0            # Initial Epsilon for Exploration - Exploitation
EPS_END = 0.025            # minimum value of epsilon
EPS_DECAY = 0.995          # linear decay of epsilon each timestep

ALPHA_INITIAL = 0.6        # exponent of priorization https://arxiv.org/pdf/1511.05952.pdf
BETA_INITIAL = 0.4         # exponent of importance sampling weights https://arxiv.org/pdf/1511.05952.pdf

ENV_SOLVED = 13.0          # average score of agent to consider environment solved

RANDOM_SEED = 2            # seed integer for random number generator
```

### Check if training on GPU is possible, else use CPU


```python
# If GPU is available use for training otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    print("Available: CPU only")
else:
    print("Available: GPU cuda")
```

    Available: GPU cuda


### Implementation of the standard neural network architecture
Standard neural network with three fully connected layers with sizes of: 37 (states) -> 256 (hidden) -> 256 (hidden) -> 4 (actions)
- Linear fully connected layers: https://pytorch.org/docs/master/generated/torch.nn.Linear.html
- Exponential linear unit activation function: https://pytorch.org/docs/stable/nn.html#torch.nn.ELU
- Dropout between fully connected layersfor regularization: https://pytorch.org/docs/master/generated/torch.nn.Dropout.html


```python
class QNetwork(nn.Module):

    def __init__(self):

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(RANDOM_SEED)
    
        self.fc1 = nn.Linear(STATE_SIZE, 2*FC_SIZE) # Size 37 -> 256
        self.fc2 = nn.Linear(2*FC_SIZE, 2*FC_SIZE) # Size 256 -> 256
        self.fc3 = nn.Linear(2*FC_SIZE, ACTION_SIZE) # Size 256 -> 4
        
        self.dropout = nn.Dropout(P_DROPOUT)

    def forward(self, state):
        
        x = F.elu(self.fc1(state)) # Exponential linear unit
        x = self.dropout(x) # Dropout for regularization
        
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        
        action = self.fc3(x)
        
        return action # Yield action for agent
```

### Implementation of the duelling neural network architecture
Duelling neural network (https://arxiv.org/abs/1511.06581)with two shared fully connected layers of size: 37 (states) -> 256 (hidden) -> 256 (hidden) and two seperated fully connected layers with one of sizes 256 (hidden) -> 128 (hidden) -> 1 (value branch) and one with sizes 256 (hidden) -> 128 (hidden) -> 4 (advantage branch)
- Linear fully connected layers: https://pytorch.org/docs/master/generated/torch.nn.Linear.html
- Exponential linear unit activation function: https://pytorch.org/docs/stable/nn.html#torch.nn.ELU
- Dropout between fully connected layersfor regularization: https://pytorch.org/docs/master/generated/torch.nn.Dropout.html
- Combination of value and advantage branches according to original paper implementation: https://arxiv.org/abs/1511.06581


```python
class DuelQNetwork(QNetwork):

    def __init__(self):

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(RANDOM_SEED)
    
        self.fc1 = nn.Linear(STATE_SIZE, 2*FC_SIZE) # Size 37 -> 256
        self.fc2 = nn.Linear(2*FC_SIZE, 2*FC_SIZE) # Size 256 -> 256
        
        self.val_1 = nn.Linear(2*FC_SIZE, FC_SIZE) # Size 256 -> 128
        self.val_2 = nn.Linear(FC_SIZE, 1) # Size 128 -> 1
        
        self.adv_1 = nn.Linear(2*FC_SIZE, FC_SIZE) # Size 256 -> 128
        self.adv_2 = nn.Linear(FC_SIZE, ACTION_SIZE) # Size 128 -> 4
        
        self.dropout = nn.Dropout(P_DROPOUT)
        
    def forward(self, state):
        
        x = F.elu(self.fc1(state)) # Exponential linear unit
        x = self.dropout(x) # Dropout for regularization
        
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        
        # Value branch
        val = F.elu(self.val_1(x))
        val = self.val_2(val).expand(state.size(0), ACTION_SIZE)
        
        # Advantage branch
        adv = F.elu(self.adv_1(x))
        adv = self.adv_2(adv)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        action = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), ACTION_SIZE)
        
        return action # Yield action for agent
```

### Implementation of experience replay buffer
Implementation follows the standard replay buffer implementation as described in the Udacity course videos regarding the DQN algorithm and the Lunar Lander example.


```python
class ReplayBuffer:

    def __init__(self):

        self.memory = deque(maxlen = BUFFER_SIZE)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(RANDOM_SEED)
    
    # Add an experience to the memory
    def add(self, state, action, reward, next_state, done):

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    # Randomly sample experiences from the memory
    def sample(self):

        # Randomly draw a sample of size BATCH_SIZE
        experiences = random.sample(self.memory, k=BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    # Calculate length of memory
    def __len__(self):

        return len(self.memory)
```

### Implementation of prioritized experience replay buffer

This implementation follows the Paper of https://arxiv.org/abs/1511.05952 and was inspired by https://github.com/ucaiado/banana-rl.


```python
class PriorityReplayBuffer(ReplayBuffer):

    def __init__(self):

        super(ReplayBuffer, self).__init__()
        
        self.memory = deque(maxlen = BUFFER_SIZE)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(RANDOM_SEED)
        
        self.priority = deque(maxlen = BUFFER_SIZE)
        self.cum_priority = 0.0 # Initialize cumulative priorities with 0.0
        self.eps = 1e-6
        self.indices = []

    # Add an experience to the memory
    # Update the total sum of priorities
    def add(self, state, action, reward, next_state, done):

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        if len(self.priority) >= BUFFER_SIZE:
            self.cum_priority -= self.priority[0]

        self.priority.append(1.0)
        self.cum_priority += self.priority[-1]

    def sample(self):

        N = len(self.memory)
        na_probs = None
        
        # Get fraction of priorities to cumulative priority
        if self.cum_priority:
            na_probs = np.array(self.priority) / self.cum_priority
        
        # Randomly draw a sample of size BATCH_SIZE but with account to priorities
        l_index = np.random.choice(N, size=min(N, BATCH_SIZE),p=na_probs)
        self.indices = l_index

        # Sample from indices
        experiences = [self.memory[ii] for ii in l_index]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    # Helper function for computing weights
    def calc_w(self, f_prio, beta, max_w, N):

        # Calculate weight
        f_w = (N * f_prio / self.cum_priority)
        
        # Bias correction
        result = (f_w ** (-beta)) / max_w
        return result

    # Calculate the weights of for each memory according to beta
    def calculate_weights(self, inp_beta):

        N = len(self.memory)
        
        # Maximum weight including bias correction
        max_w = (N * min(self.priority) / self.cum_priority) ** (-inp_beta)

        # Compute weight with helper function
        w = [self.calc_w(self.priority[ii], inp_beta, max_w, N) for ii in self.indices]
        
        result = torch.tensor(w, device = device, dtype = torch.float).reshape(-1, 1)
        return result

    # Update priorities
    def update_priority(self, td_err):

        for i, f_tderr in zip(self.indices, td_err):
            f_tderr = float(f_tderr)
            self.cum_priority -= self.priority[i]
            self.priority[i] = ((abs(f_tderr) + self.eps) ** ALPHA_INITIAL)
            self.cum_priority += self.priority[i]
        self.max_priority = max(self.priority)
        self.indices = []

    # Calculate length of memory
    def __len__(self):

        return len(self.memory)
```

### Implementation of the Single Agent:
Implementation follows the standard agent implementation as described in the Udacity course videos regarding the DQN algorithm and the Lunar Lander example.

Standard implementation of Agent using:
- Standard neural QNetwork with three fully connected layers, exponential linear unit and dropout regularization
- Standard Q-learning algorithm
- Standard epsilon greedy policy
- Standard experience replay with random sampling
- Standard Mean-squared-error loss function
- Adam optimizer: https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam


```python
class SingleAgent():

    def __init__(self):
        
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.seed = random.seed(RANDOM_SEED)

        self.qnetwork_local = QNetwork().to(device) # Use standard network
        self.qnetwork_target = QNetwork().to(device) # Use standard network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)  #Initialize Optimizer with new network

        self.memory = ReplayBuffer() # Use standard experience replay
        self.t_step = 0

    # Perform step
    def step(self, state, action, reward, next_state, done):
        
        # Add to memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Propagate time
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        # Learn from experiences
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    # Choose action for agent according to given policy
    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # epsilon greedy policy for exploration and exploitation
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # Learn from experiences
    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences

        # Implement standardQ-learning
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Calculate target Q
        Q_targets = rewards + (GAMMA * Q_targets_next * (1.0 - dones))

        # Get expected Q
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Standard mean-squared-error loss function
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Reset gradients
        self.optimizer.zero_grad()
        # perform backpropagation
        loss.backward()
        # Optimize weights
        self.optimizer.step()

        # Update networks
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU)*target_param.data)
```

### Implementation of the Double Agent:
Same as the Single Agent with two changes:
- Double Q-Learning algorithm instead of single Q-Learning algorithm: https://arxiv.org/abs/1509.06461
- Duelling neural DuelQNetwork instead of standard QNetwork


```python
class DoubleAgent(SingleAgent): # Inherit from SingleAgent

    def __init__(self):

        super(SingleAgent, self).__init__() # Inherit from SingleAgent

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.seed = random.seed(RANDOM_SEED)
        
        self.qnetwork_local = DuelQNetwork().to(device) # Use Duelling network
        self.qnetwork_target = DuelQNetwork().to(device) # Use Duelling network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) # Initialize Optimizer with new network

        self.memory = ReplayBuffer() # use standard experience replay
        self.t_step = 0
        

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        # Implement double Q-learning
        Q_argmax = self.qnetwork_local(next_states).detach()
        _, a_prime = Q_argmax.max(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, a_prime.unsqueeze(1))
        
        # Calculate target Q
        Q_targets = rewards + (GAMMA * Q_targets_next * (1.0 - dones))
        
        # Get expected Q
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Standard mean-squared-error loss function
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Reset gradients
        self.optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        # Optimize weights
        self.optimizer.step()

        # Update networks
        self.soft_update(self.qnetwork_local, self.qnetwork_target)  
```

### Implementation of the Triple Agent:
Same as SingleAgent with two changes:
 - Prioritized experience replay: https://arxiv.org/abs/1511.05952
 - Weighted mean-squared-error loss function


```python
class TripleAgent(SingleAgent): # Inherit from SingleAgent

    def __init__(self):

        super(SingleAgent, self).__init__() # Inherit from SingleAgent

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.seed = random.seed(RANDOM_SEED)
        
        self.qnetwork_local = QNetwork().to(device) # Use standard network
        self.qnetwork_target = QNetwork().to(device) # Use standard network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) # Initialize Optimizer with new network

        self.memory = PriorityReplayBuffer() # use prioritized experience replay
        self.t_step = 0

    # Calculate beta for bias correction of prioritized experience replay memory
    def calculate_beta(self, time):

        result = BETA_INITIAL + min(float(time) / MAX_T, 1.0) * (1.0 - BETA_INITIAL)
        return result

    # Compute individual loss function: weighted-mean-squared-error
    def calculate_loss(self, inp, exp, w):

        # source: http://forums.fast.ai/t/how-to-make-a-custom-loss-function-pytorch/9059/20
        mse = (inp - exp) ** 2.0
        result = mse * w.expand_as(mse)
        loss = result.mean(0)
        return loss
    
    # Adaption of learning algorithm for prioritized experience replay
    def learn(self, experiences, t=MAX_T):

        states, actions, rewards, next_states, dones = experiences

        # Implement double Q-learning
        Q_argmax = self.qnetwork_local(next_states).detach()
        _, a_prime = Q_argmax.max(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, a_prime.unsqueeze(1))
        
        # Calculate target Q
        Q_targets = rewards + (GAMMA * Q_targets_next * (1.0 - dones))
        
        # Get expected Q
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Update beta for bias correction
        beta = self.calculate_beta(t)
        # Compute weights
        weights = self.memory.calculate_weights(inp_beta=beta)

        # Compute error
        td_err = Q_targets - Q_expected
        # Update memory with error
        self.memory.update_priority(td_err)

        # Weighted mean-squared-error loss function
        loss = self.calculate_loss(Q_targets, Q_expected, weights)
        # Reset gradients
        self.optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        # Optimize weights
        self.optimizer.step()

        # Update weights
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
```

### Implementation of the Quadruple Agent:
Same as TripleAgent with one change:
- Duelling neural DuelQNetwork instead of standard QNetwork

Note that the QuadrupleAgent features many of the improvements used in the Rainbow implementation (https://arxiv.org/abs/1710.02298), which is a state-of-the-art list of improvements and combined implementation for Deep Reinforcement learning.


```python
class QuadrupleAgent(TripleAgent): # Inherit from TripleAgent

    def __init__(self):

        super(TripleAgent, self).__init__() # Inherit from TripleAgent

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.seed = random.seed(RANDOM_SEED)
        
        self.qnetwork_local = DuelQNetwork().to(device) # Use Duelling network
        self.qnetwork_target = DuelQNetwork().to(device) # Use Duelling network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) # Initialize Optimizer with new network

        self.memory = PriorityReplayBuffer() # use prioritized experience replay
        self.t_step = 0
```

### Training of agent with Q-learning algorithm
Algorithm loops over a maximumN_EPISODES=2000.0 and trains and agent in each episode for up to MAX_T=1000.0 time.

- Every 100 episodes the current average score is outputted and the weights are written in a temporary file
- Epsilon for the epsilon-greedy policy decays with time
- The environment is considered solved if the average score is greater than ENV_SOLVED=13.0.
- If the environment is solved, the training is stopped and the final weights are written in a file


```python
def dqn(agent,filename):
    
    scores = []                          # list containing scores from each episode
    scores_window = deque(maxlen = 100)  # last 100 scores
    eps = EPS_START                      # initialize epsilon
    
    for i_episode in range(1, N_EPISODES + 1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]         # get the current state
        score = 0 
        
        for t in range(MAX_T):
            
            action = agent.act(state, eps)                 # select an action     
            
            env_info = env.step(action.astype(int))[brain_name]        # send the action to the environment
            
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            agent.step(state, action, reward, next_state, done) # agent step
            
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        eps = max(EPS_END, EPS_DECAY * eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        # Output temporary weights file every 100 episodes.
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), filename+'_network_temp.pth')
            
        # Output final weights file at end of run.    
        if np.mean(scores_window) > ENV_SOLVED:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), filename+'_network_final.pth')
            break
    return scores
```

### Main-loop of the program:
In the following cells all the previous implementations come into play.
Each agent architecture is subsequently trained and the weights of the final solutions are stored in files.
The average training scores are written into files and stored.


```python
if device == "cpu":
    print("Running on CPU.")
else:
    print("Running on GPU.")
```

    Running on GPU.



```python
print("\nSingleAgent Run:\n")
agent_single = SingleAgent()
scores_single = dqn(agent_single, 'single')
f = open('scores_single.pckl', 'wb')
pickle.dump(scores_single, f)
f.close()
```

    
    SingleAgent Run:
    
    Episode 100	Average Score: 1.06
    Episode 200	Average Score: 4.62
    Episode 300	Average Score: 7.57
    Episode 400	Average Score: 10.09
    Episode 500	Average Score: 12.17
    Episode 557	Average Score: 13.02
    Environment solved in 557 episodes!	Average Score: 13.02



```python
print("\nDoubleAgent Run:\n")
agent_double = DoubleAgent()
scores_double = dqn(agent_double, 'double')
f = open('scores_double.pckl', 'wb')
pickle.dump(scores_double, f)
f.close()
```

    
    DoubleAgent Run:
    
    Episode 100	Average Score: 1.19
    Episode 200	Average Score: 4.87
    Episode 300	Average Score: 8.46
    Episode 400	Average Score: 10.18
    Episode 500	Average Score: 12.53
    Episode 600	Average Score: 12.59
    Episode 634	Average Score: 13.08
    Environment solved in 634 episodes!	Average Score: 13.08



```python
print("\nTripleAgent Run:\n")
agent_triple = TripleAgent()
scores_triple = dqn(agent_triple, 'triple')
f = open('scores_triple.pckl', 'wb')
pickle.dump(scores_triple, f)
f.close()
```

    
    TripleAgent Run:
    
    Episode 100	Average Score: 0.62
    Episode 200	Average Score: 4.77
    Episode 300	Average Score: 7.46
    Episode 400	Average Score: 9.34
    Episode 500	Average Score: 12.50
    Episode 522	Average Score: 13.06
    Environment solved in 522 episodes!	Average Score: 13.06



```python
print("\nQuadrupleAgent Run:\n")
agent_quadruple = QuadrupleAgent()
scores_quadruple = dqn(agent_quadruple, 'quadruple')
f = open('scores_quadruple.pckl', 'wb')
pickle.dump(scores_quadruple, f)
f.close()
```

    
    QuadrupleAgent Run:
    
    Episode 100	Average Score: 0.62
    Episode 200	Average Score: 4.52
    Episode 300	Average Score: 7.49
    Episode 400	Average Score: 10.50
    Episode 500	Average Score: 12.82
    Episode 503	Average Score: 13.06
    Environment solved in 503 episodes!	Average Score: 13.06


The average scores are loaded back from files and plotted.


```python
# Definition of helper function to calculate rolling mean
def runningMean(y, N):
    x = np.asarray(y)
    result = np.zeros(len(x))
    for i in range(1,len(x)):
        if i < int(N):
            result[i] = np.mean(x[0:i]) # Fill up first data points where index is smaller than window size
        else:
            result[i] = np.mean(x[i-N:i]) # Calculate rolling mean where index is larger than window size
    return result

f = open('scores_single.pckl', 'rb')
scores_single = pickle.load(f)
f.close()
f = open('scores_double.pckl', 'rb')
scores_double = pickle.load(f)
f.close()
f = open('scores_triple.pckl', 'rb')
scores_triple = pickle.load(f)
f.close()
f = open('scores_quadruple.pckl', 'rb')
scores_quadruple = pickle.load(f)
f.close()

# Calculate average scores of all four Agents with the same window=100 as used in the Q-learning algorithm
mean_single = runningMean(scores_single, 100)
mean_double = runningMean(scores_double, 100)
mean_triple = runningMean(scores_triple, 100)
mean_quadruple = runningMean(scores_quadruple, 100)

# Make four subplots and show results of each agent in one plot
fig, axs = plt.subplots(2, 2, figsize=(12,12))
axs[0, 0].plot(np.arange(len(scores_single)), scores_single, color="Black")
axs[0, 0].plot(np.arange(len(mean_single)), mean_single, 'tab:orange')
axs[0, 0].axhline(y=13.0, color="Black")
axs[0, 0].set_title('Single Agent')

axs[0, 1].plot(np.arange(len(scores_double)), scores_double, color="Black")
axs[0, 1].plot(np.arange(len(mean_double)), mean_double, 'tab:blue')
axs[0, 1].axhline(y=13.0, color="Black")
axs[0, 1].set_title('Double Agent')

axs[1, 0].plot(np.arange(len(scores_triple)), scores_triple, color="Black")
axs[1, 0].plot(np.arange(len(mean_triple)), mean_triple, 'tab:red')
axs[1, 0].axhline(y=13.0, color="Black")
axs[1, 0].set_title('Triple Agent')

axs[1, 1].plot(np.arange(len(scores_quadruple)), scores_quadruple, color="Black")
axs[1, 1].plot(np.arange(len(mean_quadruple)), mean_quadruple, 'tab:green')
axs[1, 1].axhline(y=13.0, color="Black")
axs[1, 1].set_title('Quadruple Agent')

for ax in axs.flat:
    ax.set(xlabel='Episode #', ylabel='Score')

for ax in axs.flat:
    ax.label_outer()

# Show averages of all fou agents in a single plot for comparison
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.plot(np.arange(len(mean_single)), mean_single, 'tab:orange')
plt.plot(np.arange(len(mean_double)), mean_double, 'tab:blue')
plt.plot(np.arange(len(mean_triple)), mean_triple, 'tab:red')
plt.plot(np.arange(len(mean_quadruple)), mean_quadruple, 'tab:green')

plt.axhline(y=13.0, color="Black")

plt.title('Comparison of all four Agents')
plt.xlabel('Episode #')
plt.ylabel('Score')
plt.show()
```


![png](output_47_0.png)



![png](output_47_1.png)


# Conclusions and future improvements


In this project I have implemented four different Agents with two different neural network architectures to solve the Banana Navigation OpenAI Gym problem.

The performances are as follows:
- SingleAgent requires 557 episodes to solve the problem
- DoubleAgent requires 634 episodes to solve the problem
- TripleAgent requires 522 episodes to solve the problem
- QuadrupleAgent requires 503 episodes to solve the problem

However, I want to stress out, that the complexity of the algorithms of QuadrupleAgent are higher than of the SingleAgent. Therefore, solving in lesser episodes does not necessarily mean that the solution requires less computational time. Because training the QuadrupleAgent requires more time per episode than training the SingleAgent.

Nevertheless, it is clear that more advanced algorithms, with double q-learning, duelling neural networks, exponential linear units, dropout regularization or prioritized experience replay as shown in this project work are better than simple standard approaches.

It can also be seen in the RainBow https://arxiv.org/abs/1710.02298 implementation of DeepMind, which combines even more improvements for Deep Reinforcement Learning.
That would include to add parametric noise to the weights as described by https://arxiv.org/abs/1706.10295.

In the future, it would very well be worthwhile to try more of these improvements or a full RainBow approach.

Further future work might include solving the problem from pixels. THen the agent, would not recieve a numerical input state vector, but an array of pixels, which need to be processed with convolutional layers. As described in the original DeepMind Nature paper https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning superhuman-control can be reached.

Of course, at last, a complete study of exploration of the hyperparameter space is necessary, which would lead to some questions like this: How many layers perform best? How many neurons? Which activation functions? Linear or exponential epsilon decay?  How to choose the discount factor for reward?

More ideas:
- Port the algorithm into C++...
- Run the training on Nvidia Jetson...


```python

```
