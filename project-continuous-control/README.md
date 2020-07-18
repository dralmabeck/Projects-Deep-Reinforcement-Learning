[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project: Continuous Control

### Introduction

For this project, I will train 20 agents with double-jointed arms to move to target locations within the Unity Reacher Environment.
 
 This projects is part of my project work on the Deep Reinforcement Learning Nanodegree by Udacity. See https://www.udacity.com/course/  deep-reinforcement-learning-nanodegree--nd893 and https://github.com/udacity/deep-reinforcement-learning

![Trained Agent][image1]

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation state space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1.0 and 1.0.

The task is continous, and in order to solev the environment, the average score of all agents must exceed 30 over 100 consecutive episodes.

### Distributed Training

For this project, I was provided with two separate versions of the Unity environment:
- The first version contains a single agent (not solved in this project).
- The second version contains 20 identical agents, each with its own copy of the environment (solved in this project).

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

#### Instructions for solving the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents). Also more agents will collect experiences in parallel , which can be used for the Experience Replay Buffer.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the "project-continous-control" subdirectory of this GitHub repository, and unzip (or decompress) the file. 

### Installation instructions
 
 This project requires Python in version >= 3.5. Furthermore the following Python packages are required.
 
 - Jupyter
 - Notebook
 - time
 - pickle
 - math
 - datetime
 - random
 - numpy
 - sys
 - copy
 - collections
 - matplotlib
 - torch
 - unityagents
 
 I recommend using the Anaconda Python distribution (https://www.anaconda.com/products/individual) for a thourough installation of a     usable Python environment. Then from the console a missing package can easily be installed with **`conda install <package_name>`**.
 
 Furthermore, with the pip package manager missing packages can be installed via the console with **`pip install <package_name>`**.
 
 This repository can be downloaded with Git using **`git checkout https://github.com/dralmabeck/Udacity-Deep-Reinforcement-Learning`**.
 
 ### How to use
 
 1. From the main directory of this repository in the console go into the "project-continous-control" subdirectory. There execute **      `jupyter notebook`**.
 
 2. Next, open the **`Continous_Control.ipynb`** notebook.
 
 3. Follow the instructions and descriptions in `Continous_Control.ipynb` to get started with training your own agent. It lists all the         algorithms and there sources on scientific papers.
 
 4. Read the **`Report.md`** for insights on my solution on this problem should you get stuck.
 
 ### Files included
 
 1. **`README.md`** - This file
 2. **`Report.md`** - Project report based on Navigation.ipynb and including all executed cells and output
 3. **`output_47_1.png`** - Picture file for Report.md
 4. **`Navigation.ipynb`** - Main program. Contains entire implementation of agent and data processing
 5. **`unity_environment.log`** - Log File of Unity Environment
 6. **`scores.pckl`** - Training scores of Agent
 7. **`checkpoint_actor.pth`** - Final network parameters for Actor Network
 8. **`checkpoint_critic.pth`** - Final network parameters for Critic Network
 9. **`Crawler.ipynb`** - Template for future project to train an agent in the Unity Crawler Environment
