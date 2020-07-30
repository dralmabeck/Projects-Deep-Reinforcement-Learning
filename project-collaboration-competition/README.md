[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Project: Collaboration and Competition

### Introduction

For this project, I will train two agents to play tennis within [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

This projects is part of my project work on the Deep Reinforcement Learning Nanodegree by Udacity.
See https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893 and https://github.com/udacity/deep-reinforcement-learning

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the "project-collaboration-competition" subdirectory of this GitHub repository, and unzip (or decompress) the file. 

### Installation instructions
 
 This project was developed with Python version 3.6. It requires the following packages with the corresponding versions.
 
 - jupyter==1.0.0
 - notebook==6.0.3
 - pypickle==0.1.0
 - numpy==1.19.1
 - matplotlib==3.3.0
 - torch==1.6.0
 - unityagents==0.4.0
 
 Please note, that currently, **`unityagents`** has very specific requirements (such as Tensorflow in version 1.7.1, which will need Python 3.6), so the version numbers are essential for setting up a Python environment to compile and execute this notebook.
 
 All packages used by me in a freshly set up development environment can be found in the **`requirements.txt`**.
 
 I recommend using the Anaconda Python distribution (https://www.anaconda.com/products/individual) for a thourough installation of a usable Python environment. Then from the console a missing package can easily be installed with **`conda install <package_name>`**.
 
 Furthermore, with the pip package manager missing packages can be installed via the console with **`pip install <package_name>`**.
 
 This repository can be downloaded with Git using **`git checkout https://github.com/dralmabeck/Udacity-Deep-Reinforcement-Learning`**.
 
 ### How to use
 
 1. From the main directory of this repository in the console go into the "project-collaboration-competition" subdirectory. There execute **`jupyter notebook`**.
 
 2. Next, open the **`Tennis.ipynb`** notebook.
 
 3. Follow the instructions and descriptions in `Tennis.ipynb` to get started with training your own agent. It lists all the algorithms and there sources on scientific papers.
 
 4. Read the **`Report.md`** for insights on my solution on this problem should you get stuck.
 
 ### Files included
 
 1. **`README.md`** - This file
 2. **`Report.md`** - Project report based on Tennis.ipynb and including all executed cells and output
 3. **`output_33_0.png`** - Picture file for Report.md
 4. **`Tennis.ipynb`** - Main program. Contains entire implementation of agent and data processing
 5. **`Soccer.ipynb`** - Template for future possible project work to train an agent in the Unity Soccer Environment
 6. **`unity_environment.log`** - Log File of Unity Environment
 7. **`requirements.txt`** - Description of fully functional development and execution environment
 8. **`output/scores.pckl`** - Training scores of Agent, maximum of both agents
 9. **`output/scores_avg.pckl`** - Training scores of Agent, average values
 10. **`output/checkpoint_actor.pth`** - Final network parameters for Actor Network
 11. **`output/checkpoint_critic.pth`** - Final network parameters for Critic Network
 12. **`source/workspace_utils.py`** - Utility function provided by Udacity to keep Jupyter workspace alive during long GPU training sessions
 13. **`source/ddpg.py`** - File containing Deep Deterministic Policy Gradient Algorithm
 14. **`source/agent.py`** - File containing the Actor-Critic-Agent implementation
 15. **`source/network.py`** - File containing the neural network architectures used by agent
 16. **`source/replay_buffer.py`** - File containing replay buffer implementation used by agent
 17. **`source/noise_generator.py`** - File containing a stochastic noise process used by agent
 18. **`source/__init.py__`** - Initialization file required by Python to import modules from source directory
