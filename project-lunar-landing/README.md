# Project: Lunar Landing

### Introduction

For this project, I will train an agent to land a space ship on the Moon.
This environment is part of the OpenAI Gym, more details can be read here: https://gym.openai.com/envs/LunarLander-v2/

The work is based on a project by the Udacity Deep Reinforcement Learning Nanodegree: https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893

![](https://github.com/dralmabeck/Udacity-Deep-Reinforcement-Learning/blob/master/project-lunar-landing/lunar_landing.gif)

A landing pad is at coordinates (0,0). Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. The environment is considered solved when 200 points are reached. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. However, a usually a timelimit of 1000 is used for each episode in training.

Four discrete actions available:
1. do nothing
2. fire left orientation engine
3. fire main engine
4. fire right orientation engine.

The state vector consists of:
1. horizontal coordinate
2. vertical coordinate
3. horizontal speed
4. vertical speed
5. angle
6. angular speed
7. 1 if first leg has contact, else 0
8. 1 if second leg has contact, else 0

### Installation instructions

This project requires Python in version >= 3.5. Furthermore the following Python packages are required:

Jupyter, Notebook, Gym, Time, Pickle, Math, Datetime, Random, Numpy, Collections, Matplotlib, Torch

I recommend using the Anaconda Python distribution (https://www.anaconda.com/products/individual) for a thourough installation of a usable Python environment. Then from the console a missing package can easily be installed with **`conda install <package_name>`**.

Furthermore, with the pip package manager missing packages can be installed via the console with **`pip install <package_name>`**.

This repository can be downloaded with Git using **`git checkout https://github.com/dralmabeck/Udacity-Deep-Reinforcement-Learning`**.

### How to use

1. Navigate into the main directory of this repository on your local machine in the console and go into the "project-lunar-lander" subdirectory. There execute **`jupyter notebook & `** as command.

2. Next, in your browser window open the **`Lunar_Lander.ipynb`** notebook.

3. Follow the instructions and descriptions in `Lunar_Lander.ipynb` to get started with training your own agent. It lists all the algorithms and their sources on scientific papers.

4. See the executed cells and description in `Lunar_Lander.html` about insights on my solution on this problem should you get stuck.

### Files included

1. **`README.md`** - This file
2. **`Lunar_Lander.ipynb`** - Main program. Contains entire implementation of agent and data processing
3. **`Lunar_Lander.html`** - HTML-copy of the notebook with all cells executed and results displayed
4. **`scores_single.pckl`** - Training scores of SingleAgent
5. **`scores_double.pckl`** - Training scores of DoubleAgent
6. **`scores_triple.pckl`** - Training scores of TripleAgent
7. **`scores_quadruple.pckl`** - Training scores of QuadrupleAgent
8. **`single_network_final.pth`** - Final network parameters for SingleAgent
9. **`double_network_final.pth`** - Final network parameters for DoubleAgent
10. **`triple_network_final.pth`** - Final network parameters for TripleAgent
11. **`quadruple_network_final.pth`** - Final network parameters for QuadrupleAgent
12. **`dqn_agent.py`** - Original Udacity file. Only used to display an untrained agent
13. **`model.py`** - Original Udacity file. Only used to display an untrained agent
14. **`lunar_landing.gif`** - Movie of three episodes of trained agent
