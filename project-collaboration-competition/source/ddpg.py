import pickle
import numpy as np
from collections import deque

import torch


def ddpg(agent, config, env):
    """Deep Deterministic Policy Gradient Algorithm for Training an ML model.
    Params
    ======
        agent (ActorCritic): agent playing in Unity Environment
        config (Config): Configuration hyperparameters of training and network
        env (UnityEnvironment): Unity Environment used in this project
    """
        
    # Empty variables for all scores
    scores_window = deque(maxlen = 100)
    scores_max = []
    scores_avg = []
    
    brain_name = env.brain_names[0]

    for i_episode in range(1, config.n_episodes + 1):

        # Get state of environment
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        states = env_info.vector_observations                  # get current state
        scores = np.zeros(config.n_agents)                     # reset scores
        agent.reset()

        while True:

            # Get actions for agents
            actions = agent.act(states)                        # select an action (for each agent)

            # Perform steps within the environment
            env_info = env.step(actions)[brain_name]           # send the action to the environment
            
            # Read variables after step
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished (for each agent)

            # Perform step for the agents incl. learning
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)

            # Sum up rewards
            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            if np.any(dones):                                  # exit loop if episode finished
                break

        episode_score = np.max(scores)                         # episode reward as maximum score of both agents
        scores_max.append(episode_score)                       # save most recent score
        scores_window.append(episode_score)                    # save most recent score
        scores_avg.append(np.mean(scores_window))              # save most recent score

        # Output the simulation status every episode
        if i_episode % 1 == 0:
            print('\rEpisode {}\tCurrent Score: {:.4f}\tAverage Score: {:.4f}'.format(i_episode, episode_score, np.mean(scores_window)))

        # Every 20 episodes write the simualtions status to files for backup
        if i_episode % 20 == 0:
            
            # Saving of network weights
            torch.save(agent.actor_local.state_dict(), 'output/checkpoint_actor_temp.pth')
            torch.save(agent.critic_local.state_dict(), 'output/checkpoint_critic_temp.pth')

            # Saving of scores
            f = open('output/scores_temp.pckl', 'wb')
            pickle.dump(scores_max, f)
            f.close()
            f = open('output/scores_temp_avg.pckl', 'wb')
            pickle.dump(scores_avg, f)
            f.close()


        # Output final weights after environment is solved
        if np.mean(scores_window) > config.env_solved:
            
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'output/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'output/checkpoint_critic.pth')
            
            # Terminate the simulations, because environment has been solved
            break

    return scores_max, scores_avg