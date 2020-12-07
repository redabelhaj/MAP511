# Reinforcement learning for the game of Snake

The goal of this project is to apply Reinforcement Learning (RL) methods to solve the game of Snake. We propose two different algorithms and different approaches for modeling the underlying Markov Decision Process (MDP). We also use various reward-shaping approaches to enhance the performance of the algorithm.

The models used are in the folder "model" :
 - a2c.py corresponds to the A2C algorithm
 - ppo.py  corresponds to the PPO algorithm
 - simple_ppo.py corresponds to the PPO with a simple state space
 - render_ppo.py shows the behaviour of an agent trained with ppo or a2c 
 - render_ppo_rot.py shows the behaviour of an agent trained with the technique of state rotations (see report for details)

Sometimes the path to the directory must be added with sys with sys.path.append(path)


## Requirements : 
numpy
torch
tqdm


### Authors 
Hugo Artigas, Reda Belhaj-Soullami & Mohamed Mimouna