import sys

from ppo_rotate import ActorCriticNet, PPO_ROT
import torch
from time import sleep
import numpy as np

size = (12,12)
ppo = PPO_ROT(size, 'ppo_img', n_iter=10000, batch_size=64, seed=10,hunger=17)
ppo.net.load_state_dict(torch.load("saved_models/ppo_rotate_state_dict.txt"))

lens, tots = [],[]
n_ep = 10
for i in range(n_ep):
    new_obs, dire = ppo.env.reset()
    obs = new_obs
    net_action, prob, net_tens = ppo.get_action_prob_state(obs, dire)
    done = False
    true_rewards = []
    this_len =0
    while not(done):
        action = ppo.get_real_action(net_action,dire)
        new_tuple_obs, reward, done, _ = ppo.env.step(action)
        new_obs, dire = new_tuple_obs
        obs = new_obs
        net_action, prob,net_tens = ppo.get_action_prob_state(obs, dire)
        ppo.env.render()
        sleep(.06)
        this_len+=1
        true_r, _  = reward 
        true_rewards.append(true_r)
        
    lens.append(this_len)
    tots.append(sum(true_rewards))
print('mean length', sum(lens)/n_ep)
print('mean reward', sum(tots)/n_ep)

