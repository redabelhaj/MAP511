import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")

# from ppo_rs import ActorCriticNet, PPO_RS
from ppo_rs_rotate import ActorCriticNet, PPO_RS_ROT
from sneks.envs.snek import SingleSnek
import torch
from time import sleep
import numpy as np


size = (12,12)
ppo = PPO_RS_ROT(size, 'ppo_img', n_iter=10000, batch_size=64, seed=10,hunger=17)
ppo.net.load_state_dict(torch.load("saved_models/ppo_rotate_test_state_dict.txt"))


new_obs, dire = ppo.env.reset()
obs = new_obs
net_action, prob, net_tens = ppo.get_action_prob_state(obs, dire)
done = False
true_rewards = []
old_dist = .5
while not(done):
    action = ppo.get_real_action(net_action,dire)
    new_tuple_obs, reward, done, _ = ppo.env.step(action)
    new_obs, dire = new_tuple_obs
    obs = new_obs
    net_action, prob,net_tens = ppo.get_action_prob_state(obs, dire)
    ppo.env.render()
    sleep(.06)

    true_r, _  = reward 
    true_rewards.append(true_r)
    if true_r>0: print(true_r)
    
print(sum(true_rewards))


