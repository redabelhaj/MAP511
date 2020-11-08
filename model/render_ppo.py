import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")

from ppo import ActorCriticNet, PPO
from simple_ppo import SimpleACNet, SimplePPO
from sneks.envs.snek import SingleSnek
import torch
from time import sleep
import numpy as np

size = (12, 12)

env = SingleSnek(size = (12,12), add_walls=True, obs_type="simplest")
ppo = SimplePPO(size, 'ppo', walls=True, n_iter=10000, batch_size=64)
ppo.net.load_state_dict(torch.load("ppo_state_dict.txt"))


for _ in range(10):
    obs = env.reset()
    done = False
    while not done:
        tens = torch.tensor(obs, dtype = torch.float32)
        logits, value = ppo.net(tens)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze().detach().numpy()
        act = np.argmax(probs)
        obs, rewards, done, info = env.step(act)
        env.render(mode='human')
        sleep(0.08)