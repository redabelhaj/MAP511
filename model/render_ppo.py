import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")

from ppo import ActorCriticNet, PPO
from sneks.envs.snek import SingleSnek
import torch
from time import sleep
import numpy as np


env = SingleSnek(size = (15,15), add_walls=False, obs_type="rgb")
size = (15, 15)
ppo = PPO(size, 'ppo', walls=False, n_iter=500, batch_size=32)
ppo.net.load_state_dict(torch.load("ppo_nowalls_state_dict.txt"))

obs = env.reset()
done = False
while not done:
    tens = torch.tensor(obs, dtype = torch.float32).permute(2,0,1)
    logits, value = ppo.net(tens)
    probs = torch.softmax(logits, dim=-1)
    probs = probs.squeeze().detach().numpy()
    act = np.argmax(probs)
    obs, rewards, done, info = env.step(act)
    env.render(mode='human')
    sleep(0.1)