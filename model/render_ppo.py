import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")

from ppo import ActorCriticNet, PPO
import torch
from time import sleep
import numpy as np

size = (12, 12)
ppo = PPO(size, 'ppo_img', n_iter=10000, batch_size=64, seed=10, hunger  = 17)
ppo.net.load_state_dict(torch.load("saved_models/ppo_1730_vanilla_state_dict.txt"))

lens, tots = [],[]
n_ep = 10
for i in range(n_ep):
    obs = ppo.env.reset()
    done = False
    lenep = 0
    totrew = 0
    
    while not done:
        lenep+=1
        tens = torch.tensor(obs, dtype = torch.float32).permute(2,0,1)
        logits, value = ppo.net(tens)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze().detach().numpy()
        act = np.random.choice(4,p=probs)
        obs, rewards, done, _ = ppo.env.step(act)
        realrew, _ = rewards
        totrew+= realrew
        ppo.env.render(mode='human')
        sleep(0.08)
   
    lens.append(lenep)
    tots.append(totrew)
print('mean length', sum(lens)/n_ep)
print('mean reward', sum(tots)/n_ep)