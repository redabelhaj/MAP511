import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")

from ppo_rs import ActorCriticNet, PPO_RS
from simple_ppo import SimpleACNet, SimplePPO
from sneks.envs.snek import SingleSnek
import torch
from time import sleep
import numpy as np

size = (12, 12)

# env = SingleSnek(size = (12,12), add_walls=True, obs_type="simplest")
# ppo = SimplePPO(size, 'ppo', walls=True, n_iter=10000, batch_size=64)
# ppo.net.load_state_dict(torch.load("ppo_state_dict.txt"))


# for _ in range(10):
#     obs = env.reset()
#     done = False
#     while not done:
#         tens = torch.tensor(obs, dtype = torch.float32)
#         logits, value = ppo.net(tens)
#         probs = torch.softmax(logits, dim=-1)
#         probs = probs.squeeze().detach().numpy()
#         act = np.argmax(probs)
#         obs, rewards, done, info = env.step(act)
#         env.render(mode='human')
#         sleep(0.08)


ppo = PPO_RS(size, 'ppo_img', n_iter=10000, batch_size=64, seed=10, hunger  = 17)
# ppo.net.load_state_dict(torch.load("saved_models/ppo_no_loop_h17_b30_newvers_state_dict.txt"))
# ppo.net.load_state_dict(torch.load("saved_models/ppo_no_loop_h17_b30_beta_state_dict.txt"))
# ppo.net.load_state_dict(torch.load("saved_models/ppo_no_loop_h17_b30_fixedseed_v3_state_dict.txt"))
# ppo.net.load_state_dict(torch.load("saved_models/ppo_no_loop_h17_b30_test2_state_dict.txt"))
ppo.net.load_state_dict(torch.load("saved_models/ppo_1730_vanilla_state_dict.txt"))

# ppo_no_loop_h17_b30_test2
# ppo_no_loop_h17_b30_fixedseed_v2
# ppo_img_hunger10_newrew
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
        obs, rewards, done, info = ppo.env.step(act)
        realrew, _ = rewards
        totrew+= realrew
        ppo.env.render(mode='human')
        sleep(0.08)
    # print('episode',i+1, 'length : ', lenep)
    # print('episode', i+1, 'rewards : ', totrew, '\n')
    lens.append(lenep)
    tots.append(totrew)
print('mean length', sum(lens)/n_ep)
print('mean reward', sum(tots)/n_ep)