import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")

from dqn import DQN, get_action_eps_greedy
from sneks.envs.snek import SingleSnek
import torch
from time import sleep


env = SingleSnek(size = (15,15), add_walls=True, obs_type="rgb")
size = env.SIZE
q_net = DQN(size)
q_net.load_state_dict(torch.load("network"))

obs = env.reset()
done = False
while not done:
    tens = torch.tensor(obs, dtype = torch.float32).permute(2,0,1)
    action_values = q_net(tens)
    action = torch.argmax(action_values)
    obs, rewards, done, info = env.step(action)
    env.render(mode='human')
    sleep(0.1)


