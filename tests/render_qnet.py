import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")

from dqn import DQN, get_action_eps_greedy
from sneks.envs.snek import SingleSnek
import torch
from time import sleep


env = SingleSnek(size = (28,28),add_walls=True)
state = env.reset()
size = env.SIZE
q_net = DQN(size)
q_net.load_state_dict(torch.load("network"))

done = False
while not done:
    action_values = q_net(state)
    action = torch.argmax(action_values)
    obs, rewards, done, info = env.step(action)
    env.render(mode='human')
    sleep(0.1)
