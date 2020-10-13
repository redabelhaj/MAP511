import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")
import gym
import sneks
from time import sleep
import argparse
from sneks.envs.sneks import MultiSneks
from sneks.envs.snek import SingleSnek
import numpy as np


### EXEMPLE : les sneks evitent les murs

def take_action(snek, env):
    d = snek.current_direction_index #0,1,2,3
    head = snek.my_blocks[0]
    if d==0 and head[0]==1:
        if head[1]==1:
            return 1
        else:
            return 3
    if d==1 and head[1]==env.SIZE[0]-2:
        if head[0]==1:
            return 2
        else:
            return 0
    if d==2 and head[0]==env.SIZE[1]-2:
        if head[1]==1:
            return 1
        else:
            return 3
    if d==3 and head[1]==1:
        # print("je ss tout a gauche")
        if head[0]==1:
            return 2
        else:
            return 0

def get_new_head(head, d,potential_dir):
    DIRECTIONS = [np.array([-1,0]), np.array([0,1]), np.array([1,0]), np.array([0,-1])]
    if (potential_dir == (d+2)%len(DIRECTIONS)):
            return -1
    else:
        new_head = tuple(np.array(head) + DIRECTIONS[potential_dir])
    return new_head


def take_action2(snek, env):
    d = snek.current_direction_index #0,1,2,3
    head = snek.my_blocks[0]
    dirs = [0,1,2,3]
    ok_dirs =[]
    for potential_dir in dirs:
        new_head = get_new_head(head, d, potential_dir)
        if new_head!= -1 and (1 <= new_head[0] <= env.SIZE[0]-2) and (1 <= new_head[1] <= env.SIZE[1]-2):
            ok_dirs.append(potential_dir)
    s = len(ok_dirs)
    if d in ok_dirs:
        rd = np.random.rand()
        if rd>.2:
            return d
       
    i = np.random.randint(0,s)
    return ok_dirs[i]
            
        



N_SNEKS = 2

env = MultiSneks(n_sneks= 1, size=(24, 24), obs_type='rgb', add_walls=True)
obs = env.reset()

dones = [False] * env.N_SNEKS
r = [0, 0]
while not all(dones):
    actions = []
    for snek in env.world.sneks:
        actions.append(take_action2(snek, env))
    obs, rewards, dones, info = env.step(actions)
    r = map(lambda x,y: x+y, zip(r, rewards))
    env.render(mode='human')
    sleep(0.1)













