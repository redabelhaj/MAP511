import gym
import sys
import sneks
from sneks.envs.snek import SingleSnek
import numpy as np
import random as rand
import time






class RandomActor:
    def __init__(self, size, name, n_iter = 500):
        self.name = name
        self.n_iter = n_iter
        self.env = SingleSnek(size = size, add_walls=False, obs_type='simplest')

    def get_action(self):
        return np.random.choice(4)

    def play_one_episode(self):
        transitions = []
        self.env.reset()
        action = self.get_action()
        done = False
        rews = []
        while not(done):
            _, reward, done, _ = self.env.step(action)
            rews.append(reward)
            action =self.get_action()
        return rews
        

    
        



if __name__ == "__main__":
    size = (10, 10)
    random_actor = RandomActor(size, 'random',n_iter=500)    
    debut = time.time()
    rews, lengths = [], []

    
    for it in range(random_actor.n_iter):
        rewards = random_actor.play_one_episode()
        lengths.append(len(rewards))
        rews.append(sum(rewards))
    mean_rew = sum(rews)/random_actor.n_iter
    mean_length = sum(lengths)/random_actor.n_iter
    print("mean length : ", mean_length)
    print("mean reward : ",mean_rew )