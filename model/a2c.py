import gym
import sys
sys.path.append("/Users/redabelhaj/Desktop/MAP511")
import sneks
from sneks.envs.snek import SingleSnek
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import time

class ActorCriticNet(torch.nn.Module):

    def __init__(self, size):
        super(ActorCriticNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2)
        self.conv2 = torch.nn.Conv2d(3, 2, 2, stride=2)
        out_size = (size[0]- 1)
        out_size = int((out_size -2 )/2 +1)

        self.actor = torch.nn.Linear(2*out_size**2, 4)
        self.critic = torch.nn.Linear(2*out_size**2, 1)

    def forward(self, obs):
        s = len(obs.size())
        if s ==2:
            tens2 = obs.unsqueeze(0).unsqueeze(0)
        if s == 3:
            tens2 = obs.unsqueeze(0)
        if s ==4 :
            tens2 = obs
        out = self.conv1(tens2)
        out = torch.sigmoid(out)
        out = self.conv2(out)
        out = out.flatten(1)
        logits, values = self.actor(out), self.critic(out)
        return logits, values


class A2C:
    def __init__(self, size, name, walls=True, n_iter = 500, batch_size=32, gamma=.99):
        self.net = ActorCriticNet(size)
        self.name = name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.env = SingleSnek(size = size, add_walls=walls, obs_type="rgb")
        self.gamma = gamma

    def get_action(self, state):
        tens = torch.tensor(state, dtype = torch.float32).permute(2,0,1)
        logits, _ = self.net(tens)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze()
        return np.random.choice(4, p = probs.detach().numpy())

    def play_one_episode(self):
        transitions = []
        new_frame = self.env.reset()
        state = new_frame
        frame = new_frame
        action = self.get_action(state)
        done = False
        sts, acts, rews = [],[], [] 
        while not(done):
            new_frame, reward, done, _ = self.env.step(action)
            new_state = new_frame-frame
            a_t = torch.tensor([action], dtype = torch.int64)
            s_t = torch.tensor(state, dtype = torch.float32).permute(2,0,1) 
            sts.append(s_t)
            acts.append(a_t)
            rews.append(reward)

            frame = new_frame
            state = new_state
            action =self.get_action(state)
        
        gamma = self.gamma
        len_ep = len(rews)
        for i in range(len_ep):
            gammas = torch.tensor([gamma**j for j in range(len_ep-i)])
            rewards = torch.tensor(rews[i:], dtype = torch.float32)
            g = torch.dot(gammas,rewards)
            s_t, a_t = sts[i], acts[i]
            transitions.append((s_t, a_t, g))

        return transitions
    
    def get_dataset(self, map_results):
        full_list = []
        list_rewards = []
        for transitions in map_results:
            full_list += transitions
            for _,_,g in transitions:
                list_rewards.append(g)
        gt_tens = torch.tensor(list_rewards, dtype = torch.float32)
        mean, std = torch.mean(gt_tens),torch.std(gt_tens)
        gt_tens = (gt_tens-mean)/(std + 1e-8)

        final_list  = [(s,a,gt_tens[i]) for i,(s,a,_) in enumerate(full_list) ]
        return final_list
        

    def get_stats(self, map_results):
        n_batch = len(map_results)
        reward_ep, len_ep  = [], []
        for i in range(n_batch):
            transitions = map_results[i]
            len_ep.append(len(transitions))
            gts = [ g.item() for _,_,g in transitions]
            r = (1-self.gamma)*sum(gts) + self.gamma*gts[0]
            reward_ep.append(r)
    
        return sum(reward_ep)/n_batch, sum(len_ep)/n_batch

    def one_training_step(self, map_results):
        dataset = self.get_dataset(map_results)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
        optimizer = torch.optim.Adam(self.net.parameters())
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        mse = torch.nn.MSELoss()
        for s,a,g in dataloader:
            optimizer.zero_grad()
            out,values = self.net(s)
            values = values.squeeze()
            values2 = values.detach()
            loss_actor = (ce(out, a.squeeze())*(g-values2)).mean()            
            loss_critic = mse(values, g)
            loss_actor.backward(retain_graph = True)
            loss_critic.backward()
            optimizer.step()



if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.manual_seed(0)
    size = (20, 20)
    a2c = A2C(size, 'a2c',walls=True, n_iter=500, batch_size=16)
    bs = a2c.batch_size
    best_reward = -1
    best_length = 0

    a2c.net.load_state_dict(torch.load(a2c.name + '_state_dict.txt'))
    with open("ep_rewards_"+a2c.name+".txt","r+") as f:
            f.truncate(0)
    with open("ep_lengths_"+a2c.name+".txt","r+") as f:
            f.truncate(0)
    debut = time.time()
    with mp.Pool() as pool:
        for it in range(a2c.n_iter):
            
            args = bs*[a2c]
            map_results = pool.map_async(A2C.play_one_episode, args).get()
            a2c.one_training_step(map_results)
            mean_reward, mean_length = a2c.get_stats(map_results)
            if mean_reward > best_reward:
                print('\n', "********* new best reward ! *********** ", round(mean_reward, 3), '\n')
                best_reward = mean_reward
                torch.save(a2c.net.state_dict(), a2c.name + '_state_dict.txt')
            if mean_length > best_length:
                print('\n', "********* new best length ! *********** ", round(mean_length, 3), '\n')
                best_length = mean_length
                torch.save(a2c.net.state_dict(), a2c.name + '_state_dict.txt')

            with open("ep_rewards_"+a2c.name+".txt","a") as f:
                f.write(str(round(mean_reward, 3))+ '\n')
            with open("ep_lengths_"+a2c.name+".txt","a") as f:
                f.write(str(round(mean_length, 3))+ '\n')
            print('iteration : ', it, 'reward : ', round(mean_reward, 3),'length : ', round(mean_length, 3),'temps : ', round(time.time()-debut, 3), '\n')
