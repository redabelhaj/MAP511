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


class SimpleACNet(torch.nn.Module):
    """
    Actor critic net in the case of simple observation (head pos, fruit pos, current direction ) 
    observation is a vector of 5 components
    """

    def __init__(self, hidden_size):
        super(SimpleACNet,self).__init__()
        self.layer1 = torch.nn.Linear(5, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.actor = torch.nn.Linear(hidden_size, 4)
        self.critic = torch.nn.Linear(hidden_size, 1)

    def forward(self, obs):
        out = self.layer1(obs)
        out = torch.sigmoid(out)
        out = self.layer2(out)
        out = torch.sigmoid(out)
        logits = self.actor(out)
        value = self.critic(out)
        return logits, value

class SimplePPO:
    def __init__(self, size, name, hidden_size = 30,walls = True,n_iter = 500, batch_size = 32,gamma = .99, n_epochs=5, eps=.2, target_kl=1e-2):
        self.name = name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.env = SingleSnek(size = size, add_walls=walls, obs_type='simplest')
        self.net = SimpleACNet(hidden_size)
        self.n_epochs = n_epochs
        self.eps = eps
        self.gamma = gamma
        self.target_kl = target_kl
    
    def get_action_prob(self, state):
        tens = torch.tensor(state, dtype = torch.float32)
        logits, _ = self.net(tens)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze().detach()
        act = np.random.choice(4, p = probs.numpy())
        return act, probs[act]

    def play_one_episode(self):
        transitions = []
        new_obs = self.env.reset()
        obs = new_obs
        action, prob = self.get_action_prob(obs)
        done = False
        sts, ats, pts, rts = [], [], [], []
        while not(done):
            new_obs, reward, done, _ = self.env.step(action)
            s_t  = torch.tensor(obs, dtype = torch.float32)
            a = 4*[0]
            a[action] =1
            a_t = torch.tensor(a, dtype = torch.float32)
            p_t = torch.tensor([prob], dtype = torch.float32)

            true_rew, _ = reward
            sts.append(s_t)
            ats.append(a_t)
            pts.append(p_t)
            rts.append(true_rew)

            obs = new_obs
            action, prob = self.get_action_prob(obs)
        len_ep = len(rts)
        for i in range(len_ep):
            gammas = torch.tensor([self.gamma**j for j in range(len_ep-i)])
            rewards = torch.tensor(rts[i:], dtype = torch.float32)
            g = torch.dot(gammas,rewards)
            s_t, a_t, p_t = sts[i], ats[i], pts[i]
            transitions.append((s_t, a_t, p_t, g))
        return transitions
    
    def get_dataset(self,map_results):
        full_list = []
        list_rewards = []
        for transitions in map_results:
            full_list += transitions
            for _,_,_,g in transitions:
                list_rewards.append(g)
        gt_tens = torch.tensor(list_rewards, dtype = torch.float32)
        # mean, std = torch.mean(gt_tens),torch.std(gt_tens)
        # gt_tens = (gt_tens-mean)/(std + 1e-8)

        final_list  = [(s,a,p,gt_tens[i]) for i,(s,a,p,_) in enumerate(full_list) ]
        return final_list

    def get_actor_loss(self, states, actions, probs, r):
        logits, vals = self.net(states)
        advs = r.unsqueeze(dim =-1)-vals.detach()
        new_probs = torch.unsqueeze(torch.diag(torch.matmul(actions, torch.softmax(logits, dim=-1).T)), dim=-1)
        ratio = new_probs/(probs.detach() + 1e-8)
        clip_advs = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advs
        return -torch.min(ratio*advs, clip_advs).mean()

    def get_stats(self, map_results):
        n_batch = len(map_results)
        reward_ep, len_ep  = [], []
        for i in range(n_batch):
            transitions = map_results[i]
            len_ep.append(len(transitions))
            gts = [ g.item() for _,_,_,g in transitions]
            r = (1-self.gamma)*sum(gts) + self.gamma*gts[0]
            reward_ep.append(r)
    
        return sum(reward_ep)/n_batch, sum(len_ep)/n_batch




    def one_training_step(self, map_results):
        dataset = self.get_dataset(map_results)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
        kl_data = DataLoader(dataset, batch_size= len(dataset), shuffle=True, num_workers=1)
        optimizer = torch.optim.Adam(self.net.parameters())
        n_epochs = self.n_epochs

        for i in range(n_epochs):
            running_loss = 0
            for s,c,p,r in dataloader:
                optimizer.zero_grad()
                loss = self.get_actor_loss(s,c,p,r)
                running_loss+=loss
                loss.backward()
                optimizer.step()
            
            kl=0
            for s,a,p,r in kl_data:
                logits, _ = self.net(s)
                new_probs = torch.unsqueeze(torch.diag(torch.matmul(a, torch.softmax(logits, dim=-1).T)), dim=-1)
                kl += (torch.log(p) - torch.log(new_probs)).mean().item()
            if kl > 1.5*self.target_kl:
                # print(f'Early stopping at step {i} due to reaching max kl {kl}')
                # print('loss : ', running_loss)
                break
            
            optimizer.zero_grad()
            loss_critic=0
            mse = torch.nn.MSELoss()
            for s,_,_,r in dataset:
                r_tens = torch.tensor([r], dtype=torch.float32).squeeze()
                _,v = self.net(s)
                v = v.squeeze()
                
                loss_critic+= mse(r_tens,v)
            # print('loss critic : ', float(loss_critic))
            loss_critic.backward()
            optimizer.step()


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (12, 12)
    ppo = SimplePPO(size, 'ppo_debug', walls=True, n_iter=10000, batch_size=64)
    bs = ppo.batch_size
    best_reward = -1
    best_length = 0

    ppo.net.load_state_dict(torch.load(ppo.name + '_state_dict.txt'))
    with open("ep_rewards_"+ppo.name+".txt","r+") as f:
            f.truncate(0)
    with open("ep_lengths_"+ppo.name+".txt","r+") as f:
            f.truncate(0)
    debut = time.time()
    
    for it in range(ppo.n_iter):
        
        args = bs*[ppo]
        map_results = list(map(SimplePPO.play_one_episode, args))
        ppo.one_training_step(map_results)
        mean_reward, mean_length = ppo.get_stats(map_results)
        if mean_reward > best_reward:
            print('\n', "********* new best reward ! *********** ", round(mean_reward, 3), '\n')
            best_reward = mean_reward
            torch.save(ppo.net.state_dict(), ppo.name + '_state_dict.txt')
        if mean_length > best_length:
            print('\n', "********* new best length ! *********** ", round(mean_length, 3), '\n')
            best_length = mean_length
            torch.save(ppo.net.state_dict(), ppo.name + '_state_dict.txt')

        with open("ep_rewards_"+ppo.name+".txt","a") as f:
            f.write(str(round(mean_reward, 3))+ '\n')
        with open("ep_lengths_"+ppo.name+".txt","a") as f:
            f.write(str(round(mean_length, 3))+ '\n')
        print('iteration : ', it, 'reward : ', round(mean_reward, 3),'length : ', round(mean_length, 3),'temps : ', round(time.time()-debut, 3), '\n')
