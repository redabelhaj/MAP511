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


class PPO:
    def __init__(self, size, name,walls = True,n_iter = 500, batch_size = 32,gamma = .99, n_epochs=5, eps=.2, target_kl=1e-2):
        self.net = ActorCriticNet(size)
        self.name = name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.env = SingleSnek(size = size, add_walls=walls, obs_type="rgb")
        self.n_epochs = n_epochs
        self.eps = eps
        self.gamma = gamma
        self.target_kl = target_kl
    
    def get_action_prob_val(self, state):
        tens = torch.tensor(state, dtype = torch.float32).permute(2,0,1)
        logits, value = self.net(tens)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze()
        act = np.random.choice(4, p = probs.detach().numpy())
        return act, probs[act], value

    def play_one_episode(self):
        transitions = []
        new_frame = self.env.reset()
        state = new_frame
        frame = new_frame
        action, prob, val = self.get_action_prob_val(state)
        done = False
        sts, ats, pts, rts, vts = [], [], [], [], []
        while not(done):
            new_frame, reward, done, _ = self.env.step(action)
            new_state = new_frame-frame
            
            s_t  = torch.tensor(state, dtype = torch.float32).permute(2,0,1)
            a = 4*[0]
            a[action] =1
            a_t = torch.tensor(a, dtype = torch.float32)
            p_t = torch.tensor([prob], dtype = torch.float32).detach()
            v_t = val.detach()

            sts.append(s_t)
            ats.append(a_t)
            pts.append(p_t)
            rts.append(reward)
            vts.append(v_t)

            frame = new_frame
            state = new_state
            action, prob, val = self.get_action_prob_val(state)
        gamma = self.gamma
        len_ep = len(rts)
        for i in range(len_ep):
            gammas = torch.tensor([gamma**j for j in range(len_ep-i)])
            rewards = torch.tensor(rts[i:], dtype = torch.float32)
            g = torch.dot(gammas,rewards)
            s_t, a_t, p_t, v_t = sts[i], ats[i], pts[i], vts[i]
            transitions.append((s_t, a_t, p_t, g, v_t))
        return transitions
    
    def get_dataset(self,map_results):
        ###  : LES REWARDS DONT NORMALISES SUR UN BATCH
        full_list = []
        list_rewards = []
        for transitions in map_results:
            full_list += transitions
            for _,_,_,g, _ in transitions:
                list_rewards.append(g)
        gt_tens = torch.tensor(list_rewards, dtype = torch.float32)
        mean, std = torch.mean(gt_tens),torch.std(gt_tens)
        gt_tens = (gt_tens-mean)/(std + 1e-8)

        final_list  = [(s,a,p,gt_tens[i],v) for i,(s,a,p,_,v) in enumerate(full_list) ]
        return final_list

    def get_actor_loss(self, states, actions, probs, advs):
        logits, _ = self.net(states)
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
            gts = [ g.item() for _,_,_,g,_ in transitions]
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
            for s,c,p,r,v in dataloader:
                optimizer.zero_grad()
                a = r-v.detach()
                loss = self.get_actor_loss(s,c,p,a)
                running_loss+=loss
                loss.backward()
                optimizer.step()
            
            kl=0
            for s,a,p,r,v in  kl_data:
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
            for s,_,_,r,_ in dataset:
                r_tens = torch.tensor([r], dtype=torch.float32).squeeze()
                _,v = self.net(s)
                v = v.squeeze()
                
                loss_critic+= mse(r_tens,v)
            # print('loss critic : ', float(loss_critic))
            loss_critic.backward()
            optimizer.step()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.manual_seed(0)
    size = (20, 20)
    ppo = PPO(size, 'ppo',walls=True, n_iter=500, batch_size=16)
    bs = ppo.batch_size
    best_reward = -0.75
    best_length = 57

    # ppo.net.load_state_dict(torch.load(ppo.name + '_state_dict.txt'))
    # with open("ep_rewards_"+ppo.name+".txt","r+") as f:
    #         f.truncate(0)
    # with open("ep_lengths_"+ppo.name+".txt","r+") as f:
    #         f.truncate(0)
    debut = time.time()
    with mp.Pool() as pool:
        for it in range(ppo.n_iter):
            
            args = bs*[ppo]
            map_results = pool.map_async(PPO.play_one_episode, args).get()
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
