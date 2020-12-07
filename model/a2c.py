import gym
import sys
import sneks
from sneks.envs.snek import SingleSnek
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
import time

class ActorCriticNet(torch.nn.Module):
    """
    The Actor-Critic neural network class
    It is a Convolutional Neural Network with two heads : 
    - the actor returns the scores for the 4 actions (up, right, down, left)
    - The critic returs the value of the state
    Notice that the actor does not return a probability distribution : softmax should be used
    """

    def __init__(self, size):
        super(ActorCriticNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 2, stride = 2)
        self.conv2 = torch.nn.Conv2d(6, 9, 2)
        out_size = 1+ int((size[0] -2 )/2)
        out_size = out_size-1 
        self.actor = torch.nn.Linear(9*out_size**2, 4)
        self.critic = torch.nn.Linear(9*out_size**2, 1)

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
    """
    - Class for the A2C algorithm that enables the possibility to use reward shaping based on the distance 
    from the agent to the fruit. 
    - The state corresponds to the raw image
    """
    def __init__(self, size, name, hunger = 120, hidden_size = 30, walls=True, n_iter = 500, batch_size=32,rs = 'none',dist_bonus = .1, gamma=.99, seed=-1):
        self.net = ActorCriticNet(size)
        self.name = name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.env = SingleSnek(size = size, dynamic_step_limit=hunger, add_walls=walls, obs_type="rgb",seed=seed)

        self.rs = rs ## reward shaping :  'none', 'close bonus' or 'diff dist bonus'
        self.dist_bonus = dist_bonus # used if rs = 'diff dist bonus'

        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def get_action(self, state):
        """
        given the state return the action taken by the actor 
        """
        tens = torch.tensor(state, dtype = torch.float32).permute(2,0,1)
        logits, _ = self.net(tens)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze()
        return np.random.choice(4, p = probs.detach().numpy())

    def play_one_episode(self):
        """
        play one episode in the environment and return (s,a,g,tr) transitions where : 
        - s is the state 
        - a is the action taken
        - g is the discounted return from state s 
        - tr is the true reward (useful for stats plots when using reward shaping - not used for training)
        """
        transitions = []
        new_state = self.env.reset()
        state = new_state
        action = self.get_action(state)
        done = False
        sts, acts, rews = [],[], [] 
        true_rewards = []
        old_dist = -1
        while not(done):
            new_state, reward, done, _ = self.env.step(action)
            true_rew, dist = reward

            if old_dist==-1: diff_dist=0
            else: diff_dist = dist - old_dist
            old_dist = dist

            if diff_dist<0:
                close_rew = 1
            else:
                close_rew = -2

            if self.rs == 'none': newrew = true_rew
            elif self.rs == 'close bonus' : newrew =  true_rew + close_rew
            elif self.rs == 'diff dist bonus': newrew = true_rew - self.dist_bonus*diff_dist
            else: raise Exception('unrecognized reward shaping mode')




            a_t = torch.tensor([action], dtype = torch.int64)
            s_t = torch.tensor(state, dtype = torch.float32).permute(2,0,1)
            sts.append(s_t)
            acts.append(a_t)
            rews.append(newrew)
            true_rewards.append(true_rew)
            state = new_state
            action =self.get_action(state)
        
        gamma = self.gamma
        len_ep = len(rews)
        for i in range(len_ep):
            gammas = torch.tensor([gamma**j for j in range(len_ep-i)])
            rewards = torch.tensor(rews[i:], dtype = torch.float32)
            g = torch.dot(gammas,rewards)
            s_t, a_t = sts[i], acts[i]
            tr = true_rewards[i]
            transitions.append((s_t, a_t, g, tr))

        return transitions
    
    def get_dataset(self, map_results):
        """
        from map_results (list of batch_size lists of transitions given by play_episode)  :
        returns a dataset of transitions (s,a,g) to be used for training 
        s: state 
        a: action
        g: discounted return from state s
        """
        full_list = []
        list_rewards = []
        for transitions in map_results:
            full_list += transitions
            for _,_,g,_ in transitions:
                list_rewards.append(g)
        gt_tens = torch.tensor(list_rewards, dtype = torch.float32)
        mean, std = torch.mean(gt_tens),torch.std(gt_tens)
        gt_tens = (gt_tens-mean)/(std + 1e-8)

        final_list  = [(s,a,gt_tens[i]) for i,(s,a,_,_) in enumerate(full_list) ]
        return final_list   

    def get_stats(self, map_results):
        """
        from map_results (list of batch_size lists of transitions given by play_episode)  :
        returns the average true reward per episode and the average episode length
        used for monitoring the behaviour of the agent
        """
        n_batch = len(map_results)
        reward_ep, len_ep  = [], []
        for i in range(n_batch):
            transitions = map_results[i]
            len_ep.append(len(transitions))
            gts = [ tr_t for _,_,_,tr_t in transitions]
            r = sum(gts)
            reward_ep.append(r)
    
        return sum(reward_ep)/n_batch, sum(len_ep)/n_batch

    def one_training_step(self, map_results):
        """
        from map_results (list of batch_size lists of transitions given by play_episode)  :
        does one training step of A2C
        """

        # get training dataset from map_results
        dataset = self.get_dataset(map_results)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=1)
        ce = torch.nn.CrossEntropyLoss(reduction='none')
        mse = torch.nn.MSELoss()
        tot_loss_critic = 0
        
        ## compute loss + optimize 
        self.optimizer.zero_grad()
        total_loss=0
        for s,a,g in dataloader:
            
            out,values = self.net(s)
            values = values.squeeze()
            values2 = values.detach()
            loss_actor = (ce(out, a.squeeze())*(g-values2)).mean()   
            loss_critic = mse(values, g)
            tot_loss_critic+= loss_critic.item()
            total_loss += .5*loss_actor + .5*loss_critic
        total_loss.backward()
        self.optimizer.step()

        with open("plots/text_files/loss_critic_"+self.name+".txt","a") as f:
                f.write(str(round(tot_loss_critic, 3))+ '\n')
        
    def truncate_all_files(self):
        name = self.name
        with open("plots/text_files/ep_rewards_"+name+".txt","r+") as f:
            f.truncate(0)
        with open("plots/text_files/ep_lengths_"+name+".txt","r+") as f:
                f.truncate(0)
        with open("plots/text_files/loss_critic_"+str(name)+'.txt', "r+") as f:
                f.truncate(0)

    def write_rew_len(self, rew, length):
        """
        saves the average reward per episode/ average length per episode
        """
        name = self.name
        with open("plots/text_files/ep_rewards_"+name+".txt","a") as f:
            f.write(str(round(rew, 3))+ '\n')
        with open("plots/text_files/ep_lengths_"+name+".txt","a") as f:
            f.write(str(round(length, 3))+ '\n')


if __name__ == "__main__":
    size = (12, 12)
    a2c = A2C(size, 'debug',hunger = 17, walls=True, n_iter=3, batch_size=3, gamma=.99, seed = 10)
    bs = a2c.batch_size
    best_reward = -1
    best_length = 0

    ### uncomment to resume training from a saved model 
    # a2c.net.load_state_dict(torch.load('saved_models/' +ppo.name + '_state_dict.txt'))

    a2c.truncate_all_files() # uncomment to delete the files corresponding to the name
    debut = time.time()

    for it in range(a2c.n_iter):
        args = bs*[a2c]
        map_results = list(map(A2C.play_one_episode, args))
        a2c.one_training_step(map_results)
        mean_reward, mean_length = a2c.get_stats(map_results)
        if mean_reward > best_reward:
            print('\n', "********* new best reward ! *********** ", round(mean_reward, 3), '\n')
            best_reward = mean_reward
            torch.save(a2c.net.state_dict(), 'saved_models/' +a2c.name + '_state_dict.txt')
        if mean_length > best_length:
            print('\n', "********* new best length ! *********** ", round(mean_length, 3), '\n')
            best_length = mean_length
            torch.save(a2c.net.state_dict(), 'saved_models/' +a2c.name + '_state_dict.txt')
        a2c.write_rew_len(mean_reward,mean_length)
        print('iteration : ', it, 'reward : ', round(mean_reward, 3),'length : ', round(mean_length, 3),'temps : ', round(time.time()-debut, 3), '\n')