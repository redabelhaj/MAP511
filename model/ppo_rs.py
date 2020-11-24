import gym
import sys
sys.path.append("/Users/redabelhaj/Desktop/MAP511")
import sneks
from sneks.envs.snek import SingleSnek
import numpy as np
import random as rand
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

class PPO_RS:
    """
    - Class for the PPO algorithm that enables the possibility to use reward shaping based on the distance 
    from the agent to the fruit. 
    - The state corresponds to the raw image
    """
    def __init__(self, size,name, hunger = 120, walls = True,n_iter = 500, batch_size = 32,dist_bonus = .1,gamma = .99, n_epochs=5, eps=.2, target_kl=1e-2, seed=-1, use_entropy = False,beta = 1e-2):
        self.net = ActorCriticNet(size)
        self.name = name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.env = SingleSnek(size = size,dynamic_step_limit=hunger,add_walls=walls, obs_type="rgb",seed = seed)
        self.n_epochs = n_epochs
        self.eps = eps
        self.gamma = gamma
        self.target_kl = target_kl
        self.dist_bonus = dist_bonus # used in the case of reward shaping with differential distance

        # parameters for entropy bonus
        self.beta = beta
        self.use_entropy=use_entropy
    
    def get_action_prob(self, state):
        """
        given the state return the action taken by the actor and the probability of this action
        """
        tens = torch.tensor(state, dtype = torch.float32).permute(2,0,1)
        logits, _ = self.net(tens)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze().detach()
        ## parfois il y a eu des bugs (probabilités = nan) donc j'ai ajouté ça just in case
        try:
            act = np.random.choice(4, p = probs.numpy())
        except:
            print(tens)
            print(logits)
            print(probs)
            torch.save(self.net.state_dict(), 'bad training - needs to be debugged')
            exit()
        return act, probs[act]

    def play_one_episode(self):
        """
        play one episode in the environment and return (s,a,p,g,tr) transitions where : 
        - s is the state 
        - a is the action taken
        - p is the probability of this action
        - g is the discounted return from state s 
        - tr is the true reward (useful for stats plots when using reward shaping - not used for training)

        """
        transitions = []
        new_obs = self.env.reset()
        obs = new_obs
        action, prob = self.get_action_prob(obs)
        done = False
        sts, ats, pts, rts = [], [], [], []
        true_rewards = []
        old_dist = -1
        while not(done):
            new_obs, reward, done, _ = self.env.step(action)
            s_t  = torch.tensor(obs, dtype = torch.float32).permute(2,0,1)
            a = 4*[0]
            a[action] =1
            a_t = torch.tensor(a, dtype = torch.float32)
            p_t = torch.tensor([prob], dtype = torch.float32)
            
            true_rew, dist = reward
            if old_dist==-1: diff_dist=0
            else: diff_dist = dist - old_dist
            old_dist = dist
            
            if diff_dist<0:
                close_rew = 1
            elif diff_dist==0:
                close_rew = 0
            else:
                close_rew = -2
            ## commenter/décommenter selon reward shaping ou pas / quel type de reward shaping 
             
            newrew = true_rew + close_rew ### reward shaping avec un bonus de +1 si on s'approche, -2 si on s'éloigne
            # newrew = true_rew ## pas de reward shaping
            # newrew = true_rew - self.dist_bonus*diff_dist # reward shaping basé sur un bonus basé sur la différence de distance 
            
            sts.append(s_t)
            ats.append(a_t)
            pts.append(p_t)
            rts.append(newrew)
            true_rewards.append(true_rew)

            obs = new_obs
            action, prob = self.get_action_prob(obs)
        len_ep = len(rts)
        for i in range(len_ep):
            gammas = torch.tensor([self.gamma**j for j in range(len_ep-i)])
            rewards = torch.tensor(rts[i:], dtype = torch.float32)
            g = torch.dot(gammas,rewards)
            s_t, a_t, p_t = sts[i], ats[i], pts[i]
            tr_t = true_rewards[i]
            transitions.append((s_t, a_t, p_t, g, tr_t))
        return transitions
    
    def get_dataset(self,map_results):
        """
        from map_results (list of batch_size lists of transitions given by play_episode)  :
        returns a dataset of transitions (s,a,p,g) to be used for training 
        s: state 
        a: action
        p: probability of this action
        g: discounted return from state s
        """
        full_list = []
        list_rewards = []
        for transitions in map_results:
            full_list += transitions
            for _,_,_,g, _ in transitions:
                list_rewards.append(g)
        gt_tens = torch.tensor(list_rewards, dtype = torch.float32)
        mean, std = torch.mean(gt_tens),torch.std(gt_tens)
        gt_tens = (gt_tens-mean)/(std + 1e-8)

        final_list  = [(s,a,p,gt_tens[i]) for i,(s,a,p,_,_) in enumerate(full_list) ]
        return final_list

    def get_actor_loss(self, states, actions, probs, r):
        """
        from a tensor of states, a tensor of actions, a tensor of probabilities, a tensor of discounted rewards r 
        returns the corresponding PPO actor loss  (inspired by PPO-CLIP implementation of OpenAI)
        """
        logits, vals = self.net(states)
        advs = r.unsqueeze(dim =-1)-vals.detach()
        new_probs = torch.unsqueeze(torch.diag(torch.matmul(actions, torch.softmax(logits, dim=-1).T)), dim=-1)
        ratio = new_probs/(probs.detach() + 1e-8)
        clip_advs = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advs
        return -torch.min(ratio*advs, clip_advs).mean()

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
            gts = [ tr_t for _,_,_,_,tr_t in transitions]
            r = sum(gts)
            reward_ep.append(r)
        return sum(reward_ep)/n_batch, sum(len_ep)/n_batch

    def one_training_step(self, map_results):
        """
        from map_results (list of batch_size lists of transitions given by play_episode)  :
        does one training step of PPO
        """

        # get training dataset from map_results
        dataset = self.get_dataset(map_results)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
        kl_data = DataLoader(dataset, batch_size= len(dataset), shuffle=True, num_workers=1)
        optimizer = torch.optim.Adam(self.net.parameters())
        n_epochs = self.n_epochs

        for i in range(n_epochs):

            # compute actor loss and optimize
            running_loss = 0
            for s,c,p,r in dataloader:
                optimizer.zero_grad()
                loss = self.get_actor_loss(s,c,p,r)
                running_loss+=loss
                loss.backward()
                optimizer.step()

            # check if kl between old and new distribution is too high
            kl=0
            for s,a,p,r in kl_data:
                logits, _ = self.net(s)
                new_probs = torch.unsqueeze(torch.diag(torch.matmul(a, torch.softmax(logits, dim=-1).T)), dim=-1)
                kl += (torch.log(p) - torch.log(new_probs)).mean().item()
            if kl > 1.5*self.target_kl:
                ## kl is too high : stop training 
                break

            

        # compute the entropy for stats / entropy bonus if applicable
        entropy = 0
        optimizer.zero_grad()
        for s,_,_,_ in  kl_data:
            logits, _ = self.net(s)
            probs = torch.softmax(logits, dim=-1)
            entropy  += (probs*torch.log(probs)).mean()
        with open("plots/text_files/plot_entropy_"+str(self.name)+'.txt', "a") as f:
            f.write(str(round(float(entropy), 3)) + '\n')
        entropy = self.beta*entropy
        if self.use_entropy:
            entropy.backward()
            optimizer.step()
    
        ## compute the critic loss and optimize
        optimizer.zero_grad()
        loss_critic=0
        mse = torch.nn.MSELoss()
        for s,_,_,r in kl_data:
            _,v = self.net(s)
            v = v.squeeze()
            loss_critic+= mse(r,v)
        
        # save critic loss in file
        with open("plots/text_files/loss_critic_"+str(self.name)+'.txt', "a") as f:
            f.write(str(round(float(loss_critic), 3)) + '\n')
        loss_critic.backward()
        optimizer.step()

    def truncate_all_files(self):
        name = self.name
        with open("plots/text_files/ep_rewards_"+name+".txt","r+") as f:
            f.truncate(0)
        with open("plots/text_files/ep_lengths_"+name+".txt","r+") as f:
                f.truncate(0)
        with open("plots/text_files/plot_entropy_"+str(name)+'.txt', "r+") as f:
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
    ppo = PPO_RS(size, 'debug', hunger=17, n_iter=3, batch_size=3,seed = 10, beta=1, use_entropy=True)
    bs = ppo.batch_size
    best_reward = -1
    best_length = 0
    ### uncomment to resume training from a saved model 
    # ppo.net.load_state_dict(torch.load('saved_models/' +ppo.name + '_state_dict.txt'))

    
    ppo.truncate_all_files() # uncomment to delete the files corresponding to the name
    debut = time.time()
    
    for it in range(ppo.n_iter):
        args = bs*[ppo]
        map_results = list(map(PPO_RS.play_one_episode, args))
        ppo.one_training_step(map_results)
        mean_reward, mean_length = ppo.get_stats(map_results)
        if mean_reward > best_reward:
            print('\n', "********* new best reward ! *********** ", round(mean_reward, 3), '\n')
            best_reward = mean_reward
            torch.save(ppo.net.state_dict(), 'saved_models/' +ppo.name + '_state_dict.txt')
        if mean_length > best_length:
            print('\n', "********* new best length ! *********** ", round(mean_length, 3), '\n')
            best_length = mean_length
            torch.save(ppo.net.state_dict(), 'saved_models/' +ppo.name + '_state_dict.txt')
        ppo.write_rew_len(mean_reward,mean_length)
        print('iteration : ', it, 'reward : ', round(mean_reward, 3),'length : ', round(mean_length, 3),'temps : ', round(time.time()-debut, 3), '\n')
