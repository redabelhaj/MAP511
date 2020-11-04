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
        probs = torch.softmax(logits, dim = -1)
        return probs, values


def get_action(state, net):
    tens = torch.tensor(state, dtype = torch.float32).permute(2,0,1)
    probs, _ = net(tens)
    probs = probs.squeeze()
    return np.random.choice(4, p = probs.detach().numpy())
    

def play_episode(env, net, n_episodes=1):

    transitions = []

    for _ in range(n_episodes):
        new_frame = env.reset()
        state = new_frame
        frame = new_frame
        action = get_action(state, net)
        done = False
        while not(done):
            new_frame, reward, done, _ = env.step(action)
            new_state = new_frame-frame
            transition = (state, action,reward)
            transitions.append(transition)
            frame = new_frame
            state = new_state
            action =get_action(state, net)
            # env.render()
    return transitions


def make_tensors(transitions, gamma):

    states, actions, gt = [],[],[]
    i=0
    n = len(transitions)
     
    for state, action, rew in transitions:
        states.append(torch.tensor(state, dtype = torch.float32).permute(2,0,1))
        actions.append(torch.tensor([action], dtype = torch.int64))
        # rewards.append(torch.tensor(rew, dtype = torch.float32))
        gammas = torch.tensor([gamma**j for j in range(n-i)])
        rewards = torch.tensor([r for _,_,r in transitions[i:]])
        g = torch.dot(gammas,rewards)
        gt.append(g)
        i+=1

    return torch.stack(states), torch.stack(actions), torch.stack(gt)



def train(env, n_episodes, gamma, batch_size = 32):
    size = env.SIZE
    net = ActorCriticNet(size)
    opt = torch.optim.RMSprop(net.parameters())
    ep_lengths, total_rewards = [], []
    ce = torch.nn.CrossEntropyLoss(reduction='mean')
    mse = torch.nn.MSELoss()

    ep_lengths = []
    total_rewards = []
    for i_ep in tqdm.tqdm(range(n_episodes)):
        
        opt.zero_grad()
        transitions = play_episode(env, net, n_episodes=batch_size)
        states, actions, gt = make_tensors(transitions, gamma)
        out,values = net(states)
        actions2 = actions.squeeze()
        
        loss =  ce(out, actions2)*(gt-values.squeeze()).mean()
        loss2 = mse(values.squeeze(), gt)

        loss.backward(retain_graph=True)
        loss2.backward()
        opt.step()

        length = len(transitions)/batch_size
        all_rewards = [t for _,_,t in transitions]
        this_reward = sum(all_rewards)/batch_size
        ep_lengths.append(length)
        total_rewards.append(this_reward)

        with open("ep_lengths_a2c.txt", 'a') as f:
            f.write(str(round(length, 3)) + '\n')
        with open("ep_rewards_a2c.txt", 'a') as f:
            f.write(str(round(this_reward, 3)) + '\n')
    return net, ep_lengths, total_rewards


if __name__ == "__main__":
    env = SingleSnek(size = (15,15), add_walls=True, obs_type="rgb")
    n_episodes=5
    gamma = .99
    with open("ep_rewards_a2c.txt","r+") as f:
            f.truncate(0)
    with open("ep_lengths_a2c.txt","r+") as f:
            f.truncate(0)
    net, ep_lengths, total_rewards = train(env, n_episodes, gamma)
    torch.save(net.state_dict(), "a2c_actor.txt")