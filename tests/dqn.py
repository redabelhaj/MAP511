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


class DQN(torch.nn.Module):

    def __init__(self, size):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2)
        self.conv2 = torch.nn.Conv2d(3, 2, 2, stride=2)
        out_size = (size[0]- 1)
        out_size = int((out_size -2 )/2 +1)

        self.layer = torch.nn.Linear(2*out_size**2, 4)

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
       
        return self.layer(out)

def get_action_eps_greedy(state, q_net,epsilon):

    tens = torch.tensor(state, dtype = torch.float32).permute(2,0,1)
    action_values = q_net(tens)
    r = np.random.rand()
    if r <epsilon:
        return np.random.choice(4)
    else:
        return torch.argmax(action_values)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition (tuple)."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return rand.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train_dqn_offpolicy(env, n_episodes, gamma, epsilon):
    size = env.SIZE
    q_net = DQN(size)
    opt = torch.optim.Adam(q_net.parameters(), lr = 1e-3)

    ep_lengths, total_rewards = [], []

    for _ in tqdm.tqdm(range(n_episodes)):
        # play ep while storing state & actions
        done = False
        state = env.reset()
        action = get_action_eps_greedy(state, q_net, epsilon)

        sar = []
        while not(done):
            new_state, reward, done, _ = env.step(action)
            sar.append((state, action, reward))
            state = new_state
            action = get_action_eps_greedy(state, q_net, epsilon)
        
        # 1 step backprop of Q 
        opt.zero_grad()
        loss = 0
        n_steps = len(sar)

        for i in range(n_steps):
            l = n_steps-i
            gamma_vect = torch.tensor([gamma**t for t in range(l)])
            r_vect = torch.tensor([r for _,_,r in sar[i:]], dtype = torch.float32)
            sample_val = torch.dot(gamma_vect,r_vect)
            s,a,_ = sar[i]
            predicted_val = q_net(s)[a]
            loss+= (sample_val-predicted_val)**2
        loss.backward()
        opt.step()

        all_rewards = [t for _,_,t in sar]

        total_rewards.append(sum(all_rewards))
        ep_lengths.append(len(sar))

        with open("ep_lengths", 'a') as f:
            f.write(str(round(len(sar), 3)) + '\n')
        with open("ep_rewards", 'a') as f:
            f.write(str(round(sum(all_rewards), 3)) + '\n')


    return q_net, ep_lengths, total_rewards

def train_dqn(env, n_episodes, batch_size, gamma, capacity = 10000, target_update=20, eps_rate= 1e-2, eps_deb = 0.9, eps_fin = 0.05):
    size = env.SIZE
    policy_net = DQN(size)
    target_net = DQN(size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opt = torch.optim.RMSprop(policy_net.parameters())
    replay_buffer = ReplayMemory(capacity)
    ep_lengths, total_rewards = [], []
    for i_ep in tqdm.tqdm(range(n_episodes)):
        # play ep while storing transition in replay buffer
        done = False
        new_frame = env.reset()
        state = new_frame
        frame = new_frame
        alpha_t =  1-np.exp(-eps_rate*i_ep)
        eps_i = alpha_t*eps_deb + (1-alpha_t)*eps_fin
        action = get_action_eps_greedy(state, policy_net, eps_i)
        length,this_reward = 0,0
        while not(done):
            length+=1
            new_frame, reward, done, _ = env.step(action)
            this_reward+=reward
            new_state = new_frame-frame
            transition = (state, action,new_state,reward)
            replay_buffer.push(transition)
            frame = new_frame
            state = new_state
            action = get_action_eps_greedy(state, policy_net, eps_i)
            env.render(mode = 'human')
        
        if len(replay_buffer)>=batch_size:
            # sample a batch from the buffer and optimize
            opt.zero_grad()
            loss = 0
            transitions = replay_buffer.sample(batch_size)
            states, actions, new_states, rewards = make_tensors(transitions) 
            state_action_values = policy_net(states).gather(1, actions)
            new_states_values = target_net(new_states).max(1)[0].detach()           
            expected_state_action_values = (new_states_values * gamma) + rewards
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            loss.backward()
            opt.step()
        
        ep_lengths.append(length)
        total_rewards.append(this_reward)
        if i_ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        with open("ep_lengths", 'a') as f:
            f.write(str(round(length, 3)) + '\n')
        with open("ep_rewards", 'a') as f:
            f.write(str(round(this_reward, 3)) + '\n')
    return policy_net, ep_lengths, total_rewards


def make_tensors(transitions):
    states, actions, new_states, rewards = [],[],[],[]
    for state, action, new_s, rew in transitions:
        states.append(torch.tensor(state, dtype = torch.float32).permute(2,0,1))
        actions.append(torch.tensor([action], dtype = torch.int64))
        new_states.append(torch.tensor(new_s, dtype = torch.float32).permute(2,0,1))
        rewards.append(torch.tensor(rew, dtype = torch.float32))

    return torch.stack(states), torch.stack(actions), torch.stack(new_states), torch.stack(rewards)


if __name__ == "__main__":
    env = SingleSnek(size = (15,15), add_walls=True, obs_type="rgb")
    n_episodes=500
    gamma = .99
    epsilon = .4
    with open("ep_rewards","r+") as f:
            f.truncate(0)
    with open("ep_lengths","r+") as f:
            f.truncate(0)

    q_net, ep_lengths, total_rewards = train_dqn(env, n_episodes, 32, gamma, capacity = 10000,target_update=100)
    torch.save(q_net.state_dict(), "network")
