import sys
sys.path.append("/Users/redabelhaj/Desktop/Sneks-master")
import gym
import sneks

from sneks.envs.snek import SingleSnek
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm


class DQN(torch.nn.Module):

    def __init__(self, size):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3)
        out_size = (size[0]- 2)*(size[1]- 2)

        self.layer = torch.nn.Linear(out_size, 4)

    def forward(self, obs):
        tens = torch.tensor(obs, dtype = torch.float32)
        tens2 = tens.unsqueeze(0).unsqueeze(0)
        out = self.conv1(tens2)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return self.layer(out)

def get_action_eps_greedy(state, q_net,epsilon):
    action_values = q_net(state)
    r = np.random.rand()
    if r <epsilon:
        return np.random.choice(4)
    else:
        return torch.argmax(action_values)


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



if __name__ == "__main__":
    env = SingleSnek(size = (28,28), add_walls=True)
    n_episodes=5000
    gamma = .9
    epsilon = .1
    with open("ep_rewards","r+") as f:
            f.truncate(0)
    with open("ep_lengths","r+") as f:
            f.truncate(0)

    q_net, ep_lengths, total_rewards = train_dqn_offpolicy(env, n_episodes, gamma, epsilon)
    torch.save(q_net.state_dict(), "network")
