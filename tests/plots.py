import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        rew, batch = [], []
        for i, l in enumerate(lines):
            r = float(l)
            batch.append(1+i)
            rew.append(r)
    return batch, rew

def running_mean(batch, rew, N=4):
    m = len(batch)
    cumsum = np.cumsum(np.insert(rew, 0, 0))
    running_mean = (cumsum[N:] - cumsum[:-N]) / float(N)
    s = len(running_mean)
    batch2 = np.linspace(0, m, num = s)
    return batch2, running_mean

def plot(path, label, N=50):
    b,r = load_data(path)
    b, r = running_mean(b,r, N=N)
    plt.plot(b,r, label = label)



plot("ep_lengths", label = 'episode lengths')
plt.xlabel("# episode")
plt.ylabel("length")
plt.legend()
plt.title("episode length")
plt.savefig("episode_lengths")
plt.clf()

plot("ep_rewards", label = 'episode rewards')
plt.ylabel("rewards")
plt.xlabel("# episode")
plt.title("sum of rewards during the episode")
plt.legend()
plt.savefig("episode_rewards")
plt.clf()