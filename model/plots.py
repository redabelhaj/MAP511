import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    with open('plots/text_files/'+path, 'r') as f:
        lines = f.read().splitlines()
        rew, batch = [], []
        for i, l in enumerate(lines):
            r = float(l)
            batch.append(1+i)
            rew.append(r)
    return batch, rew

def running_mean(batch, rew, N=1):
    m = len(batch)
    cumsum = np.cumsum(np.insert(rew, 0, 0))
    running_mean = (cumsum[N:] - cumsum[:-N]) / float(N)
    s = len(running_mean)
    batch2 = np.linspace(0, m, num = s)
    return batch2, running_mean

def plot(path, label, N=20):
    b,r = load_data(path)
    b, r = running_mean(b,r, N=N)
    plt.plot(b,r, label = label)


plot("ep_lengths_a2c_debug.txt", label = 'ppo - raw image NEW reward')

plt.xlabel("# episode")
plt.ylabel("length")
plt.legend()
plt.title("episode length")
plt.savefig("plots/images/episode_lengths")
plt.clf()

plot("ep_rewards_a2c_debug.txt", label = 'ppo - raw image')


plt.ylabel("rewards")
plt.xlabel("# episode")
plt.title("sum of rewards during the episode")
plt.legend()
plt.savefig("plots/images/episode_rewards")
plt.clf()


# plot("loss_critic_a2c.txt", label = 'critic loss', N=50)
# plt.ylabel("loss")
# plt.xlabel("# episode")
# plt.title("Loss of the critic")
# plt.legend()
# plt.savefig("loss_critic_a2c.png")
# plt.clf()