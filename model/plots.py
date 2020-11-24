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

plot("ep_lengths_ppo_1730_vanilla_2.txt", label = 'vanilla2')
plot("ep_lengths_ppo_1730_vanilla.txt", label = 'vanilla')
# plot("ep_lengths_ppo_1730_noloop.txt", label = "RS : closer +1; further -2")
# plot("ep_lengths_ppo_1730_noloop2.txt", label = "RS : closer +1; further -2 ; 2")
# plot("ep_lengths_ppo_1730_distRS10.txt", label = 'RS distance : bonus = 10')
plot("ep_lengths_ppo_rotate.txt", label = 'vanilla + rotation')


plt.xlabel("# batch")
plt.ylabel("length")
plt.legend()
plt.title("episode length - MaxHunger = 15 - Batch size = 30 ")
plt.savefig("plots/images/episode_lengths")
plt.clf()

# plot("ep_rewards_ppo_1730_vanilla_2.txt", label = 'vanilla2')
plot("ep_rewards_ppo_1730_vanilla.txt", label = 'vanilla')
# plot("ep_rewards_ppo_rotate.txt", label = 'vanilla + rotation')

# plot("ep_rewards_ppo_1730_noloop.txt", label = "RS : closer +1; further -2")
# plot("ep_rewards_ppo_1730_noloop2.txt", label = "RS : closer +1; further -2; 2")
# plot("ep_rewards_ppo_1730_distRS10.txt", label = 'RS distance : bonus = 10')
# plot("ep_rewards_ppo_1730_distRS5.txt", label = 'RS distance : bonus = 5')
# plot("ep_rewards_ppo_1730_distRS15.txt", label = 'RS distance : bonus = 15')
# plot("ep_rewards_ppo_1730_vanilla_entropy.txt", label = 'vanilla w/ entropy bonus 0.2')
# plot("ep_rewards_ppo_1730_vanilla_entropy_1.txt", label = 'vanilla w/ entropy bonus 1')
# plot("ep_rewards_ppo_1730_vanilla_entropy_10.txt", label = 'vanilla w/ entropy bonus 10')
# plot("ep_rewards_ppo_1730_vanilla_entropy_100.txt", label = 'vanilla w/ entropy bonus 100')

# plot("ep_rewards_ppo_rotate_noloops_hunger50.txt", label = 'rotation, hunger 50, RS : closer +1, further -4')

plt.ylabel("rewards")
plt.xlabel("# batch")
plt.title("Reward per episode- MaxHunger = 17 - Batch size = 30")
plt.legend()
plt.savefig("plots/images/episode_rewards")
plt.clf()


# plot("loss_critic_ppo_no_loop_h17_b30_test2.txt", label = 'critic loss')
# plt.ylabel("loss")
# plt.xlabel("# episode")
# plt.title("Loss of the critic")
# plt.legend()
# plt.savefig("plots/images/loss_critic")
# plt.clf()

plot("plot_entropy_ppo_1730_vanilla_entropy_10.txt", label = 'vanilla w/ entropy bonus 10')
plot("plot_entropy_ppo_1730_vanilla_entropy_1.txt", label = 'vanilla w/ entropy bonus 1')

plot("plot_entropy_ppo_1730_vanilla_entropy_100.txt", label = 'vanilla w/ entropy bonus 100')

plt.ylabel("entropy")
plt.xlabel("# episode")
plt.title("Entropy")
plt.legend()
plt.savefig("plots/images/entropy")
plt.clf()
