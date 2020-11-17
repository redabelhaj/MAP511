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


# plot("ep_lengths_ppo_h15_bs30.txt", label = 'ppo ')
# # plot("ep_lengths_a2c_h15_bs30.txt", label = 'a2c ')
# plot("ep_lengths_ppo_h15_bs30_diffrew.txt", label = 'ppo - differential rewards : bonus = .3')
# plot("ep_lengths_ppo_h15_bs30_diffrew_b1.txt", label = 'ppo - differential rewards : bonus = 1')
# plot("ep_lengths_ppo_h15_bs30_diffrew_b10.txt", label = 'ppo - differential rewards : bonus = 10')
# plot("ep_lengths_ppo_h15_bs30_diffrew_b3_bigrew.txt", label = 'ppo - differential rewards : bonus = .3 - big reward')
# plot("ep_lengths_ppo_h15_bs30_diffrew_b3_debug.txt", label = 'ppo - differential rewards : bonus = .3 - debug')
# plot("ep_lengths_ppo_h15_bs30_db3_g09.txt", label = 'ppo - differential rewards : gamma =.9')
# plot("ep_lengths_ppo_bigboard.txt", label = 'bigger board ?')
# plot("ep_lengths_ppo_bigboard_newarch.txt", label = 'bigger board , new arch ?')
# plot("ep_lengths_ppo_new_rewards.txt", label = 'ppo, new rewards ')
# plot("ep_lengths_ppo_new_rewards_newtest.txt", label = 'ppo, new rewards  - new test')
plot("ep_lengths_ppo_no_loop_h17_b30.txt", label = 'ppo, new rewards, h17')
plot('ep_lengths_ppo_no_loop_h17_b30_fixedseed.txt', label = 'fixed seed')

# ppo_bigboard_newarch
plt.xlabel("# batch")
plt.ylabel("length")
plt.legend()
plt.title("episode length - MaxHunger = 15 - Batch size = 30 ")
plt.savefig("plots/images/episode_lengths")
plt.clf()

# plot("ep_rewards_ppo_h15_bs30.txt", label = 'ppo ')
# plot("ep_rewards_a2c_h15_bs30.txt", label = 'a2c ')


# plot("ep_rewards_ppo_h15_bs30_diffrew.txt", label = 'ppo - differenrtial rewards : bonus = .3')
# plot("ep_rewards_ppo_h15_bs30_diffrew_b1.txt", label = 'ppo - differential rewards : bonus = 1')
# plot("ep_rewards_ppo_h15_bs30_diffrew_b10.txt", label = 'ppo - differential rewards : bonus = 10')
# plot("ep_rewards_ppo_h15_bs30_diffrew_b3_bigrew.txt", label = 'ppo - differential rewards : bonus = .3 - big reward')
# plot("ep_rewards_ppo_h15_bs30_diffrew_b3_debug.txt", label = 'ppo - differential rewards : bonus = .3 - debug')
# plot("ep_rewards_ppo_h15_bs30_diffrew_b3_debug_optim.txt", label = 'ppo - differential rewards : bonus = .3 - debug optim')
# plot("ep_rewards_ppo_h15_bs30_db3_g09.txt", label = 'ppo - differential rewards : gamma =.9')
# plot("ep_rewards_ppo_bigboard.txt", label = 'bigger board')
# plot("ep_rewards_ppo_bigboard_newarch.txt", label = 'bigger board , new arch ?')
# plot("ep_rewards_ppo_new_rewards.txt", label = 'new rewards')
# plot("ep_rewards_ppo_new_rewards_newtest.txt", label = 'ppo, new rewards  - new test')
# plot("ep_rewards_simple_ppo_debug.txt", label = 'debug simple ppo')
plot("ep_rewards_ppo_no_loop_h17_b30.txt", label = 'ppo, new rewards, h17')
plot("ep_rewards_ppo_no_loop_h17_b30_fixedseed.txt", label = 'fixed seed')

# ppo_h15_bs30_diffrew_b3_debug_optim
# plot("ep_rewards_a2c_h15_bs30.txt", label = 'a2c ')

plt.ylabel("rewards")
plt.xlabel("# batch")
plt.title("Reward per episode- MaxHunger = 15 - Batch size = 30")
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