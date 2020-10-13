x = np.arange(n_episodes)

plt.plot(x, total_rewards, label = 'rewards')
plt.legend()
plt.savefig('rewards')
plt.clf()

plt.plot(x, ep_lengths, label = 'episode lengths')
plt.legend()
plt.savefig('episode lengths')

