from ppo import DT_PPO
import matplotlib.pyplot as plt
import gym

ENV = 'CartPole-v1'
env = gym.make(ENV)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

LEN_EP = int(2e4)

env.close()


agent = DT_PPO(state_dim=state_dim, action_dim=action_dim, hidden_size=32, clip=0.2, epoch=10, gamma=0.99, lr=3e-6, nhead = 1, nlayer = 1, normalize = False, max_norm = 1, max_timestep=5000)


def get_pm(model):
    return sum(p.numel() for p in model.parameters()) / 1e3

get_pm(agent.dt)

r, l, r_, r__ = agent.Learn(LEN_EP, ENV, notebook=False)

L = len(r)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Absolute reward
axs[0, 0].plot(range(L), r, label='Absolute reward', color='blue')
axs[0, 0].set_title('Absolute Reward')
axs[0, 0].set(xlabel='Number of Episodes', ylabel='Reward')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Loss evolution
axs[0, 1].plot(range(L), l, label='Loss Evolution', color='orange')
axs[0, 1].set_title('Loss Evolution')
axs[0, 1].set(xlabel='Number of Episodes', ylabel='Loss')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Average reward
axs[1, 0].plot(range(L), r_, label='Average Reward', color='green')
axs[1, 0].set_title('Average Reward')
axs[1, 0].set(xlabel='Number of Episodes', ylabel='Reward')
axs[1, 0].grid(True)
axs[1, 0].legend()

# Average reward (cut 5)
axs[1, 1].plot(range(L), r__, label='Average Reward Cut 5', color='red')
axs[1, 1].set_title('Average Reward Cut 5')
axs[1, 1].set(xlabel='Number of Episodes', ylabel='Reward')
axs[1, 1].grid(True)
axs[1, 1].legend()

# Common title
plt.suptitle('Agent Learning Performance', fontsize=16)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()
