from decision_transformer import *
import json
#torch.autograd.set_detect_anomaly(True)
#torch.manual_seed(2023)

def check_nan(tensor):
    if torch.isnan(tensor).any():
        raise ValueError(f'NaN values encountered')

def schedule(s : torch.optim.lr_scheduler.StepLR, step, max_step):
    if (step / max_step) < 0.8:
        return s.step()

class DT_PPO:
    def __init__(self, state_dim, action_dim, hidden_size, lr = 3e-4, gamma = 0.99, clip = 0.2, epoch = 10,  max_timestep = 2048, nlayer = 3, nhead = 3, max_norm = 0.2, normalize = True):
        self.dt = DecisionTransformer(state_dim, action_dim, hidden_size, nlayer=nlayer, nhead=nhead)
        self.optimizer = torch.optim.Adam(self.dt.parameters(), lr=lr)
        self.max_norm = max_norm
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.normalize = normalize
        self.epoch = epoch
        self.eps_clip = clip
        self.max_timestep = max_timestep
        self.rollout_buffer = RolloutBuffer(buffer_size=max_timestep, state_dim=state_dim, action_dim=action_dim)
    
    def get_action(self, state, action,rtg, timestep):
        action_dist = self.dt.get_action(
            state, 
            action,
            None,
            rtg,
            timesteps=timestep
            )
        return action_dist
    
    def update(self):
        for _ in range(self.epoch):
            batch = self.rollout_buffer.get_batch()
            state = batch['states']
            old_action_prob = batch['actions']
            reward = batch['rewards']
            next_state = batch['next_states']
            done = batch['dones']
            rtg = batch['rtg']
            timestep = batch['timestep']
            action = batch['great_action']
            rtg = (rtg - rtg.mean()) / (rtg.std() + 1e-7) if self.normalize else rtg

            _, action_preds, return_preds, value = self.dt.forward(
                    state,
                    old_action_prob,
                    None,
                    rtg,
                    timestep
                )
            s_, _, _1, next_value = self.dt.forward(
                    next_state,
                    action_preds,
                    None,
                    rtg,
                    timestep
                )
            advantages = rtg + self.gamma * (1-done) * next_value - value
            ratio = (action_preds - old_action_prob).mean()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1+ self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(value, rtg + self.gamma * (1-done) * next_value.detach())
            entropy = -torch.mean(-action_preds)#-torch.sum(action_preds * (action_preds + 1e-8), dim=1).mean()
            loss = actor_loss + 0.01 * entropy + 0.5 *  critic_loss#actor_loss + 0.5 * critic_loss - 0.01 * entropy            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #torch.nn.utils.clip_grad_norm_(self.dt.parameters(), max_norm=self.max_norm)

        return loss.item()
    
    
    def save(self, path):
        torch.save(self.dt, f'{path}/model.pt')
        with open(f'{path}/param.json', 'w+') as f:
            d = {
                'lr' : self.lr,
                'gamma' : self.gamma,
                'state_dim' : self.state_dim,
                'action_dim' : self.action_dim,
                'eps_clip' : self.eps_clip
            }
            json.dump(d, f)
        

    def load(path):
        dt = torch.load(f'{path}/model.pt')
        with open(f'{path}/param.json', 'r+') as f:
            param = json.load(f)
        lr = param['lr']
        gamma = param['gamma']
        state_dim = param['state_dim']
        action_dim = param['action_dim']
        eps_clip = param['eps_clip']
        ppo = DT_PPO(state_dim, action_dim, 12, lr, gamma, eps_clip)
        ppo.dt = dt
        return ppo
    
    def Learn(self, timesteps, env, notebook = False, reward_scale = 1e-4):
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=timesteps//10, gamma=0.1)
        env = Env(env, reward_scale=reward_scale, reward_method=BASIC_METHOD)
        i = 0
        r, l, r_, r__ = [], [], [], []
        losses = []
        f = tqdm.tqdm(total=timesteps) if not notebook else tqdm_notebook(total=timesteps)
        while i <= timesteps:
            self.rollout_buffer.reset()
            state, action, rtg, timestep = env.reset()
            rewards = []
            state_std, state_mean = state.std(), state.mean()
            #print(self.optimizer.param_groups)
            for _ in range(self.max_timestep):
                old_state = state.clone()
                action_dist = self.get_action(
                state=state,
                action=action,
                rtg=rtg,
                timestep=timestep
                )
                state, action, reward, rtg, timestep, done, great_action = env.step_(action_dist, _)
                rewards.append(reward.squeeze(2)[0][-1].item())
                
                terminal_value = self.dt.get_value(
                states=state,
                actions=action,
                returns_to_go=rtg,
                timesteps=timestep
                )
                reward += self.gamma * terminal_value
                self.rollout_buffer.add_experience(old_state, action, reward, state, done, rtg, timestep, great_action)
                loss = self.update()
                #schedule(self.schedule, i, timesteps)
                losses.append(loss)

                if done:
                    r.append(np.sum(rewards))
                    r_.append(np.mean(r))
                    r__.append(np.mean(r[-5:]))
                    l.append(np.mean(losses))
                
                i += 1
                if i % (timesteps // 10) == 0: 
                    print(f'timestep : {i}, reward_mean_sum : {np.mean(r[-2:])}, loss : {loss}')
                f.update(1)

                
            

        return r, l, r_, r__

"""
import gym

ENV = 'CartPole-v1'
env = gym.make(ENV)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

LEN_EP = int(3e4)

env.close()


agent = DT_PPO(state_dim=state_dim, action_dim=action_dim, hidden_size=3, clip=0.2, epoch=10, gamma=0.99, buffer_size=500_000, lr=3e-3)
r, l, r_, r__ = agent.Learn(LEN_EP, ENV, notebook=False,reward_scale=1e-3)

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

#agent.save('bin') 

"""