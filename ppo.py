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
    """
    Proximal Policy Optimization with Decision Transformer
    stqt_dim : int : state dimension
    action_dim : int : action dimension
    hidden_size : int : hidden size
    lr : float : learning rate
    gamma : float : discount factor
    clip : float : clip value
    epoch : int : number of epoch
    max_timestep : int : max timestep
    nlayer : int : number of layer
    nhead : int : number of head
    max_norm : float : max norm
    normalize : bool : normalize
    """
    def __init__(self, state_dim,
                action_dim : int, 
                hidden_size : int, 
                lr : float = 3e-4, 
                gamma : float = 0.99, 
                clip : float = 0.2, 
                epoch : int = 10,  
                max_timestep : int = 2048, 
                nlayer : int = 3,
                nhead : int = 3, 
                max_norm : float = 0.2, 
                normalize : bool = True
        ):
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
        """
        Get the action distribution.
        state : torch.Tensor : state
        action : torch.Tensor : action
        """
        action_dist = self.dt.get_action(
            state, 
            action,
            None,
            rtg,
            timesteps=timestep
            )
        return action_dist
    
    def update(self):
        """
        Update the model.
        """
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
            entropy = -torch.mean(-action_preds)
            loss = actor_loss + 0.01 * entropy + 0.5 *  critic_loss         
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()
    
    
    def save(self, path : str):
        """
        Save the model.
        path : str : path to save the model
        """
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
        

    def load(path : str):
        """
        Load the model.
        path : str : path to load the
        """
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
    
    def Learn(self, timesteps : int, env : str, notebook : bool = False, reward_scale : float = 1e-4):
        """"
        Learn the model.
        timesteps : int : number of timesteps
        env : str : environment id
        notebook : bool : notebook
        reward_scale : float : reward scale
        """
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
            for _ in range(self.max_timestep):
                old_state = state.clone()
                action_dist = self.get_action(
                state=state,
                action=action,
                rtg=rtg,
                timestep=timestep
                )
                state, action, reward, rtg, timestep, done, great_action = env.step(action_dist, _)
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



