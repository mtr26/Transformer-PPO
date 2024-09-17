import gym
import torch
import random
import numpy as np


SEED = 0#2026


MAX_EPISODE_LEN = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

action_selector = {
    'argmax' : lambda action : torch.argmax(action, dim=0).item()
}

def basic_reward_process(rtg, reward, reward_scale, all_reward):
    return rtg + (reward * reward_scale)

def adventages_reward_process(rtg, reward, reward_scale, all_reward):
    return rtg 


def basic_reward_method(current_rtg, current_rewards, scale_factor, previous_rewards):
    # Calculate the difference between current reward and previous return-to-go
    pred_return = current_rtg - (current_rewards * scale_factor)
    return pred_return.float()

def baseline_reward_method(current_rtg, current_rewards, scale_factor, previous_rewards):
    # Calculate the difference between current rewards and a moving average baseline
    baseline = previous_rewards.mean(dim=1, keepdim=True)
    reward_difference = current_rewards - baseline

    # Scale the difference by a factor
    scaled_difference = scale_factor * reward_difference

    # Add the scaled difference to the previous return-to-go
    pred_return = current_rtg + scaled_difference

    return pred_return

def discounted_sum_reward_method(current_rtg, current_rewards, scale_factor, previous_rewards):
    # Calculate the discounted sum of future rewards
    current_rewards = torch.tensor(current_rewards).unsqueeze(0).to(device)
    discount_factor = 0.95
    discounted_sum = torch.cumsum(current_rewards.flip(dims=[0]), dim=0).flip(dims=[0])
    
    # Scale the discounted sum by a factor
    scaled_discounted_sum = scale_factor * discounted_sum

    # Add the scaled discounted sum to the previous return-to-go
    pred_return = current_rtg + scaled_discounted_sum

    return pred_return

def moving_average_reward_method(current_rtg, current_rewards, scale_factor, previous_rewards):
    # Calculate the moving average of recent rewards
    window_size = 10
    moving_avg = torch.mean(previous_rewards[:, -window_size:], dim=1, keepdim=True)
    
    # Scale the moving average by a factor
    scaled_avg = scale_factor * moving_avg

    # Add the scaled moving average to the previous return-to-go
    pred_return = current_rtg + scaled_avg

    return pred_return

def reward_clipping_method(current_rtg, current_rewards, scale_factor, previous_rewards):
    # Clip the current rewards within a certain range
    current_rewards = torch.tensor([current_rewards]).to(device)
    clipped_rewards = torch.clamp(current_rewards, -1, 1)
    
    # Scale the clipped rewards by a factor
    scaled_clipped_rewards = scale_factor * clipped_rewards

    # Add the scaled clipped rewards to the previous return-to-go
    pred_return = current_rtg + scaled_clipped_rewards

    return pred_return

def exp_moving_average_reward_method(current_rtg, current_rewards, scale_factor, previous_rewards):
    # Calculate the exponential moving average of recent rewards
    alpha = 0.1
    exp_moving_avg = alpha * current_rewards + (1 - alpha) * previous_rewards[:, -1].unsqueeze(1)
    
    # Scale the exponential moving average by a factor
    scaled_exp_avg = scale_factor * exp_moving_avg

    # Add the scaled exponential moving average to the previous return-to-go
    pred_return = current_rtg + scaled_exp_avg

    return pred_return

reward_used = {
    'basic' : basic_reward_method,
    'adventages' : adventages_reward_process,
    'baseline' : baseline_reward_method,
    'discounted' : discounted_sum_reward_method,
    'moving' : moving_average_reward_method,
    'clipping' : reward_clipping_method,
    'exp' : exp_moving_average_reward_method

}

def make_env(env_id, seed = 0):
    def _f():
        env = gym.make(env_id)
        return env
    return _f

class Env:
    def __init__(self, env_id, num_env = 1, action_select = 'argmax', reward_method = 'basic',reward_scale = 1e-2):
        self.num_envs = num_env
        self.env = gym.make(env_id)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.state_std = 1.0
        self.state_mean = 0.0
        self.selector = action_selector[action_select]
        self.rewards_scale = reward_scale
        self.use_mean = False
        self.reward_method = reward_used[reward_method]
        self.action_range = [
            torch.tensor(0),
            torch.tensor(self.env.action_space.n),
        ]
        f_states, info = self.env.reset(seed=SEED)
        self.states, self.action, self.rewards, self.timesteps, self.rtg = self._init_output(f_states)
        self._init_return()


    def step(self, action_dist, epoch):
        action = self.selector(action_dist)
        if self.use_mean:
            action = action_dist.mean.reshape(self.num_envs, -1, self.act_dim)[:, -1]
        #action = action.clamp(*self.action_range)
        states, rewards, done, _, info = self.env.step(action)
        states = (
            torch.from_numpy(states).to(device=device).reshape(self.num_envs, -1, self.state_dim)
        )
        self._process(epoch)
        self.states = torch.cat([self.states, states], dim=1)
        self.rewards[:, - 1] = torch.tensor(rewards).to(device=device).reshape(self.num_envs, 1)
        self.action[:, -1] = action
        pred_return = self.reward[:, -1]#self.reward_method(self.rtg[:, -1], rewards, self.rewards_scale, self.rewards)
        self.rtg = torch.cat(
            [self.rtg, pred_return.reshape(self.num_envs, -1, 1)], dim=1
        )
        return self.states, self.action, self.rewards, self.rtg, self.timesteps, done
    
    def step_(self, action_dist, epoch):
        action = self.selector(action_dist)
        if self.use_mean:
            action = action_dist.mean.reshape(self.num_envs, -1, self.act_dim)[:, -1]
        #action = action.clamp(*self.action_range)
        states, rewards, done, _, info = self.env.step(action)
        states = (
            torch.from_numpy(states).to(device=device).reshape(self.num_envs, -1, self.state_dim)
        )
        #self._process(epoch)
        self.states = torch.cat([states], dim=1)
        self.rewards[:, -1] = torch.tensor(rewards).to(device=device).reshape(self.num_envs, 1)
        self.action[:, -1] = action
        pred_return = self.reward_method(self.rtg[:, 0], rewards, self.rewards_scale, self.rewards)
        self.rtg = torch.cat(
            [pred_return.reshape(self.num_envs, -1, 1)], dim=1
        )
        return self.states, self.action, self.rewards, self.rtg, self.timesteps, done, action
        
    def _reset_env(self):
        f_states, info = self.env.reset(seed=SEED)
        self.states, self.action, self.rewards, self.timesteps, self.rtg = self._init_output(f_states)
        return f_states
    
    def _init_return(self):
        self._process()

    def _process(self, timestep=None):
        self.action = torch.cat(
            [
                self.action,
                torch.zeros((self.num_envs, self.action_dim), device=device).reshape(
                    self.num_envs, -1, self.action_dim
                ),
            ],
            dim=1,
        )
        self.rewards = torch.cat(
            [
                self.rewards,
                torch.zeros((self.num_envs, 1), device=device).reshape(self.num_envs, -1, 1),
            ],
            dim=1,
        )
        if timestep != None:
            self.timesteps = torch.cat(
                [
                    self.timesteps,
                    torch.ones((self.num_envs, 1), device=device, dtype=torch.long).reshape(
                        self.num_envs, 1
                    )
                    * (timestep + 1),
                ],
                dim=1,
            )



    def _init_output(self, state):
        states = (
        torch.from_numpy(state)
        .reshape(self.num_envs, self.state_dim)
        .to(device=device, dtype=torch.float32)
        ).reshape(self.num_envs, -1, self.state_dim)
        
        actions = torch.zeros(0, device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        timesteps = torch.tensor([0] * self.num_envs, device=device, dtype=torch.long).reshape(
            self.num_envs, -1
        )
        target_return = torch.tensor([1]*self.num_envs, dtype=torch.float32).to(device)
        ep = target_return
        target_return = torch.tensor(ep, device=device, dtype=torch.float32).reshape(
            self.num_envs, -1, 1
        )
        return states, actions, rewards, timesteps, target_return


    def reset(self):
        state = self._reset_env()
        self.states = (
        torch.from_numpy(state)
        .reshape(self.num_envs, self.state_dim)
        .to(device=device, dtype=torch.float32)
        ).reshape(self.num_envs, -1, self.state_dim)
        #self.states = torch.cat([self.states, state], dim=1)
        self._init_return()
        return self.states, self.action, self.rtg.float(), self.timesteps
    



