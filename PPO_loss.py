import torch
import torch.nn as nn
import torch.nn.functional as F


def classic_ppo(states, action_target, action_preds, rtg, next_state, reward, gamma = 0.99, gae_gamma = 1.0, clip_range = 0.3, entropy_coef=0.01, value_coef=0.5):
    # Compute the adventages
    delta = reward + gamma * rtg - action_preds
    adventages = delta
    # Ratio
    ratio = torch.exp(action_target - action_preds)

    # Clipped surrogate loss
    surr1 =  adventages * ratio
    surr2 = adventages * torch.clamp(ratio, 1 - clip_range, 1+clip_range)
    policy_loss = -torch.min(surr1, surr2)

    value_loss = F.mse_loss(action_preds, action_target)
    entropy = -torch.mean(-action_preds)
    loss = policy_loss + entropy_coef * entropy + value_coef * value_loss
    return loss.mean()
