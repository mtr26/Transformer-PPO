"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import random

class Custom_Buffer:
    def __init__(self, mem_capacity = 5000, batch_size = 64):
        self.traj = []
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size

    def __len__(self):
        return len(self.traj)
    
    def _is_full(self):
        return self.__len__() > self.mem_capacity

    def push(self, states, actions, rewards, dones, rtg, timesteps):
        if self._is_full():
            self.traj.pop(0)
        self.traj.append((states, actions, rewards, dones, rtg, timesteps))

    def sample(self):
        return random.sample(self.traj, self.batch_size)
    

