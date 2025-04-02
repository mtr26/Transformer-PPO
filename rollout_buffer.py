import torch
import numpy as np


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class RolloutBuffer:
    """
    Rollout Buffer for storing experiences.
    buffer_size : int : size of the buffer
    state_dim : int : dimension of the state space
    action_dim : int : dimension of the action space
    """
    def __init__(self, buffer_size : int, state_dim : int, action_dim : int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = [0]*self.buffer_size
        self.actions = [0]*self.buffer_size
        self.rewards = [0]*self.buffer_size
        self.next_states = [0]*self.buffer_size
        self.dones = [0]*self.buffer_size
        self.rtg = [0]*self.buffer_size
        self.timestep = [0]*self.buffer_size
        self.great_action = [0]*self.buffer_size

        self.index = 0
        self.full = False

    def reset(self):
        """
        Reset the buffer.
        """
        self.states = [0]*self.buffer_size
        self.actions = [0]*self.buffer_size
        self.rewards = [0]*self.buffer_size
        self.next_states = [0]*self.buffer_size
        self.dones = [0]*self.buffer_size
        self.rtg = [0]*self.buffer_size
        self.timestep = [0]*self.buffer_size
        self.great_action = [0]*self.buffer_size
        self.index = 0
        self.full = False

    def add_experience(self, state, action, reward, next_state, done, rtg, timestep, great_action):
        """
        Add an experience to the buffer.
        state : torch.Tensor : state
        action : torch.Tensor : action
        reward : float : reward
        next_state : torch.Tensor : next state
        done : bool : episode done
        rtg : float : return to go
        timestep : int : timestep
        great_action : int : great action
        """
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.rtg[self.index] = rtg
        self.timestep[self.index] = timestep
        self.great_action[self.index] = great_action

        self.index = (self.index + 1) % self.buffer_size
        if self.index == 0:
            self.full = True

    def __get_sample__(self, indice : int):
        """
        Get a sample from the buffer.
        indice : int : index of the sample
        """
        batch = {
            'states': self.states[indice],
            'actions': self.actions[indice],
            'rewards': self.rewards[indice],
            'next_states': self.next_states[indice],
            'dones': self.dones[indice],
            'rtg' : self.rtg[indice],
            'timestep' : self.timestep[indice],
            'great_action' : self.great_action[indice]
        }
        return batch
    
    def __generate__(self):
        """
        Generate a random index.
        """
        n = np.random.randint(0, self.index if not self.full else self.buffer_size)
        return n
    

    def get_batchs(self, batch_size : int):
        """
        Get a batch of samples from the buffer.
        batch_size : int : size of the
        """
        g = [self.__get_sample__(np.random.randint(0, self.index if not self.full else self.buffer_size)) for i in range(batch_size)]
        st = [i['states'].tolist() for i in g]
        ac = [i['actions'].tolist() for i in g]
        re = [i['rewards'].tolist() for i in g]
        ns = [i['next_states'].tolist() for i in g]
        do = [i['dones'] for i in g]
        rt = [i['rtg'].tolist() for i in g]
        ti = [i['timestep'].tolist() for i in g]
        ga = [i['great_action'] for i in g]

        

        batchs = {
            'states': torch.tensor(st).to(DEVICE),
            'actions': torch.tensor(ac).to(DEVICE),
            'rewards': torch.tensor(re).to(DEVICE),
            'next_states': torch.tensor(ns).to(DEVICE),
            'dones': torch.tensor(do).to(DEVICE),
            'rtg' : torch.tensor(rt).to(DEVICE),
            'timestep' : torch.tensor(ti).to(DEVICE),
            'great_action' : torch.tensor(ga).to(DEVICE)
        }
        return batchs
        

    def get_batch(self):
        """
        Get a sample from the buffer.
        """
        return self.__get_sample__(self.__generate__())