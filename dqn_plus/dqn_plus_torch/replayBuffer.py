import config
from utils import SumTree

import numpy as np
import random
import torch

TD_INIT = config.td_init
EPSILON = config.epsilon
ALPHA = config.alpha

class Replay_buffer:
    '''
    basic replay buffer
    '''
    def __init__(self, capacity = int(1e6), batch_size = None):
        self.capacity = capacity
        self.memory = [None for _ in range(capacity)] #list to save tuples
        self.ind_max = 0  #how many transitions have been stored

    def remember(self, state, action, reward, next_state, done):
        ind = self.ind_max % self.capacity
        self.memory[ind] = (state, action, reward, next_state, done)
        self.ind_max += 1

    def sample(self, k):
        '''
        return sampled transitions. make sure there are at least k transitions
        '''
        index_set = random.sample(list(range(len(self))), k)
        states = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        actions = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        dones = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()


        return states, actions, rewards, next_states, dones

    def __len__(self):
        return min(self.ind_max, self.capacity)


class Proportion_replay_buffer:
    '''
    proportion-based replay buffer
    '''
    def __init__(self, capacity = int(1e6), batch_size = None):
        self.capacity = capacity
        self.alpha = ALPHA
        self.weights = SumTree(self.capacity)
        self.memory = [None for _ in range(capacity)] #list to save tuples
        self.ind_max = 0  #how many transitions have been stored
        self.default = TD_INIT

    def remember(self, state, action, reward, next_state, done):
        ind = self.ind_max % self.capacity
        self.memory[ind] = (state, action, reward, next_state, done)
        delta = self.default + EPSILON - self.weights.vals[ind + self.capacity - 1]
        self.weights.update(delta, ind)
        self.ind_max += 1

    def sample(self, batch_size):
        '''
        return sampled transitions. make sure there are at least k transitions
        '''
        index_set = [self.weights.retrive(self.weights.vals[0]*random.random()) for _ in range(batch_size)]
        #print(f'size of index_set is: {len(index_set)},{index_set[0]}')
        #for each index, normalized probability<->stochastic prioritization
        probs = torch.from_numpy(np.vstack([self.weights.vals[ind+self.capacity-1]/self.weights.vals[0] for ind in index_set])).float()
        #print(f'type of probs is: {type(probs)}')
        states = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        actions = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        dones = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()

        return index_set, states, actions, rewards, next_states, dones, probs

    def insert(self, error, index):
        delta = error + EPSILON - self.weights.vals[index + self.capacity - 1]
        self.weights.update(delta, index)

    def __len__(self):
        return min(self.ind_max, self.capacity)

