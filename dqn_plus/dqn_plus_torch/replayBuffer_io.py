import config 
from utils import SumTree

import numpy as np
import random
import torch
import json

TD_INIT = config.td_init
EPSILON = config.epsilon
ALPHA = config.alpha
FILE_PATH = config.file_path 


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
    proportion-based replay buffer, modification to read and write memory data from disk, not from ram
    '''
    def __init__(self, capacity = int(1e6), batch_size = None):
        self.capacity = capacity
        self.alpha = ALPHA
        self.weights = SumTree(self.capacity)
        #self.memory = [None for _ in range(capacity)] #list to save tuples
        self.ind_max = 0  #how many transitions have been stored
        self.default = TD_INIT

    def remember(self, state, action, reward, next_state, done):
        ind = self.ind_max % self.capacity
        '''
        write memory data to json file and saved it to checkpoints folder,
        memory data could be very large and ram cannot hold all of them thus system would crash
        '''
        #import ipdb;ipdb.set_trace()
        dict = {}
        dict['state'] = state.tolist()
        dict['action'] = action.tolist()
        dict['reward'] = reward
        dict['next_state'] = next_state.tolist()
        dict['done'] = done
        temp_dict = json.dumps(dict, indent = 5)
        with open(FILE_PATH+'memory'+str(ind)+'.json', 'w') as file:
            file.write(temp_dict)

        #self.memory[ind] = (state, action, reward, next_state, done)
        #import ipdb;ipdb.set_trace()
        delta = self.default + EPSILON - self.weights.vals[ind + self.capacity - 1]
        self.weights.update(delta, ind)
        self.ind_max += 1
        #print(f'ind is:{ind},ind_max is:{self.ind_max}\n')

    def sample(self, batch_size):
        '''
        return sampled transitions. make sure there are at least batch_size transitions
        '''
        index_set = [self.weights.retrive(self.weights.vals[0]*random.random()) for _ in range(batch_size)]
        #print(f'size of index_set is: {len(index_set)},{index_set[0]}')
        #for each index, normalized probability<->stochastic prioritization
        probs = torch.from_numpy(np.vstack([self.weights.vals[ind+self.capacity-1]/self.weights.vals[0] for ind in index_set])).float()
        #print(f'type of probs is: {type(probs)}')
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for ind in index_set:
            with open(FILE_PATH+'memory'+str(ind)+'.json','r') as openfile:
                json_temp = json.load(openfile)
                #import ipdb;ipdb.set_trace()
                states.append(json_temp['state'])
                actions.append(json_temp['action'])
                rewards.append(json_temp['reward'])
                next_states.append(json_temp['next_state'])
                dones.append(json_temp['done'])
        
        #import ipdb;ipdb.set_trace()
        states = torch.tensor(states).float()
        states = torch.squeeze(states) #change dimension from bs,1,2,80,80 to bs,2,80,80
        actions = torch.tensor(actions).long()
        actions = actions.unsqueeze(1)
        #actions = actions.resize_((,1))
        rewards = torch.tensor(rewards).float()
        rewards = rewards.unsqueeze(1)
        #rewards = rewards.resize_((,1))
        next_states = torch.tensor(next_states).float()
        next_states = torch.squeeze(next_states)
        dones = torch.tensor(dones).float()
        dones = dones.unsqueeze(1)
        #dones = dones.resize_((,1))
        
        #import ipdb;ipdb.set_trace()        
        #states_test = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        #actions_test = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        #rewards_test = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        #next_states_test = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        #dones_test = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()

        return index_set, states, actions, rewards, next_states, dones, probs

    def insert(self, error, index):
        delta = error + EPSILON - self.weights.vals[index + self.capacity - 1]
        self.weights.update(delta, index)

    def __len__(self):
        return min(self.ind_max, self.capacity)

