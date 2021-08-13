import argparse
import os
import time
import gym

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

ENV_ID = 'CartPole-v1'
RANDOM_SEED = 1
RENDER = False

ALG_NAME = 'PG'
TRAIN_EPIOSDES = 200
TEST_EPISODES = 10
MAX_STEPS = 500


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size, states, actions):
        '''
        hidden_size
        states: dimension of states
        actions: action spaces
        '''
        super(PolicyNetwork, self).__init__()
        self.action_space = actions
        
        self.linear1 = nn.Linear(states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.action_space)

    
    def forward(self, x):
        import ipdb;ipdb.set_trace()
        x = F.relu(self.linear1(x))
        pref_act = self.linear2(x)
        return F.softmax(pref_act,dim=-1)

class REINFORCE:
    def __init__(self, hidden_size, states, actions):
        self.action_space = actions
        self.model = PolicyNetwork(self, hidden_size, states, actions)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
	# self.model.train()

    # def select_action(self, state):
    #     '''
    #     return cur
    #     '''
    #     probs = self.model(state)
    #     action = 
    #     probs = self.model(Variable(state).cuda())       
    #     action = probs.multinomial().data
    #     prob = probs[:, action[0,0]].view(1, -1)
    #     log_prob = prob.log()
    #     entropy = - (probs*probs.log()).sum()

    #     return action[0], log_prob, entropy

    # def update_parameters(self, rewards, log_probs, entropies, gamma):
    #     R = torch.zeros(1, 1)
    #     loss = 0
    #     for i in reversed(range(len(rewards))):
    #         R = gamma * R + rewards[i]
    #         loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
    #     loss = loss / len(rewards)
		
    #     self.optimizer.zero_grad()
    #     loss.backward()
	# utils.clip_grad_norm(self.model.parameters(), 40)
    #     self.optimizer.step()

if __name__ == "__main__":
    import ipdb;ipdb.set_trace()
    policy = PolicyNetwork(4,3,2)
    x = Variable(torch.rand(3))
    # x = torch.tensor(np.random.normal(size=(3)))
    res = policy(x)


