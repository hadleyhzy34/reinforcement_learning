import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c import Actor

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]

model = Actor(state_space)
model.load_state_dict(torch.load('a2c_model.pth'))

# model policy performance visualization
for i_episode in count(1):
    state = env.reset()
    for _step in range(1000):
        # import ipdb;ipdb.set_trace()
        action, action_prob = model.select_action(state)
        state, r, done, _ = env.step(action)

        env.render()
        if done:
            break


