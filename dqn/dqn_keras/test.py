

import gym
import random

env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n


import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers



def build_model(states, actions):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(actions,activation='linear'))
    return model


del model


model = build_model(states,actions)

model.summary()



from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy = policy, nb_actions = actions, nb_steps_warmup=10,target_model_update=1e-2)
    return dqn


dqn = build_agent(model,actions)
dqn.compile(optimizers.Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000,visualize=False,verbose=1)


