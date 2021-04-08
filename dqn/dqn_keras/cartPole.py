import numpy as np
import gym
import random


env_name = 'CartPole-v0'
env = gym.make(env_name)

states = env.observation_space.shape[0]
actions = env.action_space.n
print(f'number of states in the current selected environment is: {env.observation_space.shape[0]}')
print(f'number of available actions is: {env.action_space}')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adam

#from keras.models import Sequential
#from keras.layers import Dense,Flatten
#from keras import optimizers
#from keras.optimizers import Adam

#build the model
def build_model(states,actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(actions,activation='linear'))
    return model
    #print(model.summary())

try:
    model
except NameError:
    pass
else:
    del model

#del model
model = build_model(states,actions)
model.summary()

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


#agent
def build_agent(model,actions):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    #policy = BoltzmannQPolicy()
    #memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model = model, memory = memory, policy = policy, nb_actions = actions, nb_steps_warmup = 10, target_model_update = 1e-2 )
    #dqn = DQNAgent(model=model, memory=memory, policy = policy, nb_actions = actions, nb_steps_warmup=10,target_model_update=1e-2)
    return dqn

dqn = build_agent(model,actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000,visualize=False,verbose=1)


scores = dqn.test(env,nb_episodes=100,visualize=False)
print(np.mean(scores.history['episode_reward']))

#visualize model
_ = dqn.test(env, nb_episodes=5, visualize = True)
