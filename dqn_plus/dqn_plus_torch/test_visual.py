import numpy as np
import gym
from utils import *
from agent import *
from config import *

Path = 'LunarLander-v2_weights.pth'
#testing Process
env = gym.make(RAM_ENV_NAME)
states = env.observation_space.shape[0]
actions = env.action_space.n
print(f'state space is: {env.observation_space.shape[0]}, action space is: {env.action_space.n}')
agent = Agent(actions, states, BATCH_SIZE, LEARNING_RATE, GAMMA, False, True, True)
agent.Q_local.load_state_dict(torch.load(Path))



total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    #env.render()
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        state = state.reshape(1,-1)
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        env.render()
        #if reward == -10:
        #    penalties += 1

        penalties += reward

        epochs += 1

    print(f'current episode, steps: {epochs}, rewards: {penalties}')
    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
