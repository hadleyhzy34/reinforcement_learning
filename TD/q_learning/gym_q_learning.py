import numpy as np
import gym

env = gym.make("Taxi-v3").env
env.render()

env.reset()
env.render()

print(f'action space {env.action_space}')
print(f'state space {env.observation_space}')


state = env.encode(3,1,1,1)
print('state:',state)

env.s = state
env.render()

#test for reward
state = env.encode(0,4,1,0)
print('state: ',state)

env.s = state
env.render()

print(env.P[state])

env.reset()
state = env.encode(0,4,4,1)
print(f'state: {state}')

env.s = state
env.render()
print(env.P[state])


env.reset()
state = env.encode(0,4,1,0)
env.s = state
env.render()
print(env.P[state])



#initialize q table
qt = np.zeros([env.observation_space.n, env.action_space.n])
print(f'shape of q table is: {qt.shape}')

#training the target
import random

alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

for i in range(1,100001):
    state = env.reset()

    epochs, penalties, reward = 0,0,0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(qt[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = qt[state, action]
        next_max = np.max(qt[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        qt[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        #clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(qt[state])
        state, reward, done, info = env.step(action)

        #if reward == -10:
        #    penalties += 1
        
        penalties += reward

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
