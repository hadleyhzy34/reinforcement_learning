import gym
import time

env = gym.make('CartPole-v1')
observation = env.reset() #position of cart,velocity of cart, angle of pole, rotation rate of pole

for _ in range(1000):
    env.render()
    time.sleep(0.1)
    import ipdb; ipdb.set_trace()
    action = env.action_space.sample()
    print(f'action is: {action}, type of action is: {type(action)}')
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
