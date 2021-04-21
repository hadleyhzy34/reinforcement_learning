import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    #env.render()
    done = False
    while not done:
        state, rewards, done, info = env.step(env.action_space.sample()) # take a random action
        env.render()
    state = env.reset()
env.close()
