import numpy as np
import gym
from agent import *
from config import *

def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, 1 + num_episode):

        episodic_reward = 0
        done = False
        state = env.reset()
        t = 0

        while not done and t < max_t:

            t += 1
            action = agent.act(state, eps)
            #print(f'action is : {action}, and type of action is: {type(action)}')
            next_state, reward, done, _ = env.step(action)
            agent.memory.remember(state, action, reward, next_state, done)

            if t % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()
                agent.soft_update(0.001)

            state = next_state.copy()
            episodic_reward += reward

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]), end='')
        if i % 50 == 0:
            print()

        eps = max(eps * eps_decay, eps_min)

    return rewards_log

if __name__ == '__main__':
    env = gym.make(RAM_ENV_NAME)
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    print(f'state space is: {env.observation_space.shape[0]}, action space is: {env.action_space.n}')
    agent = Agent(actions,states,BATCH_SIZE,LEARNING_RATE,GAMMA)
    rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    agent.Q_local
