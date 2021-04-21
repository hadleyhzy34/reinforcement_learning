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
        #import ipdb;ipdb.set_trace()
        state = env.reset()
        t = 0

        while not done and t < max_t:

            t += 1
            #print(f'state and next before action is: {state.shape}')
            #reshape state to make sure it could make for both batch and single sample
            state = state.reshape(1,-1)
            action = agent.act(state, eps)
            #print(f'action is : {action}, and type of action is: {type(action)}')
            next_state, reward, done, _ = env.step(action)
            agent.memory.remember(state, action, reward, next_state, done) #insert new memories
            #print(f'agent memory is: {agent.memory.ind_max}')
            if t % 4 == 0 and len(agent.memory) >= agent.batch_size:
                #print(f'agent started learning process')
                agent.learn()
                agent.soft_update(0.001)
            
            #print(f'current state and next_state is: {state.shape}, {next_state.shape}')
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
    agent = Agent(actions,states,BATCH_SIZE,LEARNING_RATE,GAMMA,False,True,True)
    rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    #torch.save(agent.Q_local.state_dict(), '{}_weights.pth'.format(RAM_ENV_NAME))
    torch.save(agent.Q_local.state_dict(), f'{RAM_ENV_NAME}_test_weights.pth')
    #agent.Q_local
