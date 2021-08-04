import numpy as np
import matplotlib.pyplot as plt
import gym
import random

# hyper parameters
# test 1
# alpha = 0.5
# gamma = 0.95
# epsilon = 0.1

epsilon = 0.1
alpha = 0.1
gamma = 0.1

#monte carlo method
env = gym.make("Taxi-v3")

def get_probs(q, env, epsilon):
    '''
    get the probability of taking the best known action according to epsion
    '''
    # import ipdb;ipdb.set_trace()
    actions = np.argmax(q,axis=1)
    policy_s = np.ones((env.observation_space.n, env.action_space.n))/env.action_space.n
    for i in range(env.observation_space.n):
        if q[i].any()!=0:
            policy_s[i] = policy_s[i] * epsilon
            policy_s[i,actions[i]] = 1 - epsilon + (epsilon / env.action_space.n)
    return policy_s

def update_Q(env, episodes, Q, q):
    for (s,a,r) in episodes:
        sum = 0
        for x in Q[(s,a)]:
            sum += x
        avg = sum/len(Q[(s,a)])
        q[s,a] = avg

def run(env, Q, q, epsilon, gamma):
    episodes = []
    state = env.reset()
    G = 0
    r = 0
    # import ipdb;ipdb.set_trace()
    probs = get_probs(q, env, epsilon)
    while True:# loop for each step of episode
        # import ipdb;ipdb.set_trace()
        # probs = get_probs(q, env, epsilon) # get the current behavior policy
        action = np.random.choice(np.arange(env.action_space.n),p=probs[state])\
                if True in q[state]!=0 else env.action_space.sample()
        
        next_state, reward, done, _ = env.step(action)
        episodes.append((state, action, reward))

        G = gamma * G + reward
        if (state,action) in Q:
            Q[(state, action)].append(G)
        else:
            Q[(state,action)] = [G]

        r += reward
        # print(f'current status:r:{r},state:{state},action:{action}')

        state = next_state
        if done:
            break
    return episodes,r

# initialize q table
q = np.zeros((env.observation_space.n, env.action_space.n))
Q = {}

# reward and error record
q_reward_record = []
q_error_record = []

# import ipdb;ipdb.set_trace()
# loop for each episode:
for episode in range(5000):
    state = env.reset()
    episodes,r = run(env,Q,q,epsilon,gamma)
    update_Q(env,episodes,Q,q)
    
    if episode%100 == 0:
        print(f'{episode}th episode: {r}')
