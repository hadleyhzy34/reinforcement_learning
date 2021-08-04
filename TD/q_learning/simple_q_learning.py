#http://mnemstudio.org/path-finding-q-learning-tutorial.htm
#provide two different criterion for q-learning
import numpy as np

#initialize q function
q = np.matrix(np.zeros([6,6]))

# r is the tabular representation for rewards
r = np.matrix([[-1,-1,-1,-1,0,-1],
               [-1,-1,-1,0,-1,100],
               [-1,-1,-1,0,-1,-1],
               [-1,0,0,-1,0,-1],
               [0,-1,-1,0,-1,100],
               [-1,0,-1,-1,0,100]])

#hyperparameter
gamma = 0.8
epsilon = 0.4
alpha = 0.1 #learning rate

#the main training loop
for episode in range(101):
    #random initial state
    state = np.random.randint(0,6)

    while (state!=5):
        possible_actions = []
        possible_q = []
        for action in range(6):
            #loop through all actions, choose rules-allowed actions
            if r[state,action] >= 0:
                possible_actions.append(action)
                possible_q.append(q[state,action])
        
        #step next state, here we use epsilon-greedy algo
        action = -1
        if np.random.random() < epsilon:
            #choose random action
            action = possible_actions[np.random.randint(0,len(possible_actions))]
        else:
            #greedy
            action = possible_actions[np.argmax(possible_q)]

        #update q value
        #method1
        #q[state,action] = r[state,action] + gamma * q[action].max()
    
        #method2
        rs = r[state,action]
        re = gamma * q[action].max()
        cur_r = q[state, action]
        td = rs+re-cur_r
        print(f'next immediate reward is: {r[action,np.argmax(q[action])]}')
        print(f'future reward is: {gamma*q[action].max()}')
        print(f'current future reward is: {q[state,action]}')
        #q[state,action] += alpha * [ rs + re - cur_r
        #q[state,action] += alpha * (r[action,np.argmax(q[action])] + gamma * q[action].max() - q[state,action])
        q[state,action] += alpha * td
        #go to the next state
        state = action

    if episode % 10 ==0:
        print('_________________________________________')
        print('training episode: %d' % episode)
        print(q)


for i in range(10):
    print('episode: %d' %i)

    #random initial state
    state = np.random.randint(0,6)
    print(f'robot starts at {state}')
    for _ in range(20):
        if state == 5:
            break
        action = np.argmax(q[state])
        print(f'the robot goes to {action}')
        state = action




