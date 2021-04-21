from networks import *
import numpy as np
from collections import deque
import torch
import torch.optim as optim
import random
from replayBuffer_io import Replay_buffer,Proportion_replay_buffer



class Agent:
    def __init__(self, actions, states, batch_size, lr, gamma, visual = False, double = True, per = False):
        '''
        when dealing with visual inputs, state_size should work as num_of_frame
        '''
        self.actions = actions
        self.states = states
        self.actions = actions
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.double = double
        self.per = per

        if not visual:    
            self.Q_local = Q_Network(self.states, self.actions)
            self.Q_target = Q_Network(self.states, self.actions)
        else:
            self.Q_local = V_Q_Network(self.states, self.actions)
            self.Q_target = V_Q_Network(self.states, self.actions)
        
        #self.soft_update(1)
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        #buffer: 1.state 2.action 3.reward 4.next state 5.done
        #self.memory = deque(maxlen=100000)
        if not self.per:
            self.memory = Replay_buffer(int(1e5), self.batch_size)
        else:
            '''
            1e6 is too large for 16g ram, temporarily set it to be 1e4 at most, otherwise system would crash
            '''
            self.memory = Proportion_replay_buffer(int(1e4), self.batch_size)

    def act(self, state, eps=0):
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32)
            #import ipdb;ipdb.set_trace()
            #print(f'not randomly\n')
            with torch.no_grad():
                action_values = self.Q_local(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            #print(f'randomly, eps is: {eps}, action space is: {self.actions}\n')
            return random.choice(np.arange(self.actions))

    def learn(self):
        if not self.per:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            w = torch.ones(actions.size())
        else:
            index_set, states, actions, rewards, next_states, dones, probs = self.memory.sample(self.batch_size)
            #import ipdb;ipdb.set_trace()
            w = 1/len(self.memory)/probs #reducing weights of the often seen samples
            #print(f'current w shape is: {w.shape}, {w[0]}, {torch.max(w)}')
            #import ipdb;ipdb.set_trace()
            w = w/torch.max(w) #make sure all samples distribution at most equal to 1
        #exp = random.sample(self.memory, self.batch_size)
        #print(f'current exp size is: {len(exp)}')
        #import ipdb; ipdb.set_trace()
        #states = torch.from_numpy(np.vstack([e[0] for e in exp])).float()
        #actions = torch.from_numpy(np.vstack([e[1] for e in exp])).long()
        #rewards = torch.from_numpy(np.vstack([e[2] for e in exp])).float()
        #next_states = torch.from_numpy(np.vstack([e[3] for e in exp])).float()
        #dones = torch.from_numpy(np.vstack([e[4] for e in exp]).astype(np.uint8)).float()
        #print(f'size of states and next states is: {states.shape}, {next_states.shape}')
        #import ipdb;ipdb.set_trace()
        Q_values = self.Q_local(states)
        #print(f'current q values based on current states size is: {Q_values.shape}')
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions)

        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            #print(f'q target values based next states is: {Q_targets.shape}')
            #double dqn algo
            if not self.double:
                Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True)
            else:
                #import ipdb;ipdb.set_trace()
                #actions_test = self.Q_local(next_states)
                #inner_actions_test = np.argmax(self.Q_local(next_states).cpu().data.numpy(),axis=1)
                inner_actions = torch.max(input=self.Q_local(next_states), dim=1, keepdim=True)[1]
                #if inner_actions_test == inner_actions:
                    #printf('two different calculations are the same')
                #else:
                    #printf('two dif cal are not the same')
                Q_targets = torch.gather(input=Q_targets, dim=1, index=inner_actions)
                #print(f'size of q targets after inner actions is: {Q_targets.shape}, {states.shape},{next_states.shape}')
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets

        deltas = Q_values - Q_targets
        loss = ((w*deltas).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update experiences of sumtree for current batch
        if self.per:
            deltas = np.abs(deltas.detach().numpy().reshape(-1))
            for i in range(self.batch_size):
                self.memory.insert(deltas[i],index_set[i])

    #for every certain constant episodes, copy nn parameters from local to target to update target
    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':
    test = Agent(2,4,128,0.01,0.99)
