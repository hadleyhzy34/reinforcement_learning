from networks import *
import numpy as np
from collections import deque
import torch
import torch.optim as optim
import random

class Agent:
    def __init__(self, actions, states, batch_size, lr, gamma):
        self.actions = actions
        self.states = states
        self.actions = actions
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
            
        self.Q_local = Q_Network(self.states, self.actions)
        self.Q_target = Q_Network(self.states, self.actions)
        
        #self.soft_update(1)
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        #buffer: 1.state 2.action 3.reward 4.next state 5.done
        self.memory = deque(maxlen=100000)

    def act(self, state, eps=0):
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_values = self.Q_local(state)
            #print(f'not randomly\n')
            return np.argmax(action_values.cpu().data.numpy())
        else:
            #print(f'randomly, eps is: {eps}, action space is: {self.actions}\n')
            return random.choice(np.arange(self.actions))

    def learn(self):
        exp = random.sample(self.memory, self.batch_size)
        #print(f'current exp size is: {len(exp)}')
        #import ipdb; ipdb.set_trace()
        states = torch.from_numpy(np.vstack([e[0] for e in exp])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in exp])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in exp])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in exp])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in exp]).astype(np.uint8)).float()

        #import ipdb;ipdb.set_trace()
        Q_values = self.Q_local(states)
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions)

        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True)
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets

        loss = (Q_values - Q_targets).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    #for every certain constant episodes, copy nn parameters from local to target to update target
    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':
    test = Agent(2,4,0.01,0.99)
