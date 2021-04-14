import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self, states, actions, hidden=[64,64], duel=True):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(states, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], actions)
        self.duel = duel
        if self.duel:
            self.fc4 = nn.Linear(hidden[1], 1)

    def forward(self,state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.duel:
            import ipdb;ipdb.set_trace()
            x1 = self.fc3(x)
            x1 = x1 - torch.max(x1, dim=1 , keepdim=True)[0]
            x2 = self.fc4(x)
            return x1 + x2
        else:
            x = self.fc3(x)
            return x
