import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self, states, actions, hidden=[64,64]):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(states, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], actions)

    def forward(self,state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
