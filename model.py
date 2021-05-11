import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, seed, hidden_size=128, hidden_size2=256):
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2) 
        self.fc3 = nn.Linear(hidden_size2, action_size)
        self.set_weights()
        
    def set_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        
        out = F.relu(self.fc1(state))
        out = self.bn(out)
        out = F.relu(self.fc2(out))
        return F.tanh(self.fc3(out))
    
class Critic(nn.Module):
    
    def __init__(self, state_size, action_size, seed, hidden_size=128, hidden_size2=128):
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size+(action_size), hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.set_weights() 
        
    def set_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        if state.dim == 1:
            state = torch.unsqueeze(state, 0)
        
        out1 = F.relu(self.fc1(state))
        out1 = self.bn(out1)
        out = torch.cat((out1, action), dim=1)
        out = F.relu(self.fc2(out))
        return self.fc3(out)