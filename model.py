import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
    
class GetTemplate(nn.Module):
    """
    Predict template 
    """
    def __init__(self, state_dim, t_dim):
        super(GetTemplate, self).__init__()
        self.state_dim = state_dim
        self.t_dim = t_dim

        modules = []
        modules.append(nn.Linear(state_dim, 512))
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())
        
        modules.append(nn.Linear(512, 512))
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())
        
        modules.append(nn.Linear(512, 512))
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())
        
        modules.append(nn.Dropout(0.5))
        modules.append(nn.Linear(512, t_dim))
        
        self.layers = nn.Sequential(*modules)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
    
    def forward(self, s, T_mask, temp):
        f = self.layers(s)
        T = torch.tanh(f)
        T = T * T_mask
        T_hot = gumbel_softmax(T, tau=temp)
        return T_hot
    
    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
        
    def load(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
        
def gumbel_softmax(logits, tau, hard=False):
    dim = -1
    g_ratio = 1e-3
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels * g_ratio) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
    
class GetAction(nn.Module):
    """
    Predict another reactant
    """
    def __init__(self, state_dim, t_dim, act_dim):
        super(GetAction, self).__init__()
        self.state_dim = state_dim
        self.t_dim = t_dim
        
        modules = []
        modules.append(nn.Linear(state_dim + t_dim, 512))
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())
        
        modules.append(nn.Linear(512, 512))
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())
        
        modules.append(nn.Linear(512, 512))
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())
        
        modules.append(nn.Dropout(0.5))
        modules.append(nn.Linear(512, act_dim))
        
        self.layers = nn.Sequential(*modules)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        
    def forward(self, s, T):
        s = s.reshape(-1, self.state_dim)
        T = T.reshape(-1, self.t_dim)
        x = torch.cat((s, T), -1)
        a = torch.tanh(self.layers(x))
        
        return a

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
        
    def load(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
        
class Critic(nn.Module):
    
    def __init__(self, state_dim, t_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.t_dim = t_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + t_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, s, t, a):
        s = s.reshape(-1, self.state_dim)
        t = t.reshape(-1, self.t_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, t, a), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
        
    def load(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
