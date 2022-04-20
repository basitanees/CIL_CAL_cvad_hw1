import torch.nn as nn
import torch.nn.functional as F
import torch


class MultiLayerPolicy(nn.Module):
    """An MLP based policy network"""
    def __init__(self):
        super().__init__()
        action_dim = 2
        state_size = 8 #change depending on features
        self.branch1 = nn.Sequential(nn.Linear(state_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.branch2 = nn.Sequential(nn.Linear(state_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.branch3 = nn.Sequential(nn.Linear(state_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.branch4 = nn.Sequential(nn.Linear(state_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.branchList = nn.ModuleList([self.branch1, self.branch2, self.branch3, self.branch4])

        self.action_range = 1

    def forward(self, features, command):

        branches_outputs = []
        for branch in self.branchList:
            branches_outputs.append(branch(features))
        branches_outputs = torch.stack(branches_outputs, dim = 1)
        
        x = branches_outputs[torch.arange(branches_outputs.shape[0]), command[0], :]
        
        x = self.action_range * torch.tanh(x) 
        return x.unsqueeze(0) if len(x.shape) == 1 else x
