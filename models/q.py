import torch.nn as nn
import torch.nn.functional as F
import torch


class MultiLayerQ(nn.Module):
    """Q network consisting of an MLP."""
    def __init__(self, config):
        super().__init__()
        action_dim = 2 # action_size
        state_size = 8 # len(config.features)
        self.linear1 = nn.Linear(state_size+action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, features, actions):

        x = torch.cat((features, actions),dim=1)#.squeeze(0)
        q = F.relu(self.linear1(x))
        q = F.relu(self.linear2(q))
        q = F.relu(self.linear3(q))
        return torch.squeeze(q, -1)
