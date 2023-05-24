import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, training=True):
        super(DQN, self).__init__()
        self.fe = nn.Linear(state_dim, 100)
        self.fc_value = nn.Linear(100, 50)
        self.fc_adv = nn.Linear(100, 50)
        self.value = nn.Linear(50, 1)
        self.adv = nn.Linear(50, action_dim)
        self.training = training

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fe(x))
        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))
        value = self.value(value)
        adv = self.adv(adv)
        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage
        return Q