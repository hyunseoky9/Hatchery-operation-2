import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np

class Critic2(nn.Module):
    """Critic (Value) Model.
    Unlike Critic, this version merges state and action features from the start.
    
    """

    def __init__(self, state_size, action_size, state_hidden_size, state_hidden_num, action_hidden_size, action_hidden_num, trunk_hidden_size, trunk_hidden_num, lrdecayrate, learning_rate,fstack):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.state_hidden_size = state_hidden_size if isinstance(state_hidden_size, list) else (np.ones(state_hidden_num)*state_hidden_size).astype(int)
        self.state_hidden_num = state_hidden_num
        self.action_hidden_size = action_hidden_size if isinstance(action_hidden_size, list) else (np.ones(action_hidden_num)*action_hidden_size).astype(int)
        self.action_hidden_num = action_hidden_num
        self.trunk_hidden_size = trunk_hidden_size if isinstance(trunk_hidden_size, list) else (np.ones(trunk_hidden_num)*trunk_hidden_size).astype(int)
        self.trunk_hidden_num = trunk_hidden_num
        self.learning_rate = learning_rate
        self.lrdecayrate = lrdecayrate
        self.fstack = fstack

        # Build the model
        layers = [nn.Linear(self.state_size + self.action_size, self.trunk_hidden_size[0]), nn.ReLU()]
        for i in range(self.trunk_hidden_num - 1):
            layers.append(nn.Linear(self.trunk_hidden_size[i], self.trunk_hidden_size[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.trunk_hidden_size[-1], 1))           # single output for Q
        self.trunk = nn.Sequential(*layers)

    def forward(self, states, actions):
        sa = torch.cat([states, actions], dim=-1)
        x = self.trunk(sa)               # optional deeper trunk
        return x

