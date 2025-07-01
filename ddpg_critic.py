import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, state_hidden_size, state_hidden_num, action_hidden_size, action_hidden_num, trunk_hidden_size, trunk_hidden_num, lrdecayrate, learning_rate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.state_hidden_size = state_hidden_size if isinstance(state_hidden_size, list) else np.ones(state_hidden_num)*state_hidden_size
        self.state_hidden_num = state_hidden_num
        self.action_hidden_size = action_hidden_size if isinstance(action_hidden_size, list) else np.ones(action_hidden_num)*action_hidden_size
        self.action_hidden_num = action_hidden_num
        self.learning_rate = learning_rate
        self.lrdecayrate = lrdecayrate

        # Build the model
        # state layers 
        slayers = [nn.Linear(self.state_size, self.state_hidden_size[0]), nn.ReLU()]
        for i in range(self.state_hidden_num - 1):
            slayers.append(nn.Linear(self.state_hidden_size[i], self.state_hidden_size[i + 1]))
            slayers.append(nn.ReLU())
        self.state_net = nn.Sequential(*slayers)        # registers parameters
        # action layers 
        alayers = [nn.Linear(self.action_size, self.action_hidden_size[0]), nn.ReLU()]
        for i in range(self.action_hidden_num - 1):
            alayers.append(nn.Linear(self.action_hidden_size[i], self.action_hidden_size[i + 1]))
            alayers.append(nn.ReLU())
        self.action_net = nn.Sequential(*alayers)      # registers parameters
        # Both encoders must end with the same width for element-wise add
        fused_dim = state_hidden_size[-1]
        # Combine state and action layers
        if len(trunk_hidden_num) == 0:                # identity passthrough
            self.trunk = nn.Identity()
            last_dim = fused_dim
        else:
            t_layers, in_dim = [], fused_dim
            for h in trunk_hidden_size:
                t_layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            self.trunk = nn.Sequential(*t_layers)
            last_dim = trunk_hidden_size[-1]
        # Final output layer
        self.q_out = nn.Linear(last_dim, 1)           # single output for Q

        # optimizer
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-8)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lrdecayrate)  # Exponential decay
        #self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)  # Halve LR every 10 steps


    def forward(self, states, actions):
        s_feat = self.state_net(states)
        a_feat = self.action_net(actions)
        x = F.relu(s_feat + a_feat)     # fusion + non-linearity
        x = self.trunk(x)               # optional deeper trunk
        return self.q_out(x)
        
