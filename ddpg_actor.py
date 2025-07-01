import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import numpy as np
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size, hidden_num, lrdecayrate, learning_rate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size if isinstance(hidden_size, list) else np.ones(hidden_num)*hidden_size
        self.hidden_num = hidden_num

        # build the model
        layers = [nn.Linear(self.state_size, self.hidden_size[0]), nn.ReLU()]
        for i in range(self.hidden_num - 1):
            layers.append(nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_size[-1], self.action_size))

        # Creating the Sequential module
        self.linear_relu_stack = nn.Sequential(*layers)

        # optimizer
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-8)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lrdecayrate)  # Exponential decay
        #self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)  # Halve LR every 10 steps
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        probs  = torch.softmax(logits, dim=-1)  # simplex projection
        return probs
