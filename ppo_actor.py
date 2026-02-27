import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, CosineAnnealingLR

class Actor_beta_dirichlet(nn.Module):
    def __init__(self, input_dims, n_actions, 
                 hidden_size, hidden_num, 
                    lrdecayrate, lr,
                    min_lr, lrdecaytype,
                    scheduler_info, device):
        
        super(Actor_beta_dirichlet, self).__init__()

        # build the model
        layers = [nn.Linear(input_dims, hidden_size[0]), nn.ReLU()]
        for i in range(hidden_num - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[-1], n_actions))

        # Creating the Sequential module
        self.actor = nn.Sequential(*layers)
        # set up optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-8)
        self.device = T.device(device)
        self.to(self.device)
        # set up learning rate scheduler
        if lrdecaytype == 'exp':
            self.scheduler = ExponentialLR(self.optimizer, gamma=lrdecayrate)  # Exponential decay
        elif lrdecaytype == 'multistep':
            self.scheduler = MultiStepLR(self.optimizer, milestones=scheduler_info['lr_drop_ep'],
                                          gamma=scheduler_info['lr_drop_gamma'])

    def forward(self, state):
        x = self.actor(state)
        # process beta dist paramgeters
        a = F.softplus(x[:, 0]) + 1e-3
        b = F.softplus(x[:, 1]) + 1e-3
        a = T.clamp(a, max=1e3)
        b = T.clamp(b, max=1e3)
        # process dirichlet dist parameters
        c = F.softplus(x[:, 2:]) + 1e-3
        c = T.clamp(c, max=1e3)
        # merge a,b, and c back together
        x = T.cat((a.unsqueeze(1), b.unsqueeze(1), c), dim=1)
        return x
