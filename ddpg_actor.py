import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
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
        self.body = nn.Sequential(*layers)

        # optimizer
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        #self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)  # Halve LR every 10 steps
    
    def forward(self, x):
        logits = self.body(x)
        probs  = torch.softmax(logits, dim=-1)  # simplex projection
        return probs

    def act(self, state, ou_process, device="cpu"):
        with torch.no_grad():
            # 1. tensor-ise state and add batch dim
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = self.body(s).squeeze(0) # shape [K]
            # 3. OU noise in logit space
            noise  = torch.as_tensor(ou_process.sample(),
                                    dtype=logits.dtype,
                                    device=logits.device)
            noisy_logits = logits + noise
            # 4. project onto simplex
            action = torch.softmax(noisy_logits, dim=-1) # stays on delta simplex
        return action.cpu().numpy()