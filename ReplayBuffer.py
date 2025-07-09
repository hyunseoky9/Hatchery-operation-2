import random
from collections import namedtuple, deque
import torch
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.
        buffer for DDPG/TD3 algorithms."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(
            torch.as_tensor(state,      dtype=torch.float32).cpu(),
            torch.as_tensor(action,     dtype=torch.float32).cpu(),
            torch.as_tensor(reward,     dtype=torch.float32).cpu(),
            torch.as_tensor(next_state, dtype=torch.float32).cpu(),
            torch.as_tensor(done,       dtype=torch.float32).cpu()
        )
        self.memory.append(e)

        #e = self.experience(state, action, reward, next_state, done)
        #self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        batch = random.sample(self.memory, k=self.batch_size)
        
        states      = torch.stack([e.state       for e in batch], dim=0)
        actions     = torch.stack([e.action      for e in batch], dim=0)
        rewards     = torch.stack([e.reward      for e in batch], dim=0).unsqueeze(-1)
        dones       = torch.stack([e.done        for e in batch], dim=0).unsqueeze(-1)
        next_states = torch.stack([e.next_state  for e in batch], dim=0)
        
        return states, actions, rewards, dones, next_states

        #return random.sample(self.memory, k=batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)