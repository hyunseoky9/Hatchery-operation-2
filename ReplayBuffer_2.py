import random
from collections import deque
import torch

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples for DDPG/TD3."""

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # Track the last unpaired t=0 transition (by index in the deque)
        self._pending_t0_idx = None

    # ### NEW: call at the start of each episode
    def start_episode(self):
        self._pending_t0_idx = None

    # ### CHANGED: add(t=...)
    def add(self, state, action, reward, next_state, done, t):
        """Add a new experience to memory and, if t==1, link it to the previous t==0."""
        item = {
            "state":      torch.as_tensor(state,      dtype=torch.float32).cpu(),
            "action":     torch.as_tensor(action,     dtype=torch.float32).cpu(),
            "reward":     torch.as_tensor(reward,     dtype=torch.float32).cpu(),
            "next_state": torch.as_tensor(next_state, dtype=torch.float32).cpu(),
            "done":       torch.as_tensor(done,       dtype=torch.float32).cpu(),
            "t":          torch.as_tensor([t],        dtype=torch.float32).cpu(),   # shape [1]
            # pair fields (filled when we later observe the t=1 step)
            "r_next":     torch.zeros(1, dtype=torch.float32),   # [1]
            "s2":         torch.zeros_like(torch.as_tensor(state, dtype=torch.float32)),  # [state_dim]
            "done_next":  torch.zeros(1, dtype=torch.float32),   # [1]
            "has_pair":   torch.zeros(1, dtype=torch.float32),   # [1] {0,1}
        }

        # Append
        self.memory.append(item)
        idx = len(self.memory) - 1  # current item index (in deque view)

        # Pairing logic
        if int(t) == 0:
            # Remember this t=0 to be paired with the very next t=1
            self._pending_t0_idx = idx
        else:
            # t==1; if the last was t=0, link it
            if self._pending_t0_idx is not None:
                prev = self.memory[self._pending_t0_idx]
                prev["r_next"]    = torch.as_tensor(item["reward"],    dtype=torch.float32).view(1)
                prev["s2"]        = item["next_state"].clone()
                prev["done_next"] = torch.as_tensor(item["done"],      dtype=torch.float32).view(1)
                prev["has_pair"]  = torch.ones(1, dtype=torch.float32)
                self._pending_t0_idx = None

        if bool(done):
            # Episode ended; clear pending pointer
            self.start_episode()

    # ### CHANGED: return the extra fields for mixed 1-/2-step targets
    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)

        states      = torch.stack([e["state"]      for e in batch], dim=0)
        actions     = torch.stack([e["action"]     for e in batch], dim=0)
        rewards     = torch.stack([e["reward"]     for e in batch], dim=0).unsqueeze(-1)
        dones       = torch.stack([e["done"]       for e in batch], dim=0).unsqueeze(-1)
        next_states = torch.stack([e["next_state"] for e in batch], dim=0)

        # ### NEW (n=2)
        tbits       = torch.stack([e["t"]         for e in batch], dim=0)          # [B,1,1] -> squeeze later
        r_next      = torch.stack([e["r_next"]    for e in batch], dim=0)          # [B,1]
        s2          = torch.stack([e["s2"]        for e in batch], dim=0)          # [B,state_dim]
        done_next   = torch.stack([e["done_next"] for e in batch], dim=0)          # [B,1]
        has_pair    = torch.stack([e["has_pair"]  for e in batch], dim=0)          # [B,1]

        # squeeze t to [B,1]
        tbits = tbits.squeeze(1)

        return states, actions, rewards, dones, next_states, tbits, has_pair, r_next, s2, done_next

    def __len__(self):
        return len(self.memory)