import torch
import torch.nn.functional as F
from random import randint, random
import numpy as np
from collections import deque

from .abstract_agent import Agent

class DQNAgent(Agent):
    def __init__(self, action_space, Q, optimizer, *,
                  gamma=0.99, eps=0.1, buffer_size=10000, batch_size=64, start_sampling=None,
                  device=None, random_seed=None):
        
        super().__init__(action_space, random_seed)
        
        self.Q = Q.to(device)
        self.optimizer = optimizer
        self.device = device
        
        self.gamma = gamma
        self.eps = eps
        self.buffer = Buffer(buffer_size, self.rng)
        self.batch_size = batch_size
        self.start_sampling = batch_size if start_sampling is None else start_sampling
        
        self.updates_counter = 0
        
    
    def observe(self, state, action, next_state, reward):
        # Append to buffer observed transition as torch tensors - No need to convert during sampling
        self.buffer.append((torch.tensor(state, dtype=torch.float32), 
                            torch.tensor(action, dtype=torch.int64), 
                            torch.tensor(next_state, dtype=torch.float32), 
                            torch.tensor(reward, dtype=torch.float32)))

    def select_action(self, state, eps = None):
        # Epsilon-greedy policy
        if self.rng.random() < self.eps(self.updates_counter) if eps is None else eps:
            # Exploration - Random Action
            return self.rng.choice(self.action_space)
        else:
            # Greedy action - no gradient computation
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                q_values = self.Q(state_tensor)
                max_q_value = torch.max(q_values)
                max_indices = torch.nonzero(q_values == max_q_value).flatten().cpu().numpy()
                action = self.action_space[self.rng.choice(max_indices)]
            return action
    
    def update(self):
        self.updates_counter += 1
        
        if len(self.buffer) < self.start_sampling:
            return None
        
        # Sample from buffer
        state, action, next_state, reward = self.buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        
        # Compute the Q-values for the current state and the next state
        q_values = self.Q(state).gather(1, action.view(-1, 1)).squeeze(dim=-1) # (N, 3) -> (N, 1) -> (N,)
        
        # Compute next_state maximal Q-values with no gradients
        with torch.no_grad():
            next_q_values = self.Q(next_state).max(dim=1)[0] # (N, 3) -> (N,)
            # Compute the target Q-values
            target_q_values = reward + self.gamma * next_q_values

        # Compute the MSE loss
        loss = F.mse_loss(q_values, target_q_values, reduction='mean') # (N,) (N,) -> scalar

        # Optimize the Q-network
        self.optimizer.zero_grad()  # Clear the gradients
        loss.backward()             # Compute the gradients
        # torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 0.01) # Clip the gradients

        self.optimizer.step()       # Update the weights

        return loss.item()
    
    @property
    def eps(self):
        return self._eps
    @eps.setter
    def eps(self, eps):
        if callable(eps):
            self._eps = eps
        else:
            self._eps = lambda t: eps


# Implementing a replay buffer like this is faster than using a naive list
class Buffer:
    def __init__(self, buffer_size, rng):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.rng = rng

    def append(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # For speed we sample indeces instead of the list itself
        batch_idxs = self.rng.integers(len(self), size=batch_size)
        # Stack the tensors into batched form
        batch = [self.buffer[idx] for idx in batch_idxs]
        state, action, next_state, reward = map(torch.stack, zip(*batch))
        action = action.squeeze(dim=-1)
        reward = reward.squeeze(dim=-1)
        return state, action, next_state, reward # (N, 2), (N,), (N, 2), (N,)
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]