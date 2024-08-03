import torch
import torch.nn.functional as F
from random import randint, random
import numpy as np
from collections import deque
import time

from .abstract_agent import Agent

class DQNRNDAgent(Agent):
    def __init__(self, action_space, Q, predictor, target, optimizer, optimizer_RND, *,
                  gamma=0.99, eps=0.1, buffer_size=10000, batch_size=64, start_sampling=None, 
                  reward_factor=0.01, start_RND_reward = 200,
                  device=None, random_seed=None):
        
        super().__init__(action_space, random_seed)
        
        self.Q = Q.to(device)
        self.predictor = predictor.to(device)
        self.target = target.to(device)
        self.optimizer = optimizer
        self.optimizer_RND = optimizer_RND
        self.device = device
        
        self.gamma = gamma
        self.eps = eps
        self.buffer = Buffer(buffer_size, self.rng)
        self.batch_size = batch_size
        self.start_sampling = batch_size if start_sampling is None else start_sampling
        
        self.updates_counter = 0

        self.reward_factor = reward_factor
        self.start_RND_reward = start_RND_reward
        
        # Running mean and var for the RND normalization
        self.state_running_mean = np.array([0.,0.])
        self.state_running_mean2 = np.array([0.,0.])
        self.RND_reward_running_mean = 0
        self.RND_reward_running_mean2 = 0
        self.stats_counter = 0
        

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
    
    def observe(self, state, action, next_state, reward):
        # Compute RND_reward
        RND_reward = self.RND_reward(next_state)
        
        # Start to use RND_reward after start_RND_reward updates
        if self.updates_counter >= self.start_RND_reward:
            reward += self.reward_factor * RND_reward
        else:
            RND_reward = 0.
        
        # Append to buffer observed transition as torch tensors - No need to convert during sampling
        self.buffer.append((torch.tensor(state, dtype=torch.float32), 
                            torch.tensor(action, dtype=torch.int64), 
                            torch.tensor(next_state, dtype=torch.float32), 
                            torch.tensor(reward, dtype=torch.float32)))
        
        return RND_reward
    
    def update(self):
        self.updates_counter += 1
        
        if len(self.buffer) < self.start_sampling:
            return None, None
        
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
        self.optimizer.step()       # Update the weights
        
        # Eval target network with no gradients
        with torch.no_grad():   
            target_output = self.target(next_state)
        # Eval predictor network
        predictor_output = self.predictor(next_state)
        
        # Compute the MSE loss
        loss_RND = F.mse_loss(predictor_output, target_output, reduction='mean') # (N,) (N,) -> scalar

        # Optimize the RND network
        self.optimizer_RND.zero_grad()
        loss_RND.backward()
        self.optimizer_RND.step()
        
        return loss.item(), loss_RND.item()

    @torch.no_grad()
    def RND_reward(self, state):
        # Normalize the state
        norm_state = self.normalize_state(state)
        norm_state = torch.tensor(norm_state, dtype=torch.float32, device=self.device)
        
        # Compute loss_RND
        target_output = self.target(norm_state).item()
        predictor_output = self.predictor(norm_state).item()
        loss_RND = (predictor_output - target_output)**2
        
        # Normalize loss_RND before updating the statistics
        norm_loss_RND = self.normalize_loss_RND(loss_RND)
        
        # Update the statistics
        self.update_stats(state, loss_RND)
        
        # Clip the norm_loss_RND
        return np.clip(norm_loss_RND, -5, 5)
    
        
    def normalize_state(self, state):
        return (state - self.state_running_mean) / np.sqrt(self.state_running_var)
    
    def normalize_loss_RND(self, loss_RND):
        return (loss_RND - self.RND_reward_running_mean) / np.sqrt(self.RND_reward_running_var)
    
    def update_stats(self, state, loss_RND):
        self.state_running_mean = (self.stats_counter/(self.stats_counter+1)) * self.state_running_mean + (1/(self.stats_counter+1)) * state
        self.state_running_mean2 = (self.stats_counter/(self.stats_counter+1)) * self.state_running_mean2 + (1/(self.stats_counter+1)) * state**2
        self.RND_reward_running_mean = (self.stats_counter/(self.stats_counter+1)) * self.RND_reward_running_mean + (1/(self.stats_counter+1)) * loss_RND
        self.RND_reward_running_mean2 = (self.stats_counter/(self.stats_counter+1)) * self.RND_reward_running_mean2 + (1/(self.stats_counter+1)) * loss_RND**2
        
        self.stats_counter += 1

    @property
    def state_running_var(self):
        return np.array([1.,1.]) if self.stats_counter <= 1 else (self.state_running_mean2 - self.state_running_mean**2)
    @property
    def RND_reward_running_var(self):
        return 1. if self.stats_counter <= 1 else (self.RND_reward_running_mean2 - self.RND_reward_running_mean**2)
    
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