# Dyna Agent class
from .abstract_agent import Agent
import numpy as np

class DynaAgent(Agent):
    def __init__(self, action_space, state_space=np.array([[-1.2, -0.07], [0.6, 0.07]]), discr_step=np.array([0.025, 0.0005]), k=1,
                   gamma=0.99, eps=0.9, buffer_size=10000, start_sampling=1000, random_seed=None):
        
        super().__init__(action_space, random_seed)
        
        self.state_space = state_space
        self.discr_step = discr_step
        self.k = k
        
        self.gamma = gamma
        self.eps = eps
        self.buffer = DynaBuffer(self.rng, buffer_size)
        self.start_sampling = start_sampling
        
        self.num_actions = len(action_space)
        
        # Initialize probability, reward, action, matrices
        self.num_states_vect = np.round(np.diff(state_space, axis=0).squeeze(axis=0) / discr_step).astype(int)  # Number of states per state dimension (pos - vel)
        self.num_states = int(np.prod(self.num_states_vect)) # Total number of states

        # Probability Matrix
        self.transition_counter = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float64)
        self.state_action_counter = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.total_reward = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        
        self.P = self._P(self)
        self.R = self._R(self)

        # Q-values Matrix
        self.Q = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        
        self.updates_counter = 0
    
    def select_action(self, state, eps = None):
        # Epsilon-greedy policy
        if self.rng.random() < self.eps(self.updates_counter) if eps is None else eps:
            # Exploration - Random Action
            return self.rng.choice(self.action_space)
        else:
            # Greedy action 
            state_idx = self.state_to_index(state)
            action = self.rng.choice(np.where(self.Q[state_idx, :] == np.max(self.Q[state_idx, :]))[0])
            return action
    
    def observe(self, state, action, next_state, reward):
        # Add transition to buffer
        state_idx = self.state_to_index(state)
        self.buffer.append(state_idx, action)
    
    def update(self, state, action, next_state, reward):
        self.updates_counter += 1
        
        # Save state-action pairs that were already encountered
        # Convert states into linear indices
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)

        # Update estimate of P , R
        self.transition_counter[state_idx, action, next_state_idx] += 1
        self.state_action_counter[state_idx, action] += 1
        self.total_reward[state_idx, action] += reward

        # Update Q-values: (1, 1, num_states) * (num_states,) = (1, )
        update_step = self.Q[state_idx, action]
        self.Q[state_idx, action] = reward + self.gamma * (self.P[state_idx, action, :] * np.max(self.Q, axis=1)).sum()
        update_step = self.Q[state_idx, action] - update_step
        
        # Sample k transitions from buffer
        if len(self.buffer) >= self.start_sampling:
            transitions = self.buffer.sample(self.k) # index form (k, 2)
            for i in range(self.k):
                state_idx = transitions[i, 0]
                action = transitions[i, 1]
                # Update Q-values
                self.Q[state_idx, action] = self.R[state_idx, action] + self.gamma * (self.P[state_idx, action, :] * np.max(self.Q, axis=1)).sum()
        
        return update_step

    def state_to_index(self, state):
        idx_vect = np.floor_divide(np.clip(state, self.state_space[0], self.state_space[1]) - self.state_space[0, :], self.discr_step)
        state_idx = int(idx_vect[0] * self.num_states_vect[1] + idx_vect[1])
        return state_idx
    
    @property
    def eps(self):
        return self._eps
    @eps.setter
    def eps(self, eps):
        if callable(eps):
            self._eps = eps
        else:
            self._eps = lambda t: eps
    
    class _P:
        def __init__(self, parent):
            self._parent = parent
        def __getitem__(self, index):
            (i,j,k) = index
            return self._parent.transition_counter[i,j,k]/self._parent.state_action_counter[i,j]
    
    class _R:
        def __init__(self, parent):
            self._parent = parent
        def __getitem__(self, index):
            (i,j) = index
            return self._parent.total_reward[i,j]/self._parent.state_action_counter[i,j]
            
class DynaBuffer:
    def __init__(self, rng, buffer_size=10000):
        self.buffer_size = buffer_size
        # First entry discrete state bin index (linearized), second action index
        self.buffer = np.empty((self.buffer_size, 2), dtype=np.int64) # (N, 2) 
        self.index = 0
        self.size = 0
        self.rng = rng
    
    def append(self, state_idx, action):
        self.buffer[self.index, 0] = state_idx 
        self.buffer[self.index, 1] = action
        self.index = (self.index + 1) % self.buffer_size
        self.size = max(self.size, self.index)

    def sample(self, k):
        idx_list = self.rng.integers(self.size, size=k)
        return self.buffer[idx_list, :]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.buffer[idx]