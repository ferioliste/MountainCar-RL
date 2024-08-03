from abc import ABC, abstractmethod
from numpy.random import default_rng

class Agent(ABC):
    def __init__(self, action_space, random_seed):
        self.action_space = action_space
        self.rng = default_rng(random_seed)
    
    def reset_rng(seed = None):
        self.rng = default_rng(seed)
    
    @abstractmethod
    def observe(self, state, action, next_state, reward):
        raise NotImplementedError("This method must be overridden by the subclass")

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError("This method must be overridden by the subclass")

    @abstractmethod
    def update(self):
        raise NotImplementedError("This method must be overridden by the subclass")