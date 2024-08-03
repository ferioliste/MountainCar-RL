from .abstract_agent import Agent
import numpy as np

class RandomAgent(Agent):
    def __init__(self, action_space, random_seed = None):
        super().__init__(action_space, random_seed)
    
    def observe(self, state, action, next_state, reward):
        # RandomAgent does NOT learn
        pass

    def select_action(self, state):
        # Selects a random action from the available action space.
        return self.rng.choice(self.action_space)

    def update(self):
        # RandomAgent does NOT update its policy
        pass