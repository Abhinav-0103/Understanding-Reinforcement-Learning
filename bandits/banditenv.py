import numpy as np

class Bandit :
    def __init__(self, k: int=10) :
        self.k = k
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=k)
        self.optimal_action = np.argmax(self.q_star)
    
    def step(self, action: int) -> float :
        true_value = self.q_star[action]
        return np.random.normal(loc=true_value, scale=1.0)