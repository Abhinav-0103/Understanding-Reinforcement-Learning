import numpy as np

class EpsilonGreedyAgent :
    def __init__(self, k: int, epsilon: float) :
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
    
    def select_action(self) -> int :
        if np.random.rand() < self.epsilon :  # Exploration
            return np.random.randint(self.k)
        else :  # Exploitation
            return np.argmax(self.Q)
        
    def update(self, action: int, reward: float) :
        self.N[action] += 1
        alpha = 1 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])