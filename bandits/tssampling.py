import numpy as np

class ThompsonSamplingAgent :
    def __init__(self, k: int) :
        self.k = k
        self.alpha = np.ones(k)
        self.beta = np.ones(k)
    
    def select_action(self) :
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, action: int, reward: int) :
        if reward == 1:
            self.alpha[action] += 1
        else :
            self.beta[action] += 1

class BernoulliBandit :
    def __init__(self, k: int) :
        self.k = k
        self.q_star = np.random.rand(k)
        self.optimal_action = np.argmax(self.q_star)
    
    def step(self, action: int) -> int :
        return np.random.rand() < self.q_star[action]