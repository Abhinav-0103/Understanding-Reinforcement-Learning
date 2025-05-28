import numpy as np

class BanditEnv :
    def __init__(self, k: int = 10) :
        self.k = k
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        self.optimal_action = np.argmax(self.q_star)
    
    def step(self, action: int) :
        q_star = self.q_star[action]
        return np.random.normal(loc=q_star, scale=1.0)