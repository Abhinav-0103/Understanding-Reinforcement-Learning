import numpy as np

class UCBAgent :
    def __init__(self, k: int, c: float) :
        self.k = k
        self.c = c
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.time = 0
    
    def select_action(self) -> int :
        self.time += 1
        ucb_values = np.zeros(self.k)

        for a in range(self.k) :
            if self.N[a] == 0 :
                return a # Exploration of All Arms
            bonus = self.c * np.sqrt(np.log(self.time) / self.N[a])
            ucb_values[a] = self.Q[a] + bonus
        
        return np.argmax(ucb_values)
    
    def update(self, action: int, reward: float) :
        self.N[action] += 1
        alpha = 1 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])