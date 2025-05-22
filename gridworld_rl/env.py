import numpy as np

class GridWorld :
    def __init__(self, size=3, terminal_states=[(0,0), (2,2)]) :
        self.size = size
        self.terminal_states = terminal_states
        self.actions = ["up", "right", "down", "left"]
        self.reset()

    def step(self, action) :
        if self.state in self.terminal_states :
            return self.state, 0, True
        
        r, c = self.state 
        if action == "up" : r = max(0, r - 1)
        if action == "right" : c = min(self.size - 1, c + 1)
        if action == "down" : r = min(self.size - 1, r + 1)
        if action == "left" : c = max(0, c - 1)

        self.state = (r, c)
        reward = 0 if self.state in self.terminal_states else -1
        done = self.state in self.terminal_states

        return self.state, reward, done

    def reset(self) :
        while True :
            r, c = np.random.randint(0, self.size, 2)
            if (r, c) not in self.terminal_states :
                break

        self.state = (r, c)
        return self.state