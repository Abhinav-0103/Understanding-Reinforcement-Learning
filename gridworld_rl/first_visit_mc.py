import numpy as np
from collections import defaultdict

def first_visit_mc(env, policy, episodes=5000, gamma=0.9) :
    returns = defaultdict(list)
    V = defaultdict(float)

    for ep in range(episodes) :
        state = env.reset()
        episode = []

        done = False

        while not done :
            action = policy(state)
            next_state, reward, done = env.step(action)
            episode.append((state, reward))
            state = next_state
        
        G = 0
        visited = set()
        for t in reversed(range(len(episode))) :
            s, r = episode[t]
            G = gamma * G + r

            if s not in visited :
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)
    
    return V

def random_policy(state) :
    return np.random.choice(["up", "right", "down", "left"])