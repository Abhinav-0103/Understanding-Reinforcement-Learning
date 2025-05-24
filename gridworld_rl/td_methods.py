import numpy as np
from collections import defaultdict

def td0_prediction(env, policy, episodes=10000, alpha=0.1, gamma=0.9) :
    V = defaultdict(float)
    delta_trace = []

    for ep in range(episodes) :
        state = env.reset()
        done = False
        delta = 0

        while not done :
            old_v = V[state]
            action = policy(state)
            next_state, reward, done = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            delta = max(delta, abs(V[state] - old_v))
            state = next_state
        delta_trace.append(delta)
    return V, delta_trace