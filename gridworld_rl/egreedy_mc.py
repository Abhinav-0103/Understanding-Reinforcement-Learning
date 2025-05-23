import numpy as np
from collections import defaultdict

def egreedy_policy(Q, state, actions, epsilon) :
    if np.random.rand() < epsilon :
        return np.random.choice(actions)
    else :
        q_values = [Q[(state, a)] for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return np.random.choice(best_actions)

def mc_egreedy(env, episodes, gamma=0.9, epsilon=0.2, decay=0.999) :
    Q = defaultdict(float)
    V = defaultdict(float)
    returns = defaultdict(list)
    policy = defaultdict(str)
    actions = env.actions

    for ep in range(episodes) :
        state = env.reset()
        episode = []

        done = False

        while not done :
            action = egreedy_policy(Q, state, actions, epsilon)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited_q = set()

        for t in reversed(range(len(episode))) :
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited_q :
                returns[(s, a)].append(G)
                Q[(s, a)] = np.mean(returns[(s, a)])
                visited_q.add((s, a))
        
    for state in [(r, c) for r in range(env.size) for c in range(env.size)] :
        if state in env.terminal_states :
            continue
        
        q_values = [Q[(state, a)] for a in actions]
        best_action = actions[np.argmax(q_values)]
        policy[state] = best_action

    return Q, policy