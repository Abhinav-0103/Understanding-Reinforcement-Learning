import numpy as np
from collections import defaultdict

def sarsa_control(env, policy, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2, decay=0.999) :
    Q = defaultdict(float)
    actions = env.actions 
    snapshots = []

    for ep in range(episodes) :
        state = env.reset()
        action = policy(Q, state, actions, epsilon)

        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = policy(Q, next_state, actions, epsilon)

            td_target = reward + gamma * Q[(next_state, next_action)]
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state
            action = next_action
        
    policy = {}
    for state in [(r,c) for r in range(env.size) for c in range(env.size) ]:
        r, c = state
        if state in env.terminal_states:
            continue
        q_vals = [Q[(state, a)] for a in actions]
        best_action = actions[np.argmax(q_vals)]
        policy[state] = best_action

    return Q, policy, snapshots