'''
    Montecarlo On-Policy solution. 

Gamma   Epsilon  Error
0.01     0.01    65.0 %
0.02     0.01    59.00000000000001 %
0.4      0.01    65.0 %
0.5      0.01    61.0 %
0.6      0.01    62.0 %
0.9      0.01    56.00000000000001 %
0.95     0.01    70.0 %
0.01     0.1     62.0 %
0.02     0.1     62.0 %
0.4      0.1     64.0 %
0.5      0.1     51.0 %
0.6      0.1     68.0 %
0.9      0.1     61.0 %
0.95     0.1     60.0 %
0.01     0.5     59.00000000000001 %
0.02     0.5     59.00000000000001 %
0.4      0.5     65.99999999999999 %
0.5      0.5     58.00000000000001 %
0.6      0.5     62.0 %
0.9      0.5     61.0 %
0.95     0.5     65.99999999999999 %
0.01     0.9     60.0 %
0.02     0.9     59.00000000000001 %
0.4      0.9     62.0 %
0.5      0.9     60.0 %
0.6      0.9     58.00000000000001 %
0.9      0.9     61.0 %
0.95     0.9     68.0 %
'''

import gym
import numpy as np

env = gym.make('Blackjack-v0')
posible_states = 704
posible_actions = 2


def policy(q, s, e_enabled):
    if e_enabled:
        return np.random.choice(range(posible_actions), p=q[s])
    return np.argmax(q[s])

def parse_state(s):
    return (1+ s[0]) * (1+ s[1]) + (1 if s[2] else 2)

def run_episode(q, render=False, e_enabled=True):
    state = parse_state(env.reset())
    states = []
    rewards = []
    actions = []
    win = False
    while True:
        action = policy(q, state, e_enabled)

        states.append(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        state = parse_state(state)
        if render:
            env.render()
        rewards.append(reward)

        if done:
            if reward == 1:
                win = True
            break
    return states, rewards, actions, win

def on_policy_improvement(episodes, gamma, epsilon):
    ## Define e-soft policy
    policy_data = np.array([epsilon/posible_actions for i in range(posible_actions - 1)] + [1 - epsilon + epsilon/posible_actions]) + np.zeros((posible_states, posible_actions))
    Q  = np.random.random((posible_states, posible_actions))
    returns = [[{ 'sum': 0, 'n': 0 } for i in range(posible_actions)] for j in range(posible_states)]
    
    for e in range(episodes):
        states, rewards, actions, _ = run_episode(policy_data)
        G = 0
        visited = {}
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            reward = rewards[t]
            action = actions[t]

            G = gamma*G + reward
            key = str(state) + '-' + str(action)
            if (not (key in visited)):
                visited[key] = True
                returns[state][action]['sum'] += G
                returns[state][action]['n'] += 1
                Q[state, action] = returns[state][action]['sum'] / returns[state][action]['n']
                A = np.argmax(Q[state])
                for a in range(posible_actions):
                    if a == A:
                        policy_data[state][a] = 1 - epsilon + epsilon/posible_actions
                    else:
                        policy_data[state][a] = epsilon/posible_actions
            else:
                break
    return policy_data

print('Gamma\tEpsilon\tError')

for epsilon in [0.5]:
    for gamma in [0.4]:
        data = on_policy_improvement(5000, gamma, epsilon)
        wins = 0
        for i in range(100):
            _, _, _, win = run_episode(data, False, False)
            if win:
                wins +=1
        print(gamma,'\t',epsilon,'\t', (1 - wins/100)*100, '%')