'''
    Montecarlo On-Policy solution. 
    |Iterations|Score|slippery|
    |10000     |0.772|True   |

NOT SLippery (Easy) (5000 episodes)
Gamma   Epsilon Error
0.01     0.01    100.0 %
0.01     0.1     0.0 %
0.01     0.5     0.0 %
0.01     0.9     0.0 %
0.1      0.01    100.0 %
0.1      0.1     0.0 %
0.1      0.5     0.0 %
0.1      0.9     0.0 %
0.5      0.01    100.0 %
0.5      0.1     0.0 %
0.5      0.5     0.0 %
0.5      0.9     0.0 %
0.9      0.01    100.0 %
0.9      0.1     100.0 %
0.9      0.5     0.0 %
0.9      0.9     0.0 %


Slippery
Gamma   Epsilon Error
0.01     0.5     60.0 %
0.1      0.01    40.0 %
0.1      0.1     90.0 %
0.1      0.25    90.0 %
0.1      0.6     70.0 %
0.1      0.9     80.0 %
0.5      0.01    80.0 %
0.5      0.1     19.999999999999996 %
0.5      0.25    40.0 %
0.5      0.5     30.000000000000004 %
0.5      0.75    80.0 %
0.5      0.6     90.0 %
0.5      0.9     70.0 %
0.9      0.5     70.0 %
0.9      0.9     40.0 %
'''

import gym
import numpy as np

env = gym.make('FrozenLake-v0', is_slippery=True)
posible_states = env.nS
posible_actions = env.nA


def policy(q, s, e_enabled):
    if e_enabled:
        return np.random.choice(range(posible_actions), p=q[s])
    return np.argmax(q[s])

def run_episode(q, render=False, e_enabled=True):
    state = env.reset()
    states = []
    rewards = []
    actions = []
    win = False
    while True:
        action = policy(q, state, e_enabled)

        states.append(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
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

def print_policy(data):
    for s in range(0, 16, 4):
        text = ''
        for t in range(s, s+4):
            v = np.argmax(data[t]) 
            if v == 2:
                text += '> '
            if v == 0:
                text += '< '
            if v == 1:
                text += 'v '
            if v == 3:
                text += '> '
        print(text)

print('Gamma\tEpsilon\tError')

for gamma in [0.01, 0.1, 0.5, 0.9]:
    for epsilon in [0.01, 0.1, 0.5, 0.9]:
        data = on_policy_improvement(5000, gamma, epsilon)
        # Error < 0.3,  data = [[0.925 0.025 0.025 0.025], [0.925 0.025 0.025 0.025], [0.925 0.025 0.025 0.025], [0.925 0.025 0.025 0.025], [0.925 0.025 0.025 0.025], [0.025 0.025 0.025 0.925], [0.025 0.025 0.925 0.025], [0.025 0.025 0.025 0.925], [0.025 0.025 0.025 0.925], [0.025 0.925 0.025 0.025], [0.925 0.025 0.025 0.025], [0.025 0.025 0.025 0.925], [0.025 0.025 0.025 0.925], [0.025 0.025 0.925 0.025], [0.025 0.925 0.025 0.025],[0.025 0.025 0.025 0.925]]
        wins = 0
        for i in range(100):
            _, _, _, win = run_episode(data, False, False)
            if win:
                wins +=1
        if wins/100 > 0.5:
            print(data)
        print(gamma,'\t',epsilon,'\t', (1 - wins/100)*100, '%')