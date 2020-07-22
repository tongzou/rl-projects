'''
    Montecarlo Off-Policy solution. 

Gamma   Error
0.01     60.0 %
0.1      59.00000000000001 %
0.25     59.599999999999994 %
0.5      61.4 %
0.75     61.7 %
0.9      58.099999999999994 %
0.99     60.9 %
'''

import gym
import numpy as np

env = gym.make('Blackjack-v0')
posible_states = 704
posible_actions = 2


def policy(q, s):
    if not isinstance(q[s], np.float64):
        return np.argmax(q[s])
    return int(q[s])

def parse_state(s):
    return (1+ s[0]) * (1+ s[1]) + (1 if s[2] else 2)


def run_episode(q, render=False):
    state = parse_state(env.reset())
    states = []
    rewards = []
    actions = []
    win = False
    while True:
        action = policy(q, state)

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

def test_policy(data):
    wins = 0
    for i in range(1000):
        _, _, _, win = run_episode(data, False)
        if win:
            wins +=1
    return wins/1000

def off_policy_improvement(episodes=1000, gamma=0.1):
    ## Define e-soft policy
    Q  = np.zeros((posible_states, posible_actions))
    policy_data = np.zeros((posible_states,))
    C = np.zeros((posible_states, posible_actions), dtype=np.int32)

    for e in range(episodes):
        b = np.random.random((posible_states, posible_actions))
        states, rewards, actions, _ = run_episode(b)
        G = 0
        W = 1
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            reward = rewards[t]
            action = actions[t]

            G = gamma*G + reward
            C[state, action] += 1
            Q[state, action] += (W / C[state, action]) * (G - Q[state, action])
            policy_data[state]  = np.argmax(Q[state])
            if action != policy_data[state]:
                break
                
            W = W * (1 / b[state, action])

    return policy_data

def print_policy(data):
    for s in range(0, 16, 4):
        text = ''
        for t in range(s, s+4):
            v = data[t]
            if v == 2:
                text += '> '
            if v == 0:
                text += '< '
            if v == 1:
                text += 'v '
            if v == 3:
                text += '> '
        print(text)



print('Gamma\tError')

for gamma in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
    data = off_policy_improvement(100000, gamma)
    # Error < 0.4 [0. 3. 0. 0. 0. 0. 2. 0. 3. 1. 0. 0. 0. 2. 2. 0.]
    win = test_policy(data)
    print(gamma,'\t', (1 - win)*100, '%')