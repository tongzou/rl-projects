'''
    Montecarlo Off-Policy solution. 
    |Iterations|Score|slippery|
    |10000     |1    |False   |
    |10000     |0.519|True   |
'''

import gym
import numpy as np

env = gym.make('FrozenLake-v0', is_slippery=True)
posible_states = env.nS
posible_actions = env.nA


def policy(q, s):
    if not isinstance(q[s], np.float64):
        return np.argmax(q[s])
    return q[s]


def run_episode(q, render=False):
    state = env.reset()
    states = []
    rewards = []
    actions = []
    win = False
    while True:
        action = policy(q, state)

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

def test_policy(data):
    wins = 0
    for i in range(1000):
        _, _, _, win = run_episode(data, False)
        if win:
            wins +=1
    return wins/1000

def off_policy_improvement(episodes=1000, gamma=0.1, allow_test_policy = True):
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

        if allow_test_policy and e % (episodes / 10) == 0:
            print(e, ' -->', test_policy(policy_data))
    if allow_test_policy:
        print(e, ' -->', test_policy(policy_data))
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


data = off_policy_improvement(1000000, gamma=0.1, allow_test_policy = True)
print_policy(data)