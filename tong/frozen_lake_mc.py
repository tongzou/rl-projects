# Colors for console
import numpy as np
import gym

env = gym.make('FrozenLake-v0')

def get_action(policy, state):
    if (isinstance(policy[state], list)):
        return np.random.choice(range(env.nA), p=policy[state])
    return policy[state]

def get_policy(Q = None, epsilon_soft = True, epsilon = 0.1):
    # if Q is None, return a random policy
    if (Q is None):
        return [[1/env.nA for j in range(env.nA)] for i in range(env.nS)]

    if (epsilon_soft):
        policy = [[1/env.nA for j in range(env.nA)] for i in range(env.nS)]
        for state in range(env.nS):
            best_action = np.argmax(Q[state])
            for action in range(env.nA):
                if action == best_action:
                    policy[state][action] = 1 - epsilon + epsilon/env.nA
                else:
                    policy[state][action] = epsilon/env.nA
    else:
        policy = [0 for i in range(env.nS)]
        for state in range(env.nS):
            q = np.array(Q[state])
            max = q.max()
            max_index = np.where(q == max)[0]
            policy[state] = [1/len(max_index) if i in max_index else 0 for i in range(env.nA)]
            # policy[state] = np.argmax(Q[state])
    return policy

def print_policy(policy):
    for s in range(0, 16, 4):
        text = ''
        for t in range(s, s+4):
            v = np.argmax(policy[t]) 
            if v == 2:
                text += '> '
            if v == 0:
                text += '< '
            if v == 1:
                text += 'v '
            if v == 3:
                text += '> '
        print(text)

def argmax_rand(arr):
    return np.random.choice(np.flatnonzero(arr == np.max(arr)))

def argmax(arr):
    q = np.array(arr)
    max = q.max()
    max_index = np.where(q == max)[0]
    return [1/len(max_index) if i in max_index else 0 for i in range(env.nA)]

def run_episode(policy, render = False):
    state = env.reset()
    states = []
    rewards = []
    actions = []
    win = False
    while True:
        action = get_action(policy, state)
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

def on_policy_improvement(episodes = 1000, first_visit = True, gamma = 0.9, epsilon = 0.1):
    ## initialize random policy    
    Q = np.random.random((env.nS, env.nA))
    policy = get_policy(Q, True, epsilon)
    returns = [[{ 'sum': 0, 'n': 0 } for j in range(env.nA)] for i in range(env.nS)]

    for e in range(episodes):
        states, rewards, actions, _ = run_episode(policy)
        G = 0
        visited = {}
        if (first_visit):
            for t in range(len(states)): # initialze visited
                state = states[t]
                reward = rewards[t]
                action = actions[t]
                key = str(state) + '-' + str(action)
                if (not key in visited):
                    visited[key] = t

        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            reward = rewards[t]
            action = actions[t]
            key = str(state) + '-' + str(action)

            G = gamma * G + reward
            if (not first_visit or visited[key] == t):
                returns[state][action]['sum'] += G
                returns[state][action]['n'] += 1
                Q[state, action] = returns[state][action]['sum'] / returns[state][action]['n']
                policy = get_policy(Q, True, epsilon)
    print(returns)
    return Q

def off_policy_improvement(episodes = 1000, gamma = 0.9, epsilon = 0.1):
    ## initialize random policy
    bpolicy = get_policy()
    Q = np.zeros((env.nS, env.nA))
    C = [[0 for j in range(env.nA)] for i in range(env.nS)]
    tpolicy = get_policy(Q)

    for e in range(episodes):
        states, rewards, actions, _ = run_episode(bpolicy)
        G = 0
        W = 1

        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            reward = rewards[t]
            action = actions[t]

            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W/C[state][action]) * (G - Q[state][action])
            # tpolicy[state] = argmax(Q[state])
            tpolicy = get_policy(Q)
            if (tpolicy[state][action] == 0):
                break
            W = W / bpolicy[state][action]
    return Q


def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps=0
        while True:
            action = get_action(policy, observation)
            observation, reward, done, _ = env.step(action)
            steps+=1
            if done and reward == 1:
                # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    print('----------------------------------------------')
    if (misses != episodes):
        print('You took an average of {:.0f} steps to get the frisbee'.format(np.mean(steps_list)))
    print('And you fell in the hole {:.2f} % of the times'.format((misses/episodes) * 100))
    print('----------------------------------------------')

Q = on_policy_improvement(10000, True, 0.99, 0.1)
# Q = off_policy_improvement(20000, 0.99)
print(Q)
policy = get_policy(Q, False)
print_policy(policy)
get_score(env, policy)