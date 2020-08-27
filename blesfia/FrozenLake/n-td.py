'''
    n-step TD to estimate V

'''

import gym
import numpy as np
env = gym.make('FrozenLake-v0', is_slippery=True)
posible_states = env.nS
posible_actions = env.nA

def policy(policy, state):
    if (isinstance(policy[state], list)):
      return np.random.choice(range(posible_actions), p=policy[state])
    return policy[state]

def get_policy(Q = None, epsilon_soft = True, epsilon = 0.1):
    # if Q is None, return a random policy
    if (Q is None):
      return [[1/posible_actions for j in range(posible_actions)] for i in range(posible_states)]

    if (epsilon_soft):
      policy = [[1/posible_actions for j in range(posible_actions)] for i in range(posible_states)]
      for state in range(posible_states):
        q = np.array(Q[state])
        max = q.max()
        max_index = np.where(q == max)[0]
        for action in range(posible_actions):
          if Q[state][action] == max:
            policy[state][action] = (1 - (posible_actions - len(max_index)) * epsilon / posible_actions) / len(max_index)
          else:
            policy[state][action] = epsilon / posible_actions
    else:
      policy = [0 for i in range(posible_states)]
      for state in range(posible_states):
        q = np.array(Q[state])
        max = q.max()
        max_index = np.where(q == max)[0]
        policy[state] = [1/len(max_index) if i in max_index else 0 for i in range(posible_actions)]
        # policy[state] = np.argmax(Q[state])
    return policy


def s_step_improvement(episodes, n=2, step_size=0.01, gamma=0.9, alpha=0.1):
    # Random policy
    V  = np.zeros((posible_states))
    # Load model
    for e in range(episodes):
        S = env.reset()
        T = 1000
        t = 0
        tao = 0
        rewards = [0]
        states = [S]
        while True:
            if t < T: # Still playing
                Q  = np.random.random((posible_states, posible_actions))
                S, R, done, _ = env.step(policy(Q, S, False))
                rewards.append(R)
                states.append(S)
                if done:
                    T = t + 1
            tao = t -n -1
            if tao >= 0: # Time to update!
                G = np.sum([(gamma**(i-tao-1))*rewards[i] for i in range(tao + 1, min(tao + n, T) + 1)])
                if tao + n < T:
                    G = G + gamma**n * V[states[tao + n]]
                V[states[tao]] += alpha*(G - V[states[tao]])
            t += 1
            if tao == T - 1:
                break
    return V

def sarsa_improvement(episodes, n=2, step_size=0.01, gamma=0.9, alpha=0.1):
    Q = np.zeros((posible_states, posible_actions))
    policy_data = get_policy(Q)
    # Load model
    for e in range(episodes):
        states = []
        actions = []
        rewards = [0]
        S = env.reset()
        states.append(S)
        A = policy(policy_data, S)
        actions.append(A)
        

        T = 1000
        t = 0
        tao = 0
        while True:
            if t < T: # Still playing
                S, R, done, _ = env.step(A)
                rewards.append(R)
                states.append(S)

                if done:
                    T = t + 1
                else:
                    A = policy(policy_data, S)
                    actions.append(A)
            tao = t - n +1
            if tao >= 0: # Time to update!
                G = np.sum([(gamma**(i-tao-1))*rewards[i] for i in range(tao + 1, min(tao + n, T) + 1)])
                if tao + n < T:
                    G = G + gamma**n * Q[states[tao + n], actions[tao+n]]
                Q[states[tao], actions[tao]] += alpha*(G - Q[states[tao], actions[tao]])
                policy_data = get_policy(Q)
            t += 1
            if tao == T - 1:
                break
    return Q, policy_data

Q, _ = sarsa_improvement(5000)
policy_data = get_policy(Q, False)
print('Q', Q)

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

win = test_policy(policy_data)
print(win)