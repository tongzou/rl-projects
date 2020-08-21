'''
    n-step TD to estimate V

'''

import gym
import numpy as np
env = gym.make('FrozenLake-v0', is_slippery=True)
posible_states = env.nS
posible_actions = env.nA

def policy(q, s, e_enabled):
    if e_enabled:
        return np.random.choice(range(len(q[s])), p=q[s])
    return np.argmax(q[s])


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


V = s_step_improvement(1000)
print('V', V)