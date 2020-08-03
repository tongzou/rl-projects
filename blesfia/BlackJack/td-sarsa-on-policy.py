'''
    Sarsa On-Policy solution. 

99900: 34% win rate (66% Error rate)
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

def sarsa_improvement(episodes, epsilon=0.1, step_size=0.01, gamma=0.9):
    ## Define e-soft policy
    policy_data = [np.array([1/posible_actions for i in range(posible_actions)]) for i in range(posible_states)]
    Q  = np.zeros((posible_states, posible_actions))
    
    for e in range(episodes):
        S = env.reset()
        S = parse_state(S)
        A = policy(policy_data, S, True)
        while True:
            S2, R, done, _ = env.step(A)
            S2 = parse_state(S2)
            A2 = policy(policy_data, S2, True)
            Q[S, A] += step_size*(R + (gamma*Q[S2, A2]) - Q[S,A])

            # Generate E-policy
            if np.max(Q[S]) != 0:
                for a in range(posible_actions):
                    policy_data[S][a] = epsilon/posible_actions
                policy_data[S][np.argmax(Q[S])] = 1 - epsilon + epsilon/posible_actions

            # Update env
            S = S2
            A = A2

            if done:
                break
        
        if e % 100 == 0:
            attempts = 100
            wins = 0
            for i in range(attempts):
                _, _, _, win = run_episode(Q, False, False)
                if win:
                    wins +=1
            print('With', e, 'episodes: ', (wins/attempts*100) if wins > 0 else 0)
    return policy_data, Q

policy_data, q = sarsa_improvement(100000)
