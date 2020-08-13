'''
    Sarsa On-Policy solution. 

'''

import gym
import numpy as np
from env import TicTacToeEnv
env = TicTacToeEnv()
posible_states = (3**9)
posible_actions = 9

19683
# o o x
# 1 1 2 = 2
# x o o
# 2 1 1 = 2
def mark_to_number(mark):
    if mark == 'O':
        return '1'
    if mark == 'X':
        return '2'
    return '0'
def parse_state (board):
    # Convert to trinary
    trinary=''
    for s in board:
        trinary += mark_to_number(s)
    return trinaryToDecimal(int(trinary))

def trinaryToDecimal(trinary): 
    decimal = 0
    i = 0
    n = 0
    while(trinary != 0): 
        dec = trinary % 10
        decimal = decimal + dec * pow(3, i) 
        trinary = trinary//10
        i += 1
    return decimal


def policy(q, s, e_enabled):
    if e_enabled:
        return np.random.choice(range(posible_actions), p=q[s])
    return np.argmax(q[s])

def run_episode(q, render=False, e_enabled=True):
    turn = 'ia' if np.random.random() < 0.5 else 'random'
    board, mark = env.reset()
    state = parse_state(board)
    win = False
    attempts = 4
    while True:
        if attempts == 0:
            break
        if turn == 'ia':
            action = policy(q, state, e_enabled)
        else:
            action = env.get_random_action()
        old_state = state
        state, reward, done, _ = env.step(action)
        state = parse_state(state[0])
        if old_state == state:
            attempts -= 1
            continue
        else:
            attempts = 4
        if done:
            if reward == 1:
                win = turn == 'ia'
            elif reward == -1:
                win = True
            break
        turn = 'ia' if turn == 'random' else 'random'
    return win
def improve(Q, S, A, R, S2, policy_data, epsilon, gamma, step_size):
    Q[S, A] += step_size*(R + (gamma*np.max(Q[S2]) - Q[S,A]))
    # Generate E-policy
    if np.max(Q[S]) != 0:
        for a in range(posible_actions):
            policy_data[S][a] = epsilon/posible_actions
        policy_data[S][np.argmax(Q[S])] = 1 - epsilon + epsilon/posible_actions

def q_improvement(episodes, epsilon=0.1, step_size=0.01, gamma=0.9):
    ## Define e-soft policy
    policy_data = [np.array([1/posible_actions for i in range(posible_actions)]) for i in range(posible_states)]
    Q  = np.zeros((posible_states, posible_actions))
    # Load models
    """ with open('model.npy', 'rb') as f:
        Q = np.load(f)
    with open('policy_data.npy', 'rb') as f:
        policy_data = np.load(f) """
    # Load model
    for e in range(episodes):
        S, mark = env.reset()
        turn = 'ia'
        attempts = 4
        S = parse_state(S)
        while True:
            if attempts == 0:
                break
            
            if turn == 'ia':
                A = policy(policy_data, S, True)
            else:
                A = env.get_random_action()
            old_state = S
            (board, mark), R, done, _ = env.step(A)
            S2 = parse_state(board)
            if turn == 'ia':
                if  S == S2:
                    attempts -= 1
                    improve(Q, S, A, 0, S2, policy_data, epsilon, gamma, step_size)
                    S = S2
                    continue
                if R == 0:
                    improve(Q, S, A, 0.2, S2, policy_data, epsilon, gamma, step_size)
                else:
                    improve(Q, S, A, R, S2, policy_data, epsilon, gamma, step_size)
            else:
                attempts = 4
                if  S != S2:
                    if R == 0:
                        improve(Q, S, A, 0.2, S2, policy_data, epsilon, gamma, step_size)
                    elif R == 1:
                        improve(Q, S, A, -1, S2, policy_data, epsilon, gamma, step_size)
            S = S2

            turn = 'ia' if turn == 'random' else 'random'

            if done:
                # Wins
                if turn == 'ia':
                    if R == 1:
                        improve(Q, S, A, 1, S2, policy_data, epsilon, gamma, step_size)
                
            # If draw, half or reward!
            if R == -1 and done:
               R = 0.5

            if done:
                break
        
        if e % 1000 == 0:
            attempts = 100
            wins = 0
            for i in range(attempts):
                win = run_episode(Q, False, False)
                if win:
                    wins +=1
            print('With', e, 'episodes: ', (wins/attempts*100) if wins > 0 else 0)
    return policy_data, Q


policy_data, q = q_improvement(200000)
attempts = 100
wins = 0
for i in range(attempts):
    win = run_episode(q, i == 0, False)
    if win:
        wins +=1
print('episodes: ', (wins/attempts*100) if wins > 0 else 0)
with open('model.npy', 'wb') as f:
    np.save(f, q)
with open('policy_data.npy', 'wb') as f:
    np.save(f, policy_data)
