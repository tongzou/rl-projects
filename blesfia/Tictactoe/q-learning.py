'''
    Sarsa On-Policy solution. 

'''

import gym
import numpy as np
from env import TicTacToeEnv
env = TicTacToeEnv()
posible_states = (3**9)*2
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

def parse_state (board, mark):
    # Convert to trinary
    trinary=''
    for s in board:
        trinary += mark_to_number(s)
    return trinaryToDecimal(int(trinary)) * int(mark_to_number(mark))

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
    turn = 'random' if np.random.random() > 0.5 else 'ia'
    board, mark = env.reset()
    state = parse_state(board, mark)
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
        state = parse_state(state[0], state[1])
        if old_state == state:
            attempts -= 1
            continue
        else:
            attempts = 4
        if done:
            if reward == 1:
                win = turn == 'ia'
            elif reward == 0:
                win = True
            break
        turn = 'ia' if turn == 'random' else 'random'
    return win


def improve(Q, S, A, R, S2, policy_data, epsilon, gamma, step_size):
    Q[S, A] += step_size*(R + (gamma*np.max(Q[S2]) - Q[S,A]))
    max = np.max(Q[S])
    max_index = np.where(Q[S] == max)[0]

    for action in range(posible_actions):
        if Q[S][action] == max:
            policy_data[S][action] = (1 - (posible_actions - len(max_index)) * epsilon / posible_actions) / len(max_index)
        else:
            policy_data[S][action] = epsilon / posible_actions
# -1 0 1
#  0 0.3 0.66 --> SUM => 1
def q_improvement(episodes, epsilon=0.1, step_size=0.01, gamma=0.9):
    ## Define e-soft policy
    policy_data = [np.array([1/posible_actions for i in range(posible_actions)]) for i in range(posible_states)]
    Q  = np.zeros((posible_states, posible_actions))
    # Load model
    for e in range(episodes):
        who_start = 'random' if np.random.random() > 0.5 else 'ia'
        S, mark = env.reset()
        if who_start == 'random':
            A = env.get_random_action()
            (S, mark), R, done, _ = env.step(A)
        S = parse_state(S, mark)
        i = -2
        while True:
            i += 2
            if S == 16440:
                a = Q[S]
                # print('Holi', a)

            A = policy(policy_data, S, True)
            (board, mark), R, done, _ = env.step(A)
            S2 = parse_state(board, mark)
            if done: # This means it wins or draws
                if S == S2:
                    improve(Q, S, A, -1, S2, policy_data, epsilon, gamma, step_size)
                elif R == 0:
                    improve(Q, S, A, 1, S2, policy_data, epsilon, gamma, step_size)
                if S == 16440:
                    a = Q[S]
                    b = policy_data[S]
                    # print('Hola', a, b)
                break

            # Random player
            A2 = env.get_random_action()
            (board, mark), R, done, _ = env.step(A2)
            S2 = parse_state(board, mark)

            if done:
                if R == 1: # Random wins
                    improve(Q, S, A, -1, S2, policy_data, epsilon, gamma, step_size)
                elif R == 0: # Draws
                    improve(Q, S, A, 0, S2, policy_data, epsilon, gamma, step_size)
                break
            else:
                improve(Q, S, A, 0, S2, policy_data, epsilon, gamma, step_size)

            S = S2
        
        if e % 1000 == 0:
            attempts = 100
            wins = 0
            for i in range(attempts):
                win = run_episode(Q, False, False)
                if win:
                    wins +=1
            print('With', e, 'episodes: ', (wins/attempts*100) if wins > 0 else 0)
    return policy_data, Q


policy_data, q = q_improvement(120000)
attempts = 100
wins = 0
for i in range(attempts):
    win = run_episode(q, i == 0, False)
    if win:
        wins +=1
print(policy_data[16440], q[16440])
print('episodes: ', (wins/attempts*100) if wins > 0 else 0)
with open('model.npy', 'wb') as f:
    np.save(f, q)
with open('policy_data.npy', 'wb') as f:
    np.save(f, policy_data)

print('Hola')