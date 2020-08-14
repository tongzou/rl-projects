'''
    Sarsa On-Policy solution. 

'''

import gym
import numpy as np
from env import TicTacToeEnv
env = TicTacToeEnv()
policy1 = 0
with open('model.npy', 'rb') as f:
    policy1 = np.load(f)
print(policy1[0])
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


def run_episode(q, render=False, e_enabled=True):
    turn = 'human' if np.random.random() < 0.5 else 'ia'
    print(turn, 'starts!')
    board, mark = env.reset()
    state = parse_state(board, mark)

    attempts = 4
    while True:

        if attempts == 0:
            print('  ', turn, ' lose the game')
            break
        if render:
            env.render()
        
        if turn == 'ia':
            action = np.argmax(q[state])
            print('IA:', state, q[state], action)
        else:
            action = int(input('Choose your action [0-9): '))
            print('Human:', action)
        old_state = state
        state, reward, done, _ = env.step(action)
        state = parse_state(state[0], state[1])
        if old_state == state:
            attempts -= 1
            print('  ', turn, 'took an invalid action. Try again')
            continue
        else:
            attempts = 4
        if done:
            if reward == 1:
                print(turn, 'WINS!')
            if render:
                env.render()
            print(reward, turn)
            break
        turn = 'ia' if turn == 'human' else 'human'

while True:
    print('----- Start Game -----')
    run_episode(policy1, True)
