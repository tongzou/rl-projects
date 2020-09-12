import gym
import gym_gomoku
import os
from algorithms import q_improvement, get_action
import numpy as np
env = gym.make('TicTacToe-v0')
""" 
env.player_color = 0 # 1
env.opponent = 'random'

observation = env.reset()
action = [1, 1]
action = env.coordinate_to_action(observation.shape[-1], action)
print(action)
print(observation)
 """

class Env:

    def trinaryToDecimal(self, trinary): 
        decimal = 0
        i = 0
        n = 0
        while(trinary != 0): 
            dec = trinary % 10
            decimal = decimal + dec * pow(3, i) 
            trinary = trinary//10
            i += 1
        return decimal

    def parse_state(self, S):
        fp = S[0].reshape((1, 9))[0]
        sp = S[1].reshape((1, 9))[0]
        trinary=''
        for i in range(0, 9):
            if (fp[i] == 1):
                trinary += '1'
            elif (sp[i] == 1):
                trinary += '2'
            else:
                trinary += '0'
        state = self.trinaryToDecimal(int(trinary) + (self.player_color + 1))
        return state

    def render(self):
        env.render()
 
    def reset(self):
        env.player_color = np.random.choice([0, 1])
        self.player_color = env.player_color
        env.opponent = 'random'
        return self.parse_state(env.reset())
    
    def step(self, A):
        observation, reward, done, _ = env.step(A)
        return self.parse_state(observation), reward, done, 0

tongTacToe = Env()
policy_data, q = q_improvement(100000, tongTacToe, 9, (3**9)*2, name='tong-tac-toe')

def run_episode(q, render=False):
    state = tongTacToe.reset()
    win = False
    while True:
        action = get_action(q, state, False)
        state, reward, done, _ = tongTacToe.step(action)
        if render:
            print('Action', action)
            print('Reward', reward)
            tongTacToe.render()
        if done:
            if reward == 1:
                win = True
            break
    return win


attempts = 1000
wins = 0
for i in range(attempts):
    win = run_episode(q, i == 0)
    if win:
        wins +=1
print('episodes: ', (wins/attempts*100) if wins > 0 else 0)
