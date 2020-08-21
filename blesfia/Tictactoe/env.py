import logging

import gym
from gym import spaces
import numpy as np


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent


def check_game_status(board):
    """Return game status by current board status.
    Args:
        board (list): Current board state
    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    for t in ['X', 'O']:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j+3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j+3] == t and board[j+6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0


class TicTacToeEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Discrete(9)
        self.marks = ['O', 'X']
        self.reset()

    
    def play(self, position):
        if self.board[position] == 0:
            self.board[position] = self.player

            if self.player == 'O':
                self.player = 'X'
            else:
                self.player = 'O'
            return True
        return False

    def reset(self):
        self.board = [0] * 9
        self.player = 'O'
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return tuple(self.board), self.player

    def return_step(self, reward):
        return self._get_obs(), reward, self.done, None

    def step(self, action):
        assert self.action_space.contains(action)

        # Invalid movement
        if not self.play(action):
            self.done = True
            return self.return_step(-1)
        
        status = check_game_status(self.board)
        # Still playing
        if status == -1:
            return self.return_step(0)
        
        self.done = True
        # Draw
        if status == 0:
            return self.return_step(0)
        # Winner
        return self.return_step(1)

    def play_auto(self):
        if self.done:
            assert False
        state, reward, done, _ = self.step(self.get_random_action())
        # Draw
        if reward == -1:
            return state, -1, done, _
        # Playing
        if reward == 0:
            return state, 0, done, _
        # Win
        if reward == 1:
            return state, -1, done, _
        
    def render(self):
        for i in range(3):
            start = 3 * int(((i)*3)/3)
            end = 3 + start
            line = ''
            for s in self.board[start:end]:
                if s == 0:
                    line += '_'
                else:
                    line += s
                line += ' '
            print(line)
        print()

    def get_random_action(self):
        available_options = []
        for i in range(0, 9):
            if self.board[i] == 0:
                available_options.append(i)
        return np.random.choice(available_options)
    
    def mark_to_number(self, mark):
        if mark == 'O':
            return '1'
        if mark == 'X':
            return '2'
        return '0'

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

    def parse_state (self, board, mark):
        # Convert to trinary
        trinary=''
        for s in board:
            trinary += self.mark_to_number(s)
        return self.trinaryToDecimal(int(trinary)) * int(self.mark_to_number(mark))