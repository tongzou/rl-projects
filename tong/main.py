import numpy as np
from algorithms import RL
import gym
import gym_gomoku
from gym_gomoku.envs import GomokuEnv


def check_score(rl, policy, episodes = 1000):
    wins = 0
    loses = 0
    for i in range(episodes):
        _, _, _, win = rl.run_episode(policy, False)
        if win == 1:
            wins += 1
        elif win == -1:
            loses += 1
    print('win: {}, loses: {}'.format(wins/episodes*100, loses/episodes*100))

def print_frozen_lake_policy(policy):
    for s in range(0, 16, 4):
        text = ''
        for t in range(s, s+4):
            v = np.argmax(policy[t])
            if v == 0:
                text += '< '
            if v == 1:
                text += 'v '
            if v == 2:
                text += '> '
            if v == 3:
                text += '^ '
        print(text)


def frozen_lake():
    rl = RL('FrozenLake-v0')
    Q = rl.q_learning(10000, 0.9, 0.02)
    policy = rl.get_policy(Q, False)
    print_frozen_lake_policy(policy)
    rl.get_score(policy)


def blackjack_state_mapper(s):
    return (1 + s[0]) * (1 + s[1]) + (1 if s[2] else 2)


def blackjack():
    rl = RL('Blackjack-v0', 704, 2, blackjack_state_mapper)
    Q = rl.on_policy_mc(30000, True, 1, 0.01)
    policy = rl.get_policy(Q, False)
    check_score(rl, policy)


def tictactoe_state_mapper(s):
    sum = 0
    for i in range(3):
        for j in range(3):
            sum += s[1, i, j] * 3**(3 * i + j) + 2 * s[2, i, j] * 3**(3 * i + j)

    return sum


def tictactoe():
    env = gym.make('TicTacToe-v0')
    env.player_color = 0
    # env.opponent = 'random'
    rl = RL(env, 3**9, 9, tictactoe_state_mapper)
    Q = rl.q_learning(200, True, 1, 0.01)
    policy = rl.get_policy(Q, False)
    # env.opponent = 'random'
    check_score(rl, policy, 100)


# frozen_lake()
# blackjack()
tictactoe()