# Colors for console
import numpy as np
import gym
import os

def cls():
  os.system('cls' if os.name == 'nt' else 'clear')

def argmax_rand(arr):
  return np.random.choice(np.flatnonzero(arr == np.max(arr)))

class RL:
  def __init__(self, id, num_states = None, num_actions = None, state_mapper = None):
    self.env = gym.make(id)
    self.num_states = self.env.nS if num_states is None else num_states
    self.num_actions = self.env.nA if num_actions is None else num_actions
    self.state_mapper = state_mapper
    return

  def get_action(self, policy, state):
    if (isinstance(policy[state], list)):
      return np.random.choice(range(self.num_actions), p=policy[state])
    return policy[state]

  def get_policy(self, Q = None, epsilon_soft = True, epsilon = 0.1):
    # if Q is None, return a random policy
    if (Q is None):
      return [[1/self.num_actions for j in range(self.num_actions)] for i in range(self.num_states)]

    if (epsilon_soft):
      policy = [[1/self.num_actions for j in range(self.num_actions)] for i in range(self.num_states)]
      for state in range(self.num_states):
        q = np.array(Q[state])
        max = q.max()
        max_index = np.where(q == max)[0]
        for action in range(self.num_actions):
          if Q[state][action] == max:
            policy[state][action] = (1 - (self.num_actions - len(max_index)) * epsilon / self.num_actions) / len(max_index)
          else:
            policy[state][action] = epsilon / self.num_actions
    else:
      policy = [0 for i in range(self.num_states)]
      for state in range(self.num_states):
        q = np.array(Q[state])
        max = q.max()
        max_index = np.where(q == max)[0]
        policy[state] = [1/len(max_index) if i in max_index else 0 for i in range(self.num_actions)]
        # policy[state] = np.argmax(Q[state])
    return policy

  def print_policy(self, policy):
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

  def argmax(self, arr):
    q = np.array(arr)
    max = q.max()
    max_index = np.where(q == max)[0]
    return [1/len(max_index) if i in max_index else 0 for i in range(self.num_actions)]

  def run_episode(self, policy, render = False):
    state = self.map_state(self.env.reset())
    states = []
    rewards = []
    actions = []
    win = False
    while True:
      action = self.get_action(policy, state)
      states.append(state)
      actions.append(action)
      state, reward, done, _ = self.env.step(action)
      state = self.map_state(state)
      if render:
        self.env.render()
      rewards.append(reward)

      if done:
        if reward == 1:
          win = True
        break
    return states, rewards, actions, win

  def on_policy_mc(self, episodes = 1000, first_visit = True, gamma = 0.9, epsilon = 0.1):
    ## initialize random policy    
    Q = np.zeros((self.num_states, self.num_actions))
    policy = self.get_policy(Q, True, epsilon)
    returns = [[{ 'sum': 0, 'n': 0 } for j in range(self.num_actions)] for i in range(self.num_states)]

    for e in range(episodes):
      states, rewards, actions, _ = self.run_episode(policy)
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
          policy = self.get_policy(Q, True, epsilon)
    print(returns)
    return Q

  def off_policy_mc(self, episodes = 1000, gamma = 0.9, epsilon = 0.1):
    ## initialize random policy
    bpolicy = self.get_policy()
    Q = np.zeros((self.num_states, self.num_actions))
    C = [[0 for j in range(self.num_actions)] for i in range(self.num_states)]
    tpolicy = self.get_policy(Q)

    for e in range(episodes):
      states, rewards, actions, _ = self.run_episode(bpolicy)
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
        tpolicy = self.get_policy(Q)
        if (tpolicy[state][action] == 0):
          break
        W = W / bpolicy[state][action]
    return Q

  def sarsa(self, episodes = 1000, gamma = 0.9, alpha = 0.5, epsilon = 0.1):
    Q = np.zeros((self.num_states, self.num_actions))
    policy = self.get_policy(Q, True, epsilon) 

    for e in range(episodes):
      s = self.map_state(self.env.reset())
      a = self.get_action(policy, s)
      while True:
        s1, reward, done, _ = self.env.step(a)
        s1 = self.map_state(s1)
        a1 = self.get_action(policy, s1)
        Q[s, a] += alpha * (reward + gamma * Q[s1, a1] - Q[s, a])
        policy = self.get_policy(Q, True, epsilon)
        s = s1
        a = a1
        if done:
            break

    return Q

  def q_learning(self, episodes = 1000, gamma = 0.9, alpha = 0.5, epsilon = 0.1):
    Q = np.zeros((self.num_states, self.num_actions))
    
    for e in range(episodes):
      s = self.map_state(self.env.reset())
      
      while True:
        policy = self.get_policy(Q, True, epsilon)
        a = self.get_action(policy, s)
        s1, reward, done, _ = self.env.step(a)
        s1 = self.map_state(s1)
        Q[s, a] += alpha * (reward + gamma * np.max(Q[s1]) - Q[s, a])
        s = s1
        if done:
          break

    return Q
      
  def get_score(self, policy, episodes=1000):
    misses = 0
    steps_list = []
    for e in range(episodes):
      observation = self.env.reset()
      steps=0
      while True:
        action = self.get_action(policy, self.map_state(observation))
        observation, reward, done, _ = self.env.step(action)
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

  def map_state(self, state):
    return state if self.state_mapper is None else self.state_mapper(state)
