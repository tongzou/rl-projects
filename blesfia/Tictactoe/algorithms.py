import numpy as np
import pickle
from os import path, getcwd

class Policy:
    def __init__(self, states, actions, default_value):
        self.data = {}
        self.actions = actions
        self.default_value = default_value

    def get(self, state):
        if not (state in self.data):
            return np.full(self.actions, self.default_value, dtype=np.dtype(float))
        return self.data[state]

    def setValue(self, state, action, value):
        if (value != self.default_value):
            actions = self.get(state)
            actions[action] = value
            self.data[state] = actions

    def get_greedy_action(self, state):
        actions = self.get(state)
        return np.random.choice(range(0, self.actions), p=actions)
    
    def get_action(self, state):
        return np.argmax(self.get(state))

def get_action(q, s, e_enabled):
    if e_enabled:
        return np.random.choice(range(0, q.actions), p=q.get(s))
    if np.mean(q.get(s)) == 0:
        return np.random.choice(range(0, q.actions))
    return np.argmax(q.get(s))

def improve(Q, S, A, R, S2, policy, epsilon, gamma, step_size):
    posible_actions = Q.actions
    Q.setValue(S, A, Q.get(S)[A] + step_size*(R + (gamma*np.max(Q.get(S2)) - Q.get(S)[A])))
    max = np.max(Q.get(S))
    max_index = np.where(Q.get(S) == max)[0]

    for action in range(posible_actions):
        if Q.get(S)[action] == max:
            policy.setValue(S, action, (1 - (posible_actions - len(max_index)) * epsilon / posible_actions) / len(max_index))
        else: 
            policy.setValue(S, action, epsilon / posible_actions)


def q_improvement(episodes, env, posible_actions, posible_states, epsilon=0.1, step_size=0.01, gamma=0.9, name='model'):
    q_name = getcwd() + '/' + name +'_q.model'
    if path.exists(q_name):
        with open(q_name, 'rb') as f:
            Q = pickle.load(f)
        with open(name + '_policy_data.model', 'rb') as f:
            policy = pickle.load(f)
    else:
        Q  = Policy(posible_states, posible_actions, 0)
        policy = Policy(posible_states, posible_actions, 1/posible_actions)

    # Load model
    for e in range(episodes):
        S = env.reset()
        
        while True:
            A = policy.get_greedy_action(S)
            S2, reward, done, _ = env.step(A)
            
            improve(Q, S, A, reward, S2, policy, epsilon, gamma, step_size)
            if done: # This means it wins or draws
                break
            S = S2
        
        if e % (episodes / 10) == 0:
            print('Training... (' + str(e / episodes *100) + ')')
            with open(name +'_q.model', 'wb') as f:
                pickle.dump(Q, f)
            with open(name + '_policy_data.model', 'wb') as f:
                pickle.dump(policy, f)
    with open(name +'_q.model', 'wb') as f:
        pickle.dump(Q, f)
    with open(name + '_policy_data.model', 'wb') as f:
        pickle.dump(policy, f)
    return policy, Q
