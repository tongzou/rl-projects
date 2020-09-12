import numpy as np

def get_action(q, s, e_enabled):
    if e_enabled:
        return np.random.choice(range(0, len(q[0])), p=q[s])
    return np.argmax(q[s])

def improve(Q, S, A, R, S2, policy_data, epsilon, gamma, step_size):
    posible_actions = len(Q[0])
    Q[S, A] += step_size*(R + (gamma*np.max(Q[S2]) - Q[S,A]))
    max = np.max(Q[S])
    max_index = np.where(Q[S] == max)[0]

    for action in range(posible_actions):
        if Q[S][action] == max:
            policy_data[S][action] = (1 - (posible_actions - len(max_index)) * epsilon / posible_actions) / len(max_index)
        else:
            policy_data[S][action] = epsilon / posible_actions

def create_e_policy(posible_actions, posible_states):
    policy_data = [np.array([1/posible_actions for i in range(posible_actions)]) for i in range(posible_states)]
    return policy_data

def q_improvement(episodes, env, posible_actions, posible_states, epsilon=0.1, step_size=0.01, gamma=0.9, name='model'):
    policy_data = create_e_policy(posible_actions, posible_states)
    Q  = np.zeros((posible_states, posible_actions))
    # Load model
    for e in range(episodes):
        S = env.reset()
        
        while True:
            A = get_action(policy_data, S, True)
            S2, reward, done, _ = env.step(A)
            
            improve(Q, S, A, reward, S2, policy_data, epsilon, gamma, step_size)
            if done: # This means it wins or draws
                break
            S = S2
    with open(name +'_q.npy', 'wb') as f:
        np.save(f, Q)
    with open(name + '_policy_data.npy', 'wb') as f:
        np.save(f, policy_data)
    return policy_data, Q
