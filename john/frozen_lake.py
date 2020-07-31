import gym
import numpy as np

env = gym.make('FrozenLake-v0') 

def sarsa(episodes = 10000, gamma = 0.95, alpha = 0.5, epsilon = 0.1):

    #Initializing the Q-matrix 
    Q = np.zeros((env.observation_space.n, env.action_space.n)) 

    #Function to choose the next action 
    def choose_action(state): 
        action=0
        if np.random.uniform(0, 1) < epsilon: 
            action = env.action_space.sample() 
        else: 
            action = np.argmax(Q[state, :]) 
        return action 
    
    #Function to learn the Q-value 
    def update(state, state2, reward, action, action2): 
        predict = Q[state, action] 
        target = reward + gamma * Q[state2, action2] 
        Q[state, action] = Q[state, action] + alpha * (target - predict)  

    #Initializing the reward 
    goal=0
    misses=0
    
    # Starting the SARSA learning 
    for episode in range(episodes): 
        t = 0
        state1 = env.reset() 
        action1 = choose_action(state1) 
    
        while True: 
            #Visualizing the training 
            # env.render() 
            
            #Getting the next state 
            state2, reward, done, _ = env.step(action1) 
            #Choosing the next action 
            action2 = choose_action(state2) 
            
            #Learning the Q-value 
            update(state1, state2, reward, action1, action2) 
    
            state1 = state2 
            action1 = action2 
            
            #Updating the respective vaLues 
            t += 1     
            
            #If at the end of learning process 
            if done and reward == 1:
                #yei
                goal +=1 
                break
            elif done:
                # print("You fell in a hole!")
                misses += 1
                break

    return Q, goal, misses


episodes = 100000
Q, reward, misses = sarsa(episodes)

#performance 
print ('win: {0} / {1}'.format(reward, episodes))
print ('loss: {0} / {1}'.format(misses, episodes))

print(Q) 
