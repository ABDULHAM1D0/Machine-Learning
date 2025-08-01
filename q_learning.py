import numpy as np
import random
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt


environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

env_states  = environment.observation_space.n
env_actions = environment.action_space.n 
q_table = np.zeros((env_states, env_actions))

print("Q table")
print(q_table)


action = environment.action_space.sample()

new_state, reward, done, info, _ = environment.step(action)

episode = 1000
alpha = 0.5
gamma = 0.9

outcomes = []

for i in tqdm(range(episode)):
    
    state, _ = environment.reset()
    done = False
    outcomes.append("Failure")
    
    while not done:
        
        if np.max(q_table[state]) > 0:
            action = np.argmax(q_table[state])
        else:
            action = environment.action_space.sample()
            
        new_state, reward, done, info, _ = environment.step(action)
            
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state, action])
        
        state = new_state
        
        
        if reward:
            outcomes[-1] = "Succes"
            
            
print("Q_table after training: ")
print(q_table)

plt.bar(range(episode), outcomes)
    
    

#%%

#------------------------------ TEST --------------------------------------

episode = 100
number_succes = 0

for i in tqdm(range(episode)):
    
    state, _ = environment.reset()
    done = False
    
    while not done:
        
        if np.max(q_table[state]) > 0:
            action = np.argmax(q_table[state])
        else:
            action = environment.action_space.sample()
            
        new_state, reward, done, info, _ = environment.step(action)
                    
        state = new_state
        
        number_succes += reward
        
print("Succes rate: ", 100*number_succes/episode)
    
    
    
    
    
    
    
    
    
    