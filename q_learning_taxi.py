import gymnasium as gym 
import numpy as np
from tqdm import tqdm
import random


env = gym.make("Taxi-v3", render_mode = "ansi")
env.reset()
print(env.render())


number_act = env.action_space.n 
state_space = env.observation_space.n 

q_table = np.zeros((state_space, number_act))

alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in tqdm(range(1, 100001)):
    
    state, _ = env.reset()
    
    done = False
    
    while not done:
        
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, done, info, _ = env.step(action)
        
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state
        
print("Train finished")

# ---------------------------------------TEST----------------------------------

total_epoch = 0
total_penalties = 0
episodes = 100


for i in tqdm(range(episodes)):
    
    state, _ = env.reset()
    epoch = 0
    penalty = 0
    reward = 0
    done = False
    
    while not done:
        
        action = np.argmax(q_table[state])
            
        next_state, reward, done, info, _ = env.step(action)
        
        state = next_state
        
        if reward == -10:
            penalty += 1
        epoch += 1
    
    total_epoch += epoch
    total_penalties += penalty

print("Result after {} episodes".format(episodes))
print("Average timestep per episode: ", total_epoch/episodes)
print("Average penalty per episode: ", total_penalties/episodes)



















