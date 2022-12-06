import Environment
import numpy as np
import pandas as pd


# the values is optimized using the differential evolution method
threshold = np.array([360, 128, 19.])


def action_transfer(h, t):
    if h > t[0]:
        action = 0
    elif h > t[1]:
        action = 1
    elif h > t[2]:
        action = 2
    else:
        action = 3
    return action


num_episodes = 20

env = Environment.Env()
headway_cost_list = []

for i in range(num_episodes):
    
    s = env.reset()
    
    # ep_r = 0
    while True:
        # select and perform action
        headway = s[4] - s[2]
        action = action_transfer(headway, threshold)
        
        s_, r, done, info = env.step(action)
        
        if done:
            break
        
        s = s_
