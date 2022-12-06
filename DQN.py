import Environment

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = Environment.Env()

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Network
class Net(nn.Module):
    def __init__(self, n_states, n_neuron, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_neuron)
        self.out = nn.Linear(n_neuron, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)
    

# Trainging
# Hyperparameters and utilities
NEURON = 256
BATCH_SIZE = 64
LR = 0.001  # learning rate
GAMMA = 1  # reward discount
TARGET_UPDATE = 20  # target update frequency
MEMORY_CAPACITY = 10000

n_action = len(env.action_space)
n_state = 3

policy_net = Net(n_state, NEURON, n_action)
target_net = Net(n_state, NEURON, n_action)

target_net.load_state_dict(policy_net.state_dict())

criterion = nn.MSELoss()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_CAPACITY)


EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 200


t = 0


def select_action_stc(state, ini_e,  decay):
    global t
    state = torch.unsqueeze(torch.FloatTensor(state), 0)
    sample = random.random()
    eps_threshold = ini_e/(1 + math.pow(t, 2) / (decay + t))
    t += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_action)]])


# Trainging loop
def optimize_model():
    if len(memory) < 2*BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Get a batch of experiences (s, a, s_, r)
    state_batch = torch.tensor(batch.state, dtype=torch.float)
    action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float).view(-1, 1)
    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float)

    # Compute Q(s, a) - the model computes Q(s_t), then we select the columns of actions taken.
    # These are the actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_) for all next states.
    # Expected values of actions for next_states are computed based on the "older" target_net;
    # selecting their best reward with max(1)[0].

    # DQN
    next_state_values = target_net(next_state_batch).max(1)[0].detach().view(BATCH_SIZE, 1)

    # # double DQN
    # q_eval_next_maxaction = policy_net(next_state_batch).argmax(1).view(BATCH_SIZE, 1)  # shape (batch, 1)
    # next_state_values = target_net(next_state_batch).detach().gather(1, q_eval_next_maxaction)  # detach from graph, don't backpropagate; shape (batch, 1)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == "__main__":
    print('\nCollecting experience...')
    reward = []
    num_episodes = 300
    for i_episode in range(num_episodes):
        
        # Update the target network from the policy network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("training episode: %d" % i_episode)

        s = env.reset()

        ep_r = 0
        while True:
            # select and perform action
            state = np.array([s[1], s[4] - s[2], s[6]])

            # stc selection
            action = select_action_stc(state, 1, 1e8)

            s_, r, done, info = env.step(action.item())

            # perform one step of the optimization
            optimize_model()
            
            if done:

                break

            next_state = np.array([s_[1], s_[4] - s_[2], s_[6]])

            # Store the transition in memory and obtain the accumulative reward
            if state[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                memory.push(state, action, next_state, r)
                ep_r += r

            # Move to the next state
            s = s_



