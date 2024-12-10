########################################################################################
# Description: This file contains the Actor and Critic classes for the DDPG algorithm.
# The Actor class is used to approximate the policy function, while the Critic class is
# used to approximate the value function. Both classes are implemented as neural networks
# using PyTorch.
########################################################################################


########################################################################################
# Libraries
########################################################################################
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ipdb import set_trace as debug


########################################################################################
# Functions
########################################################################################

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

########################################################################################
# Classes
########################################################################################

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.path = 'output/PandaPickAndPlace-v3-run114/actor.pkl'

        #Check and load pretrained model if exist
        if os.path.exists(self.path):
            self.load_weights()
        else:
            self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def load_weights(self):
        print(f"Loading weights from {self.path}")
        self.load_state_dict(torch.load(self.path))

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.path = 'output/PandaPickAndPlace-v3-run114/critic.pkl'

        #Check and load pretrained model if exist
        if os.path.exists(self.path):
            self.load_weights()
        else:
            self.init_weights(init_w)    
        # self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    def load_weights(self):
        print(f"Loading weights from {self.path}")
        self.load_state_dict(torch.load(self.path))

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out