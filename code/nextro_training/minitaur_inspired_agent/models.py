'''
Pytorch models for continuous control.

All models assume that a feature representing the
current timestep is used in addition to the features
received from the environment.
'''
import numpy as np
from all import nn

def fc_q(env, hidden1=516, hidden2=516):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear0(hidden2, 1),
    )

def fc_v(env, hidden1=516, hidden2=516):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear0(hidden2, 1),
    )

def fc_soft_policy(env, hidden1=516, hidden2=516):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear0(hidden2, env.action_space.shape[0] * 2),
    )
