'''
Pytorch models for continuous control.

All models assume that a feature representing the
current timestep is used in addition to the features
received from the environment.
'''
import numpy as np
from all import nn
from torch.nn.functional import relu, dropout
from torch import add

FREEZE_SIDE = True

class ParallelNet(nn.Module):
    def __init__(self,
                inp_size,
                out_size,
                hidden_m,
                hidden_s,
                ):
        super().__init__()
        self.inp_layer = nn.Linear(inp_size, hidden_m[0])
        self.hidden_00 = nn.Linear(hidden_m[0],hidden_m[1])
        self.hidden_01 = nn.Linear(hidden_m[1], out_size)
        self.hidden_10 = nn.Linear(hidden_s[0], hidden_s[1])
        self.hidden_11 = nn.Linear(hidden_s[1], out_size)
        self.final_layer = nn.Linear(out_size, out_size)
    
    def forward(self, x):
        inp0 = relu(self.inp_layer(x))
        inp0 = dropout(inp0, p=0.1)

        h00 = relu(self.hidden_00(inp0))
        h00 = relu(self.hidden_01(h00))
        
        h10 = relu(self.hidden_10(inp0))
        h10 = relu(self.hidden_11(h10))

        out = self.final_layer(h00.add(h10))
        return out

def fc_q(env, hidden1=516, hidden2=516):
    print("Custom Q loaded")
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear0(hidden2, 1),
    )

def fc_v(env, hidden1=516, hidden2=516):
    print("Custom V loaded")
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear0(hidden2, 1),
    )

def fc_soft_policy(env, hidden1=516, hidden2=20):
    print("Custom PI loaded")
    net =  ParallelNet(env.state_space.shape[0] + 1,
        env.action_space.shape[0] * 2,
        hidden_m=[hidden1,hidden1], 
        hidden_s=[hidden1,hidden2],)

    if FREEZE_SIDE:
        net.hidden_10.weight.requires_grad = False
        net.hidden_10.bias.requires_grad = False
        net.hidden_11.weight.requires_grad = False
        net.hidden_11.bias.requires_grad = False
    else:
        net.hidden_00.weight.requires_grad = False
        net.hidden_00.bias.requires_grad = False
        net.hidden_01.weight.requires_grad = False
        net.hidden_01.bias.requires_grad = False
    return net
