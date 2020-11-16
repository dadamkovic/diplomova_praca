#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:27:59 2020

@author: daniel
"""

# from class_utils import tflog2pandas

import all.presets
from all.experiments import SingleEnvExperiment, load_and_watch, ParallelEnvExperiment
from all.environments import GymEnvironment
import all
import nextro_env
import pickle
import torch



device = 'cuda'
env = GymEnvironment('nextro-v0', device)
env.render(mode='train')
env.reset()
agent = all.presets.continuous.sac(device=device)
exp = SingleEnvExperiment(agent, env)

exp.train(episodes=5)

torch.save(exp._agent.agent.policy.model, 'someObj.obj')

with open(r'policy.obj','wb') as fh:
    pickle.dump(exp._agent.agent.policy.model, fh)
with open(r'q1.obj','wb') as fh:
    pickle.dump(exp._agent.agent.q_1.model , fh)
with open(r'q2.obj','wb') as fh:
    pickle.dump(exp._agent.agent.q_2.model, fh)
with open(r'val.obj','wb') as fh:
    pickle.dump(exp._agent.agent.v.model, fh)
