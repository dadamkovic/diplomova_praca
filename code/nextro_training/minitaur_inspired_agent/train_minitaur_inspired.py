#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:27:59 2020

@author: daniel
"""

# from class_utils import tflog2pandas

from all.experiments import SingleEnvExperiment
from all.environments import GymEnvironment
import all
import nextro_env
from sac_minitaur_inspired import sac_minitaur_inspired



device = 'cuda'
env = GymEnvironment('nextro-v0', device)
env.render(mode='train')
env.reset()
agent = sac_minitaur_inspired(device=device)
exp = SingleEnvExperiment(agent, env)

exp.train(episodes=10)
