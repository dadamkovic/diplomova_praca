#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:27:59 2020

@author: daniel
"""

# from class_utils import tflog2pandas
import matplotlib.pyplot as plt
import all.presets
from all.experiments import SingleEnvExperiment, ExperimentWriter
from all.environments import GymEnvironment
from nbcap import ShowVideoCallback, ScreenRecorder, OutputManager, DisplayProcess
import numpy as np

import all
import gym
import time
import numpy as np




device = 'cuda'
env = GymEnvironment('nextro_env:nextro-v0', device)
env.reset();
agent = all.presets.continuous.sac(device=device)
exp = SingleEnvExperiment(agent, env)

exp.train(episodes=250000)
    
exp.test(episodes=5)
    
# df = tflog2pandas(exp._writer.logdir)
# df_episode_ret = df[df['metric'] == 'AntBulletEnv-v0/evaluation/returns/episode']

# plt.plot(df_episode_ret['step'], df_episode_ret['value'].rolling(25, min_periods=1).mean())
# plt.xlabel('episode')
# plt.ylabel('total reward')
# plt.grid(ls='--')
