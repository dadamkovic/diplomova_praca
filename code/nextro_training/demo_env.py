#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:27:59 2020

@author: daniel
"""

import all
import gym
import time
import numpy as np

env = gym.make('nextro_env:nextro-v0')
#agent = all.presets.continuous.sac()
env.render(mode='human')
state = np.array([])
state = env.reset()
done = False
actions = np.zeros(36)
count = 0
while True:
    if not done:
        actions += np.random.default_rng().normal(0,0.01,36)
        state, reward, done, _ = env.step(actions)
        if count > 5: 
            print(reward)
            count = 0
        else:
            count += 1
    else:
        env.reset()
        done = False
        actions = np.zeros(36)