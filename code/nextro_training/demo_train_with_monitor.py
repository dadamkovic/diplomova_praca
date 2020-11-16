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
import os

import pybullet
import pybullet_envs
import torch.nn as nn
import torch


import all
import gym
import time
import numpy as np


#@title -- Auxiliary Functions -- { display-mode: "form" }
display_size = (900, 700)
show_video = ShowVideoCallback(dimensions=(700, 500))

# make sure that only one instance
# of the display is ever created
try:
    DISP_PROC
except NameError:
    DISP_PROC = DisplayProcess(display_size=display_size)

def make_screen_recorder(max_gui_outputs=10):
    video_path="output"
    segment_time=1

    output_manager = OutputManager(max_gui_outputs=max_gui_outputs)
    video_callback=output_manager(show_video)
    display = DISP_PROC.id

    screen_recorder = ScreenRecorder(
        display, display_size, video_path,
        segment_time=segment_time, video_callback=video_callback
    )

    return screen_recorder

SCREEN_RECORDER = make_screen_recorder()

class CapturedExperiment(SingleEnvExperiment):
    def __init__(self, agent, env, screen_recorder=SCREEN_RECORDER,
                 render=False, quiet=False, write_loss=True,
                 capture_train=False, capture_test=True):
        super().__init__(agent, env, render=render, quiet=quiet,
                         write_loss=write_loss)
        if screen_recorder is None and (capture_train or capture_test):
            self.screen_recorder = make_screen_recorder()
        else:
            self.screen_recorder = screen_recorder

        self.capture_train = capture_train
        self.capture_test = capture_test

    def _run_training_episode(self):
        if self.capture_train and not self.screen_recorder is None:
            render = self._render
            self._render = True

            try:
                with self.screen_recorder:
                    return super()._run_training_episode()
            except:
                self._render = render
                raise
        else:
            return super()._run_training_episode()

    def _run_test_episode(self):
        if self.capture_test and not self.screen_recorder is None:
            render = self._render
            self._render = True

            try:
                with self.screen_recorder:
                    return super()._run_test_episode()
            except:
                self._render = render
                raise
        else:
            return super()._run_test_episode()


device = 'cuda'
env = GymEnvironment('nextro_env:nextro-v0', device)
env.reset();
agent = all.presets.continuous.sac(device=device)
exp = CapturedExperiment(agent, env)

screen_recorder = make_screen_recorder(max_gui_outputs=1)

with screen_recorder:
    exp.train(frames=250000)

exp.test(episodes=5)

# df = tflog2pandas(exp._writer.logdir)
# df_episode_ret = df[df['metric'] == 'AntBulletEnv-v0/evaluation/returns/episode']

# plt.plot(df_episode_ret['step'], df_episode_ret['value'].rolling(25, min_periods=1).mean())
# plt.xlabel('episode')
# plt.ylabel('total reward')
# plt.grid(ls='--')
