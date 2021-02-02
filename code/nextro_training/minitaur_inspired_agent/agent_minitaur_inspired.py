#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:27:59 2020

@author: daniel
"""

from all.experiments import SingleEnvExperiment, ParallelEnvExperiment
from all.environments import GymEnvironment
import nextro_env
from sac_minitaur_inspired import sac_minitaur_inspired
from argparse import ArgumentParser
import gym
from utils import resolve_arguments, load_weights, get_existing_runs


USAGE = """"The program launches the training of an agent in either pybullet GUI
            or DIRECT mode."""

if __name__ == '__main__':
    parser = ArgumentParser(usage=USAGE)
    args = resolve_arguments(parser)

    previous_runs = set(get_existing_runs())

    env = gym.make('nextro-v0',
                   set_loc=args.loc,
                   man_mod=args.man_mod,
                   logging=args.logging)
    env = GymEnvironment(env,
                         args.device)
    agent = sac_minitaur_inspired(device=args.device,
                                  last_frame=args.frames)
    exp = SingleEnvExperiment(agent,
                              env,
                              render=args.render)

    all_runs = set(get_existing_runs())
    current_run_name = list(all_runs - previous_runs)[0]

    if args.loc != '':
        exp = load_weights(args.loc,
                            exp)

    if args.mode == 'train':
        if args.episodes != 0:
            exp.train(episodes=args.episodes)
        else:
            exp.train(frames=args.frames)
    else:
        exp.test(episodes=args.episodes)

    exp._env._env.store_settings(current_run_name)
