#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:27:59 2020

@author: daniel
"""

from all.experiments import SingleEnvExperiment
from all.environments import GymEnvironment
import nextro_env
from sac_minitaur_inspired import sac_minitaur_inspired
from argparse import ArgumentParser
import gym
from utils import resolve_arguments, load_weights, get_existing_runs


USAGE = """Launches the training of an agent in GUI or DIRECT mode.

            When doing TRAINING the FRAMES parameter SHOULD be used.
            When doing TESTING the EPISODES parameter HAS TO be used."""

# TODO it is worth thinking whether the episodes shouldn't just be scrapped
# by manually transforming episodes into the requested frames

if __name__ == '__main__':
    parser = ArgumentParser(usage=USAGE)
    args = resolve_arguments(parser)

    previous_runs = set(get_existing_runs())

    env = gym.make('nextro-v0',
                   c_args=args)

    env = GymEnvironment(env,
                         args.device)

    pretrained_models = None
    if args.loc != '':
        pretrained_models = load_weights(args.loc)

    agent = sac_minitaur_inspired(device=args.device,
                                  last_frame=args.frames,
                                  pretrained_models=pretrained_models)
    exp = SingleEnvExperiment(agent,
                              env,
                              render=args.render)

    all_runs = set(get_existing_runs())
    current_run_name = list(all_runs - previous_runs)[0]



    if args.mode == 'train':
        if args.episodes != 0:
            exp.train(episodes=args.episodes)
        else:
            exp.train(frames=args.frames)
    else:
        if args.episodes == 0:
            raise Exception('You HAVE TO speficy EPISODES when using test!!!')
        exp.test(episodes=args.episodes)

    exp._env._env.store_settings(current_run_name)
