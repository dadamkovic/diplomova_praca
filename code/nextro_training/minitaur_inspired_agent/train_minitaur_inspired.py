#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:27:59 2020

@author: daniel
"""

# from class_utils import tflog2pandas

from all.experiments import SingleEnvExperiment, ParallelEnvExperiment
from all.environments import GymEnvironment
import nextro_env
from sac_minitaur_inspired import sac_minitaur_inspired
from argparse import ArgumentParser
import os
from torch import load as t_load

NET_FILES = ['policy.pt', 'q_1.pt', 'q_2.pt','v.pt']
USAGE = """"The program launches the training of an agent in either pybullet GUI
            or DIRECT mode."""
parser = ArgumentParser(usage=USAGE)

def resolve_arguments():

    parser.add_argument('--render', default=False, action='store_true',
                        help="Set this flag to render the environment.")

    parser.add_argument('--device', default='cuda', required=False,
                        type=str, choices = ['cuda', 'cpu'],
                        help="Can be either 'cuda' (default) or 'cpu', will determine if pytorch can use CUDA cores.")

#TODO: that latest thing isnt implemented yet but it would be cool to do it eventually
    parser.add_argument('--regime', default='train', required=False,
                        type=str, choices = ['train', 'test'],
                        help="Determines if agent starts training or just test a pretrained network. " +
                        "Both training and testing can pre-load a trained network if its location is specified " +
                        "using the -loc flag.")

    parser.add_argument('--loc', default='', type=str, required=False,
                        help="When loading a pretrained network specify its location directory here " +
                        "(needs to be set when using -regime 'test'). Use 'latest' when using the " +
                        "latest network in runs dir. Files that hold network weights must be named " +
                        "q_1.pt, q_2.pt, policy.pt, v.pt")

    parser.add_argument('--episodes', type=int, required=False, default=10,
                        help="Determines the numer of training/testing episodes.")

    parser.add_argument('--frames', type=int, required=False, default=0,
                        help="Determins the number of training/testing frames.")

    args = parser.parse_args()
    return args

def load_weights(location, exp):
    try:
        filenames = os.listdir(location)
    except FileNotFoundError():
        raise Exception("You have to specify correct directory path!")

    for net_file in NET_FILES:
        if net_file not in filenames:
            raise Exception("The location you specified does not include the network files. " +
                          "Use --help for more info.")


    nn_paths = []
    for weight_file in NET_FILES:
        nn_paths.append(os.path.join(location, weight_file))

    #TODO: figure out how this plays with the CUDA settings, until then hope it works
    exp._agent.agent.policy.model = t_load(nn_paths[0])
    exp._agent.agent.q_1.model = t_load(nn_paths[1])
    exp._agent.agent.q_2.model = t_load(nn_paths[2])
    exp._agent.agent.v.model = t_load(nn_paths[3])
    print('----------------------------')
    print("WEIGHTS LOADED!")
    print('----------------------------')
    return exp

if __name__ == '__main__':
    args = resolve_arguments()

    env = GymEnvironment('nextro-v0', args.device)
    agent = sac_minitaur_inspired(device=args.device)
    exp = SingleEnvExperiment(agent, env, render=args.render)
    if args.loc != '':
        exp = load_weights(args.loc, exp)

    if args.regime == 'train':
        if args.frames != 0:
            exp.train(frames=args.frames)
        else:
            exp.train(episodes=args.episodes)
    else:
        if args.frames != 0:
            exp.test(frames=args.frames)
        else:
            exp.test(episodes=args.episodes)
