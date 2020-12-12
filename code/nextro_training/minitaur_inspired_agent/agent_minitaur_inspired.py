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
import gym

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
    parser.add_argument('--mode', default='train', required=False,
                        type=str, choices = ['train', 'test'],
                        help="Determines if agent starts training or just test a pretrained network. " +
                        "Both training and testing can pre-load a trained network if its location is specified " +
                        "using the -loc flag.")

    parser.add_argument('--loc', default='', type=str, required=False,
                        help="When loading a pretrained network specify its location directory here " +
                        "(needs to be set when using --mode 'test'). Use 'latest' when using the " +
                        "latest network in runs dir. Files that hold network weights must be named " +
                        "q_1.pt, q_2.pt, policy.pt, v.pt")

    parser.add_argument('--frames', type=int, required=False, default=2e6,
                        help="Determins the number of training frames. Test "+
                        "episodes can be set during the program's run.")

    parser.add_argument('--man_mod', default=False, required=False, action='store_true',
                        help='Set this flag if you want to be presented with option'+
                        ' for manual modification of settings.')

    parser.add_argument("--logging", default=False, required=False, action='store_true',
                        help="Set flag to enable advanced logging into the running terminal.")
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

def get_existing_runs():
    return os.listdir('./runs')

def get_new_run(old_folders):
    curr_folders = os.listdir('./runs')
    for folder in curr_folders:
        if folder not in old_folders:
            return folder
    return None


if __name__ == '__main__':
    args = resolve_arguments()
    exist_runs = get_existing_runs()

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
    if args.loc != '':
        exp = load_weights(args.loc,
                           exp)

    if args.mode == 'train':
        exp.train(frames=args.frames)
    else:
        episodes = int(input('Testing episodes: '))
        exp.test(episodes=episodes)

    new_run_folder = get_new_run(exist_runs)

    exp._env._env.store_settings(new_run_folder)
