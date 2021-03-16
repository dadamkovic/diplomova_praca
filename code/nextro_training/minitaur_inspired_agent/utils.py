#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:21:34 2021

@author: daniel
"""

import os
from torch import load as t_load
from collections import namedtuple

#wirghts of the networks will be saved in files named like this
NET_FILES = ['policy.pt', 'q_1.pt', 'q_2.pt','v.pt']

#lists all folder in /runs (used to rename the file to more sensible name)
def get_existing_runs():
    return os.listdir('./runs')

#loads the pretrained weights into the model
#TODO: find why the loaded weights cannot be trained further only observed
def load_weights(location):
    try:
        filenames = os.listdir(location)
    except FileNotFoundError():
        raise FileNotFoundError("You have to specify correct directory path!")

    for net_file in NET_FILES:
        if net_file not in filenames:
            raise FileNotFoundError("The location you specified does not include the network files. " +
                          "Use --help for more info.")

    nn_paths = []
    for weight_file in NET_FILES:
        nn_paths.append(os.path.join(location, weight_file))

    pretrained_models = namedtuple("models", "q_1 q_2 v policy")

    #TODO: figure out how this plays with the CUDA settings, until then hope it works
    pretrained_models.policy = t_load(nn_paths[0])
    pretrained_models.q_1 = t_load(nn_paths[1])
    pretrained_models.q_2 = t_load(nn_paths[2])
    pretrained_models.v = t_load(nn_paths[3])
    print('----------------------------')
    print("WEIGHTS LOADED!")
    print('----------------------------')
    return pretrained_models


def resolve_arguments(parser):

    parser.add_argument('--render', default=False, action='store_true',
                        help="Set this flag to render the environment.")

    parser.add_argument('--device', default='cuda', required=False,
                        type=str, choices = ['cuda', 'cpu'],
                        help="Can be either 'cuda' (default) or 'cpu', will determine if pytorch can use CUDA cores.")

    parser.add_argument('--mode', required=True,
                        type=str, choices = ['train', 'test'],
                        help="Determines if agent starts training or just test a pretrained network. " +
                        "Both training and testing can pre-load a trained network if its location is specified " +
                        "using the -loc flag.")
    #the latest feature not yet implemented
    parser.add_argument('--loc', default='', type=str, required=False,
                        help="When loading a pretrained network specify its location directory here " +
                        "(needs to be set when using --mode 'test'). Use 'latest' when using the " +
                        "latest network in runs dir. Files that hold network weights must be named " +
                        "q_1.pt, q_2.pt, policy.pt, v.pt")

    parser.add_argument('--frames', type=int, required=False, default=2e6,
                        help="Determins the number of training frames. Testing "+
                        "with FRAMES specified will not work see --episodes!")

    parser.add_argument('--episodes', type=int, required=False, default=0,
                        help="Determins the number of training/testing episodes. If " +
                        "both episodes AND frames are specified episodes will"+
                        " be used prefferentaily. Episodes HAVE to be used "+
                        "for testing.")

    parser.add_argument('--man_mod', default=False, required=False, action='store_true',
                        help='Set this flag if you want to be presented with option'+
                        ' for manual modification of settings.')

    parser.add_argument("--logging", default=False, required=False, action='store_true',
                        help="Set flag to enable advanced logging into the running terminal.")

    parser.add_argument("-rew_params", nargs=4, required=False, help="List of "+\
                        "rewards in the form <FORWARD, ENERGY, DRIFT, SHAKE>")

    parser.add_argument("--death_wall", default=True, required=False,
                        action='store_false',
                        help="If set death wall will be disabled.")

    args = parser.parse_args()
    return args
