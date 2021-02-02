#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:21:34 2021

@author: daniel
"""

import os
from torch import load as t_load

#wirghts of the networks will be saved in files named like this
NET_FILES = ['policy.pt', 'q_1.pt', 'q_2.pt','v.pt']

def get_existing_runs():
    return os.listdir('./runs')

#TODO: most liekly not needed anymore
def get_new_run(old_folders):
    curr_folders = os.listdir('./runs')
    for folder in curr_folders:
        if folder not in old_folders:
            return folder
    return None

#loads the pretrained weights into the model
#TODO: find why the loaded weights cannot be trained further only observed
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


def resolve_arguments(parser):

    parser.add_argument('--render', default=False, action='store_true',
                        help="Set this flag to render the environment.")

    parser.add_argument('--device', default='cuda', required=False,
                        type=str, choices = ['cuda', 'cpu'],
                        help="Can be either 'cuda' (default) or 'cpu', will determine if pytorch can use CUDA cores.")

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

    parser.add_argument('--episodes', type=int, required=False, default=0,
                        help="Determins the number of training episodes. Will" +
                        "be used for training if specified (doesn't apply to tests).")

    parser.add_argument('--man_mod', default=False, required=False, action='store_true',
                        help='Set this flag if you want to be presented with option'+
                        ' for manual modification of settings.')

    parser.add_argument("--logging", default=False, required=False, action='store_true',
                        help="Set flag to enable advanced logging into the running terminal.")
    args = parser.parse_args()
    return args
