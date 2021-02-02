#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:10:25 2020

@author: daniel
"""
import pickle
import os
import git
from pprint import pprint
from consolemenu import *
from consolemenu.items import *

# default maximal length in seconds of a single episode
DEFAULT_MAX_TIME = 20

# defines the FPS that the simulation will run at
FRAMES_PER_SECOND = 30

NUM_JOINTS = 18

# (NUM_JOINTS joint angles + NUM_JOINTS joint velocities + yaw + pitch + roll)
SENZOR_OUTPUT_SIZE = (NUM_JOINTS*2 + 3)

# ((NUM_JOINTS old joint angles + NUM_JOINTS old joint velocities +
#  NUM_JOINTS previous joint angles) + (yaw + pitch + roll))
STORED_OBSERVATION_SIZE = NUM_JOINTS*3 + 3

# number of prevous observations supplied on the input
PREV_OBS_ON_INPUT = 5

# observation consists of current information and a number of observations
# from history
OBSERVATION_SIZE = SENZOR_OUTPUT_SIZE + STORED_OBSERVATION_SIZE * PREV_OBS_ON_INPUT
# if PIDs are to be used thsi determins [kp, ki, kd] parameters
PID_PARAMS = [0.1, 0, 0.003]
# when true PIDs with the above parameters will be applied to joints
PID_ENABLED = False
# enables/disables self-collision
COLLISION = True
# multiple of the dist eward
DIST_CONST = 5000
# multiple of the acceleration punishment
ACCEL_CONST = 0.01

POS_GAIN_START = 0.1
POS_GAIN_FINAL = 0.2

YPR_CONST = 1
# listed to remove ambiguity when addressing joints
JOINT_NAMES = ['left_back_hip_joint',
               'left_back_knee_joint',
               'left_back_ankle_joint',
               'left_center_hip_joint',
               'left_center_knee_joint',
               'left_center_ankle_joint',
               'left_front_hip_joint',
               'left_front_knee_joint',
               'left_front_ankle_joint',
               'right_back_hip_joint',
               'right_back_knee_joint',
               'right_back_ankle_joint',
               'right_center_hip_joint',
               'right_center_knee_joint',
               'right_center_ankle_joint',
               'right_front_hip_joint',
               'right_front_knee_joint',
               'right_front_ankle_joint']

def get_default_settings(manual_modify=False):
    set_dict = {}
    set_dict['DEFAULT_MAX_TIME'] = DEFAULT_MAX_TIME
    set_dict['FRAMES_PER_SECOND'] = FRAMES_PER_SECOND
    set_dict['NUM_JOINTS'] = NUM_JOINTS
    set_dict['SENZOR_OUTPUT_SIZE'] = SENZOR_OUTPUT_SIZE
    set_dict['STORED_OBSERVATION_SIZE'] = STORED_OBSERVATION_SIZE
    set_dict['PREV_OBS_ON_INPUT'] = PREV_OBS_ON_INPUT
    set_dict['OBSERVATION_SIZE'] = OBSERVATION_SIZE
    set_dict['PID_PARAMS'] = PID_PARAMS
    set_dict['PID_ENABLED'] = PID_ENABLED
    set_dict['COLLISION'] = COLLISION
    set_dict['DIST_CONST'] = DIST_CONST
    set_dict['ACCEL_CONST'] = ACCEL_CONST
    set_dict['YPR_CONST'] = YPR_CONST
    set_dict['POS_GAIN_FINAL'] = POS_GAIN_FINAL
    set_dict['POS_GAIN_START'] = POS_GAIN_START
    set_dict['DIFF_DIST_REW'] = True
    set_dict['PUNISH_Y'] = False
    set_dict['TOTAL_DIST_REW'] = False
    if manual_modify:
        set_dict = manually_mod_settings(set_dict)
    set_dict['JOINT_NAMES'] = JOINT_NAMES
    return set_dict

def save_default_settings(settings, loc):
    file_name_pic = os.path.join(loc, 'settings.p')
    file_name_txt = os.path.join(loc, 'settings.txt')
    pickle.dump(settings, open(file_name_pic, 'wb'))
    with open(file_name_txt,'w') as fh:
        for key, item in settings.items():
            if key == 'JOINT_NAMES':
                continue
            fh.write('{} \t {} \n'.format(str(key), str(item)))
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            fh.write('CURRENT GIT HASH: {}'.format(sha))
        except Exception:
            print('Failed to fetch GIT INFO!')

def isInt(number):
    tempN = int(number)
    if (number - tempN) > 0:
        return False
    return True

def manually_mod_settings(set_dict):
    menu = ConsoleMenu("Tune parameters", "([] brackets show original value)")
    menu_items = []
    for idx, (key, val) in enumerate(set_dict.items()):

        desc = f"{key}[{val}]"
        prompt = f"{key}[{val}] : "

        def prompt_func(text, set_key):
            value = input(text)
            if set_key in ['PID_ENABLED','COLLISION']:
                set_dict[set_key] = True if value.startswith(('t','T')) else False
            elif set_key == 'PID_PARAMS':
                set_dict[set_key] = [float(x) for x in value.replace(' ',',').split(',')]
            elif isInt(float(value)):
                set_dict[set_key] = int(value)
            else:
                set_dict[set_key] = float(value)
            return value

        if key in ['PID_ENABLED','COLLISION','TOTAL_DIST_REW','DIFF_DIST_REW','PUNISH_Y']:
            tmp = SelectionMenu(['True','False'])
            desc = f"{key}[{val}]"
            menu_items.append(SubmenuItem(desc, tmp, menu))
        elif key == 'PID_PARAMS':
            prompt = "PID <P I D>: "
            menu_items.append(FunctionItem(desc, prompt_func, [prompt, key]))
        else:
            menu_items.append(FunctionItem(desc, prompt_func, [prompt, key]))
        menu.append_item(menu_items[idx])

    def print_sett():
        # for m_entry, dict_key in zip(menu_items, set_dict.keys()):
        #     if m_entry.get_return() is not None:
        #         set_dict[dict_key] = float(m_entry.get_return())
        for key, item in set_dict.items():
            print(f"{key} : {item}")
        input('Press any key to return')

    menu.append_item(FunctionItem("Print all", print_sett))
    menu.show()
    return set_dict
# def manually_mod_settings(set_dict):
#     print("Current settings are:")
#     print('---------------------------')
#     pprint(set_dict)
#     while True:
#         user_choice = input('Modify?(Y/N)').lower()
#         if user_choice in ['y', 'n']:
#             break
#         else:
#             print("(Y/N)")
#     if user_choice == 'n':
#         return set_dict
#     print('-------------------')
#     print('type q/Q to exit')
#     print('-------------------')
#     while True:
#         sel_key = input('key:value   > ')
#         if sel_key.lower() == 'q':
#             return set_dict
#         key, value = sel_key.split(':')
#         if key in ['PID_ENABLED','COLLISION']:
#             set_dict[key] = True if value.startswith(('t','T')) else False
#         elif key == 'PID_PARAMS':
#             set_dict[key] = [float(x) for x in value.split(',')]
#         elif key in ['POS_GAIN_START','POS_GAIN_FINAL','YPR_CONST',
#                      'ACCEL_CONST','DIST_CONST','DEFAULT_MAX_TIME']:
#             set_dict[key] = float(value)
#         else:
#             set_dict[key] = int(value)

def load_settings(loc, manual_modify=False):
    try:
        file_name_pic = os.path.join(loc, 'settings.p')
        settings = pickle.load(open(file_name_pic, 'rb'))
        print("SETTINGS LOADED!")
        if manual_modify:
            settings = manually_mod_settings(settings)
        return settings
    except Exception:
        print("The location doesn't have a settings file. Skipping...")
        return {}
