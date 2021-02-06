#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:10:25 2020

@author: daniel
"""
import pickle
import os
import git
from consolemenu import ConsoleMenu, SelectionMenu
from consolemenu.items import SubmenuItem, FunctionItem



#################################################################
#STATIC SETTINGS
#################################################################
NUM_JOINTS = 18
# listed to remove ambiguity when addressing joints
#number of control bits for controlling direction that the robot is moving in
CONTROL_SIZE = 4
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
# ((NUM_JOINTS old joint angles + NUM_JOINTS old joint velocities +
#  NUM_JOINTS previous action) + (yaw + pitch + roll) + 4 control bits)
STORED_OBSERVATION_SIZE = NUM_JOINTS*3 + 3 + CONTROL_SIZE
# (NUM_JOINTS joint angles + NUM_JOINTS joint velocities + yaw + pitch + roll)
# + control bits
SENZOR_OUTPUT_SIZE = (NUM_JOINTS*2 + 3 + CONTROL_SIZE)


#################################################################
#USER SETTINGS
#################################################################

# default maximal length in seconds of a single episode
DEFAULT_MAX_TIME = 20
# if default_max_time and default_min_time are different, the time that episode
# will run for gets chosen as random number in the interval, this in theory
# should prevent agent from taking too long to start moving
DEFAULT_MIN_TIME = 20
# defines the FPS that the simulation will run at
FRAMES_PER_SECOND = 30
# number of prevous observations supplied on the input
PREV_OBS_ON_INPUT = 5
# if PIDs are to be used thsi determins [kp, ki, kd] parameters
PID_PARAMS = [0.1, 0, 0.003]
# multiple of the dist eward
DIST_CONST = 5000
# multiple of the acceleration punishment
ACCEL_CONST = 0.01
POS_GAIN_START = 0.1
POS_GAIN_FINAL = 0.2
YPR_CONST = 1


def add_non_user_settings(set_dict):
    set_dict['NUM_JOINTS'] = NUM_JOINTS
    set_dict['JOINT_NAMES'] = JOINT_NAMES
    set_dict['SENZOR_OUTPUT_SIZE'] = SENZOR_OUTPUT_SIZE
    set_dict['STORED_OBSERVATION_SIZE'] = STORED_OBSERVATION_SIZE

    # observation consists of current information and a number of observations
    # from history, has to be rocmputed because depth of prevobservations
    #ca be tuned by user
    set_dict['OBSERVATION_SIZE'] = set_dict['SENZOR_OUTPUT_SIZE'] + \
        (set_dict['STORED_OBSERVATION_SIZE'] * set_dict['PREV_OBS_ON_INPUT'])
    return set_dict

def remove_non_user_settings(set_dict):
    del(set_dict['NUM_JOINTS'])
    del(set_dict['JOINT_NAMES'])
    del(set_dict['SENZOR_OUTPUT_SIZE'])
    del(set_dict['STORED_OBSERVATION_SIZE'])
    del(set_dict['OBSERVATION_SIZE'])
    return set_dict

def get_default_settings(manual_modify=False):
    set_dict = {}
    set_dict['DEFAULT_MAX_TIME'] = DEFAULT_MAX_TIME
    set_dict['DEFAULT_MIN_TIME'] = DEFAULT_MIN_TIME
    set_dict['FRAMES_PER_SECOND'] = FRAMES_PER_SECOND
    set_dict['PREV_OBS_ON_INPUT'] = PREV_OBS_ON_INPUT
    set_dict['PID_ENABLED'] = False
    set_dict['PID_PARAMS'] = PID_PARAMS
    set_dict['DIST_CONST'] = DIST_CONST
    set_dict['ACCEL_CONST'] = ACCEL_CONST
    set_dict['YPR_CONST'] = YPR_CONST
    set_dict['POS_GAIN_FINAL'] = POS_GAIN_FINAL
    set_dict['POS_GAIN_START'] = POS_GAIN_START
    set_dict['COLLISION'] = True
    set_dict['DIFF_DIST_REW'] = True
    set_dict['PUNISH_Y'] = False
    set_dict['TOTAL_DIST_REW'] = False
    if manual_modify:
        set_dict = manually_mod_settings(set_dict)

    set_dict = add_non_user_settings(set_dict)
    return set_dict

def save_default_settings(settings, loc):
    file_name_pic = os.path.join(loc, 'settings.p')
    file_name_txt = os.path.join(loc, 'settings.txt')

    cleaned_settings = remove_non_user_settings(settings)
    pickle.dump(cleaned_settings, open(file_name_pic, 'wb'))

    with open(file_name_txt,'w') as fh:
        for key, item in settings.items():
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

def load_settings(loc, manual_modify=False):
    try:
        file_name_pic = os.path.join(loc, 'settings.p')
        set_dict = pickle.load(open(file_name_pic, 'rb'))
        print("SETTINGS LOADED!")
        if manual_modify:
            set_dict = manually_mod_settings(set_dict)
        set_dict = add_non_user_settings(set_dict)
        return set_dict
    except Exception:
        print("The location doesn't have a settings file. Skipping...")
        return {}

#class modifications needed for the manual input bellow, to show current value
#in the menu
class FunctionItemMod(FunctionItem):
    def __init__(self, text, function, args=None, kwargs=None, menu=None, should_exit=False):
        super().__init__(text, function, args=args, kwargs=kwargs, menu=menu, should_exit=should_exit)
        self.base_text = text

    def action(self):
        self.return_value = self.function(*self.args, **self.kwargs)
        self.text = self.base_text + f"<{self.return_value}>"

class SubmenuItemMod(SubmenuItem):
    def __init__(self, text, submenu, menu=None, should_exit=False):
        super(SubmenuItemMod, self).__init__(text=text, submenu=submenu, menu=menu, should_exit=should_exit)
        self.base_text = text

    def clean_up(self):
        super().clean_up()
        self.text = (self.base_text + '<') + ('True>' if self.get_return()==0 else 'False>')

#using a simple menu load parameters from the user
#TODO: look for a way to implement simple explanations of what the parameters do
def manually_mod_settings(set_dict):
    menu = ConsoleMenu("Tune parameters", "[original value] / <new value>")
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
            menu_items.append(SubmenuItemMod(desc, tmp, menu))
        elif key == 'PID_PARAMS':
            prompt = "PID <P I D>: "
            menu_items.append(FunctionItemMod(desc, prompt_func, [prompt, key]))
        else:
            menu_items.append(FunctionItemMod(desc, prompt_func, [prompt, key]))

        menu.append_item(menu_items[idx])

    def print_sett():
        for key, item in set_dict.items():
            print(f"{key} : {item}")
        input('\nPress any key to return!')

    menu.append_item(FunctionItem("PRINT CURRENT SETTINGS", print_sett))
    menu.show()
    return set_dict
