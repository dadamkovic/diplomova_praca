"""
This module serves to define a wrapper around the pybullet's URDFBasedRobot
class and to define a new gym class NextroEnv. Both of these were created to
allow reinforcement learning techniques to work with all package, and conform
to the stardard gym environment.
"""

import time
from collections import deque
import pybullet as p
from pybullet_envs import robot_bases as rb
import pybullet_data
import numpy as np
import gym
from .pid import PID
from gym.utils import seeding
from .settings import load_settings, get_default_settings, save_default_settings
import os
import shutil

# the urdf location has to be absolute path
NEXTRO_LOC = __file__.replace('envs/nextro_env.py', 'assets/urdf/nextro.urdf')


# exists mostly to define the calc_state and robot_specific_reset methods that
# are missing in the URDFBasedRobot definition
class NextroBot(rb.URDFBasedRobot):
    def __init__(self,
                 model_urdf,
                 robot_name,
                 settings,
                 com_args,
                 basePosition=[0, 0, 0.1],
                 baseOrientation=[0, 0, 0, 1],
                 fixed_base=False):
        self.settings = settings
        super().__init__(model_urdf=model_urdf,
                         robot_name=robot_name,
                         action_dim=self.settings['NUM_JOINTS'],
                         obs_dim=self.settings['OBSERVATION_SIZE'],
                         self_collision=self.settings['COLLISION'],
                         basePosition=basePosition,
                         baseOrientation=baseOrientation)
        self.joint_ids = []
        self.frame_count = 0
        self.com_args = com_args

        final_frame = 0
        if self.com_args.mode == 'train':
            if self.com_args.frames != 0:
                final_frame = self.com_args.frames
            else:
                final_frame = self.com_args.episodes * \
                    self.settings['DEFAULT_MAX_TIME']/self.settings['FRAMES_PER_SECOND']
        self.final_frame = final_frame

    # only called once when adding the robot urdf into environment
    # 3 fixed joints are removed from the list of joints and named dictionary
    # with joints as values
    def robot_specific_reset(self, _p):
        keylist = list(self.jdict.keys())
        joint_names = self.settings['JOINT_NAMES']
        num_joints = self.settings['NUM_JOINTS']
        for key in keylist:
            if key.endswith('jointfix'):
                fixed_joint = self.jdict[key]
                del self.jdict[key]
                self.ordered_joints.remove(fixed_joint)

        for joint in joint_names:
            self.joint_ids.append(self.jdict[joint].jointIndex)
        p.setJointMotorControlArray(1,self.joint_ids,
                                    controlMode=p.VELOCITY_CONTROL,
                                    velocityGains=[0 for _ in range(num_joints)])

    # by default called in the parent method after adding urdf, is only here to
    # prevent errors
    def calc_state(self):
        return 0

    # pybullet's Joint class doesn't accept parametrs for setting the p/vGains
    # it also dosesn't have setJointMotorControlArray-like bulk set
    def set_all_joints(self, joint_angles):
        self.frame_count += 1
        pos_gain_start = self.settings['POS_GAIN_START']
        pos_gain_final = self.settings['POS_GAIN_FINAL']
        pos_range = [pos_gain_start, pos_gain_final]

        if self.com_args.mode == 'train':
            pos_gain = np.interp(self.frame_count,
                                 [1, self.final_frame],
                                 pos_range)
        else:
            pos_gain = pos_gain_final

        num_joints = self.settings['NUM_JOINTS']
        pos_gains = [pos_gain for _ in range(num_joints)]
        #the forces are set a bit arbitratily
        p.setJointMotorControlArray(1,
                                    self.joint_ids,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_angles,
                                    forces=[10 for _ in range(num_joints)],
                                    positionGains=pos_gains)


# defines all the methods that a gym environment has to provide
class NextroEnv(gym.Env):
    def __init__(self, **kwargs):
        #load settings from supplied file
        self.settings = load_settings(loc=kwargs['c_args'].loc,
                                      manual_modify=kwargs['c_args'].man_mod)
        #if no file was supplied we load default settings manualy or automatically
        if self.settings == {}:
            self.settings = get_default_settings(kwargs['c_args'].man_mod)

        self.robot = NextroBot(model_urdf=NEXTRO_LOC,
                               robot_name='nextro',
                               settings=self.settings,
                               com_args=kwargs['c_args'],
                               basePosition=[0, 0, 0.1],
                               baseOrientation=[0, 0, 0, 1])

        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space
        self.np_random, _ = gym.utils.seeding.np_random()

        self.device = 'cuda'
        self.state = None
        self.action = None
        self.reward = None
        self.done = True
        self._info = None
        self.render_env = None
        self._time_step = 1/self.settings['FRAMES_PER_SECOND']
        self.__name__ = 'Nextro'
        self._episode_length = self.settings['DEFAULT_MAX_TIME']

        self.c_args = kwargs['c_args']
        # set true after reset, set to false after episode ends
        self._init = False
        # keeps track of how long an episode has been runnning
        self._time_elapsed = 0
        self._old_dist_travelled = 0
        self.client = None
        self.new_observation = np.zeros(self.settings['SENZOR_OUTPUT_SIZE'])

        # will hold historic observation as FIFO buffer
        obs_size = self.settings['STORED_OBSERVATION_SIZE']
        prev_obs = self.settings['PREV_OBS_ON_INPUT']
        self.obs_buffer = deque(np.zeros(obs_size) for _ in range(prev_obs))

        self.pid_regs = []
        self.steps_taken = 0
        self.logging = kwargs['c_args'].logging

    # called first regardes of render mode, should not have to be called again
    # if the mode is not 'human'
    def render(self, **kwargs):
        if self.client is not None:
            # will make sure that in human render mode camera will follow robot
            if self.render_env:
                x, y, _ = self.robot.robot_body.get_position()
                # numbers chosen for a good angle can be changed if desired
                p.resetDebugVisualizerCamera(1.2, -145, -38, [x, y, 0])
            return

        # render is callled with param first time, initialization done here
        if kwargs['mode'] == 'human':
            self.render_env = True
            self.client = p.connect(p.GUI)
            self._time_delay = True

        elif self.client is None:
            self._time_delay = False
            self.client = p.connect(p.DIRECT)
            self.render_env = False
        # used by loadURDF(plane.urdf)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        return

    # reset is called when first creating the env and after each episode ends
    def reset(self):
        self._init = True
        self.done = False
        self.steps_taken = 0
        self._time_elapsed = 0
        self._old_dist_travelled = 0

        # during testing there is no point in making the episodes last different
        # times
        if self.c_args.mode == 'train':
            # frames should be used when these default times differ
            self._episode_length = np.random.uniform(self.settings['DEFAULT_MIN_TIME'],
                                                     self.settings['DEFAULT_MAX_TIME'])

        # render doesn't have to be called in direct mode by user but it should
        # still run at least once
        if self.render_env is None:
            self.render(mode='direct')

        # if the robot is loaded we only need to move it not add it again
        if self.robot.robot_body is not None:
            self.robot.robot_body.reset_position(self._original_position)
            self.robot.robot_body.reset_orientation(self._original_orientation)
            self._reset_joints()
        else:
            p.resetSimulation(self.client)
            p.setGravity(0, 0, -10)
            p.loadURDF("plane.urdf")
            self.robot.reset(p)
            # see above how these are used
            self._original_position = self.robot.robot_body.get_position()
            self._original_orientation = self.robot.robot_body.get_orientation()
            # move camera closer to robot
            p.resetDebugVisualizerCamera(1.2, -145, -38, [0, 0, 0])
            p.setTimeStep(self._time_step, self.client)

        obs_size = self.settings['STORED_OBSERVATION_SIZE']
        prev_obs = self.settings['PREV_OBS_ON_INPUT']
        self.obs_buffer = deque(np.zeros(obs_size) for _ in range(prev_obs))

        if self.settings['PID_ENABLED']:
            self.pid_regs = []
            p_params = self.settings['PID_PARAMS']

            for idx in range(self.settings['NUM_JOINTS']):
                self.pid_regs.append(PID(*p_params, self._time_step))
        self.state = self._get_state()
        return self.state

    def _preproc_action(self, action):
        # clipping to make sure we don't set the motors to weird angles
        action = np.clip(action, a_min=-1.4, a_max=1.4)
        pid_action = 0

        # by default motors are not mirrored so negative angle on one side
        # is positive angle on the other, mirroring is done here
        for i in range(self.settings['NUM_JOINTS']//2):
            action[i] *= -1
        if self.settings['PID_ENABLED']:
            for idx in range(self.settings['NUM_JOINTS']):
                pid_action = self.pid_regs[idx].update(action[idx],
                                                       self.new_observation[idx])
                action[idx] = self.new_observation[idx] + pid_action
        return action

    def step(self, raw_action):
        if not self._init:
            raise Exception('Initialize the environment first!')
        if self.done:
            raise Exception('Reset environment after episode ends!')
        # TODO: Here I could potentially add PD regulator
        self.action = self._preproc_action(raw_action)
        self._set_joints(self.action)
        self._time_elapsed += self._time_step

        # the order is important here don't change it
        p.stepSimulation()
        self.state = self._get_state()
        self.reward = self._get_reward_update_done()
        self._update_buffer(self.action)

        # time delay only set in human render mode
        if self._time_delay:
            time.sleep(self._time_step)
        self.steps_taken += 1

        return self.state, self.reward, self.done, self._info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # pop old observation from the buufer and push new in
    def _update_buffer(self, action):
        # part of historic observation is action that was taken
        obs_2_store = np.concatenate((self.new_observation, action))
        self.obs_buffer.pop()
        self.obs_buffer.appendleft(obs_2_store)

    # the reward is calculated as postion change in 2D space (x, y axis)
    # plus the discounted total distance from starting position
    def _get_reward_update_done(self):
        x, y, z, a, b, c, d = self.robot.robot_body.get_pose()
        yaw, pitch, roll = p.getEulerFromQuaternion((a, b, c, d))

        # if the robot turns over end episode and give bad reward
        if abs(yaw) > (np.pi/2):
            self.done = True
            self._init = False
            self._time_elapsed = 0
            return -50
        # end episode if enough time has elapsed
        if self._time_elapsed >= self._episode_length:
            self.done = True
            self._init = False
            self._time_elapsed = 0

        # original position might not have been exactly [0,0] so adjust
        # current coordinates
        x = x - self._original_position[0]
        y = y - self._original_position[1]
        num_joints = self.settings['NUM_JOINTS']
        curr_velocities = self.new_observation[num_joints:(2*num_joints)]

        # distance reward besaed on change of distance between steps
        if self.settings['DIFF_DIST_REW']:
            if self.settings['PUNISH_Y']:
                new_dist_travelled = x**2 - y**2
            else:
                new_dist_travelled = x**2
            dist_change = new_dist_travelled - self._old_dist_travelled
            self._old_dist_travelled = new_dist_travelled
            dist_param = dist_change

        elif self.settings['TOTAL_DIST_REW']:
            dist_param = x**2


        if self.settings['PREV_OBS_ON_INPUT'] > 1:
            prev_obs = self.settings['PREV_OBS_ON_INPUT']
            prev_velocities = self.obs_buffer[prev_obs-1][num_joints:(2*num_joints)]
            joint_accel = (curr_velocities - prev_velocities) / (prev_obs*self._time_step)
        else:
            joint_accel = curr_velocities

        # this is better form when calculating reward as it punishes sudden movements
        joint_accel = np.linalg.norm(joint_accel)

        reward = self.settings['DIST_CONST']*dist_param \
            - self.settings['YPR_CONST']*(abs(yaw) + abs(pitch) + abs(roll)) \
            - self.settings['ACCEL_CONST']*joint_accel

        if self.steps_taken % 150 == 0 and self.logging:
            print("PREV VELO:")
            print(prev_velocities)
            print("CURR VELO:")
            print(curr_velocities)
            print("DIFFERENCE")
            print((curr_velocities - prev_velocities))
            print("FINAL IS:")
            print(joint_accel)
            print("YAW, PITCH, ROLL")
            print(yaw, pitch, roll)
            print('REWARD')
            print(reward)

        return reward

    # get the current angle of every joint and return them as numpy array
    def _get_state(self):
        if not self._init:
            raise Exception('Initialize the environment first')
        _, _, _, a, b, c, d = self.robot.robot_body.get_pose()
        body_angles = p.getEulerFromQuaternion((a, b, c, d))
        joint_angles = []
        joint_velocities = []

        for joint in self.settings['JOINT_NAMES']:
            joint_handle = self.robot.jdict[joint]
            joint_angles.append(joint_handle.get_position())
            joint_velocities.append(joint_handle.get_velocity())

        #the fixed [0,0,0,0] will later be used as control bits
        self.new_observation = np.concatenate((joint_angles,
                                               joint_velocities,
                                               body_angles,
                                               [1,0,0,0]))
        unrolled_buffer = np.ndarray.flatten(np.array(self.obs_buffer))

        return np.concatenate((self.new_observation, unrolled_buffer))

    # all joints reset to 0 position, overrides simulation constraints
    def _reset_joints(self):
        for joint in self.settings['JOINT_NAMES']:
            self.robot.jdict[joint].reset_position(0, 0)

    # set joints accoring to the ordered list of joint angles
    def _set_joints(self, angles):
        self.robot.set_all_joints(angles)
        #  for idx, joint in enumerate(JOINT_NAMES):
        #      self.robot.jdict[joint].set_position(angles[idx])

    def store_settings(self, dir_name):
        dir_names = os.listdir('./runs')
        old_name = os.path.join('./runs', dir_name)
        new_name = 'run_0'
        i = 1
        while new_name in dir_names:
            new_name = 'run_' + str(i)
            i += 1
        new_name = os.path.join('runs', new_name)
        shutil.move(old_name, new_name)
        save_default_settings(self.settings, new_name)
        print('-------------------')
        print(f'Settings saved tp {new_name}')
        print('-------------------')
