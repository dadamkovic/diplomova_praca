"""
This module serves to define a wrapper around the pybullet's URDFBasedRobot
class and to define a new gym class NextroEnv. Both of these were created to
allow reinforcement learning techniques to work with all package, and conform
to the stardard gym environment.
"""


import gym
from gym.utils import seeding
import time
import pybullet as p
from pybullet_envs import robot_bases as rb
import pybullet_data
import numpy as np
from collections import deque

#the urdf location has to be absolute path
NEXTRO_LOC = __file__.replace('envs/nextro_env.py','assets/urdf/nextro.urdf')
#default maximal length in seconds of a single episode
DEFAULT_MAX_TIME = 20
#defines the FPS that the simulation will run at
FRAMES_PER_SECOND = 30

NUM_JOINTS = 18
#(NUM_JOINTS joint angles + NUM_JOINTS joint velocities + yaw + pitch + roll)
CURRENT_SENZOR_OUTPUT_SIZE = (NUM_JOINTS*2 + 3)
#((NUM_JOINTS old joint angles + NUM_JOINTS old joint velocities + NUM_JOINTS previous joint angles) +
#(yaw + pitch + roll))
STORED_OBSERVATION_SIZE = NUM_JOINTS*3 + 3
#4 previous moves
#TODO: maybe set them spaced 2 frames from each other
OBSERVATION_SIZE = CURRENT_SENZOR_OUTPUT_SIZE + STORED_OBSERVATION_SIZE*4
#number of motorized joints on a robot


JOINT_MOVEMNT_COST_DISCOUNT = 0.2

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

#exists mostly to define the calc_state and robot_specific_reset methods that
#are missing in the URDFBasedRobot definition
class NextroBot(rb.URDFBasedRobot):
    def __init__(self,
               model_urdf,
               robot_name,
               action_dim,
               obs_dim,
               basePosition=[0, 0, 0],
               baseOrientation=[0, 0, 0, 1],
               fixed_base=False,
               self_collision=True):
        super().__init__(model_urdf, robot_name, action_dim,
                                   obs_dim, basePosition,baseOrientation,
                                   fixed_base,self_collision)

    #only called once when adding the robot urdf into environment
    #the 3 fixed joints are removed from the list of joints and named dictionary
    #with joints as values
    def robot_specific_reset(self, _p):
        keylist = list(self.jdict.keys())
        for key in keylist:
            if key.endswith('jointfix'):
                fixed_joint = self.jdict[key]
                del self.jdict[key]
                self.ordered_joints.remove(fixed_joint)

    #by default called in the parent method after adding urdf, is only here to
    #prevent errors
    def calc_state(self):
        return 0

#defines all the methods that a gym environment has to provide
class NextroEnv(gym.Env):
    def __init__(self):
        self.robot = NextroBot(model_urdf=NEXTRO_LOC,
                               robot_name='nextro',
                               action_dim=NUM_JOINTS,
                               obs_dim=OBSERVATION_SIZE,
                               basePosition=[0,0,0.1],
                               baseOrientation=[0,0,0,1])

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

        #set true after reset, set to false after episode ends
        self._init = False
        self._time_step = 1/FRAMES_PER_SECOND
        self.__name__ = 'Nextro'
        self._episode_length = DEFAULT_MAX_TIME
        #keeps track of how long an episode has been runnning
        self._time_elapsed = 0
        self._old_dist_travelled = 0
        self.client = None
        self.new_observation = np.zeros(CURRENT_SENZOR_OUTPUT_SIZE)
        self.obs_buffer = deque(
            np.zeros(STORED_OBSERVATION_SIZE) for i in range(4))

    #reset is called when first creating the env and after each episode ends
    def reset(self):
        self._init = True
        self.done = False

        if self.render_env == None:
            self._time_delay = False
            self.client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.render_env = False

        #if the robot is loaded we only need to move it not add it again
        if self.robot.robot_body is not None:
            self.robot.robot_body.reset_position(self._original_position)
            self.robot.robot_body.reset_orientation(self._original_orientation)
            self._reset_joints()
        else:
            p.resetSimulation(self.client)
            p.setTimeStep(self._time_step, self.client)
            p.setGravity(0, 0, -10)
            p.loadURDF("plane.urdf")
            self.robot.reset(p)
            #see above how these are used
            self._original_position = self.robot.robot_body.get_position()
            self._original_orientation = self.robot.robot_body.get_orientation()
            #move camera closer to robot
            p.resetDebugVisualizerCamera(1.2, -145, -38, [0,0,0])
            p.setTimeStep(self._time_step, self.client)
        #reset episode timer
        self._time_elapsed = 0
        #reset baseline position
        self._old_dist_travelled = 0
        self.obs_buffer = deque(
            np.zeros(STORED_OBSERVATION_SIZE) for i in range(4))
        self.state = self._get_state()
        return self.state


    #no need for this method
    def render(self, **kwargs):
        if self.client is not None:
            #will make sure that in human render mode camera will follow robot
            if self.render_env:
                x, y, _ = self.robot.robot_body.get_position()
                p.resetDebugVisualizerCamera(1.2, -145, -38, [x ,y ,0])
            return
        if kwargs['mode'] == 'human':
            self.render_env = True
            self.client = p.connect(p.GUI)
            self._time_delay = True
        #TODO:stuff bellow could be potentially removed
        else:
            self._time_delay = False
            self.client = p.connect(p.DIRECT)
            self.render_env = False
        #used by loadURDF(plane.urdf)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        return


    def step(self, action):
        if not self._init:
            raise Exception('Initialize the environment first!')
        if self.done:
            raise Exception('Reset environment after episode ends!')
        #TDOD: Here I could potentially add PD regulator
        #
        #
        #clipping to make sure we don't set the motors to weird angles
        action = np.clip(action, a_min=-1.4, a_max=1.4)
        self._set_joints(action)
        self._time_elapsed += self._time_step

        self.state = self._get_state()
        self.reward = self._get_reward_update_done()

        p.stepSimulation()
        self._update_buffer(action)

        #time delay only set in human render mode
        if self._time_delay:
            time.sleep(self._time_step)

        return self.state, self.reward, self.done, self._info

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]


    def _update_buffer(self, action):
        obs_2_store = np.concatenate((self.new_observation, action))
        self.obs_buffer.pop()
        self.obs_buffer.appendleft(obs_2_store)


    #the reward is calculated as postion change in 2D space (x, y axis)
    #plus the discounted total distance from starting position
    def _get_reward_update_done(self):
        x, y, z, a, b, c, d = self.robot.robot_body.get_pose()
        yaw, pitch, roll = p.getEulerFromQuaternion((a, b, c, d))
        #original position was not exactly [0,0] so adjust current coordinates
        x = x - self._original_position[0]
        y = y - self._original_position[1]
        #euclidean distance from the origin (x, y axis)
        new_dist_travelled = np.linalg.norm([x, y])
        reward = -1
        if new_dist_travelled > self._old_dist_travelled:
            reward = 1
            self._old_dist_travelled = new_dist_travelled

        #if the robot turns over end episode and give bad reward
        if abs(yaw) > (np.pi/2):
            self.done = True
            self._init = False
            self._time_elapsed = 0
            return -500
        #end episode if it is at the end
        if self._time_elapsed >= self._episode_length:
            self.done = True
            self._init = False
            self._time_elapsed = 0

        return reward


    #get the current angle of every joint and return them as numpy array
    def _get_state(self):
        if not self._init:
            raise Exception('Initialize the environment first')
        _, _, _, a, b, c, d = self.robot.robot_body.get_pose()
        body_angles = p.getEulerFromQuaternion((a, b, c, d))
        joint_angles = []
        joint_velocities = []

        for joint in JOINT_NAMES:
            joint_handle = self.robot.jdict[joint]
            joint_angles.append(joint_handle.get_position())
            joint_velocities.append(joint_handle.get_velocity())

        self.new_observation = np.concatenate((joint_angles,
                                               joint_velocities,
                                               body_angles))
        unrolled_buffer = np.ndarray.flatten(np.array(self.obs_buffer))

        return np.concatenate((self.new_observation, unrolled_buffer))


    #all joints reset to 0 position, overrides simulation constraints
    def _reset_joints(self):
        for joint in JOINT_NAMES:
            self.robot.jdict[joint].reset_position(0, 0)


    #set joints accoring to the ordered list of joint angles
    def _set_joints(self, angles):
        for idx, joint in enumerate(JOINT_NAMES):
            self.robot.jdict[joint].set_position(angles[idx])
