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


#the urdf location has to be absolute path
NEXTRO_LOC = __file__.replace('envs/nextro_env.py','assets/urdf/nextro.urdf')
#default maximal length in seconds of a single episode
DEFAULT_MAX_TIME = 30
#defines the FPS that the simulation will run at
FRAMES_PER_SECOND = 30
#determines how big of a role total travelled distance plays in the reward
#higher is stronger
DISTANCE_DISCOUNT = 0.8

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
               self_collision=False):
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
    def __init__(self, render=False, episode_length=DEFAULT_MAX_TIME):       
        self.robot = NextroBot(NEXTRO_LOC, 'nextro', 36, 18, basePosition=[0,0,0.1],
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
        
        #set true after reset, set to false after episode ends
        self._init = False
        self._time_step = 1/FRAMES_PER_SECOND
        self.__name__ = 'Nextro'
        #used when calculating reward
        self._old_position = []
        self._episode_length = episode_length
        #keeps track of how long an episode has been runnning
        self._time_elapsed = 0
        self._old_dist_travelled = 0

        self.client = None
        
        


    #reset is called when first creating the env and after each episode ends
    def reset(self):
        self._init = True
        self.done = False
        
        #if the robot is loaded we only need to move it not add it again
        if self.robot.robot_body is not None:
            self.robot.robot_body.reset_position(self._original_position)
            self.robot.robot_body.reset_orientation(self._original_orientation)
            self._reset_joints()
        else:
            p.resetSimulation(self.client)
            p.setTimeStep(self._time_step,self.client)
            p.setGravity(0,0,-10)
            p.loadURDF("plane.urdf")
            self.robot.reset(p)
            #see above how these are used
            self._original_position = self.robot.robot_body.get_position()
            self._original_orientation = self.robot.robot_body.get_orientation()
        
        #reset episode timer
        self._time_elapsed = 0
        #reset baseline position
        self._old_dist_travelled = 0
        self.state = self._getstate()
        return self.state

    def step(self, action):
        if not self._init:
            raise Exception('Initialize the environment first!')
        if self.done:
            raise Exception('Reset environment after episode ends!')
            
        angles = []
        
        #joint angles are set using the gaussian distribution
        #actions are are paired (mu1, sigma1, mu2, sigma2,...) total 18*2 items
        for idx in range(0, self.action_space.shape[0], 2):
            angles.append(np.random.default_rng().normal(action[idx],
                                                         abs(action[idx+1]),
                                                         None))
        self._set_joints(angles)
        self._time_elapsed += self._time_step
        self.state = self._getstate()
        self.reward = self._get_reward_updatedone()
        p.stepSimulation()
        
        if self._time_delay:
            time.sleep(self._time_step)
        
        return self.state, self.reward, self.done, self._info

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]

    #the reward is calculated as postion change in 2D space (x, y axis)
    #plus the discounted total distance from starting position
    def _get_reward_updatedone(self):
        x, y, z, a, b, c, d = self.robot.robot_body.get_pose()
        angle = p.getEulerFromQuaternion((a, b, c, d))
        #if the robot turns over end episode and give bad reward
        if abs(angle[0]) > (np.pi/2):
            self.done = True
            self._init = False
            self._time_elapsed = 0
            return -5
        #end episode if it is at the end
        if self._time_elapsed >= self._episode_length:
            self.done = True
            self._init = False
            self._time_elapsed = 0
        
        #original position was not exactly [0,0] so adjust current coordinates
        x = x - self._original_position[0]
        y = y - self._original_position[1]
        #distance from the origin (x, y axis)
        self._new_dist_travelled = np.sqrt(x**2+y**2)
        position_change = self._new_dist_travelled - self._old_dist_travelled
        self._old_dist_travelled = self._new_dist_travelled

        #return position_change + DISTANCE_DISCOUNT*self._new_dist_travelled
        return self._new_dist_travelled
    
    #no need for this method
    def render(self, **kwargs):
        if self.client is not None:
            return
        if kwargs['mode'] == 'human':
            self.client = p.connect(p.GUI)
            self._time_delay = True
        else:
            self._time_delay = False
            self.client = p.connect(p.DIRECT)
        p.setTimeStep(self._time_step, self.client)
        #used by loadURDF(plane.urdf)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        return

    #get the current angle of every joint and return them as numpy array
    def _getstate(self):
        if not self._init:
            raise Exception('Initialize the environment first')
        joint_angles = []
        for joint in self.robot.ordered_joints:
            joint_angles.append(joint.get_position())
        return np.array(joint_angles)

    #all joints reset to 0 position
    def _reset_joints(self):
        self._set_joints(np.zeros(self.action_space.shape[0]//2))

    #set joints accoring to the ordered list of joint angles
    def _set_joints(self, angles):
        for idx, joint in enumerate(JOINT_NAMES):
            self.robot.jdict[joint].set_position(angles[idx])
