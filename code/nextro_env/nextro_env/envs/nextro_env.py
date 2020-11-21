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
from pid import PID


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

#number of prevous observations supplied on the input
PREV_OBS_ON_INPUT = 5

#observation consists of current information and a number of observations from history
OBSERVATION_SIZE = CURRENT_SENZOR_OUTPUT_SIZE + STORED_OBSERVATION_SIZE*PREV_OBS_ON_INPUT

JOINT_MOVEMNT_COST_DISCOUNT = 0.2

#listed to remove ambiguity when addressing joints
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
        self._time_step = 1/FRAMES_PER_SECOND
        self.__name__ = 'Nextro'
        self._episode_length = DEFAULT_MAX_TIME

        #set true after reset, set to false after episode ends
        self._init = False
        #keeps track of how long an episode has been runnning
        self._time_elapsed = 0
        self._old_dist_travelled = 0
        self.client = None
        self.new_observation = np.zeros(CURRENT_SENZOR_OUTPUT_SIZE)
        #will hold historic observation as FIFO buffer
        self.obs_buffer = deque(
            np.zeros(STORED_OBSERVATION_SIZE) for _ in range(PREV_OBS_ON_INPUT))

        self.pid_regs = []
        for idx in range(NUM_JOINTS):
            self.pid_regs.append(PID(0.3, 0, 0.003, self._time_step))



    #called first regardes of render mode, shouldn not have to be called again
    #if the mode is not 'human'
    def render(self, **kwargs):
        if self.client is not None:
            #will make sure that in human render mode camera will follow robot
            if self.render_env:
                x, y, _ = self.robot.robot_body.get_position()
                #numbers chosen for a good angle can be changed if desired
                p.resetDebugVisualizerCamera(1.2, -145, -38, [x ,y ,0])
            return

        #render is callled with param first time, initialization done here
        if kwargs['mode'] == 'human':
            self.render_env = True
            self.client = p.connect(p.GUI)
            self._time_delay = True

        elif self.client is None:
            self._time_delay = False
            self.client = p.connect(p.DIRECT)
            self.render_env = False
        #used by loadURDF(plane.urdf)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        return


    #reset is called when first creating the env and after each episode ends
    def reset(self):
        self._init = True
        self.done = False

        #render doesn't have to be called in direct mode by user but it should
        #still run at least once
        if self.render_env == None:
            self.render(mode='direct')

        #if the robot is loaded we only need to move it not add it again
        if self.robot.robot_body is not None:
            self.robot.robot_body.reset_position(self._original_position)
            self.robot.robot_body.reset_orientation(self._original_orientation)
            self._reset_joints()
        else:
            p.resetSimulation(self.client)
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
            np.zeros(STORED_OBSERVATION_SIZE) for _ in range(PREV_OBS_ON_INPUT))
        for idx in range(NUM_JOINTS):
            self.pid_regs.append(PID(0.3, 0, 0.003, self._time_step))
        self.state = self._get_state()
        return self.state


    def _preproc_action(self, action):
        #clipping to make sure we don't set the motors to weird angles
        action = np.clip(action, a_min=-1.4, a_max=1.4)
        pid_action = 0
        #by default motors are not mirrored so negative angle on one side
        #is positive angle on the other, mirroring is done here
        for i in range(NUM_JOINTS//2):
            action[i] *= -1
        for idx in range(NUM_JOINTS):
            pid_action = self.pid_regs[idx].update(action[idx],
                                                  self.new_observation[idx])
            action[idx] = self.new_observation[idx] + pid_action
        return action


    def step(self, raw_action):
        if not self._init:
            raise Exception('Initialize the environment first!')
        if self.done:
            raise Exception('Reset environment after episode ends!')
        #TODO: Here I could potentially add PD regulator
        self.action = self._preproc_action(raw_action)
        self._set_joints(self.action)
        self._time_elapsed += self._time_step

        #the order is important here don't change it
        p.stepSimulation()
        self.state = self._get_state()
        self.reward = self._get_reward_update_done()
        self._update_buffer(self.action)

        #time delay only set in human render mode
        if self._time_delay:
            time.sleep(self._time_step)

        return self.state, self.reward, self.done, self._info


    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]


    #pop old observation from the buufer and push new in
    def _update_buffer(self, action):
        #part of historic observation is action that was taken
        obs_2_store = np.concatenate((self.new_observation, action))
        self.obs_buffer.pop()
        self.obs_buffer.appendleft(obs_2_store)


    #the reward is calculated as postion change in 2D space (x, y axis)
    #plus the discounted total distance from starting position
    def _get_reward_update_done(self):
        x, y, z, a, b, c, d = self.robot.robot_body.get_pose()
        yaw, pitch, roll = p.getEulerFromQuaternion((a, b, c, d))

        #original position might not have been exactly [0,0] so adjust current coordinates
        x = x - self._original_position[0]
        y = y - self._original_position[1]
        #euclidean distance from the origin (x, y axis)
        new_dist_travelled = np.linalg.norm([x, y])
        dist_change = new_dist_travelled - self._old_dist_travelled
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

        prev_velocities = self.obs_buffer[0][18:36]
        curr_velocities = self.new_observation[18:36]
        joint_accel = (curr_velocities - prev_velocities) / self._time_step
        #this is better form when calculating reward as it punishes sudden movements
        joint_accel = np.linalg.norm(joint_accel)
        if self._time_elapsed == 100000*self._time_step:
            print("PREV VELO:")
            print(prev_velocities)
            print("CURR VELO:")
            print(curr_velocities)
            print("DIFFERENCE")
            print((curr_velocities - prev_velocities))
            print("FINAL IS:")
            print(joint_accel)

        reward = 1000*dist_change - 0.001*joint_accel  #- 0.005*joint_accel

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
