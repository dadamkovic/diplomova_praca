import gym
from gym.utils import seeding
import time
import pybullet as p
from pybullet_envs import robot_bases as rb
from pybullet_envs import gym_locomotion_envs as gle
import pybullet_data
import numpy as np

NEXTRO_LOC = '/media/daniel/DanielStuff/diplomovaPraca/code/nextro_env/nextro_env/assets/urdf/nextro.urdf'
MAX_TIME = 10

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
        
    def robot_specific_reset(self, _p):
        keylist = list(self.jdict.keys())
        for key in keylist:
            if key.endswith('jointfix'):
                fixed_joint = self.jdict[key]
                del self.jdict[key]
                self.ordered_joints.remove(fixed_joint)

    def calc_state(self):
        return 0

        
class NextroEnv(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 30}
    def __init__(self, render=False):
        self.robot = NextroBot(NEXTRO_LOC, 'nextro', 36, 18, basePosition=[0,0,0.1],
                               baseOrientation=[0,0,0,1])
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space
        self.np_random, _ = gym.utils.seeding.np_random()
        
        self._time_step = 1/30
        self._init = False
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        
        self.client = p.connect(p.GUI)
        p.setTimeStep(self._time_step,self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF

        
    def reset(self):
        self._init = True
        self._done = False
        if self.robot.robot_body != None:
            self.robot.robot_body.reset_position(self._original_position)
            self.robot.robot_body.reset_orientation(self._original_orientation)
            self._reset_joints()
        else:
            p.resetSimulation(self.client)
            p.setTimeStep(self._time_step,self.client)
            self._time_elapsed = 0
            p.setGravity(0,0,-10)
            planeId = p.loadURDF("plane.urdf")
            self.robot.reset(p)
            self._state = self._get_state()
            self._original_position = self.robot.robot_body.get_position()
            self._original_orientation = self.robot.robot_body.get_orientation()
        
        
        return 
    
    def step(self, action):
        if not self._init:
            raise Exception('Initialize the environment first')
        angles = []
        for idx in range(0,self.action_space.shape[0],2):
            angles.append(np.random.default_rng().normal(action[idx],abs(action[idx+1]),None))
        
        self._set_joints(angles)
        
        self._time_elapsed += self._time_step
        self._state = self._get_state()  
        self._reward = self.reward()
        
        p.stepSimulation()
        time.sleep(self._time_step)
        
        if self._time_elapsed >= MAX_TIME:
            self._done = True
            self._init = False
            self._time_elapsed = 0
        
        return self._state, self._reward, self._done, self._info 
    
    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
    
    def reward(self):
        x,y,z = self.robot.robot_body.get_position()
        reward = np.sqrt(x**2+y**2)
        return reward
    
    def render(self):
        pass
    
    def _get_state(self):
        if not self._init:
            raise Exception('Initialize the environment first')
        joint_angles = [] 
        for joint in self.robot.ordered_joints:
            joint_angles.append(joint.get_position())
        return np.array(joint_angles)
    
    def _reset_joints(self):
        self._set_joints(np.zeros(self.action_space.shape[0]//2))
    
    def _set_joints(self,angles):
        for idx, joint in enumerate(self.robot.ordered_joints):
            joint.set_position(angles[idx])