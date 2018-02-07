import numpy as np
import gym
from gym import spaces

class BitFlipperEnv(gym.Env):
  '''Bit Flipping environment
      The state space is binary strings of length n.
      The action space is an index i from {0,1...n-1} which represents the agent flipping ith bit of the environment 
      Given an initial state the agent has to reach a goal state.
      Reward: Only goal state has reward 0,rest all states have reward -1
  '''
  
  def __init__(self,n,space_seed):
    self.n=n    
    self.action_space = spaces.Discrete(self.n)
    self.observation_space = spaces.MultiBinary(self.n)
    self.reward_range = (-1,0)
    spaces.seed(space_seed)
    self.initial_state = self.state_space.sample()
    self.goal = self.state_space.sample()  
    self.state = self.initial_state
    self.envstepcount = 0
    
  def _step(self,action):
    '''
     accepts action and returns obs,reward, b_flag(episode start), info dict(optional)
    '''
    self.state = self._bitflip(action)  ## computes s_t1
    reward = self._calculate_reward()
    self.envstepcount += 1
    done = self._compute_done(reward)
    return  np.array(self.state),reward,done,{}

  def _reset():  
    self.state=self.initial_state
    return self.state
  
  def _close():
    pass
  
  def _render(self, mode='human', close=False):
    pass 
  
  def _seed(self,seed):
    pass
  
  def _bitflip(self,index):
    s2=np.array(self.state)
    s2[index] = not s2[index]
    return s2
  
  def _calculate_reward(self):
    if(self.goal==self.state):
      return 0
    else:
      return -1
    
  def _compute_done(self,reward):
    if(reward==0 or self.envstepcount >=100):
      return True
    else:
      return False
