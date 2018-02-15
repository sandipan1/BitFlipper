# BitFlipper
BitFlipper environment in OpenAI gym format


## Problem Statement
lets say there is a n size binary array (e.g [1,0,0,0,1,1]) and we want to get different array (e.g [0,0,0,1,1,0]) of the same size. Only actions actions allowed are single bit flip at i th position and only get a reward of 0 if the final state is achieved. Otherwise you get a reward -1.

initial=[1,0,0,0,1,1]<br>
 flip at index 2 =[1,0,1,0,1,1]<br>
reward=-1

 flip at index 5=[1,0,1,0,1,0]<br>
 reward=-1 
 
 The goal is to make a agent learn to achieve final state given a initial state.The only observation the agent get are the reward and current state.
 
## Steps to run
Clone the repo<br>
In a conda env / virtualenv :<br> `pip install -e .`
<br>
To run DQN on BitFlipper environment call main() from dqn.py

To run DQN+HER  on BitFlipper environment call main() from dqn_her.py

### Related papers:
Deep Q Networks :http://www.davidqiu.com:8888/research/nature14236.pdf 
Hindsight Experience Replay:https://arxiv.org/pdf/1707.01495.pdf
