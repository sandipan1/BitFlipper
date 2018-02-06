# BitFlipper
BitFlipper environment in OpenAI gym format


## Problem Statement
lets say there is a n size binary array (e.g [1,0,0,0,1,1]) and we want to get different array (e.g [0,0,0,1,1,0]) of the same size. Only actions actions allowed are single bit flip at i th position and only get a reward of 0 if the final state is achieved. Otherwise you get a reward -1.

initial=[1,0,0,0,1,1]
 flip at index 2 =[1,0,1,0,1,1]
reward=-1

 flip at index 5=[1,0,1,0,1,0]
 reward=-1 
 
 The goal is to make a agent learn to achieve final state given a initial state.The only observation the agent get are the reward and current state.
 
