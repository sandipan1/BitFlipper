import gym
from baselines import deepq
import dqn_her.baselines.deepq.dqn_her as her
import gym_BitFlipper
import numpy as np
from gym.envs.registration import register 
import os
import tensorflow as tf


def callback(lcl, _glb):
    #for deepq training
    #stop training when mean reward for last 100 episodes <= (reward_max - reward_dist)
    is_solved = (lcl['saved_mean_reward_diff']!=None) and (lcl['saved_mean_reward_diff']<=lcl['env'].n/100*3)
    return is_solved

def make_env(n=10,space_seed=0):
  # create environment
  id = "BitFlipper"+str(n)+":"+str(space_seed)+"-v0"
  try :
    register(id=id,entry_point='gym_BitFlipper.envs:BitFlipperEnv',kwargs = {"space_seed":space_seed,"n":n})
  except :
    print("Environment with id = "+id+" already registered.Continuing with that environment.")
  env=gym.make(id)
  env.seed(0)
  return env

def train(env,save_path,optimisation_factor=1,buffer_factor=1,target_freq=1,batch_factor=1,gamma=0.98):
  #train deepq agent on env
  #agent has 1 mlp hidden layer with 256 units
  a=deepq.models.mlp([256])
  act = her.learn(env,q_func=a,lr=1e-3,max_timesteps=80000*env.n,buffer_size=1000000*buffer_factor,exploration_fraction=0.05,
      exploration_final_eps=0.01,train_freq=1,batch_size=128*batch_factor,gamma=gamma,
      print_freq=200,checkpoint_freq=100,target_network_update_freq=16*target_freq,num_optimisation_steps=env.n*optimisation_factor,
      callback=callback)
  #save trained model 
  print("Saving model to "+save_path)
  act.save(save_path)

def test(env,load_path,num_episodes=10000):
  act = deepq.load(load_path+".pkl")
  success_count=0.0
  test_render_file = open(load_path+".txt","w")
  for i in range(num_episodes):
      obs, done = env.reset(seed=i), False
      env_string = "Initial State: "+str((env.initial_state).T)+"\nGoal State: "+str((env.goal).T)+"\nMax_reward: "+str(env.reward_max)+"\n"
      print(env_string)
      test_render_file.write(env_string)  
      episode_rew = 0.0
      while not done:
          render_string = env.render(mode='ansi')+"\n"
          test_render_file.write(render_string)  
          obs, rew, done, _ = env.step(act(np.concatenate([obs,env.goal])[None])[0])
          episode_rew += rew
      render_string = env.render(mode='ansi')+"\n"
      test_render_file.write(render_string)
      if(episode_rew > -env.n):
        print("Episode successful with reward ",episode_rew)
        test_render_file.write("Episode successful with reward "+str(episode_rew)+"\n")
        success_count+=1.0
      else:
        print("Episode unsuccessful with reward ",episode_rew)
        test_render_file.write("Episode unsuccessful with reward "+str(episode_rew)+"\n")
  success_rate = success_count/num_episodes
  print("Success Rate: ",success_rate)
  test_render_file.write("Success Rate: "+str(success_rate)+"\n")
  test_render_file.close()
  return success_rate

def main(n_list=[5,10],  space_seed_list=[0],num_episodes=10000,save_path="./",batch_factor=1,optimisation_factor=1,buffer_factor=1,target_freq=1,gamma=0.98):
  test_results_file = open(save_path+"test_results.txt","w")
  for n in n_list:
    for space_seed in space_seed_list:
        print("started for "+str(n)+","+str(space_seed))
        env = make_env(n,space_seed)
        filename = "her:bitflip"+str(n)+":"+str(space_seed)
        with tf.Graph().as_default():
            train(env,save_path+filename+".pkl",optimisation_factor,buffer_factor,target_freq,batch_factor,gamma)
        with tf.Graph().as_default():
            success_rate = test(env,save_path+filename,num_episodes) 
            test_results_file.write("Bits :"+str(n)+","+"Seed :"+str(space_seed)+","+"Success :"+str(success_rate)+"\n")
  test_results_file.close()
