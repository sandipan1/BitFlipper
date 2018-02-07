import gym
from baselines import deepq

def callback(lcl, _glb):  ##  make changes
  
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def main():
  # create environment
  print("changed without uninstall")
  env=gym.make('BitFlipper-v0')
  
  # learning agent
  a=deepq.models.mlp([256])
  act = deepq.learn(env,q_func=a,lr=1e-3,max_timesteps=100000,buffer_size=50000,exploration_fraction=0.1,
      exploration_final_eps=0.02,
      print_freq=10,
      callback=callback)
  
  print("Saving model to bitflip.pkl")
  act.save("bitflip.pkl")
  
  
if __name__=='__main__':
  main()
