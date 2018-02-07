import gym
from baselines import deepq
from gym.envs.registration import register 
def callback(lcl, _glb):  ##  make changes
  
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved
  
def main():
  # create environment
  space_seed = 1
  n = 15
  id = "BitFlipper"+str(n)+":"+str(space_seed)+"-v0"
  register(
    id=id,
    entry_point='BitFlipper.gym_BitFlipper.envs:BitFlipperEnv',
    kwargs = {"space_seed":space_seed,"n":n}
  )
  env=gym.make(id)
  
  # learning agent
  a=deepq.models.mlp([256])
  act = deepq.learn(env,q_func=a,lr=1e-3,max_timesteps=10000000,buffer_size=50000,exploration_fraction=0.1,
      exploration_final_eps=0.05,train_freq=1,batch_size=64,
      print_freq=500,checkpoint_freq=1000,callback=callback)
  
  print("Saving model to bitflip.pkl")
  act.save("bitflip.pkl")
  
  
if __name__=='__main__':
  main()
