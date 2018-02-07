from gym.envs.registration import register
#id='BitFlipper-n:space_seed'
register(
    id='BitFlipper-v0',
    entry_point='BitFlipper.gym_BitFlipper.envs:BitFlipperEnv',
    kwargs = {"space_seed":0,"n":10}
)

