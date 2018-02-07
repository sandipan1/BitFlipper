from gym.envs.registration import register
#id='BitFlipper-n:space_seed'
register(
    id='BitFlipper-10:0',
    entry_point='BitFlipper.gym_BitFlipper.envs:BitFlipperEnv',
    kwargs = {"space_seed":0,"n":10}
)

register(
    id='BitFlipper-15:0',
    entry_point='BitFlipper.gym_BitFlipper.envs:BitFlipperEnv',
    kwargs = {"space_seed":0,"n":15}
)
