from gym.envs.registration import register

register(
    id='BitFlipper-v0',
    entry_point='BitFlipper.gym_BitFlipper.envs:BitFlipperEnv',
    space_seed = 0,
    n = 10
)
