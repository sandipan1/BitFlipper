from gym.envs.registration import register

register(
    id='Bitflipper-v0',
    entry_point='gym_BitFlipper.envs:BitFlipperEnv',
)
