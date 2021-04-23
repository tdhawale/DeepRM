from gym.envs.registration import register

register(
    id='MB_DeepRM-v0',
    entry_point='DeepRM.envs:Env',
)
