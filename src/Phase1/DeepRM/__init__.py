from gym.envs.registration import register

register(
    id='deeprm-v0',
    entry_point='DeepRM.envs:DeepEnv',
)

register(
    id='MB_DeepRM-v0',
    entry_point='DeepRM.envs:Env',
)
