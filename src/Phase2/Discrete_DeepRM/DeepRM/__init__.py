from gym.envs.registration import register

register(
    id='deeprm-v0',
    entry_point='DeepRM.envs:DeepEnv',
)
