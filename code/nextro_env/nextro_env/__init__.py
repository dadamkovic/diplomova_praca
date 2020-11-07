from gym.envs.registration import register

register(
    id='nextro-v0',
    entry_point='nextro_env.envs:NextroEnv',
)

