from gymnasium.envs.registration import register

register(
    id="gymnasium_env/MazeEnv-v0",
    entry_point="gymnasium_env.envs:MazeEnv",
)

register(
    id="gymnasium_env/MazeEnv-v1",
    entry_point="gymnasium_env.envs:VariableMazeEnv",
)