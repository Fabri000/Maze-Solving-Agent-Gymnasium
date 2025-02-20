from gymnasium.envs.registration import register

register(
    id="gymnasium_env/MazeEnv-v0",
    entry_point="gymnasium_env.envs:SimpleMazeEnv",
)

register(
    id="gymnasium_env/MazeEnv-v1",
    entry_point="gymnasium_env.envs:SimpleEnrichMazeEnv",
)

register(
    id="gymnasium_env/VariableMazeEnv-v0",
    entry_point="gymnasium_env.envs:SimpleVariableMazeEnv",
)

register(
    id="gymnasium_env/VariableMazeEnv-v1",
    entry_point="gymnasium_env.envs:SimpleEnrichVariableMazeEnv",
)
